"""
LEGACY / fallback garment color extraction, for the upper-body and lower-body
regions of a person via pose landmarks rather than real segmentation.

As of this version, indexer/garment_analysis.py (SegFormer clothes
segmentation) is the PRIMARY path and gives real per-pixel garment masks
instead of the proportional crop boxes this module uses -- see that module's
docstring for the rationale. This module now only runs when:
  - SegFormer can't be loaded (no internet to huggingface.co, or
    config.USE_SEGMENTATION=False / --legacy-color on the CLI), or
  - segmentation ran but found no usable person region for a given image
    (garment_analysis.py falls back to this, per-image, automatically).

Strategy (unchanged from v1):
1. Try MediaPipe Pose to locate shoulder/hip/knee landmarks -> precise crop boxes.
2. If pose detection fails (no person, or low landmark visibility), fall back to
   a fixed proportional crop (upper torso band / lower leg band of the image).
3. Within each crop, run KMeans on pixel colors, discard clusters that look like
   skin-tone or that match the image's corner pixels (proxy for background),
   and pick the largest remaining cluster as the garment color.
4. Map the resulting RGB value to the nearest name in config.COLORS using
   CIELAB distance (see common/color_space.py) rather than raw RGB distance --
   RGB Euclidean distance tracks human color perception poorly (e.g. "navy"
   can end up numerically closer to "black" than to "blue"), which was a
   flagged gap in the original version of this module.

This remains a heuristic, not a trained model -- kept around specifically so
the pipeline degrades gracefully instead of hard-failing when the
segmentation model isn't available.
"""
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from common.color_space import nearest_color_name as _lab_nearest_color_name

# Empirical LAB-distance threshold: below this, the "tie" strip's color is
# judged too similar to the shirt itself to be a real separate garment --
# we're almost certainly just re-detecting the shirt, not a tie.
_TIE_MIN_LAB_DIST = 18.0

try:
    import mediapipe as mp
    _mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=0, min_detection_confidence=0.4
    )
    _HAS_MEDIAPIPE = True
except Exception:
    _HAS_MEDIAPIPE = False


def _skin_like(rgb):
    r, g, b = rgb
    return r > 90 and g > 55 and b > 35 and r > b and 5 < (r - g) < 90


def _background_samples(full_image: Image.Image, border_frac=0.04, max_samples=300):
    """Sample pixels from a thin ring around the FULL image (not the crop) as
    the background reference. Using the crop's own corners (the old approach)
    breaks down when the person fills most of the crop -- those "corners" can
    already be garment fabric, so the real garment color gets discarded as
    "background" by mistake. The full-image border is a much safer bet for
    typical fashion photos (plain/studio backgrounds)."""
    arr = np.array(full_image.convert("RGB"))
    h, w, _ = arr.shape
    bh, bw = max(1, int(h * border_frac)), max(1, int(w * border_frac))
    ring = np.concatenate([
        arr[:bh, :, :].reshape(-1, 3), arr[-bh:, :, :].reshape(-1, 3),
        arr[:, :bw, :].reshape(-1, 3), arr[:, -bw:, :].reshape(-1, 3),
    ], axis=0)
    if len(ring) > max_samples:
        ring = ring[np.random.default_rng(0).choice(len(ring), max_samples, replace=False)]
    return ring


def _background_like(rgb, bg_samples, tol=24):
    if bg_samples is None or len(bg_samples) == 0:
        return False
    d = np.linalg.norm(bg_samples.astype(float) - np.array(rgb, dtype=float), axis=1)
    return bool(np.min(d) < tol)


def _dominant_color(crop: Image.Image, bg_samples, min_cluster_frac=0.06):
    if crop is None or crop.width < 4 or crop.height < 4:
        return None
    arr = np.array(crop.convert("RGB")).reshape(-1, 3)
    if len(arr) > 4000:
        idx = np.random.default_rng(0).choice(len(arr), 4000, replace=False)
        arr = arr[idx]
    # k=6 (was 4): finer-grained clusters separate garment from background/skin
    # more cleanly than a coarse 4-way split.
    k = min(6, len(np.unique(arr, axis=0)))
    if k < 1:
        return None
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(arr)
    counts = np.bincount(km.labels_, minlength=k)
    order = np.argsort(-counts)
    total = counts.sum()

    for i in order:
        if counts[i] / total < min_cluster_frac:
            continue  # too small a sliver of the crop to plausibly be the garment
        color = km.cluster_centers_[i]
        if _skin_like(color) or _background_like(color, bg_samples):
            continue
        return tuple(color.astype(int))
    return tuple(km.cluster_centers_[order[0]].astype(int))


def _center_strip(crop, frac=0.16):
    """A narrow vertical strip down the center of the upper-body crop --
    roughly where a necktie sits, if one is present."""
    if crop is None or crop.width < 8:
        return None
    w, h = crop.size
    half = max(2, int(w * frac / 2))
    left = max(0, int(w / 2 - half))
    right = min(w, int(w / 2 + half))
    bottom = int(h * 0.85)  # avoid the very bottom edge (more likely waistband/pants)
    if right - left < 4 or bottom < 4:
        return None
    return crop.crop((left, 0, right, bottom))


def _tie_color_guess(upper_crop, bg_samples, shirt_rgb):
    """
    HEURISTIC ONLY -- there is no dedicated "tie" class in either this
    fallback path or the SegFormer model used elsewhere in this pipeline, and
    no check for whether a tie is actually present. This assumes a roughly
    front-facing, mostly-unoccluded subject wearing a tie visibly different
    in color from their shirt -- it will legitimately miss or misfire on
    patterned ties, same-color ties, or non-frontal poses. Treat the result
    as a weak, opportunistic signal, not a confirmed detection.
    """
    import numpy as np
    if shirt_rgb is None:
        return None
    strip = _center_strip(upper_crop)
    if strip is None:
        return None
    strip_rgb = _dominant_color(strip, bg_samples, min_cluster_frac=0.10)
    if strip_rgb is None:
        return None
    from common.color_space import rgb_to_lab, lab_distance
    if lab_distance(rgb_to_lab(strip_rgb), rgb_to_lab(shirt_rgb)) < _TIE_MIN_LAB_DIST:
        return None  # too close to the shirt color to plausibly be a separate tie
    return strip_rgb


def _nearest_color_name(rgb):
    return _lab_nearest_color_name(rgb)


def _pose_crops(image: Image.Image):
    if not _HAS_MEDIAPIPE:
        return None
    arr = np.array(image.convert("RGB"))
    h, w, _ = arr.shape
    result = _mp_pose.process(arr)
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    try:
        l_sh, r_sh = pt(11), pt(12)
        l_hip, r_hip = pt(23), pt(24)
        l_knee, r_knee = pt(25), pt(26)
    except IndexError:
        return None

    if not all(lm[i].visibility > 0.3 for i in [11, 12, 23, 24]):
        return None

    x_min = max(0, int(min(l_sh[0], r_sh[0], l_hip[0], r_hip[0])))
    x_max = min(w, int(max(l_sh[0], r_sh[0], l_hip[0], r_hip[0])))
    # Inset inward (was padded outward by 10px) -- the old outward padding
    # pulled in background/arms at the sides, which is exactly the kind of
    # contamination that made KMeans pick a background-tinted "gray" instead
    # of the actual "blue" shirt.
    box_w = x_max - x_min
    inset = int(box_w * 0.10)
    x_min, x_max = x_min + inset, x_max - inset
    upper_top = max(0, int(min(l_sh[1], r_sh[1]) + 0.05 * (max(l_hip[1], r_hip[1]) - min(l_sh[1], r_sh[1]))))
    upper_bottom = int(max(l_hip[1], r_hip[1]))
    lower_bottom = min(h, int(max(l_knee[1], r_knee[1]) + 15))

    if x_max - x_min < 5 or upper_bottom - upper_top < 5:
        return None

    upper = image.crop((x_min, upper_top, x_max, max(upper_bottom, upper_top + 5)))
    lower = None
    if lower_bottom > upper_bottom:
        lower = image.crop((x_min, upper_bottom, x_max, lower_bottom))
    return upper, lower


def _fallback_crops(image: Image.Image):
    w, h = image.size
    upper = image.crop((int(w * 0.15), int(h * 0.30), int(w * 0.85), int(h * 0.60)))
    lower = image.crop((int(w * 0.15), int(h * 0.55), int(w * 0.85), int(h * 0.90)))
    return upper, lower


def extract_garment_colors(image: Image.Image) -> dict:
    """
    Returns a dict with the SAME shape as garment_analysis.GarmentAnalyzer's
    output (garment_colors/accessories/segmentation_labels are always
    present, just empty here) so build_index.py and feature_extractor.py can
    merge either source without caring which one actually ran:
        {
          "upper_color": str | None, "upper_color_conf": float,
          "lower_color": str | None, "lower_color_conf": float,
          "crop_method": "pose" | "fallback",
          "garment_colors": {},       # legacy path doesn't produce per-garment colors
          "accessories": [],          # legacy path can't detect bags/scarves/etc.
          "segmentation_labels": [],
        }
    """
    crops = _pose_crops(image)
    method = "pose"
    if crops is None:
        crops = _fallback_crops(image)
        method = "fallback"

    upper_crop, lower_crop = crops
    bg_samples = _background_samples(image)
    upper_rgb = _dominant_color(upper_crop, bg_samples)
    upper_name, upper_conf = _nearest_color_name(upper_rgb)
    lower_name, lower_conf = _nearest_color_name(_dominant_color(lower_crop, bg_samples))

    garment_colors = {}
    tie_rgb = _tie_color_guess(upper_crop, bg_samples, upper_rgb)
    if tie_rgb is not None:
        tie_name, _ = _nearest_color_name(tie_rgb)
        if tie_name:
            garment_colors["Tie"] = tie_name

    return {
        "upper_color": upper_name,
        "upper_color_conf": upper_conf,
        "lower_color": lower_name,
        "lower_color_conf": lower_conf,
        "crop_method": method,
        "garment_colors": garment_colors,
        "accessories": [],
        "segmentation_labels": [],
    }
