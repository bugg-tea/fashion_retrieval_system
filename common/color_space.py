"""
Shared CIELAB color utilities, used by BOTH the indexer (naming a dominant
garment RGB as e.g. "blue") and the retriever (scoring how close a wanted
color is to a found color, e.g. "blue" query vs a "navy" garment).

Why LAB instead of raw RGB (this was flagged as a known gap in the v1
writeup, and confirmed as the #1 easy win in the eval-run analysis):
  Euclidean distance in RGB does not track human color perception well --
  it under-weights hue differences and over-weights brightness differences,
  so e.g. "navy" (a dark blue) can end up numerically closer to "black" than
  to "blue" in raw RGB space even though a person would call it blue first.
  CIELAB was specifically designed so that Euclidean distance between two
  points approximates *perceptual* difference much more closely. Converting
  once here and reusing the same LAB table everywhere means the indexer's
  color *naming* and the retriever's color *similarity scoring* can never
  silently drift out of sync with each other.

No new dependency: this is a standard, well-documented sRGB -> XYZ -> CIELAB
conversion (D65 illuminant) implemented directly in numpy so we don't need to
pull in scikit-image / colormath / OpenCV just for one conversion.
"""
from functools import lru_cache

import numpy as np

from config import COLORS

# sRGB (D65) -> XYZ matrix, and the D65 reference white, both in the
# conventional 0-100 XYZ scale.
_M = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([95.0489, 100.0, 108.8840])
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def rgb_to_lab(rgb) -> np.ndarray:
    """Convert one or many sRGB triples (0-255 ints/floats) to CIELAB.

    Accepts shape (3,) or (..., 3); returns the same leading shape with a
    trailing (L, a, b) triple. Vectorized so it's cheap to call on whole
    pixel arrays, not just single averaged colors.
    """
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _M.T * 100.0
    t = xyz / _WHITE
    f = np.where(t > _EPS, np.cbrt(t), (_KAPPA * t + 16.0) / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_distance(lab_a, lab_b) -> float:
    """Plain Euclidean (CIE76) distance in LAB space. CIE76 is a coarser
    perceptual metric than CIEDE2000, but is symmetric, cheap, and more than
    adequate for "which broad color family is this" -- the same tradeoff the
    rest of this codebase makes elsewhere (e.g. BLIP over a bigger VLM)."""
    return float(np.linalg.norm(np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)))


# Reference palette, converted to LAB once at import time and cached -- every
# caller (indexer color naming, retriever color similarity) shares this exact
# table so the two stages can never disagree about what "blue" means.
@lru_cache(maxsize=1)
def _colors_lab_table():
    names = list(COLORS.keys())
    labs = rgb_to_lab(np.array([COLORS[n] for n in names], dtype=np.float64))
    return names, labs


# Empirically, two colors more than ~60 LAB units apart are essentially
# unrelated (e.g. black vs. white is ~100); used to normalise raw distances
# into a bounded [0, 1] confidence / similarity score.
_MAX_LAB_DIST = 100.0


def nearest_color_name(rgb):
    """Map a dominant RGB value to the closest name in config.COLORS.

    Returns (name, confidence in [0, 1]) or (None, 0.0) if rgb is None.
    """
    if rgb is None:
        return None, 0.0
    names, labs = _colors_lab_table()
    target = rgb_to_lab(rgb)
    dists = np.linalg.norm(labs - target, axis=-1)
    i = int(np.argmin(dists))
    confidence = max(0.0, 1.0 - float(dists[i]) / _MAX_LAB_DIST)
    return names[i], round(confidence, 3)


def color_name_similarity(wanted: str, found: str, falloff: float = 0.7) -> float:
    """Graded similarity between two *named* colors already in config.COLORS
    (e.g. "blue" query vs a "navy" garment), in LAB space.

    `falloff` controls how forgiving the soft match is: similarity hits 0 at
    a LAB distance of falloff * _MAX_LAB_DIST. Smaller falloff = stricter.
    """
    if not wanted or not found:
        return 0.0
    if wanted == found:
        return 1.0
    if wanted not in COLORS or found not in COLORS:
        return 0.0
    names, labs = _colors_lab_table()
    i, j = names.index(wanted), names.index(found)
    dist = lab_distance(labs[i], labs[j])
    return max(0.0, 1.0 - dist / (_MAX_LAB_DIST * falloff))
