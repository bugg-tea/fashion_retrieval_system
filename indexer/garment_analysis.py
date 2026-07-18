"""
Per-garment color & accessory extraction using human-parsing segmentation.

This is the direct fix for the two concrete color/attribute bugs surfaced in
the v1 eval run:
  - "a blue shirt got read as gray" -- because the old MediaPipe-pose crop
    was a rough proportional torso box that mixed in background and skin
    pixels along with the actual shirt fabric.
  - "no separate garment metadata" -- upper/lower was the finest granularity
    available; a belt, bag, scarf, or hat had no dedicated field at all.

How it works:
    mattmdjaga/segformer_b2_clothes is a SegFormer-B2 model fine-tuned for
    18-class human parsing (background/hair/face/limbs + Hat, Sunglasses,
    Upper-clothes, Skirt, Pants, Dress, Belt, Bag, Scarf, shoes). Feeding it
    one image gives a per-pixel label map, so we can sample the actual
    garment pixels for each class directly -- no crop-box guessing needed.
    ~28M parameters, runs on CPU in well under a second per image at the
    reduced resolution this module uses (config.SEGMENTATION_MAX_SIDE).

    We deliberately do NOT upsample the model's output back to full image
    resolution (the common approach for visualization) -- since we only need
    an average garment *color*, not a pixel-perfect mask, we instead
    downsample the ORIGINAL image to match the model's native output
    resolution and index directly into that. This skips a bilinear-upsample
    over the full photo for every single image, which is pure waste for our
    use case and meaningfully faster on CPU.

Reliability / fallback (the same philosophy as the rest of this codebase --
an enhancement model must never turn into a hard crash):
  - If the model can't be loaded at all (no internet to huggingface.co,
    disk full, etc.) the analyzer disables itself ONCE at construction time
    (with a single clear warning) and every image goes through the legacy
    indexer/color_utils.py MediaPipe-pose path instead.
  - If segmentation runs but finds no usable garment region for a specific
    image (e.g. a product-flatlay photo with no person in it, or the model
    just gets it wrong), that ONE image individually falls back to the
    legacy path -- the rest of the batch is unaffected.
  - mock=True (see indexer/feature_extractor.py's FeatureExtractor) skips
    model loading entirely and always uses the legacy path, so the full
    pipeline can be exercised offline with no Hugging Face access at all.

Known limitations (documented rather than silently hidden -- see also
WRITEUP.md "Limitations"):
  - The underlying model's own published per-class accuracy is weakest on
    small/thin classes: "Belt" (~35% acc) and "Scarf" (~63% acc) in
    particular. SEGMENTATION_MIN_MASK_FRAC filters out tiny/spurious
    detections, but false negatives (a real belt that's missed) are more
    likely than false positives for these two classes specifically.
  - The model does not have a dedicated "tie" class -- a necktie is not
    separated from "Upper-clothes", so a query like "red tie" still can't
    get a garment-specific tie color the way "red scarf" now can. This is a
    genuine remaining gap; a full solution would need either a differently
    labelled parsing model or a small object detector, which is out of
    scope for a free/CPU-only pipeline (see WRITEUP.md).
  - Trained primarily on standing, mostly-unoccluded people (ATR dataset);
    accuracy degrades on seated/heavily-occluded poses, same caveat as the
    MediaPipe-pose approach it replaces.
"""
import numpy as np
from PIL import Image

from config import (
    DEVICE, SEGMENTATION_MODEL_NAME, USE_SEGMENTATION, SEGMENTATION_MAX_SIDE,
    SEGMENTATION_MIN_MASK_FRAC, SEGMENTATION_LABELS, GARMENT_COLOR_LABELS,
)
from common.color_space import nearest_color_name
from indexer.color_utils import extract_garment_colors as _legacy_extract

_EMPTY_RESULT_EXTRA = {"garment_colors": {}, "garment_color_confidence": {}, "accessories": [], "segmentation_labels": []}


def _looks_like_real_labels(id2label: dict) -> bool:
    """A freshly-constructed (non-finetuned) Segformer config defaults to
    id2label = {0: "LABEL_0", 1: "LABEL_1", ...} -- guards against silently
    trusting that placeholder mapping instead of the documented one."""
    if not id2label:
        return False
    return not all(str(v).upper().startswith("LABEL_") for v in id2label.values())


class GarmentAnalyzer:
    def __init__(self, mock: bool = False, use_segmentation: bool = None):
        self.mock = mock
        self.use_segmentation = USE_SEGMENTATION if use_segmentation is None else use_segmentation
        self.enabled = False
        self._model = None
        self._processor = None
        self._id2label = dict(SEGMENTATION_LABELS)
        self._warned = False

        if not mock and self.use_segmentation:
            self._try_load()

    def _try_load(self):
        try:
            import torch
            from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

            self._torch = torch
            try:
                self._processor = SegformerImageProcessor.from_pretrained(
                    SEGMENTATION_MODEL_NAME,
                    size={"height": SEGMENTATION_MAX_SIDE, "width": SEGMENTATION_MAX_SIDE},
                )
            except Exception:
                # Some processor/version combinations don't accept a `size`
                # override kwarg -- fall back to the checkpoint's own default
                # rather than failing the whole analyzer over a speed knob.
                self._processor = SegformerImageProcessor.from_pretrained(SEGMENTATION_MODEL_NAME)

            self._model = AutoModelForSemanticSegmentation.from_pretrained(SEGMENTATION_MODEL_NAME).to(DEVICE).eval()

            model_labels = getattr(self._model.config, "id2label", None)
            if _looks_like_real_labels(model_labels):
                self._id2label = {int(k): v for k, v in model_labels.items()}
            # else: keep the config.py fallback map -- documented, checked-in
            # constant matching this exact checkpoint's published labels.

            self.enabled = True
        except Exception as e:
            print(f"[garment_analysis] SegFormer unavailable, falling back to legacy "
                  f"MediaPipe-pose color extraction for all images. Reason: {e}")
            self.enabled = False
            self._model = None
            self._processor = None

    # ---------------- Public API ----------------
    def process_image(self, image: Image.Image) -> dict:
        return self.process_batch([image])[0]

    def process_batch(self, images):
        images = [im.convert("RGB") for im in images]

        if self.mock or not self.enabled:
            return [_legacy_extract(im) for im in images]

        try:
            return self._process_batch_segformer(images)
        except Exception as e:
            # A batch-level failure (e.g. a transient OOM) should not take
            # down indexing -- degrade this batch to the legacy path and
            # keep going. Model stays "enabled" since this is treated as a
            # one-off, not evidence the model itself is broken.
            if not self._warned:
                print(f"[garment_analysis] segmentation batch failed ({e}); "
                      f"using legacy fallback for this batch (will retry segmentation on the next).")
                self._warned = True
            return [_legacy_extract(im) for im in images]

    # ---------------- Internal ----------------
    def _process_batch_segformer(self, images):
        torch = self._torch
        inputs = self._processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            logits = self._model(**inputs).logits  # (B, C, h, w) -- native (low) decoder resolution, not upsampled
        pred = logits.argmax(dim=1).detach().cpu().numpy()  # (B, h, w)

        results = []
        for j, image in enumerate(images):
            mask = pred[j]
            h, w = mask.shape
            small_rgb = np.array(image.resize((w, h), Image.BILINEAR))  # same coord system as `mask`
            record = self._colors_from_mask(mask, small_rgb)
            if record is None:
                # No usable garment area found (e.g. no person in frame) --
                # per-image fallback, rest of the batch is unaffected.
                results.append(_legacy_extract(image))
            else:
                results.append(record)
        return results

    def _colors_from_mask(self, mask: np.ndarray, rgb: np.ndarray):
        from config import GARMENT_LABEL_TO_OBJECT

        total = mask.size
        garment_colors, garment_conf, seg_labels = {}, {}, []

        for label_id, label_name in self._id2label.items():
            if label_name not in GARMENT_COLOR_LABELS and label_name not in GARMENT_LABEL_TO_OBJECT:
                continue
            pixel_mask = mask == label_id
            frac = float(pixel_mask.sum()) / total
            if frac < SEGMENTATION_MIN_MASK_FRAC:
                continue
            seg_labels.append(label_name)
            if label_name in GARMENT_COLOR_LABELS:
                pixels = rgb[pixel_mask]
                # Median (not mean/KMeans) -- the mask already isolates the
                # garment from background/skin (that's the whole point of
                # segmentation over a crop box), so a robust central tendency
                # is enough; no need for KMeans clustering to separate
                # "contaminants" the way the legacy crop-box approach needed.
                median_rgb = tuple(int(x) for x in np.median(pixels, axis=0))
                name, conf = nearest_color_name(median_rgb)
                if name:
                    garment_colors[label_name] = name
                    garment_conf[label_name] = conf

        if not garment_colors:
            return None  # signal "nothing usable" -> caller falls back to legacy per-image

        upper_color = garment_colors.get("Upper-clothes") or garment_colors.get("Dress")
        upper_conf = garment_conf.get("Upper-clothes", garment_conf.get("Dress", 0.0))
        lower_color = garment_colors.get("Pants") or garment_colors.get("Skirt") or garment_colors.get("Dress")
        lower_conf = garment_conf.get("Pants", garment_conf.get("Skirt", garment_conf.get("Dress", 0.0)))

        accessories = [GARMENT_LABEL_TO_OBJECT[lbl] for lbl in seg_labels if lbl in GARMENT_LABEL_TO_OBJECT]

        return {
            "upper_color": upper_color,
            "upper_color_conf": upper_conf,
            "lower_color": lower_color,
            "lower_color_conf": lower_conf,
            "crop_method": "segformer",
            "garment_colors": garment_colors,
            "garment_color_confidence": garment_conf,
            "accessories": sorted(set(accessories)),
            "segmentation_labels": sorted(set(seg_labels)),
        }
