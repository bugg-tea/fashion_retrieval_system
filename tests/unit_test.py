"""
Unit tests for the new modules added in this version: common/color_space.py
(CIELAB color utilities), indexer/garment_analysis.py (SegFormer segmentation
+ its legacy fallback), and common/llm_cache.py (disk-backed LLM cache).

These are fast, fully offline, and don't need any model download -- they
test the pure-Python/numpy logic directly (mask math, color math, cache
read/write), plus the *fallback* behaviour of GarmentAnalyzer (which is
exactly what runs in this sandboxed/offline environment, so it's exercised
for real, not mocked out).

Run:
    python tests/unit_test.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw


def test_color_space():
    from common.color_space import rgb_to_lab, lab_distance, nearest_color_name, color_name_similarity

    # Round-trip sanity: pure black and pure white should land at the
    # extremes of the L channel.
    black_lab = rgb_to_lab((0, 0, 0))
    white_lab = rgb_to_lab((255, 255, 255))
    assert black_lab[0] < 1.0, f"black L* should be ~0, got {black_lab[0]}"
    assert white_lab[0] > 99.0, f"white L* should be ~100, got {white_lab[0]}"

    # Perceptual sanity: navy should be closer to blue than black is to blue.
    from config import COLORS
    d_blue_navy = lab_distance(rgb_to_lab(COLORS["blue"]), rgb_to_lab(COLORS["navy"]))
    d_blue_black = lab_distance(rgb_to_lab(COLORS["blue"]), rgb_to_lab(COLORS["black"]))
    assert d_blue_navy < d_blue_black, "navy should be perceptually closer to blue than black is"

    # nearest_color_name should recover an exact reference color exactly.
    name, conf = nearest_color_name(COLORS["yellow"])
    assert name == "yellow" and conf > 0.99, f"expected exact yellow match, got {name} ({conf})"
    assert nearest_color_name(None) == (None, 0.0)

    # color_name_similarity: identity, symmetry, ordering.
    assert color_name_similarity("blue", "blue") == 1.0
    assert color_name_similarity("blue", "navy") == color_name_similarity("navy", "blue")
    assert color_name_similarity("blue", "navy") > color_name_similarity("blue", "black"), \
        "navy should score more similar to blue than black does"
    assert color_name_similarity(None, "blue") == 0.0
    assert color_name_similarity("not_a_color", "blue") == 0.0
    print("  OK: common/color_space.py")


def test_garment_analysis_mask_logic():
    from indexer.garment_analysis import GarmentAnalyzer
    from config import SEGMENTATION_LABELS

    ga = GarmentAnalyzer.__new__(GarmentAnalyzer)  # bypass model loading entirely
    ga._id2label = dict(SEGMENTATION_LABELS)

    mask = np.zeros((96, 96), dtype=int)
    mask[10:50, 20:70] = 4   # Upper-clothes
    mask[50:90, 20:70] = 6   # Pants
    mask[5:20, 5:20] = 16    # Bag (large enough to pass the min-fraction threshold)

    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    rgb[10:50, 20:70] = (41, 128, 185)   # blue
    rgb[50:90, 20:70] = (20, 20, 20)     # black
    rgb[5:20, 5:20] = (200, 30, 30)      # red

    rec = ga._colors_from_mask(mask, rgb)
    assert rec is not None
    assert rec["upper_color"] == "blue", rec
    assert rec["lower_color"] == "black", rec
    assert rec["garment_colors"].get("Bag") == "red", rec
    assert "handbag" in rec["accessories"], rec
    assert rec["crop_method"] == "segformer"

    # An all-background mask must signal "nothing usable" (None), not crash
    # or silently return an empty-but-truthy record -- this is exactly what
    # triggers the per-image legacy fallback in process_batch.
    empty_mask = np.zeros((96, 96), dtype=int)
    empty_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    assert ga._colors_from_mask(empty_mask, empty_rgb) is None

    # A region below SEGMENTATION_MIN_MASK_FRAC must be ignored (tests the
    # threshold logic, important since Belt/Scarf are the model's weakest
    # classes and most prone to tiny spurious detections).
    tiny_mask = np.zeros((96, 96), dtype=int)
    tiny_mask[0:2, 0:2] = 8  # Belt, 4/9216 px =~ 0.04% << 1.2% threshold
    tiny_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    tiny_rgb[0:2, 0:2] = (10, 10, 200)
    assert ga._colors_from_mask(tiny_mask, tiny_rgb) is None
    print("  OK: indexer/garment_analysis.py mask-to-color logic")


def test_garment_analysis_offline_fallback():
    """In this sandboxed environment there's no route to huggingface.co, so
    a real (non-mock) GarmentAnalyzer should detect that at construction
    time, disable itself, and STILL return correctly-shaped, usable results
    by transparently falling back to the legacy MediaPipe-pose path -- this
    is the exact reliability guarantee the module promises."""
    from indexer.garment_analysis import GarmentAnalyzer

    img = Image.new("RGB", (200, 300), (230, 230, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, 160, 150], fill=(41, 128, 185))
    d.rectangle([50, 150, 150, 280], fill=(20, 20, 20))

    ga = GarmentAnalyzer(mock=False)  # will try, and fail, to reach huggingface.co
    assert ga.enabled is False, "expected segmentation to be disabled with no network access"
    results = ga.process_batch([img, img])
    assert len(results) == 2
    for rec in results:
        for key in ("upper_color", "lower_color", "crop_method", "garment_colors", "accessories", "segmentation_labels"):
            assert key in rec, f"missing key {key} in fallback result: {rec}"
        assert rec["crop_method"] in ("pose", "fallback")
    print("  OK: indexer/garment_analysis.py graceful offline fallback")


def test_llm_cache():
    from common import llm_cache

    tmpdir = tempfile.mkdtemp()
    old_path = llm_cache._CACHE_PATH
    old_mem = llm_cache._mem_cache
    try:
        llm_cache._CACHE_PATH = os.path.join(tmpdir, "llm_cache.jsonl")
        llm_cache._mem_cache = None  # force reload from the (empty) temp path

        assert llm_cache.get("parse", "model-x", "hello") is None
        llm_cache.put("parse", "model-x", "hello", {"colors": ["blue"]})
        assert llm_cache.get("parse", "model-x", "hello") == {"colors": ["blue"]}
        # Different kind/model/prompt must be a distinct cache entry.
        assert llm_cache.get("verify", "model-x", "hello") is None

        # Simulate a fresh process picking the cache back up from disk.
        llm_cache._mem_cache = None
        assert llm_cache.get("parse", "model-x", "hello") == {"colors": ["blue"]}
        print("  OK: common/llm_cache.py")
    finally:
        llm_cache._CACHE_PATH = old_path
        llm_cache._mem_cache = old_mem
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    print("Running offline unit tests ...")
    test_color_space()
    test_garment_analysis_mask_logic()
    test_garment_analysis_offline_fallback()
    test_llm_cache()
    print("\nAll unit tests passed.")


if __name__ == "__main__":
    main()
