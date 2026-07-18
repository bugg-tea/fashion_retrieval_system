"""
Offline smoke test: generates a small synthetic image set with KNOWN garment
colors, runs the real (non-mock) color-extraction branch plus the mock-mode
CLIP/BLIP branch through indexing and retrieval, and asserts nothing crashes
and color extraction is reasonably accurate. This does NOT validate CLIP/BLIP
quality (that needs real model downloads / real photos) -- it validates that
the surrounding pipeline (indexing, FAISS, query parsing, multi-stage scoring,
reflection) is wired together correctly.

Run:
    python tests/smoke_test.py
"""
import json
import os
import random
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COLORS

TEST_DIR = "data/images/_smoke_test"
INDEX_DIR = "data/index"


def generate_images(n=30, seed=7):
    from PIL import Image, ImageDraw
    random.seed(seed)
    os.makedirs(TEST_DIR, exist_ok=True)
    names = list(COLORS.keys())
    bgs = [(230, 230, 235), (150, 200, 120), (140, 140, 145), (220, 200, 180)]
    for i in range(n):
        img = Image.new("RGB", (300, 500), random.choice(bgs))
        d = ImageDraw.Draw(img)
        u, l = random.choice(names), random.choice(names)
        d.ellipse([120, 40, 180, 100], fill=(200, 170, 140))
        d.rectangle([90, 100, 210, 300], fill=COLORS[u])
        d.rectangle([100, 300, 200, 460], fill=COLORS[l])
        img.save(os.path.join(TEST_DIR, f"img_{i:03d}_{u.replace(' ', '-')}_{l.replace(' ', '-')}.jpg"), quality=95)


def check_color_accuracy():
    correct, total = 0, 0
    with open(os.path.join(INDEX_DIR, "metadata.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            fname = os.path.basename(r["image_path"]).replace(".jpg", "")
            parts = fname.split("_")
            u_true, l_true = parts[2].replace("-", " "), parts[3].replace("-", " ")
            total += 1
            if r["upper_color"] == u_true and r["lower_color"] == l_true:
                correct += 1
    return correct, total


def main():
    print("1. Generating synthetic test images with known colors ...")
    generate_images()

    print("2. Running indexer (mock CLIP/BLIP, REAL color extraction) ...")
    ret = subprocess.run(
        [sys.executable, "-m", "indexer.build_index", "--mock", "--data_dir", TEST_DIR, "--workers", "4"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    assert ret.returncode == 0, "Indexer crashed"

    print("3. Checking color-extraction accuracy against known ground truth ...")
    correct, total = check_color_accuracy()
    acc = correct / total
    print(f"   {correct}/{total} = {acc:.0%} exact upper+lower color match")
    assert acc > 0.5, "Color extraction accuracy too low -- something is broken"

    print("4. Running retriever on assignment-style queries (mock mode) ...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retriever.search import Retriever
    r = Retriever(mock=True)
    for q in [
        "A person in a bright yellow raincoat.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
    ]:
        result = r.search(q, top_k=3)
        assert "results" in result and isinstance(result["results"], list)
        print(f"   OK: '{q}' -> {len(result['results'])} results, "
              f"top confidence={result['confidence']}")

    print("\nAll smoke tests passed.")
    print(f"NOTE: cleaning up {TEST_DIR} and {INDEX_DIR} (test artifacts only).")
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


if __name__ == "__main__":
    main()
