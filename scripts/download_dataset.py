"""
Downloads a sample of the Fashionpedia dataset from Hugging Face and saves
it as plain JPEG files under --out, ready to be pointed at with
`python -m indexer.build_index --data_dir <out>`.

Dataset: detection-datasets/fashionpedia (Jia et al., ECCV 2020; CC-BY-4.0).
It's an object-detection dataset (images + bounding boxes/segmentation), but
we only need the images themselves here -- this project does its own
attribute extraction (CLIP/BLIP/SegFormer) rather than using Fashionpedia's
bounding-box annotations.

Streams by default (datasets' `streaming=True`) rather than downloading the
full ~3.5GB dataset up front -- for a `--n` in the hundreds/low thousands
this means only the images actually sampled are ever transferred, which
matters on a metered/slow connection. Falls back to a normal (non-streaming)
download automatically if streaming isn't supported by the installed
`datasets` version.

Fashionpedia is mostly studio/runway photography (see README.md "Getting the
dataset" for why you'll want to add some general lifestyle photos too, for
environment diversity).

Run:
    pip install datasets
    python scripts/download_dataset.py --n 800 --out data/images/fashionpedia
    python scripts/download_dataset.py --n 200 --split val --out data/images/fashionpedia_val
"""
import argparse
import hashlib
import io
import os
import sys

from tqdm import tqdm

DATASET_ID = "detection-datasets/fashionpedia"


def _save_image(image, out_dir: str) -> bool:
    """Saves one PIL image as a content-hashed .jpg filename (stable and
    automatically de-duplicating if the script is re-run / interrupted and
    resumed). Returns False (and skips) for anything that fails to decode or
    re-encode -- a handful of bad rows in a 46K-image dataset shouldn't abort
    the whole download."""
    try:
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        data = buf.getvalue()
        name = hashlib.md5(data).hexdigest() + ".jpg"
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            return True  # already saved on a previous run
        with open(path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"  skipped one image ({e})", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="number of images to sample")
    ap.add_argument("--out", default="data/images/fashionpedia")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-streaming", action="store_true",
                     help="Force a full (non-streaming) dataset download instead.")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Missing dependency: run `pip install datasets` first.")

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {DATASET_ID} (split={args.split}, streaming={not args.no_streaming}) ...")
    saved = 0
    if not args.no_streaming:
        try:
            ds = load_dataset(DATASET_ID, split=args.split, streaming=True)
            ds = ds.shuffle(seed=args.seed, buffer_size=max(args.n * 4, 1000))
            for row in tqdm(ds.take(args.n), total=args.n, desc="Downloading"):
                if _save_image(row["image"], args.out):
                    saved += 1
        except Exception as e:
            print(f"Streaming mode failed ({e}); falling back to a full dataset download.")
            args.no_streaming = True

    if args.no_streaming:
        ds = load_dataset(DATASET_ID, split=args.split)
        ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
        for row in tqdm(ds, desc="Downloading"):
            if _save_image(row["image"], args.out):
                saved += 1

    print(f"Saved {saved}/{args.n} images to {args.out}")
    print("Remember (see README.md 'Getting the dataset'): Fashionpedia is mostly "
          "studio/runway photography -- add some general lifestyle photos into "
          "data/images/ too for environment diversity (office/street/park/home).")


if __name__ == "__main__":
    main()
