"""
OPTIONAL BACKFILL SCRIPT. As of this version, indexer/feature_extractor.py
already computes zero-shot "object" tags (bench, chair, dog, bicycle, ...)
directly during the normal `python -m indexer.build_index` run, unioned with
any accessories the SegFormer garment-segmentation branch detects (e.g. a
real "Bag" region -> "handbag"). You do NOT need to run this script for a
fresh index built with this version of the pipeline.

This script still exists, unchanged in spirit, for one case: backfilling
`objects` onto an index that was built with an OLDER version of this
pipeline (before object tagging was folded into the main indexing pass), so
you don't have to re-run the full CLIP/BLIP encoding stage just to pick up
this one field.

Why this is (almost) free:
    indexer/build_index.py's checkpoint file (data/index/checkpoint_records.jsonl)
    already contains the CLIP image embedding for every image, computed during
    your original run. Classifying those cached embeddings against a new
    zero-shot vocabulary (config.OBJECTS) is just a handful of short text
    encodes + one matrix multiply against vectors you already have -- seconds,
    not hours. No images are re-decoded, no BLIP caption is regenerated.

    This script does NOT touch checkpoint_records.jsonl, checkpoint_colors.jsonl,
    or the FAISS index -- it only updates the "objects" field of each record in
    metadata.jsonl (union'd with whatever's already there -- e.g. segmentation-
    detected accessories on a newer index are preserved, not overwritten -- and
    backs the old file up first).

Requires: data/index/checkpoint_records.jsonl to still be present. This is
the same checkpoint file build_index.py writes and keeps after finishing --
if you deleted it, this script can't run without re-encoding, since colors
are stripped out of the final metadata.jsonl on purpose.

Run any time after `python -m indexer.build_index` has completed:
    python -m indexer.add_object_tags

Safe to re-run repeatedly (e.g. after editing config.OBJECTS or
OBJECT_SCORE_THRESHOLD) -- it always starts fresh from the untouched
checkpoint embeddings, never from a previous run's output.
"""
import json
import os
import shutil

import numpy as np

from config import INDEX_DIR, METADATA_PATH, DEVICE, CLIP_MODEL_NAME, OBJECTS, OBJECT_SCORE_THRESHOLD

RECORDS_CKPT_PATH = os.path.join(INDEX_DIR, "checkpoint_records.jsonl")


def _load_clip_embeddings():
    if not os.path.exists(RECORDS_CKPT_PATH):
        raise SystemExit(
            f"{RECORDS_CKPT_PATH} not found.\n"
            "This script needs the checkpoint file produced by "
            "`python -m indexer.build_index` (it is kept on disk even after "
            "indexing finishes -- don't delete it). Run the indexer first."
        )
    paths, embeddings = [], []
    with open(RECORDS_CKPT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "clip_embedding" not in rec:
                continue
            paths.append(rec["image_path"])
            embeddings.append(rec["clip_embedding"])
    if not embeddings:
        raise SystemExit(f"No usable records with clip_embedding found in {RECORDS_CKPT_PATH}.")
    return paths, np.array(embeddings, dtype="float32")


def main():
    paths, emb = _load_clip_embeddings()
    print(f"Loaded {len(paths)} cached CLIP embeddings from {RECORDS_CKPT_PATH} -- no image re-encoding needed.")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP text encoder only (fast, no BLIP) ...")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    texts = [f"a photo containing a {o}" for o in OBJECTS]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.inference_mode():
        text_feats = model.get_text_features(**inputs)
        if hasattr(text_feats, "pooler_output"):
            text_feats = text_feats.pooler_output
    text_feats = torch.nn.functional.normalize(text_feats, dim=-1).detach().cpu().numpy()

    img_feats = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sims = img_feats @ text_feats.T                    # (N, len(OBJECTS)) cosine sims
    scores = 1.0 / (1.0 + np.exp(-(sims * 20 - 10)))    # same sigmoid convention as CLOTHING_TYPES

    objects_by_path = {}
    for i, path in enumerate(paths):
        tags = [OBJECTS[j] for j, s in enumerate(scores[i]) if s > OBJECT_SCORE_THRESHOLD]
        objects_by_path[path] = tags

    if not os.path.exists(METADATA_PATH):
        raise SystemExit(f"{METADATA_PATH} not found -- run the indexer first.")
    backup_path = METADATA_PATH + ".bak"
    shutil.copyfile(METADATA_PATH, backup_path)
    print(f"Backed up existing metadata to {backup_path}")

    updated = []
    with open(METADATA_PATH) as f:
        for line in f:
            rec = json.loads(line)
            existing = set(rec.get("objects", []))
            new_tags = set(objects_by_path.get(rec["image_path"], []))
            rec["objects"] = sorted(existing | new_tags)
            updated.append(rec)

    with open(METADATA_PATH, "w") as f:
        for rec in updated:
            f.write(json.dumps(rec) + "\n")

    n_with_objects = sum(1 for r in updated if r["objects"])
    print(f"Done. {n_with_objects}/{len(updated)} images got at least one object tag.")
    print(f"Updated {METADATA_PATH} in place (original backed up at {backup_path}).")


if __name__ == "__main__":
    main()
