"""
Part A -- The Indexer.

Pipeline:
  1. Discover images under --data_dir.
  2. Garment analysis IN PARALLEL across CPU cores (ProcessPoolExecutor).
     By default this runs SegFormer clothes segmentation (see
     indexer/garment_analysis.py) -- each worker process loads its own copy
     of the model ONCE (via the pool's initializer, not per image) and then
     processes images one at a time, so the (one-off) model-load cost is
     amortised across the whole shard of images that worker handles. Pass
     --legacy-color to use the older, much lighter MediaPipe-pose approach
     instead (indexer/color_utils.py) -- useful on a very RAM-constrained
     machine, or if you deliberately don't want the extra ~110MB model
     download. Whichever path is enabled, a per-image segmentation failure
     (e.g. no person detected in a product flatlay) transparently falls back
     to the legacy method for that one image -- see garment_analysis.py.
  3. BATCH-encode images through CLIP + BLIP-caption + zero-shot scene/style/
     clothing/object tags. This is done in one process because the model
     itself (not per-image overhead) is the bottleneck, and batching a
     single loaded model gives a far bigger speedup than reloading it in N
     worker processes. On a GPU this alone gives roughly a 10-20x throughput
     increase over one-image-at-a-time inference.
  4. Merge both results into one metadata record per image, write
     metadata.jsonl.
  5. Build a FAISS IndexFlatIP over L2-normalised CLIP vectors (== cosine
     similarity search) and persist it.

RAM note: with segmentation enabled (the default), each worker process holds
its own SegFormer model in memory. On a modest machine (<= 8GB RAM), prefer
`--workers 2` to `--workers 4` over a higher core count; the CLIP/BLIP stage
that follows loads its own models in the main process on top of that. Use
`--legacy-color` if you'd rather avoid the extra memory/download entirely.

Resume support:
    Garment analysis and CLIP/BLIP records are each appended to a checkpoint
    file (data/index/checkpoint_colors.jsonl, data/index/checkpoint_records.jsonl)
    as soon as they're computed, fsync'd to disk. If the process is killed
    (Ctrl+C, closed terminal, crash, power loss) at any point, simply re-run
    the exact same command -- already-finished images are detected from the
    checkpoint files and skipped, so it picks up right where it left off
    instead of starting over. Pass --fresh to ignore/delete old checkpoints
    and reprocess everything from scratch (e.g. after changing --data_dir).

Refreshing just the garment analysis (e.g. after switching --legacy-color on
or off) WITHOUT re-running the expensive CLIP/BLIP stage:
    delete only data/index/checkpoint_colors.jsonl (leave checkpoint_records.jsonl
    alone), then run the exact same build command again. CLIP/BLIP encoding is
    still fully cached and gets skipped (seconds); only garment analysis is
    recomputed and merged into the final metadata.jsonl.

Run:
    python -m indexer.build_index --data_dir data/images --workers 4 --batch_size 32
    python -m indexer.build_index --mock --limit 20   # offline pipeline smoke test
    python -m indexer.build_index --data_dir data/images --fresh   # ignore old checkpoints
    python -m indexer.build_index --data_dir data/images --legacy-color  # skip SegFormer entirely
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import faiss
from tqdm import tqdm

from config import DATA_DIR, INDEX_DIR, FAISS_INDEX_PATH, METADATA_PATH
from indexer.feature_extractor import FeatureExtractor


def _iter_checkpoint(path):
    """Yield JSON records from a checkpoint file, if it exists.

    Skips a trailing truncated line (possible if the process was killed
    mid-write) instead of crashing -- that one record is simply redone.
    """
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _append_checkpoint(path, rec):
    """Append one record and force it to disk immediately (flush + fsync) so
    a Ctrl+C, crash, or power loss right after this call still leaves the
    record safely on disk -- that's what makes resuming exact rather than
    approximate."""
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _discover_images(data_dir):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)


_ANALYZER = None  # set once per worker process by _init_worker, never in the main process


def _init_worker(use_segmentation):
    """ProcessPoolExecutor initializer: runs once when each worker process
    starts, NOT once per image. This is what makes segmentation-based
    garment analysis affordable under multiprocessing -- the ~110MB model is
    loaded #workers times total over the whole run, not once per image."""
    global _ANALYZER
    from indexer.garment_analysis import GarmentAnalyzer
    _ANALYZER = GarmentAnalyzer(mock=False, use_segmentation=use_segmentation)


def _color_worker(path):
    try:
        from PIL import Image
        image = Image.open(path).convert("RGB")
        return path, _ANALYZER.process_image(image), None
    except Exception as e:
        return path, None, str(e)


def extract_colors_parallel(image_paths, workers, legacy_color):
    results, errors = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                              initargs=(not legacy_color,)) as ex:
        futures = {ex.submit(_color_worker, p): p for p in image_paths}
        desc = "Garment analysis (legacy pose)" if legacy_color else "Garment analysis (SegFormer, parallel)"
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            path, colors, err = fut.result()
            if err:
                errors[path] = err
            else:
                results[path] = colors
    return results, errors


def build(data_dir, workers, batch_size, mock, limit=None, fresh=False, legacy_color=False):
    os.makedirs(INDEX_DIR, exist_ok=True)

    if fresh:
        for p in (os.path.join(INDEX_DIR, "checkpoint_colors.jsonl"),
                  os.path.join(INDEX_DIR, "checkpoint_records.jsonl")):
            if os.path.exists(p):
                os.remove(p)
                print(f"--fresh: removed old checkpoint {p}")

    image_paths = _discover_images(data_dir)
    if limit:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise SystemExit(f"No images found under {data_dir}")

    print(f"Found {len(image_paths)} images.")
    if not mock and not legacy_color and workers > 4:
        print(f"Note: garment analysis uses SegFormer segmentation by default, and each of your "
              f"{workers} worker processes loads its own copy of that model. If you hit memory "
              f"pressure, re-run with a smaller --workers (e.g. 2-4) or pass --legacy-color.")

    # ---------------- Stage: garment analysis (resumable) ----------------
    color_ckpt_path = os.path.join(INDEX_DIR, "checkpoint_colors.jsonl")
    t0 = time.time()
    color_results = {}
    if not mock:
        color_results = {rec["image_path"]: rec["colors"] for rec in _iter_checkpoint(color_ckpt_path)}
        remaining_for_color = [p for p in image_paths if p not in color_results]
        color_errors = {}
        if remaining_for_color:
            n_overlap = len(image_paths) - len(remaining_for_color)
            if n_overlap:
                print(f"Garment analysis: resuming from checkpoint -- "
                      f"{n_overlap}/{len(image_paths)} already done, "
                      f"{len(remaining_for_color)} remaining.")
            new_colors, color_errors = extract_colors_parallel(remaining_for_color, workers, legacy_color)
            for p, c in new_colors.items():
                _append_checkpoint(color_ckpt_path, {"image_path": p, "colors": c})
            color_results.update(new_colors)
        else:
            print("Garment analysis: all images already done (from checkpoint) -- skipping.")
        print(f"Garment analysis done in {time.time() - t0:.1f}s ({len(color_errors)} failures this run)")

    # ---------------- Stage: CLIP/BLIP batch encoding (resumable) ----------------
    records_ckpt_path = os.path.join(INDEX_DIR, "checkpoint_records.jsonl")
    already_done = {rec["image_path"] for rec in _iter_checkpoint(records_ckpt_path)}
    remaining_paths = [p for p in image_paths if p not in already_done]

    t1 = time.time()
    failed = []
    if remaining_paths:
        n_overlap = len(image_paths) - len(remaining_paths)
        if n_overlap:
            print(f"CLIP/BLIP encoding: resuming from checkpoint -- "
                  f"{n_overlap}/{len(image_paths)} already encoded, "
                  f"{len(remaining_paths)} remaining.")
        extractor = FeatureExtractor(mock=mock)
        for i in tqdm(range(0, len(remaining_paths), batch_size), desc="CLIP/BLIP batch encoding"):
            batch_paths = remaining_paths[i:i + batch_size]
            try:
                batch_records = extractor.process_batch(batch_paths, precomputed_colors=color_results)
            except Exception:
                # fall back to per-image so one bad file doesn't drop the whole batch
                batch_records = []
                for p in batch_paths:
                    try:
                        batch_records.append(extractor.process_batch([p], precomputed_colors=color_results)[0])
                    except Exception as e2:
                        failed.append((p, str(e2)))
            # Persisted to disk immediately, one record at a time -- safe to
            # Ctrl+C or close the terminal at any point; at worst you lose the
            # in-flight batch, never anything already written.
            for rec in batch_records:
                _append_checkpoint(records_ckpt_path, rec)
        print(f"Feature extraction done in {time.time() - t1:.1f}s ({len(failed)} failures this run)")
    else:
        print("CLIP/BLIP encoding: all images already done (from checkpoint) -- skipping model load.")

    # Reload from checkpoint as the single source of truth -- this makes the
    # final result identical whether it was produced in one run or resumed
    # across several, and re-orders/dedupes to match the current image list.
    by_path = {}
    for rec in _iter_checkpoint(records_ckpt_path):
        by_path[rec["image_path"]] = rec
    records = [by_path[p] for p in image_paths if p in by_path]

    # Re-merge the freshest color_results into every record, even ones whose
    # CLIP/BLIP encoding was skipped because it was already cached above.
    # Without this, re-running just the (cheap, ~minutes) color stage after
    # tweaking indexer/color_utils.py would have no effect on metadata.jsonl
    # unless the whole (expensive, ~hours) CLIP/BLIP stage also re-ran.
    if not mock:
        for rec in records:
            colors = color_results.get(rec["image_path"])
            if colors:
                rec.update(colors)

    if not records:
        raise SystemExit("No images were successfully processed.")

    dim = len(records[0]["clip_embedding"])
    vectors = np.array([r["clip_embedding"] for r in records], dtype="float32")
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_PATH, "w") as f:
        for i, rec in enumerate(records):
            rec_out = {k: v for k, v in rec.items() if k != "clip_embedding"}
            rec_out["faiss_id"] = i
            f.write(json.dumps(rec_out) + "\n")

    print(f"Indexed {len(records)} images -> {FAISS_INDEX_PATH}, {METADATA_PATH}")
    if failed:
        print(f"Failed images ({len(failed)}):")
        for p, e in failed[:10]:
            print(f"  {p}: {e}")

    return len(records), len(failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4),
                     help="Parallel worker processes for garment analysis. Kept conservative by "
                          "default since SegFormer-based analysis loads one model copy per worker "
                          "(see --legacy-color to avoid this entirely).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mock", action="store_true",
                     help="Deterministic fake embeddings, no model downloads -- pipeline testing only.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true",
                     help="Ignore/delete existing checkpoints and reprocess everything from scratch.")
    ap.add_argument("--legacy-color", action="store_true", dest="legacy_color",
                     help="Use the legacy MediaPipe-pose color extraction instead of SegFormer "
                          "clothes segmentation -- lighter weight, no extra model download, but "
                          "less accurate garment colors and no per-garment metadata.")
    args = ap.parse_args()
    build(args.data_dir, args.workers, args.batch_size, args.mock, args.limit, args.fresh, args.legacy_color)
