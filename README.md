# Multimodal Fashion & Context Retrieval

A search engine that retrieves images from a fashion database based on natural
language descriptions of *what* someone is wearing, *where* they are, and the
*vibe* of the outfit — built to go beyond vanilla CLIP's known weaknesses on
compositionality ("red shirt, blue pants" vs "blue shirt, red pants") and
fine-grained fashion attributes.


## Why not just CLIP?

CLIP produces one global embedding per image, so it has no notion of *which*
color belongs to *which* garment, and it tends to key off whichever visual
element is most salient rather than being systematically compositional. This
project keeps CLIP as the semantic backbone (it's still a very strong
zero-shot prior) but wraps it in a **structured, multi-branch pipeline** that
extracts explicit, separately-verifiable attributes per image, then combines
semantic search with attribute filtering and a real cross-attention
reranker. See `WRITEUP.md` for the full design rationale, alternatives
considered, and tradeoffs — this file is the practical setup guide.

## Architecture

```
                              IMAGE
                                │
                ┌───────────────┼──────────────────┬────────────────┐
                │               │                   │                │
                ▼               ▼                   ▼                ▼
             CLIP            BLIP              SegFormer         CLIP zero-shot
        (semantic vec)     (caption)        clothes segmentation  vs OBJECTS
                │               │            (falls back to           │
                │               │           MediaPipe+KMeans           │
                │               │            if unavailable)           │
                │               │                   │                │
                └───────┬───────┴─────────┬─────────┴────────┬───────┘
                        │                 │                  │
              CLIP zero-shot vs   per-garment colors    accessories
              ENVIRONMENTS/STYLES (upper/pants/skirt/    (bag/sunglasses)
              (top-K, not just    dress/belt/bag/scarf/  unioned into
               argmax)            hat) + accessories     clothing/object tags
                        │                 │                  │
                        └────────┬────────┴──────────────────┘
                                 ▼
                       Structured Image Record
              {embedding, caption, scene(+topk), style(+topk),
               clothing_tags, objects, upper_color, lower_color,
               garment_colors, accessories}
                                 │
                          FAISS + metadata.jsonl
════════════════════════════════════════════════════════════════════
                            USER QUERY
                                 │
                 ┌───────────────┴────────────────┐
                 │                                 │
                 ▼                                 ▼
       Rule-based attribute parser         Optional Gemini parser
       (colors/clothing/scene/style/       (constrained to same vocab,
        objects)                            discards anything hallucinated,
                 │                          disk-cached to save free-tier quota)
                 └───────────────┬─────────────────┘
                                 ▼
                       Merged structured query
                                 │
        Stage 1: FAISS semantic search                    (top 100)
                                 │
        Stage 2: attribute-match scoring (LAB color        (top 30, via
                 similarity, soft scene/style match)         hard-negative-
                                 │                            mining merge)
        Stage 3: BLIP-ITM cross-attention rerank            (top 10; LLM
                 (+ optional LLM verify on only the           verify only on
                  top few candidates)                         top 3, cached)
                                 │
        Stage 4: weighted composite score
                                 │
        Stage 5: reflection retry if confidence < 0.45
          (re-weights the attribute category that was most
           often unmatched in the current top results, then
           re-scores the SAME candidate pool -- no re-search,
           and reuses cached ITM/LLM scores from the first pass)
                                 │
         Ranked results + confidence + per-attribute
              explanation ("matched"/"missed")
```

## What each component is for

| Component | Model | Purpose |
|---|---|---|
| Semantic embedding | CLIP ViT-B/32 | Fast approximate-nearest-neighbour search over the whole database |
| Caption | BLIP (base) | Free-form description used for reranking + optional LLM context |
| Scene / style / clothing / object tags | CLIP zero-shot (reuses the same model, no extra training) | Structured attributes for filtering, at zero extra model cost. Scene/style keep top-3, not just the best guess. |
| Garment color + accessories | **SegFormer clothes segmentation** (falls back to MediaPipe Pose + KMeans if unavailable) | Per-garment color (upper-clothes/pants/skirt/dress/belt/bag/scarf/hat) from real pixel masks, not a rough crop box |
| Reranker | BLIP-ITM (base-coco) | Real image↔text cross-attention on the shortlist — this is what actually helps with compositional queries like "red tie and white shirt" |
| Query parser | Regex + controlled vocab, optional Gemini | Turns free text into the same structured schema used at index time |
| Verifier (optional) | Gemini free tier | Final sanity check, graded on caption+attributes only, never overriding without evidence — only run on the top few candidates per query, and cached, to conserve free-tier quota |

Nothing here needs a paid API. Gemini is entirely optional — the system runs
fully on local, free, open-source models without it, just with slightly
lower precision on paraphrased queries. Every model download (CLIP, BLIP x2,
SegFormer) is free and pulled automatically from Hugging Face on first run.





## Building the index (Part A)

```bash
python -m indexer.build_index --data_dir data/images --workers 4 --batch_size 32
```

- **Garment analysis** runs in a `ProcessPoolExecutor` across `--workers` CPU
  cores. By default this is SegFormer clothes segmentation — each worker
  loads its own copy of the (~110MB) model *once* when it starts (not once
  per image), so the cost is amortised across that worker's whole shard of
  images. On a machine with limited RAM, prefer `--workers 2`–`4` over your
  full core count, since each worker holds its own model copy in memory. Add
  `--legacy-color` to skip SegFormer entirely and use the older, much
  lighter MediaPipe-pose approach instead (less accurate colors, no
  per-garment metadata, but near-zero extra memory and no model download).
- CLIP + BLIP inference runs batched (`--batch_size`) in one process — this is
  where most of the real speedup comes from, since the model itself (not
  per-image Python overhead) is the bottleneck; reloading it across multiple
  processes would cost more than it saves unless you have multiple GPUs.
  Zero-shot object tagging is computed in this same pass now, so you no
  longer need a separate step for it (see `indexer/add_object_tags.py`'s
  docstring — it's kept only as a backfill tool for older indexes).
- On a single GPU, indexing ~1,000 images typically takes a few minutes, not
  a full day. On CPU only, expect it to be slower — reduce `--batch_size` if
  you hit memory limits, and consider a subset for iteration.
- **Resumable:** if the process is killed at any point (Ctrl+C, crash, closed
  terminal), just re-run the exact same command — it picks up where it left
  off via `data/index/checkpoint_*.jsonl`. Pass `--fresh` to start over.

This produces `data/index/clip.index` (FAISS) and `data/index/metadata.jsonl`
(one structured record per image).

## Querying (Part B)

```bash
python demo.py "A person in a bright yellow raincoat."
python demo.py "Professional business attire inside a modern office." --k 10
python demo.py --eval          # runs all 5 assignment evaluation queries
python demo.py --eval --json   # machine-readable output
```

Each result includes component scores (`semantic`, `attribute`, `itm`,
`llm`), a composite `confidence`, and a per-attribute `explanation` showing
exactly which requested attributes were matched or missed — this is what
makes results interpretable rather than a black-box similarity number.
Results also carry `garment_colors` (e.g. `{"Upper-clothes": "blue", "Bag":
"red"}`) whenever segmentation found per-garment detail beyond the
upper/lower summary.

### If you're using `GEMINI_API_KEY`

- Every LLM call is cached on disk at `data/index/llm_cache.jsonl` — running
  the same query twice (e.g. re-running `--eval` while iterating) costs zero
  extra quota the second time. Safe to delete any time.
- Only the top `config.MAX_LLM_VERIFY_CANDIDATES` (default 3) candidates per
  query are sent for verification, not the whole shortlist — this was a real
  bug in an earlier version of this pipeline (up to 10 verify calls **per
  query**, plus a re-verify on every low-confidence reflection retry) that
  burns a free-tier daily quota in a handful of queries. Both fixes together
  mean a 5-query eval run now costs roughly 5 (parse) + up to 15 (verify)
  Gemini calls worst-case on a cold cache, and ~0 on a warm one.
- `config.GEMINI_MODEL` defaults to `gemini-2.5-flash-lite` rather than the
  full `gemini-2.5-flash` — Flash-Lite variants have historically had a much
  higher free-tier daily request cap. Verify current numbers at
  https://ai.google.dev/gemini-api/docs/rate-limits, since Google changes
  these without much notice.
- If you still hit `429 RESOURCE_EXHAUSTED`, that's expected free-tier
  behaviour, not a bug — the system degrades gracefully (that query's
  LLM-based scoring is simply skipped, weight redistributed to the other
  signals) rather than crashing.

## Testing

```bash
python tests/unit_test.py    # fast, offline: LAB color math, segmentation
                              # mask logic, fallback behaviour, LLM cache
python tests/smoke_test.py   # generates synthetic images with known colors,
                              # runs real color extraction + FAISS indexing +
                              # full retrieval with mock CLIP/BLIP embeddings
python -m tests.e2e_test     # real end-to-end test against your actual built
                              # index, using real CLIP/BLIP/ITM (+ Gemini, if
                              # GEMINI_API_KEY is set) -- build the index first
```

`unit_test.py` and `smoke_test.py` need no model downloads at all. **Mock
mode is for pipeline testing only — it never produces meaningful search
results.**

## Scalability notes (see WRITEUP.md for detail)

- `IndexFlatIP` (exact search) is used by default — fine up to roughly
  100K–1M vectors. For 1M+, swap to `faiss.IndexIVFFlat` or `IndexHNSWFlat`
  (a one-line change in `build_index.py` / `search.py`, everything else is
  unaffected since only the FAISS index type changes).
- The attribute/ITM/LLM stages only ever run on a bounded top-K shortlist
  from stage 1, not the whole database, so per-query cost stays roughly
  constant as the database grows — only the FAISS search stage scales with N.

## Known limitations

- **Necktie color** still can't be extracted separately from the rest of the
  upper body — the segmentation model doesn't have a dedicated "tie" class,
  so a query like "red tie" relies on the caption/CLIP signal rather than a
  garment-specific color match the way "red scarf" or "red belt" now can.
- **Belt and scarf detection** are the segmentation model's weakest classes
  (per its own published per-class accuracy) — expect more missed detections
  (false negatives) than false positives for these two specifically.
- Segmentation (and its MediaPipe-pose fallback) both assume a mostly-visible
  standing or seated person; accuracy drops on heavy occlusion or unusual poses.
- No labeled evaluation set exists for this project, so "confidence" is an
  internally consistent score, not a validated precision/recall number — see
  WRITEUP.md "Future work" for how a small labeled set would change that.
- Gemini features are free-tier and therefore rate-limited by Google, not by
  this codebase — see the quota section above.

## Repository layout

```
config.py                    # all tunable settings, controlled vocabularies
common/
  color_space.py             # shared CIELAB color utilities (indexer + retriever)
  llm_cache.py                # disk-backed cache for Gemini calls
indexer/
  feature_extractor.py       # CLIP + BLIP + zero-shot scene/style/clothing/object branches
  garment_analysis.py        # SegFormer clothes segmentation (+ legacy fallback)
  color_utils.py             # legacy pose-based garment color extraction (fallback path)
  build_index.py             # Part A entry point
  add_object_tags.py         # optional backfill for indexes built with an older version
retriever/
  query_parser.py            # rule-based + optional LLM query parsing
  llm_verify.py               # optional final verification stage
  search.py                  # Part B: multi-stage retrieval logic
scripts/download_dataset.py  # dataset acquisition helper
demo.py                      # CLI for running queries
tests/
  unit_test.py                # fast offline unit tests (no model downloads)
  smoke_test.py                # offline pipeline test, no model downloads needed
  e2e_test.py                  # real end-to-end test against a built index
WRITEUP.md                   # approaches/tradeoffs/future-work for the submission PDF
```
