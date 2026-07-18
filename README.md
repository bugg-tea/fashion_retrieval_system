# Multimodal Fashion & Context Retrieval

Glance ML Internship Assignment — submission by [your name]

This is a search engine for fashion photos. You type something like *"a
person in a bright yellow raincoat"* or *"casual weekend outfit for a city
walk"* and it finds matching images from a database of ~800 photos. Nothing
fancy on the infra side (FAISS, flat file, done) — most of the effort went
into making the *matching* actually work for fashion, which turned out to be
a more interesting problem than I expected going in.

## The short version

I started with plain CLIP embeddings + cosine similarity, like the
assignment hints at. It works, but not well enough — the assignment
literally calls out the "red shirt blue pants vs blue shirt red pants"
problem, and once you test it, CLIP does exactly that. One global vector per
image means color, garment, and scene are all mashed together with no way
to tell the model "the *shirt* is red, not the pants." It also doesn't
distinguish "hoodie" from "sweatshirt" reliably, and it's biased toward
whatever's visually loudest in the frame.

So instead of one embedding, I built a pipeline that pulls out several
separate, checkable pieces of information per image — what the caption says,
what scene/style it looks like, and critically, **which color belongs to
which garment** using real per-pixel segmentation, not a rough crop. Then at
query time I combine semantic search with attribute filtering and a proper
reranker, instead of trusting one similarity score.

I'll explain the reasoning in more detail below, but that's the gist: keep
CLIP as the backbone (it's genuinely good, no reason to throw it away), and
wrap it in enough structure that compositional queries actually work.

## Repo layout

```
config.py                    # every tunable knob lives here, nothing hardcoded elsewhere
common/
  color_space.py             # LAB color math shared by indexer + retriever
  llm_cache.py                # disk cache so re-running a query doesn't burn API quota
indexer/                      # Part A
  feature_extractor.py       # CLIP embedding + BLIP caption + zero-shot scene/style/objects
  garment_analysis.py        # SegFormer clothes segmentation -> per-garment color
  color_utils.py             # older MediaPipe-based fallback, kept for when segformer isn't available
  build_index.py             # entry point, builds the FAISS index + metadata
retriever/                    # Part B
  query_parser.py            # turns free text into structured attributes
  search.py                  # the actual multi-stage retrieval logic
  llm_verify.py               # optional last-mile sanity check via Gemini
scripts/download_dataset.py  # grabs a Fashionpedia sample from HF
demo.py                      # CLI entry point, also runs the 5 eval queries
tests/                        # unit, smoke, and e2e tests
```

Part A and Part B only talk to each other through `config.py` and the files
on disk (`data/index/clip.index` + `metadata.jsonl`). You could delete
`retriever/` entirely and `indexer/` would still run fine, and vice versa.

## The dataset

I used a ~800 image sample streamed from `detection-datasets/fashionpedia`
on Hugging Face (`scripts/download_dataset.py`, no need to pull the full
3.5GB set). One thing worth flagging honestly: Fashionpedia is mostly
studio/runway shots, and the assignment specifically wants environment
diversity — office, street, park, home. So I supplemented it with a couple
hundred general lifestyle photos so the "where" part of the query actually
has something to retrieve against. If you only index runway photography,
every "professional office" query is going to struggle no matter how good
the retrieval logic is, because the images just aren't there.

## Part A — the indexer

For every image I extract:

- **CLIP ViT-B/32 embedding** — the semantic backbone, used for the initial
  nearest-neighbor search. Fast, well understood, good zero-shot prior.
- **BLIP caption** — a free-text description of the image, used later for
  reranking and as extra context if the LLM stage is on.
- **Scene / style / clothing-type / object tags** — all zero-shot CLIP
  classification against a fixed vocabulary in `config.py` (no extra model,
  reuses the CLIP embedding I already have). I keep the top-3 guesses for
  scene and style, not just the argmax, so a query can get partial credit
  if the correct scene was CLIP's second-best guess rather than its first.
- **Per-garment color** — this was the part that actually mattered most.
  Early on I was cropping a rough torso box (pose landmarks from
  MediaPipe) and running KMeans on it, and the results were bad — a shirt
  that's maybe 40% of the crop, mixed with background and skin pixels, and
  KMeans just returns whatever color dominates the box, which is often
  *not* the shirt. Swapped that for `mattmdjaga/segformer_b2_clothes`, a
  clothes-segmentation model that gives real per-pixel masks for
  upper-clothes / pants / skirt / dress / belt / bag / scarf / hat. Color
  gets sampled from the actual garment pixels now, not a bounding box. This
  one change fixed more retrieval errors than anything else I tried — see
  "what actually went wrong" below.
- **Objects** — bench, chair, bag, sunglasses, etc, again zero-shot CLIP
  classified directly against the image (not inferred from the caption —
  more on why that distinction matters below).

All of this gets packed into one structured record per image and written to
`metadata.jsonl`, alongside a FAISS `IndexFlatIP` of the CLIP embeddings.

Everything runs on CPU. CLIP + BLIP run batched in a single process (that's
where the real speedup is — the model forward pass is the bottleneck, not
per-image Python overhead). Segmentation runs across worker processes since
it's the slower step; each worker loads its own copy of the segmentation
model once and reuses it across its shard of images, rather than reloading
per image. It's also resumable — if it dies partway through (I killed mine
by accident more than once while tuning batch size), re-running the same
command picks up from a checkpoint instead of starting over.

```bash
pip install -r requirements.txt
python scripts/download_dataset.py --n 800 --out data/images/fashionpedia
python -m indexer.build_index --data_dir data/images --workers 4 --batch_size 32
```

If you don't want to deal with the segmentation model at all, `--legacy-color`
falls back to the old MediaPipe approach — less accurate, but near-zero
memory and no extra download. Not what I'd recommend, but useful if you're
just trying to get something running quickly.

## Part B — the retriever

This is a 5-stage pipeline, not a single similarity search:

**1. Query parsing.** A regex/keyword parser pulls colors, clothing types,
scene, style, and objects out of the query string against the same
controlled vocabulary used at index time. If a `GEMINI_API_KEY` is set, an
LLM parser can catch phrasing the regex misses — but it's constrained to
the same vocabulary, and anything it returns outside that vocabulary gets
thrown away. I didn't want a "creative" LLM parse inventing an attribute
that was never in the query.

**2. Semantic search (FAISS, top 100).** Plain CLIP nearest-neighbor over
the whole database. This stage alone is roughly what vanilla CLIP retrieval
gives you.

**3. Attribute filtering (top 30).** Score each of the 100 candidates
against the parsed attributes — color similarity in CIELAB space (not RGB;
LAB distance tracks human color perception much better, e.g. it won't call
navy and sky-blue "close" the way raw RGB Euclidean distance sometimes
does), soft scene/style credit using the top-3 guesses from indexing, and
set-membership matching for clothing/objects. Instead of just re-ranking by
a blended score and keeping the top 30, I keep the *union* of the top-N by
pure semantic score and top-N by pure attribute score — a candidate that's
a near-perfect attribute match but a middling semantic rank can otherwise
get silently dropped before it ever reaches reranking, and I actually hit
this as a real bug (see below).

**4. Cross-attention reranking (top 10).** The top 30 get scored with
BLIP-ITM (image-text matching), which does real cross-attention between the
image and query text rather than comparing two independently-computed
vectors. This is the stage that does the heavy lifting on compositional
queries like "red tie and white shirt" — it can actually look at the image
and the phrase together, instead of relying on two embeddings happening to
land close in space.

**5. Composite scoring + optional LLM verification.** Final score is a
weighted blend — semantic 0.35, attribute 0.30, ITM 0.25, LLM-verify 0.10
(only the top 3 candidates go to the LLM stage, not all 10, and everything
is disk-cached so re-running the same query during testing costs zero extra
API quota). If confidence on the top result comes out below 0.45, there's a
reflection step: look at which attribute category was most often missed
across the current top candidates and re-weight toward it, then re-score
the *same* shortlist — no new search, no new model calls, just a smarter
second pass over what's already been retrieved.

```bash
python demo.py "A person in a bright yellow raincoat."
python demo.py --eval          # runs all 5 assignment queries, prints per-attribute matched/missed
```

Every result comes back with its component scores broken down (semantic /
attribute / itm / llm) plus which specific attributes matched and which
didn't — I wanted this to be debuggable, not a black-box number. When
something ranks wrong, I can actually tell *why*.

## What actually went wrong (and what I changed)

I ran the 5 evaluation queries early on and two things were clearly broken:

- **"Someone wearing a blue shirt sitting on a park bench"** — the top
  result's caption said "chair," and the actually-correct image didn't make
  the top 5. Turned out the semantic search stage was already finding the
  right images — the loss was happening in attribute scoring, not FAISS.
  The fix was making sure object presence ("bench") is checked by CLIP
  directly against the image, not inferred from whatever noun BLIP happened
  to caption. Those two signals can disagree, and I was trusting the wrong
  one.
- **Shirt colors coming back gray or just wrong.** This was the pose-crop
  problem described above — fixed by switching to SegFormer segmentation
  for garment colors. This was the single biggest quality jump in the whole
  project, by a good margin.

Both of these are the kind of failure you only catch by actually running
the eval queries and looking at *why* something ranked wrong, not just
whether the final score looked reasonable. I'd rather write that up than
pretend the first version worked.

## Known limitations, said plainly

- **Necktie color still isn't separable.** The segmentation model doesn't
  have a dedicated "tie" class, so for something like "red tie," the system
  falls back on the caption/CLIP signal instead of a real garment-region
  color match, the way it can for a scarf or belt. I know about this one
  and didn't have a clean fix within scope — flagging it rather than hiding
  it.
- **Belt and scarf detection** are the weakest classes in the segmentation
  model itself — expect more misses than false positives there.
- Segmentation assumes a mostly-visible, standing or seated person. Heavy
  occlusion or unusual poses degrade it.
- There's no labeled ground-truth set for this project, so "confidence" is
  an internally consistent score across my own pipeline — not a validated
  precision/recall number against human-labeled relevance. I'd want that
  before claiming any specific accuracy figure.
- The LLM stage is free-tier Gemini, so it's rate-limited by Google, not by
  anything in this codebase — the system just degrades gracefully (skips
  that signal, redistributes its weight to the other three) if quota runs
  out mid-run.

## Does this scale to a million images?

The honest answer: the retrieval *logic* does, the index type as configured
right now doesn't quite, and that's a one-line fix, not a redesign.

- `IndexFlatIP` is exact nearest-neighbor and is fine up to somewhere around
  100K–1M vectors. Past that you'd swap to `IndexIVFFlat` or `IndexHNSWFlat`
  — same FAISS library, different index constructor, nothing else in the
  pipeline changes.
- More importantly: stages 3–5 (attribute scoring, ITM reranking, LLM
  verify) never touch the whole database — they only ever run on the
  bounded top-K shortlist that comes out of stage 1. So per-query cost past
  the initial FAISS search doesn't grow with the size of the database. Only
  stage 1 scales with N, and that's the part FAISS is already built to
  handle efficiently.

## Zero-shot capability

No part of this is trained on labeled fashion data specific to this
project. Scene, style, clothing, and object tagging are all zero-shot CLIP
classification against a controlled vocabulary — a phrase the system has
never seen verbatim (e.g. "heading out for a jog") still gets handled
because it's semantic search + BLIP-ITM cross-attention doing the work, not
string matching against training labels. The one model that *is* trained
rather than zero-shot is SegFormer, but it's trained for general clothes
segmentation (on the ATR dataset), not fine-tuned on anything from this
project — it's a fixed, general "what garment is this pixel" tool, so it
doesn't compromise zero-shot generalization to this dataset.

## Other approaches I considered and didn't use

**Fine-tuning CLIP on fashion data** (e.g. converting Fashionpedia's
attribute labels into captions and contrastively fine-tuning) would
probably close some of the fine-grained gap CLIP has on fashion vocabulary.
I didn't go this way for three reasons: it needs a GPU and real training
time I didn't want to spend on a take-home; it risks overfitting to
Fashionpedia's studio/runway look, which actively works against the
assignment's ask for environment diversity; and it doesn't even fix the
actual architectural problem — a fine-tuned CLIP is still one global vector
per image, so "red shirt blue pants" vs "blue shirt red pants" is still
ambiguous no matter how well it's trained. It's the right call for a real
production system with training infra and a properly diverse training set
— just not the right call here.

**A heavier open-vocabulary detector** (Grounding DINO, OWL-ViT) for object
detection instead of zero-shot CLIP classification. Would probably be more
accurate for "is there a bench in this photo," but it's a much bigger model
to run on CPU for marginal gain over CLIP-vs-image zero-shot classification,
which was already solving the actual bug I hit (see "bench became chair"
above).

## If I had another week

- **Necktie-specific color extraction** — probably means either a dedicated
  small detector for ties, or hand-labeling a few hundred crops and training
  a tiny classifier on top of segmentation output.
- **A real held-out labeled set** (query → relevant image pairs) so
  precision/recall is an actual measured number instead of an internally
  consistent confidence score. This also unlocks learned confidence
  calibration (logistic regression or similar over the four component
  scores) instead of the fixed hand-picked weights in `config.py` right now.
- **A genuine second search on reflection**, not just re-weighting the
  existing shortlist — e.g. if "blue shirt in a park" under-retrieves,
  re-querying just "blue shirt" in isolation and merging results, rather
  than only re-scoring what stage 1 already returned.
- **Extending to real locations and weather**, since the assignment asks
  for this explicitly:
  - *Locations:* add a geo/landmark branch — either a bigger, hierarchical
    CLIP zero-shot vocabulary (neighborhood → city → known landmarks), or
    for something production-grade, an actual landmark-recognition API. The
    query parser would also need a named-entity step to catch city/country
    names, which the current regex parser doesn't attempt.
  - *Weather:* cheapest option is a zero-shot CLIP pass against a weather
    vocabulary (rainy/sunny/snowy/overcast), same pattern as the existing
    scene/style branches. If the photos have real EXIF timestamp +
    geolocation, cross-referencing a historical weather API would be more
    accurate than guessing from pixels alone.
- **Stronger reranker** — BLIP-ITM works, but SigLIP or BLIP-2's ITM head
  would likely be better; this is a bigger swap since it probably changes
  score calibration and I'd want to re-tune `SCORE_WEIGHTS` against it
  rather than just plug it in.

## Testing

```bash
python tests/unit_test.py     # color math, segmentation fallback logic — no downloads
python tests/smoke_test.py    # synthetic images with known colors, full pipeline, mocked models
python -m tests.e2e_test      # real models against a built index, this is the "does it actually work" test
```

## Setup

```bash
git clone <this-repo>
cd fashion-retrieval
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # optional — only needed if you want the Gemini stages
```

Everything runs on CPU, including segmentation. If `mediapipe` fails to
install, drop to Python 3.10/3.11 — it tends to lag behind newer releases.
No paid API required anywhere — Gemini is opt-in, and the system runs at
slightly lower precision on paraphrased queries without it, not broken.

---

Sample outputs for all 5 evaluation queries (top-2 results with parsed
attributes) are included in this submission as `test_output/`.
