# Write-up: Multimodal Fashion & Context Retrieval

## 0. v2 changelog: what changed since the first eval run, and why

A prior eval run surfaced two concrete failure modes, plus a self-review
identified a list of further improvements (both are summarized here since
they're the direct motivation for this version's changes — see the repo's
commit history / PR description for the original writeups).

**Bug 1 — "bench became chair":** for *"someone wearing a blue shirt sitting
on a park bench,"* the top result's BLIP caption said "chair," not "bench,"
and the correct image ranked outside the top 5. **Diagnosis:** the semantic
retrieval stage was already finding the right neighborhood — the loss was in
attribute matching, not FAISS. **Fix:** object presence (`objects` field) was
already computed via zero-shot CLIP classification *directly against the
image*, independent of the BLIP caption — so "bench" vs "chair" is a direct
CLIP-vs-image comparison, not a hope that BLIP's caption used the right noun.
This version also adds a **hard-negative-mining merge** at stage 2 (union
the top candidates by pure semantic score *and* by pure attribute score,
not just their blended average) so a correct image with strong attribute
signal but middling semantic rank isn't silently dropped before reranking —
see `retriever/search.py`.

**Bug 2 — shirt color read as gray/wrong:** the old MediaPipe-pose crop was a
rough proportional torso box that mixed in background and skin pixels along
with the actual garment fabric — if a shirt occupied ~40% of the crop and
background/skin the rest, KMeans could easily return the wrong dominant
color. **Fix:** `indexer/garment_analysis.py` replaces this with **SegFormer
clothes segmentation** (`mattmdjaga/segformer_b2_clothes`), which gives real
per-pixel garment masks — color is sampled from the actual shirt pixels, not
a bounding box. This is the single highest-impact change in this version
(matches the "⭐⭐⭐⭐⭐" priority items from the self-review: clothing
segmentation AND object detection, both addressed without needing a heavy
detector like Grounding DINO — see §1.D below for why that was deliberately
avoided on a CPU-only budget).

Everything else from the self-review's priority list, and what was/wasn't
implemented this round:

| Priority | Item | Status |
|---|---|---|
| ⭐⭐⭐⭐⭐ | Clothing segmentation instead of pose crop | **Done** — `garment_analysis.py`, CPU-only, ~110MB model, graceful fallback to the old pose-crop approach if unavailable |
| ⭐⭐⭐⭐⭐ | Object detection (bench/chair/bag/...) | **Already in place / strengthened** — zero-shot CLIP-vs-image classification (no Grounding DINO/OWL-ViT needed — see §1.D), now folded into the main indexing pass instead of a separate script, and unioned with segmentation-detected accessories (bag, sunglasses) |
| ⭐⭐⭐⭐☆ | Combine BLIP caption nouns with CLIP clothing tags | Already in place (caption-substring union), now also unions in segmentation-detected garments (scarf/belt/hat/skirt/dress) |
| ⭐⭐⭐⭐☆ | Soft attribute scoring instead of binary | **Extended** — color similarity now uses CIELAB distance (was RGB) for a more perceptually accurate soft match; scene matching now gives partial credit via top-3 scene guesses, not just the argmax; style ordinal-distance matching was already in place |
| ⭐⭐⭐⭐☆ | Garment-specific metadata (shirt/jacket/tie/pants) | **Done, with one caveat** — segmentation gives Upper-clothes/Pants/Skirt/Dress/Belt/Bag/Scarf/Hat colors separately (`garment_colors` field); a necktie specifically still isn't separable from Upper-clothes since the segmentation model has no dedicated tie class (see "Known limitations" in README.md) |
| ⭐⭐⭐☆☆ | Top-k probabilities for style/scene | **Done** — `scene_topk`/`style_topk` stored at index time, scene now gets soft credit for a top-3 (not just top-1) match |
| ⭐⭐⭐☆☆ | Attribute-focused second search on reflection | **Not done this round** — the existing reflection loop (re-weight the worst-performing category, re-score the same candidate pool) was kept as-is; a genuine second FAISS query on reflection is real future work (see §4), deliberately deferred so every change shipped this round could be validated rather than adding an untested new search path |
| ⭐⭐☆☆☆ | Stronger reranker (SigLIP/BLIP-2/Florence-2/...) | **Not done** — BLIP-ITM kept; swapping rerankers is a larger, separately-scoped change (see §4) |
| ⭐⭐☆☆☆ | Learned confidence calibration | **Not done** — no labeled query→relevance dataset exists to train on (see §4) |
| — | CIELAB instead of RGB for color naming/matching | **Done** — `common/color_space.py`, shared by both the indexer (naming) and retriever (similarity scoring), so the two can't drift out of sync |
| — | Hard negative mining at stage 2 | **Done** — see Bug 1 above |
| — | LLM free-tier quota management | **Done, and was a real bug** — LLM verification was being called on every stage-3 candidate (up to 10) per query, and again on every reflection retry, against a Gemini free-tier daily quota that can be as low as ~20 requests for a given model. Fixed by (a) capping verification to the top few candidates, (b) reusing cached scores across a reflection retry instead of re-querying, and (c) a disk-backed cache so re-running the same query costs nothing the second time. Also migrated off the now-deprecated `google-generativeai` SDK to `google-genai`, and switched the default model to a Flash-Lite variant (see README.md) |

Deliberately **not** attempted, and why: a trained confidence-calibration
model and a genuinely offline-measurable precision/recall number both need a
labeled query→relevant-image dataset that doesn't exist yet for this project
— building one honestly (not just eyeballing 5 queries) is a bigger, separate
effort than a CPU/free-tier code change, and is called out explicitly in §4
rather than faked with an unvalidated heuristic.



## 1. Approaches considered

### A. Vanilla CLIP zero-shot retrieval
Embed all images and the query with CLIP, rank by cosine similarity.
- **Pro:** trivial to implement, genuinely strong zero-shot baseline, no training data needed.
- **Con:** single global embedding per image has no notion of attribute binding
  — it cannot reliably distinguish "red shirt, blue pants" from "blue shirt,
  red pants," and tends to weight whichever visual element is most salient
  rather than being systematically compositional. Also weak on fine-grained
  distinctions (navy vs sky blue, hoodie vs sweatshirt) since CLIP's training
  objective rewards coarse semantic alignment, not fashion-specific granularity.
- **When it's the right call:** small datasets, exploratory prototypes, or
  queries that are mostly about coarse scene/subject matter rather than
  precise attribute combinations.

### B. Fine-tune CLIP (or a similar dual-encoder) on fashion data
Contrastively fine-tune on a fashion image-caption dataset (e.g. Fashionpedia
attributes converted to captions).
- **Pro:** can genuinely close the fine-grained-attribute gap if done well;
  the model *learns* fashion vocabulary rather than relying on CLIP's general
  web-scale priors.
- **Con:** needs a GPU and meaningful training time, risks overfitting to
  Fashionpedia's studio/runway visual style (poor generalization to the
  office/park/street/home diversity the assignment explicitly asks for),
  and doesn't fix the *architectural* attribute-binding problem — a
  single global embedding is still one vector, no matter how well-trained.
- **When it's the right call:** production systems with resources for
  training + evaluation infrastructure, and a genuinely representative
  training set matching the target domain.

### C. Structured multi-branch pipeline (chosen)
Instead of one embedding, extract several **independently verifiable**
representations per image (semantic embedding, caption, scene, style,
clothing tags, per-region color) and combine semantic search with explicit
attribute filtering and a cross-attention reranker.
- **Pro:** directly targets the compositionality problem — color is bound to
  a specific body region rather than floating in a shared embedding; every
  score is explainable; works zero-shot (no fine-tuning, no labeled training
  set required); each branch is swappable/upgradable independently.
- **Con:** more moving parts than a single model; latency is higher per
  query than raw ANN search (though bounded, since all extra stages only
  touch a top-K shortlist, not the whole database); heuristic/statistical
  components (segmentation-based color naming, zero-shot scene/style) have a
  real, non-zero error rate that further model upgrades could reduce.
- **When it's the right call:** exactly this assignment's constraints — no
  budget for paid resources, need for genuine zero-shot generalization to
  *unseen phrasings*, and an explicit ask for compositional accuracy that
  vanilla CLIP is known to lack.

### D. VQA / vision-language model as a per-image classifier
Run a VLM (e.g. a small open VQA model) with a fixed question set per image
("What color is the top? What color are the pants? What environment is
this?") to directly generate structured labels.
- **Pro:** potentially more accurate structured labels than zero-shot CLIP
  classification, since a VQA model reasons more explicitly.
- **Con:** meaningfully slower per image at indexing time (one or more
  generation passes per question, vs one batched CLIP forward pass), and for
  color specifically it's not obviously better than direct pixel-level
  extraction (sampling the actual segmented garment pixels is a more literal,
  more auditable way to get a garment color than asking a language model to
  name one). We use this idea selectively — BLIP captioning already gives a
  VQA-adjacent signal for free — rather than adding a second heavy VLM pass
  per image.

### E. Heavy object/garment detectors (Grounding DINO, OWL-ViT, SAM)
Run a general-purpose open-vocabulary object detector to localize objects
("bench," "bag," "tree") and/or garments with bounding boxes or masks.
- **Pro:** genuinely more accurate localization than either of the two
  approaches this project uses (zero-shot CLIP-vs-image classification for
  objects, SegFormer for garments) — a real detector reasons about
  *where* something is, not just *whether the image looks like it contains
  one*.
- **Con:** an order of magnitude heavier than what this project can justify
  on a CPU-only, free-tier budget. Grounding DINO/OWL-ViT are meaningfully
  larger and slower per image than SegFormer-B2 (~28M params, sub-second CPU
  inference), and this project already needs to run CLIP + BLIP-caption +
  BLIP-ITM + SegFormer per image/query as it is — adding a full detector on
  top, per image, at indexing time for a dataset in the hundreds-to-thousands
  range would push indexing time from minutes to potentially hours on CPU.
- **What this project does instead:** for *objects* (bench, chair, bag, dog,
  ...), zero-shot CLIP classification directly against the full image is
  "good enough" — it answers "does this image contain a bench" without
  needing to localize *where*, which is exactly what attribute matching
  needs. For *garments specifically* (where per-region color, not just
  presence, matters), SegFormer's pixel-level masks are the right amount of
  localization for the cost — enough to isolate "these pixels are the
  shirt" without the overhead of a general detector. This is a deliberate
  cost/accuracy tradeoff for this project's constraints, not a claim that
  it's more accurate than a real detector would be — see §4 "Future work"
  for when it would be worth revisiting (e.g. a GPU becomes available, or
  recall on small objects like "umbrella" or "sunglasses" proves to be a
  real bottleneck in practice).

**Chosen approach: C, informed by the useful pieces of D, deliberately avoiding E's cost** — CLIP for
semantic search and zero-shot scene/style/clothing tagging, BLIP for
captioning and (separately) image-text-matching reranking, SegFormer clothes
segmentation (falling back to MediaPipe+KMeans) for per-garment color
extraction, and an optional constrained LLM layer for query parsing and final
verification. All model components (CLIP, BLIP-caption, BLIP-ITM, SegFormer,
MediaPipe) are free and run locally on CPU; the LLM layer (Gemini free tier)
is strictly optional and the system is fully functional without it.

## 2. Chosen architecture — how it handles the five evaluation queries

1. **"A person in a bright yellow raincoat."** — color ("yellow") and clothing
   type ("raincoat") are extracted by the query parser and checked against
   the image's independently-extracted `upper_color` and CLIP zero-shot
   `clothing_tags` — this is a direct attribute match, not a hope that CLIP's
   single embedding happens to weight "yellow" and "raincoat" correctly.
2. **"Professional business attire inside a modern office."** — `environment`
   ("office") and `style` ("professional"/"business casual") are matched
   against the image's zero-shot scene and style classification, both
   computed independently of the caption, so the system isn't relying on
   BLIP having mentioned "office" verbatim.
3. **"Someone wearing a blue shirt sitting on a park bench."** — color+clothing
   ("blue shirt") plus environment ("park") are checked as separate
   attributes; the BLIP-ITM reranking stage additionally checks the full
   sentence (including "sitting on a bench") against the caption via real
   cross-attention, catching detail the structured attributes don't cover.
4. **"Casual weekend outfit for a city walk."** — no literal garment is named
   ("casual weekend outfit" is a *style inference* task), so this query
   leans mainly on the `style="casual"` zero-shot classification and
   `environment="urban street"`, exactly the kind of compositional-but-not-
   literal query CLIP alone often gets partially right but imprecisely.
5. **"A red tie and a white shirt in a formal setting."** — the hardest
   compositional case (two colors, two garments, one style). This is exactly
   why colors are extracted *per body region* rather than as one embedding:
   the system can check "is there a red X and a white Y" instead of just "is
   red and white present somewhere in this image."

If the composite confidence for the top result falls below a threshold
(0.45), the **reflection stage** looks at which attribute category was most
often missed across the current top candidates and re-weights that category
before re-scoring — e.g. if every top result nails the color but misses the
scene, scene gets boosted and the pool is re-ranked. This costs nothing extra
in terms of embeddings or model calls (it re-scores the same shortlist), and
directly targets the low-confidence case rather than blindly returning the
same weak results.

**On hallucination control:** every LLM call (query parsing and final
verification) is constrained to a fixed controlled vocabulary and any output
outside it is discarded rather than trusted; the verifier only ever grades
an already-extracted caption + attribute set (it never invents new visual
facts about the image), and low-confidence results are explicitly flagged
as "no confident match" rather than silently returned as if certain.

## 3. Modularity, scalability, zero-shot capability (self-assessment against the rubric)

- **Modular code:** each branch (`feature_extractor.py`, `garment_analysis.py`,
  `color_utils.py`, `query_parser.py`, `llm_verify.py`, `search.py`) is
  independently testable and swappable — e.g. swapping the reranker touches
  only `search.py::_itm_score`, and `garment_analysis.py`/`color_utils.py`
  already demonstrate this: SegFormer is the primary path and the older
  pose-based approach is now purely a fallback module, swapped in
  automatically with no changes needed anywhere else.
- **Scalability:** the only stage that touches the *entire* database per
  query is FAISS search (stage 1); everything downstream (attribute scoring,
  ITM reranking, LLM verification) runs on a bounded top-K shortlist, so
  per-query latency past stage 1 doesn't grow with database size. `IndexFlatIP`
  is exact but O(N) — swapping to `IndexIVFFlat` (approximate, sub-linear) is
  a one-line change for datasets beyond ~100K-1M images (see Future Work).
- **Zero-shot capability:** no component is trained/fine-tuned on labeled
  fashion data. Scene, style, clothing, and object tagging are all zero-shot
  CLIP classification against a controlled vocabulary — new phrasing in a
  query ("a person heading out for a jog," say) is handled by CLIP's own
  semantic understanding (stage 1) and BLIP-ITM's cross-attention (stage 3),
  not by string-matching against a training label set. SegFormer is the one
  component that IS trained (on the ATR human-parsing dataset) rather than
  zero-shot — but it's a fixed, general-purpose "what garment is this pixel"
  model, not fine-tuned on this project's data, so it doesn't compromise the
  zero-shot-to-this-dataset property the rest of the pipeline has.

## 4. Future work

### a. Extending to locations (cities, places) and weather
- **Locations:** add a geo/landmark branch — either a CLIP zero-shot pass
  against a larger, hierarchical vocabulary of place types (neighborhood →
  city → country landmarks) or, for a real product, reverse image search /
  landmark detection APIs. The existing `scene` field would become one of
  several location-related fields (`scene_type`, `landmark`, `city_guess`),
  and the query parser would need a named-entity step to recognize city/
  country names in free text, which the current regex-based extractor
  doesn't attempt.
- **Weather:** two options: (1) a zero-shot CLIP pass against a weather
  vocabulary (rainy, sunny, snowy, overcast) exactly analogous to the
  existing scene/style branches — cheap to add, same pattern; (2) if EXIF
  timestamp + geolocation metadata is available on real photos, cross-
  reference with a historical weather API for ground truth rather than
  visual inference alone, which would be strictly more accurate than
  guessing from pixels.

### b. Improving precision (updated — see §0 for what shipped this round)
- **Color — done:** CIELAB distance (was RGB-Euclidean) for both naming and
  similarity scoring (`common/color_space.py`); SegFormer garment
  segmentation (was pose-based crop) for pixel-accurate garment color
  regions. **Still open:** a small trained color classifier on isolated
  garment crops could push accuracy further than nearest-named-color-in-LAB,
  and CIEDE2000 (vs. this project's simpler CIE76) is a more perceptually
  accurate — but notably more complex — distance metric if color precision
  proves to be a bottleneck in practice.
- **Attribute matching — partially done:** color/style/scene now use soft,
  graded scoring (CIELAB distance, ordinal formality distance, top-3 scene
  credit) instead of pure binary matched/not-matched. Clothing and object
  matching are still binary hit/miss on set membership — extending those to
  use the `*_confidence` scores already computed at index time (e.g. a
  0.51-confidence clothing-tag match counting for less than a
  0.95-confidence one) is the next natural step, not done this round.
- **Reranking / reflection — partially done:** the reflection loop's second
  pass now reuses cached ITM/LLM scores instead of recomputing them (a real
  quota-saving fix, see §0), and stage 2 now does hard-negative-mining
  candidate selection. **Still open, as originally noted:** a genuine
  *second, differently-worded* FAISS search focused on just the missed
  attribute (e.g. re-querying "blue shirt" in isolation when "blue shirt in
  a park" under-retrieves) rather than only re-weighting the existing
  candidate pool; and a small held-out validation set of
  query→relevant-image pairs to empirically tune `SCORE_WEIGHTS` and measure
  real precision/recall instead of reporting internal confidence only.
- **Reranker upgrade — not done:** swapping BLIP-ITM for a stronger
  cross-attention reranker (SigLIP, BLIP-2 ITM, Florence-2) is a larger,
  separately-scoped change (different preprocessing, likely different output
  calibration requiring re-tuning `SCORE_WEIGHTS`) — deferred rather than
  bundled into this round's changes.
- **Learned confidence calibration — not done:** training a small classifier
  (logistic regression / XGBoost) on (semantic, attribute, itm, color-match,
  style-match, scene-match) → P(relevant) needs labeled query-relevance
  pairs this project doesn't have. The honest path here is collecting that
  small labeled set first, not fabricating a calibration without ground truth.
- **Query parsing:** the rule-based parser is regex/keyword-based; a small
  fine-tuned sequence-tagging model (or a larger, more carefully prompted
  LLM pass with few-shot examples) would handle more creative phrasing than
  regex ever will, while keeping the same "discard anything outside the
  controlled vocabulary" hallucination guardrail.

## 5. Codebase

See the accompanying repository (indexer/, retriever/, config.py, demo.py,
tests/) — structured exactly as described in README.md, Part A (indexer/) and
Part B (retriever/) are cleanly separated modules sharing only `config.py`
and the on-disk index/metadata files, so either can be re-run, replaced, or
scaled independently.
