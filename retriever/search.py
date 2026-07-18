"""
Part B -- The Retriever.

Multi-stage pipeline:
  Stage 1  CLIP semantic vector search (FAISS)            -> top STAGE1_TOP_K
  Stage 2  Attribute-match scoring against parsed query, THEN a "hard
           negative mining" merge (union of the top candidates by pure
           semantic score and by pure attribute score, not just their
           blended average) -> top STAGE2_TOP_K. This matters because a
           blended sort alone can bury a candidate that's a near-perfect
           attribute match but a middling semantic match (or vice versa);
           unioning both rankings before cutting gives each signal a chance
           to rescue a candidate the other one would have dropped.
  Stage 3  BLIP image-text-matching (ITM) re-ranking        -> top STAGE3_TOP_K
           (LLM verification, if enabled, only runs on the top
           MAX_LLM_VERIFY_CANDIDATES of these -- see retriever/llm_verify.py
           and common/llm_cache.py for why: verifying every stage-3
           candidate burns free-tier Gemini quota fast for little benefit,
           since the LLM's judgment mostly only changes the ranking among
           the handful of candidates that are already close contenders.)
  Stage 4  Composite score = weighted(semantic, attribute, itm, [llm_verify])
  Stage 5  Reflection (only triggered on low confidence): find which attribute
           category is most often unmatched among the current top results,
           re-weight attribute scoring to emphasise it, and re-score the SAME
           stage-2 candidate pool -- no new FAISS search needed, and BOTH the
           ITM score AND any LLM verification score are reused from the first
           pass rather than recomputed, so a reflection retry costs no extra
           model calls (this used to silently double LLM-quota usage on every
           low-confidence query -- fixed here).

Every returned result carries its component scores and a per-attribute
explanation (matched/unmatched + wanted/found), satisfying the
explainability goal from the design doc without any hardcoded per-query logic.
"""
import json
import os

import faiss
import numpy as np

from config import (
    DEVICE, FAISS_INDEX_PATH, METADATA_PATH, STAGE1_TOP_K, STAGE2_TOP_K, STAGE3_TOP_K,
    STAGE2_HARD_NEGATIVE_TOPN, FINAL_K_DEFAULT, SCORE_WEIGHTS, LOW_CONFIDENCE_THRESHOLD,
    ITM_MODEL_NAME, CLOTHING_TYPES, OBJECTS, STYLE_ORDER, SCENE_SOFT_MATCH_CREDIT,
    MAX_LLM_VERIFY_CANDIDATES,
)

# Threshold above which a graded label-similarity score counts as "matched"
# for the per-attribute explanation (see _label_set_similarity below).
LABEL_MATCH_THRESHOLD = 0.75


def _attribute_subquery_text(attrs: dict, category: str):
    """Builds a short text snippet representing ONLY one attribute category,
    so search()'s reflection step can run a second, targeted FAISS search
    against just that signal instead of only re-weighting the same pool."""
    if category == "color" and attrs.get("colors"):
        return " and ".join(f"{c} clothing" for c in attrs["colors"])
    if category == "clothing" and attrs.get("clothing_types"):
        return " and ".join(attrs["clothing_types"])
    if category == "object" and attrs.get("objects"):
        return " and ".join(attrs["objects"])
    if category == "scene" and attrs.get("environment"):
        return f"a photo taken in a {attrs['environment']}"
    if category == "style" and attrs.get("style"):
        return f"a {attrs['style']} outfit"
    return None
from common.color_space import color_name_similarity
from indexer.feature_extractor import FeatureExtractor
from retriever.query_parser import parse_query
from retriever.llm_verify import is_enabled as llm_enabled, verify_candidate


class Retriever:
    def __init__(self, mock: bool = False):
        self.mock = mock
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError("Index not found -- run `python -m indexer.build_index` first.")

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH) as f:
            self.metadata = [json.loads(line) for line in f]

        self.extractor = FeatureExtractor(mock=mock)  # reused for query text embedding
        self._itm_model = None
        self._itm_processor = None

        # Precomputed CLIP text-label embeddings (already computed once in
        # FeatureExtractor._precompute_label_embeddings), reused here for
        # graded clothing/object similarity instead of plain set-intersection
        # hit/miss. Zero extra model calls.
        if not mock:
            self._cloth_label_emb = dict(zip(CLOTHING_TYPES, self.extractor.cloth_emb))
            self._obj_label_emb = dict(zip(OBJECTS, self.extractor.obj_emb))
        else:
            self._cloth_label_emb, self._obj_label_emb = {}, {}

    def _load_itm(self):
        if self._itm_model is None and not self.mock:
            from transformers import BlipForImageTextRetrieval, BlipProcessor
            self._itm_model = BlipForImageTextRetrieval.from_pretrained(ITM_MODEL_NAME).to(DEVICE).eval()
            self._itm_processor = BlipProcessor.from_pretrained(ITM_MODEL_NAME)

    # ---------------- Stage 1 ----------------
    def _embed_query(self, query: str):
        if self.mock:
            import hashlib
            h = int(hashlib.sha256(query.encode()).hexdigest(), 16)
            rng = np.random.default_rng(h % (2**32))
            v = rng.normal(size=512).astype("float32")
            return v / np.linalg.norm(v)

        import torch
        inputs = self.extractor.clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            feat = self.extractor.clip_model.get_text_features(**inputs)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output
        feat = torch.nn.functional.normalize(feat, dim=-1)
        return feat[0].detach().cpu().numpy().astype("float32")

    def _stage1(self, query_vec, top_k):
        qv = query_vec.reshape(1, -1).copy()
        faiss.normalize_L2(qv)
        scores, ids = self.index.search(qv, min(top_k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    # ---------------- Stage 2: attribute matching ----------------
    @staticmethod
    def _color_similarity(wanted: str, found: str) -> float:
        """Graded color match instead of exact-name-only: 'blue' vs 'navy'
        scores partway between a miss and a hit. Delegates to
        common/color_space.py, which compares colors in CIELAB rather than
        raw RGB -- CIELAB tracks human color perception much more closely
        (RGB Euclidean distance under-weights hue and over-weights
        brightness, so e.g. 'navy' can end up numerically closer to 'black'
        than to 'blue' in plain RGB space). Uses only the color *name(s)*
        already stored in metadata.jsonl -- no re-indexing needed."""
        return color_name_similarity(wanted, found)

    @staticmethod
    def _style_similarity(wanted: str, found: str) -> float:
        """Ordinal formality distance instead of exact-match-only: a
        'professional' query against a 'business casual' image scores
        partial credit instead of zero, since they're adjacent on the
        formality spectrum. Uses only the single style label already stored
        at index time -- no re-indexing needed."""
        if not wanted or not found:
            return 0.0
        if wanted == found:
            return 1.0
        if wanted not in STYLE_ORDER or found not in STYLE_ORDER:
            return 0.0
        d = abs(STYLE_ORDER.index(wanted) - STYLE_ORDER.index(found))
        span = len(STYLE_ORDER) - 1
        return max(0.0, 1.0 - d / span)

    @staticmethod
    def _scene_similarity(wanted: str, record: dict) -> float:
        """Exact match on the image's #1 scene guess scores 1.0. If the
        wanted environment instead shows up as one of the image's *other*
        top-scoring scene guesses (config.TOPK_SCENE_STYLE, stored at index
        time as scene_topk), it gets partial credit instead of an outright
        miss -- scenes genuinely are ambiguous from a single photo (e.g. a
        covered patio could plausibly be "office" or "cafe"), so an
        all-or-nothing match on only the argmax throws away real signal.
        Falls back to plain exact-match if scene_topk isn't present (e.g. an
        index built before this field existed)."""
        found = record.get("scene")
        if not wanted:
            return 0.0
        if wanted == found:
            return 1.0
        topk = record.get("scene_topk") or []
        other_names = {name for name, _ in topk[1:]}  # skip index 0, that's `found`, already checked above
        if wanted in other_names:
            return SCENE_SOFT_MATCH_CREDIT
        return 0.0

    # CLIP text embeddings for narrow same-domain vocab (all CLOTHING_TYPES /
    # OBJECTS share one sentence template) sit much closer together than raw
    # cosine similarity suggests -- unrelated pairs like "raincoat" vs "dress"
    # still score ~0.85-0.89, not near 0. Rescale so only similarity clearly
    # above that same-domain noise floor counts as a genuine near-synonym;
    # exact matches (raw sim 1.0) still rescale to 1.0 automatically.
    LABEL_SIM_NOISE_FLOOR = 0.93

    @staticmethod
    def _label_set_similarity(wanted: list, found: set, label_emb: dict) -> float:
        """Graded multi-label similarity using CLIP text embeddings instead of
        plain set-intersection, rescaled past the same-domain noise floor (see
        LABEL_SIM_NOISE_FLOOR) so unrelated-but-same-category words don't get
        counted as a match."""
        if not wanted or not found or not label_emb:
            return 0.0
        import torch
        best_raw = 0.0
        for w in wanted:
            w_vec = label_emb.get(w)
            if w_vec is None:
                continue
            for f in found:
                f_vec = label_emb.get(f)
                if f_vec is None:
                    continue
                best_raw = max(best_raw, float(torch.dot(w_vec, f_vec)))
        floor = Retriever.LABEL_SIM_NOISE_FLOOR
        return max(0.0, (best_raw - floor) / (1.0 - floor))

    def _attribute_score(self, attrs: dict, record: dict, category_weights: dict):
        parts, explain = [], {}
        caption = (record.get("caption") or "").lower()

        if attrs["colors"]:
            rec_colors = [c for c in (record.get("upper_color"), record.get("lower_color")) if c]
            # Include per-garment colors from the segmentation branch (e.g. a
            # separately-identified "Scarf": "red") on top of upper/lower --
            # richer than the old two-bucket-only matching, though a necktie
            # specifically still isn't separable (see garment_analysis.py
            # "Known limitations").
            rec_colors += list(record.get("garment_colors", {}).values())
            rec_colors = list(dict.fromkeys(rec_colors))  # de-dupe, keep order
            sim = max((Retriever._color_similarity(w, rc) for w in attrs["colors"] for rc in rec_colors), default=0.0)
            parts.append((sim, category_weights["color"]))
            explain["color"] = {"matched": sim >= 0.7, "wanted": attrs["colors"], "found": rec_colors,
                                 "similarity": round(sim, 2)}

        if attrs["clothing_types"]:
            rec_tags = set(record.get("clothing_tags", []))
            # Union in clothing words the BLIP caption itself mentions -- catches
            # garments the zero-shot CLIP classifier missed but the caption got
            # right (e.g. explanation showing found=[] despite "raincoat" being
            # right there in the caption). No re-indexing needed: caption is
            # already stored in metadata.jsonl.
            caption_tags = {c for c in CLOTHING_TYPES if c in caption}
            combined_tags = rec_tags | caption_tags
            sim = self._label_set_similarity(attrs["clothing_types"], combined_tags, self._cloth_label_emb)
            parts.append((sim, category_weights["clothing"]))
            explain["clothing"] = {"matched": sim >= LABEL_MATCH_THRESHOLD, "wanted": attrs["clothing_types"],
                                    "found": sorted(combined_tags), "similarity": round(sim, 2)}

        if attrs.get("objects"):
            # record.get("objects") is populated at index time (zero-shot CLIP
            # vs config.OBJECTS, unioned with segmentation-detected
            # accessories like a bag) -- see indexer/feature_extractor.py.
            # indexer/add_object_tags.py remains available as a standalone
            # backfill for indexes built with an older version of this
            # pipeline that predates this being automatic.
            rec_objs = set(record.get("objects", []))
            caption_objs = {o for o in attrs["objects"] if o in caption}
            combined_objs = rec_objs | caption_objs
            sim = self._label_set_similarity(attrs["objects"], combined_objs, self._obj_label_emb)
            parts.append((sim, category_weights["object"]))
            explain["object"] = {"matched": sim >= LABEL_MATCH_THRESHOLD, "wanted": attrs["objects"],
                                  "found": sorted(combined_objs), "similarity": round(sim, 2)}

        if attrs["environment"]:
            sim = Retriever._scene_similarity(attrs["environment"], record)
            parts.append((sim, category_weights["scene"]))
            explain["scene"] = {"matched": sim >= 1.0, "wanted": attrs["environment"], "found": record.get("scene"),
                                 "similarity": round(sim, 2)}

        if attrs["style"]:
            sim = Retriever._style_similarity(attrs["style"], record.get("style"))
            parts.append((sim, category_weights["style"]))
            explain["style"] = {"matched": sim >= 0.8, "wanted": attrs["style"], "found": record.get("style"),
                                 "similarity": round(sim, 2)}

        if not parts:
            return 1.0, explain  # no structured constraints requested -- don't penalise
        score = sum(v * wt for v, wt in parts) / sum(wt for _, wt in parts)
        return score, explain

    # ---------------- Stage 3: ITM rerank ----------------
    def _itm_score(self, query: str, image_path: str) -> float:
        if self.mock:
            import hashlib
            h = int(hashlib.sha256((query + image_path).encode()).hexdigest(), 16)
            return (h % 1000) / 1000.0

        import torch
        self._load_itm()
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self._itm_processor(images=image, text=query, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            out = self._itm_model(**inputs)
            probs = torch.softmax(out.itm_score, dim=1)
        return float(probs[0, 1])

    @staticmethod
    def _normalise_weights():
        w = dict(SCORE_WEIGHTS)
        if not llm_enabled():
            drop = w.pop("llm_verify")
            total = sum(w.values())
            w = {k: v + (v / total) * drop for k, v in w.items()}
        return w

    # ---------------- Public search ----------------
    def search(self, query: str, top_k: int = FINAL_K_DEFAULT, allow_reflection: bool = True):
        attrs = parse_query(query)
        query_vec = self._embed_query(query)

        stage1 = self._stage1(query_vec, STAGE1_TOP_K)
        if not stage1:
            return {"query": query, "parsed_attributes": attrs, "results": [], "confidence": 0.0,
                    "low_confidence_warning": True}

        weights = self._normalise_weights()
        base_cat_weights = {"color": 1.0, "clothing": 1.0, "scene": 1.0, "style": 1.0, "object": 1.0}

        results = self._score_pool(query, attrs, stage1, weights, base_cat_weights)

        reflected = False
        if allow_reflection and results and results[0]["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            top_n = results[:min(5, len(results))]
            miss_counts = {"color": 0, "clothing": 0, "scene": 0, "style": 0, "object": 0}
            for r in top_n:
                for cat, info in r["explanation"].items():
                    if not info["matched"]:
                        miss_counts[cat] += 1
            worst_cat = max(miss_counts, key=miss_counts.get) if any(miss_counts.values()) else None
            if worst_cat:
                # Targeted second search: embed a short snippet for ONLY the
                # most-often-missing attribute, run FAISS on it, and union
                # any new candidates into the pool before rescoring -- this
                # can recover images the full-sentence embedding ranked too
                # low to reach the original stage-1 cut.
                extra_stage1 = []
                sub_text = _attribute_subquery_text(attrs, worst_cat)
                if sub_text:
                    sub_vec = self._embed_query(sub_text)
                    extra_stage1 = self._stage1(sub_vec, STAGE1_TOP_K)

                merged_stage1 = dict(stage1)
                for faiss_id, sem_score in extra_stage1:
                    merged_stage1.setdefault(faiss_id, sem_score)
                merged_stage1 = list(merged_stage1.items())

                boosted = dict(base_cat_weights)
                boosted[worst_cat] = 2.5
                new_results = self._score_pool(query, attrs, merged_stage1, weights, boosted, reuse_itm=results)
                if new_results and new_results[0]["confidence"] > results[0]["confidence"]:
                    results = new_results
                    reflected = True

        top_results = results[:top_k]
        overall_confidence = top_results[0]["confidence"] if top_results else 0.0

        return {
            "query": query,
            "parsed_attributes": attrs,
            "results": top_results,
            "confidence": round(overall_confidence, 3),
            "low_confidence_warning": overall_confidence < LOW_CONFIDENCE_THRESHOLD,
            "reflection_triggered": reflected,
        }

    def _score_pool(self, query, attrs, stage1, weights, category_weights, reuse_itm=None):
        itm_cache = {r["image_path"]: r["itm_score"] for r in reuse_itm} if reuse_itm else {}
        # Reused across a reflection retry so a low-confidence query never
        # pays for the same Gemini verification call twice (see module
        # docstring, Stage 5) -- reuse_itm carries llm_score from the first
        # pass too, despite the name (kept as one param to avoid re-plumbing
        # two parallel "previous results" arguments through search()).
        llm_cache = {r["image_path"]: r["llm_score"] for r in reuse_itm if r.get("llm_score") is not None} \
            if reuse_itm else {}

        stage2_all = []
        for faiss_id, sem_score in stage1:
            record = self.metadata[faiss_id]
            attr_score, explain = self._attribute_score(attrs, record, category_weights)
            stage2_all.append((faiss_id, sem_score, attr_score, explain, record))

        # Hard-negative-mining merge: union the top-N by semantic score alone
        # and the top-N by attribute score alone, rather than only keeping
        # whatever a single blended sort would have kept. See module
        # docstring, Stage 2.
        by_sem = sorted(stage2_all, key=lambda x: x[1], reverse=True)[:STAGE2_HARD_NEGATIVE_TOPN]
        by_attr = sorted(stage2_all, key=lambda x: x[2], reverse=True)[:STAGE2_HARD_NEGATIVE_TOPN]
        merged = {x[0]: x for x in by_sem}
        for x in by_attr:
            merged.setdefault(x[0], x)
        stage2 = sorted(merged.values(), key=lambda x: 0.5 * x[1] + 0.5 * x[2], reverse=True)[:STAGE2_TOP_K]

        stage3 = sorted(stage2, key=lambda x: 0.5 * x[1] + 0.5 * x[2], reverse=True)[:STAGE3_TOP_K]

        scored = []
        for rank, (faiss_id, sem_score, attr_score, explain, record) in enumerate(stage3):
            path = record["image_path"]
            itm = itm_cache.get(path)
            if itm is None:
                itm = self._itm_score(query, path)

            llm_score = llm_cache.get(path)
            if llm_score is None and llm_enabled() and rank < MAX_LLM_VERIFY_CANDIDATES:
                v = verify_candidate(query, record.get("caption", ""), {
                    "colors": [record.get("upper_color"), record.get("lower_color")],
                    "clothing_tags": record.get("clothing_tags"),
                    "scene": record.get("scene"), "style": record.get("style"),
                })
                if v:
                    llm_score = v["score"]

            components = {"semantic": sem_score, "attribute": attr_score, "itm": itm}
            if llm_score is not None:
                components["llm_verify"] = llm_score
                confidence = sum(components[k] * weights[k] for k in components)
            else:
                active_w = {k: weights[k] for k in components}
                norm = sum(active_w.values())
                confidence = sum(components[k] * (active_w[k] / norm) for k in components)

            scored.append({
                "image_path": path,
                "confidence": round(float(confidence), 3),
                "semantic_score": round(float(sem_score), 3),
                "attribute_score": round(float(attr_score), 3),
                "itm_score": round(float(itm), 3),
                "llm_score": round(float(llm_score), 3) if llm_score is not None else None,
                "caption": record.get("caption"),
                "scene": record.get("scene"), "style": record.get("style"),
                "upper_color": record.get("upper_color"), "lower_color": record.get("lower_color"),
                "garment_colors": record.get("garment_colors", {}),
                "objects": record.get("objects", []),
                "explanation": explain,
            })

        scored.sort(key=lambda r: r["confidence"], reverse=True)
        return scored
