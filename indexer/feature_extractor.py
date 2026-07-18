"""
Multi-branch feature extraction for images.

Branches (all reuse ONE loaded CLIP model where possible, so scene/style/
clothing-tag/object classification is "free" zero-shot lookups rather than
several separately trained classifiers):
  1. CLIP semantic embedding      -> stage-1 ANN search vector
  2. BLIP caption                 -> free-form description for reranking/LLM
  3. Zero-shot scene (top-K)      -> CLIP image vec vs ENVIRONMENTS text vecs
  4. Zero-shot style (top-K)      -> CLIP image vec vs STYLES text vecs
  5. Zero-shot clothing tags      -> CLIP image vec vs CLOTHING_TYPES (multi-label)
  6. Zero-shot object tags        -> CLIP image vec vs OBJECTS (multi-label) --
     folded in here directly (previously a separate optional post-process
     script, indexer/add_object_tags.py -- kept around for backward
     compatibility with older indexes, but no longer a required manual step)
  7. Garment colors + accessories -> indexer/garment_analysis.py (SegFormer
     human parsing, falling back to indexer/color_utils.py automatically)

process_batch() is the real workhorse: it batches images through CLIP/BLIP in
one forward pass per batch, which is where the actual throughput win comes
from (GPU or CPU) -- see indexer/build_index.py for how this combines with
process-level parallelism for the garment-analysis stage.
"""
import hashlib

import numpy as np
from PIL import Image

from config import (
    DEVICE, CLIP_MODEL_NAME, CAPTION_MODEL_NAME, ENVIRONMENTS, STYLES,
    CLOTHING_TYPES, OBJECTS, TOPK_SCENE_STYLE, GARMENT_LABEL_TO_CLOTHING_TYPE,
)
from indexer.color_utils import extract_garment_colors
from indexer.garment_analysis import _EMPTY_RESULT_EXTRA


class FeatureExtractor:
    def __init__(self, mock: bool = False):
        """
        mock=True skips all model downloads/inference and produces deterministic
        pseudo-random (but stable, hash-seeded) embeddings instead. This exists
        purely to exercise the rest of the pipeline (FAISS indexing, attribute
        scoring, reranking, reflection, CLI) in environments without internet
        access to Hugging Face -- it is NOT a substitute for real inference and
        must never be used to produce actual results.
        """
        self.mock = mock
        if mock:
            return

        import torch
        from transformers import CLIPModel, CLIPProcessor, BlipForConditionalGeneration, BlipProcessor

        self.torch = torch
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        self.caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME).to(DEVICE).eval()
        self.caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)

        self._precompute_label_embeddings()

    # ---------------- CLIP text/label embeddings (for zero-shot) ----------------
    def _precompute_label_embeddings(self):
        with self.torch.inference_mode():
            self.env_emb = self._encode_texts([f"a photo taken in a {e}" for e in ENVIRONMENTS])
            self.style_emb = self._encode_texts([f"a {s} outfit" for s in STYLES])
            self.cloth_emb = self._encode_texts([f"a person wearing a {c}" for c in CLOTHING_TYPES])
            self.obj_emb = self._encode_texts([f"a photo containing a {o}" for o in OBJECTS])

    @staticmethod
    def _unwrap_features(output):
        """transformers >=5.0 sometimes returns a BaseModelOutputWithPooling
        object from get_text_features/get_image_features instead of a plain
        tensor. Handle both so this works across transformers versions."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output

    def _encode_texts(self, texts):
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(DEVICE)
        feats = self._unwrap_features(self.clip_model.get_text_features(**inputs))
        return self.torch.nn.functional.normalize(feats, dim=-1)

    def _encode_images_clip(self, images):
        inputs = self.clip_processor(images=images, return_tensors="pt").to(DEVICE)
        with self.torch.inference_mode():
            feats = self._unwrap_features(self.clip_model.get_image_features(**inputs))
        return self.torch.nn.functional.normalize(feats, dim=-1)

    # ---------------- Mock mode ----------------
    def _mock_vec(self, key, dim):
        h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        v = rng.normal(size=dim).astype("float32")
        return v / np.linalg.norm(v)

    def _process_mock(self, path, image):
        colors = extract_garment_colors(image)
        h = int(hashlib.sha256(path.encode()).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        scene = ENVIRONMENTS[rng.integers(0, len(ENVIRONMENTS))]
        style = STYLES[rng.integers(0, len(STYLES))]
        return {
            "clip_embedding": self._mock_vec(path, 512).tolist(),
            "caption": f"[mock] a person wearing a {colors.get('upper_color') or 'unknown'} top "
                       f"and {colors.get('lower_color') or 'unknown'} bottoms",
            "scene": scene, "scene_confidence": 0.5, "scene_topk": [[scene, 0.5]],
            "style": style, "style_confidence": 0.5, "style_topk": [[style, 0.5]],
            "clothing_tags": [], "objects": [],
            **colors,
            "image_path": path,
        }

    # ---------------- Public API ----------------
    def process_image(self, path: str) -> dict:
        return self.process_batch([path])[0]

    def process_batch(self, image_paths, precomputed_colors=None):
        images = [Image.open(p).convert("RGB") for p in image_paths]

        if self.mock:
            return [self._process_mock(p, img) for p, img in zip(image_paths, images)]

        clip_vecs = self._encode_images_clip(images)  # (B, D)

        cap_inputs = self.caption_processor(images=images, return_tensors="pt").to(DEVICE)
        with self.torch.inference_mode():
            cap_ids = self.caption_model.generate(**cap_inputs, max_new_tokens=30)
        captions = self.caption_processor.batch_decode(cap_ids, skip_special_tokens=True)

        env_scores = self.torch.softmax(clip_vecs @ self.env_emb.T * 100, dim=-1)
        style_scores = self.torch.softmax(clip_vecs @ self.style_emb.T * 100, dim=-1)
        cloth_sims = self.torch.sigmoid((clip_vecs @ self.cloth_emb.T) * 20 - 10)  # independent multi-label
        obj_sims = self.torch.sigmoid((clip_vecs @ self.obj_emb.T) * 20 - 10)      # independent multi-label

        results = []
        for j, path in enumerate(image_paths):
            scene_topk = self._topk(env_scores[j], ENVIRONMENTS, TOPK_SCENE_STYLE)
            style_topk = self._topk(style_scores[j], STYLES, TOPK_SCENE_STYLE)
            cloth_tags = set(CLOTHING_TYPES[k] for k, s in enumerate(cloth_sims[j].tolist()) if s > 0.5)
            obj_tags = set(OBJECTS[k] for k, s in enumerate(obj_sims[j].tolist()) if s > 0.5)

            colors = (precomputed_colors or {}).get(path)
            if colors is None:
                colors = extract_garment_colors(images[j])
            # extract_garment_colors / GarmentAnalyzer output always carries
            # garment_colors/accessories/segmentation_labels (possibly empty)
            # -- see indexer/garment_analysis.py's _EMPTY_RESULT_EXTRA -- but
            # guard defensively anyway in case an older checkpoint file
            # (from before this field existed) is being resumed from.
            for k, v in _EMPTY_RESULT_EXTRA.items():
                colors.setdefault(k, v)

            # Union in clothing/object words the segmentation branch found
            # directly (e.g. a real "Scarf" pixel region), on top of the pure
            # zero-shot CLIP guesses -- same "union, don't replace"
            # philosophy used for the caption-word union in retriever/search.py.
            for seg_label in colors["segmentation_labels"]:
                mapped = GARMENT_LABEL_TO_CLOTHING_TYPE.get(seg_label)
                if mapped:
                    cloth_tags.add(mapped)
            obj_tags |= set(colors["accessories"])

            results.append({
                "clip_embedding": clip_vecs[j].detach().cpu().numpy().tolist(),
                "caption": captions[j].strip(),
                "scene": scene_topk[0][0], "scene_confidence": scene_topk[0][1], "scene_topk": scene_topk,
                "style": style_topk[0][0], "style_confidence": style_topk[0][1], "style_topk": style_topk,
                "clothing_tags": sorted(cloth_tags),
                "objects": sorted(obj_tags),
                **colors,
                "image_path": path,
            })
        return results

    @staticmethod
    def _topk(scores_row, labels, k):
        """Top-k (label, rounded_score) pairs from a 1D softmax score tensor,
        sorted descending. k=1 recovers the old argmax-only behaviour."""
        vals = scores_row.detach().cpu().numpy()
        order = np.argsort(-vals)[:k]
        return [[labels[i], round(float(vals[i]), 3)] for i in order]
