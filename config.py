"""
Central configuration for the Fashion & Context Retrieval system.
All tunable knobs live here so behaviour can be changed without touching pipeline code.
"""
import os

# ---------- CPU thread configuration ----------
# On CPU-only machines, PyTorch/BLAS thread pools sometimes default to a
# lower count than the box actually has (or get pinned to 1 by an inherited
# env var). This does NOT change any numerical results -- it only controls
# how many CPU threads the existing math ops are allowed to use -- so it is
# purely a speed lever. setdefault() means it never overrides a value you've
# already set yourself in the shell.
_CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(_CPU_COUNT))
# Silences a HuggingFace tokenizers warning/slowdown that shows up when a
# process that already forked (our ProcessPoolExecutor color-extraction step)
# later touches a tokenizer -- harmless either way, just quieter and slightly
# faster.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def get_device():
    try:
        import torch
        torch.set_num_threads(_CPU_COUNT)
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE = get_device()

# ---------- Models ----------
# CLIP: the semantic backbone. ViT-B/32 is the best speed/quality tradeoff for a
# CPU-friendly assignment; swap to ViT-L/14 if you have a GPU and want better recall.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# BLIP captioning: gives every image a free-form text description, used for
# stage-3 ITM reranking and as context for optional LLM verification.
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
# BLIP ITM (image-text matching head): unlike CLIP's single global embedding,
# this model does real cross-attention between the image and text, which is
# why it is used for the *reranking* stage rather than the initial ANN search
# (too slow to run against the whole database, but fine for the top ~30-100).
ITM_MODEL_NAME = "Salesforce/blip-itm-base-coco"

# ---------- Paths ----------
DATA_DIR = os.environ.get("FASHION_DATA_DIR", "data/images")
INDEX_DIR = os.environ.get("FASHION_INDEX_DIR", "data/index")
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "clip.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.jsonl")

# ---------- Controlled vocabularies ----------
# Kept broad and generic -- NOT tailored to the 5 eval queries specifically --
# so the system generalises to unseen phrasing (the zero-shot requirement).
# Reference RGB values are rough perceptual anchors used only for nearest-name
# lookup of extracted dominant colors.
COLORS = {
    "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
    "red": (200, 30, 30), "maroon": (128, 0, 0), "orange": (230, 126, 34),
    "yellow": (241, 196, 15), "gold": (212, 175, 55), "green": (39, 174, 96),
    "olive": (128, 128, 0), "teal": (0, 128, 128), "blue": (41, 128, 185),
    "navy": (10, 25, 70), "sky blue": (135, 206, 235), "purple": (142, 68, 173),
    "lavender": (200, 162, 230), "pink": (231, 84, 128), "brown": (101, 67, 33),
    "tan": (210, 180, 140), "beige": (222, 202, 173), "khaki": (195, 176, 145),
    "cream": (255, 253, 208), "burgundy": (128, 0, 32), "silver": (192, 192, 192),
    "denim": (79, 100, 137),
}

CLOTHING_TYPES = [
    "blazer", "suit jacket", "button-down shirt", "dress shirt", "t-shirt",
    "polo shirt", "hoodie", "sweater", "sweatshirt", "cardigan", "jacket",
    "coat", "trench coat", "raincoat", "windbreaker", "vest", "jeans",
    "trousers", "chinos", "shorts", "skirt", "dress", "jumpsuit", "overalls",
    "tie", "bowtie", "blouse", "tank top", "leggings", "joggers",
    # Added for the SegFormer garment-parsing branch (see
    # indexer/garment_analysis.py) -- these are classes the segmentation
    # model can localise directly (as opposed to only being CLIP zero-shot
    # guesses), so recall on them is meaningfully better than the rest of
    # this list.
    "scarf", "belt", "hat",
]

ENVIRONMENTS = ["office", "urban street", "park", "home", "cafe", "studio", "gym", "store"]

# How many of the top-scoring ENVIRONMENTS/STYLES (by CLIP zero-shot softmax)
# to keep per image, beyond just the single argmax. Storing e.g. top-3 instead
# of only top-1 is what lets retrieval give partial credit to a "2nd place"
# scene/style guess instead of an all-or-nothing single label (see
# retriever/search.py::_scene_similarity / _style_similarity).
TOPK_SCENE_STYLE = 3

STYLES = ["formal", "business casual", "smart casual", "casual", "streetwear", "athleisure", "professional"]

# Same formality set as STYLES, but ordered along a casual->formal spectrum.
# Used only for SOFT style scoring (adjacent styles get partial credit instead
# of an all-or-nothing 0/1 match) -- see retriever/search.py::_style_similarity.
STYLE_ORDER = ["athleisure", "streetwear", "casual", "smart casual", "business casual", "professional", "formal"]
assert set(STYLE_ORDER) == set(STYLES), "STYLE_ORDER must contain exactly the STYLES values"

# "Poor man's object detection": a small vocabulary classified zero-shot
# against the CLIP image embedding we already computed at index time (see
# indexer/add_object_tags.py) -- no new model, no re-encoding of images, no
# re-running the 8-hour CLIP/BLIP stage. Catches cases (e.g. "park bench")
# that the BLIP caption or the ENVIRONMENTS/CLOTHING_TYPES classifiers miss
# entirely. Not as precise as a real detector (Grounding DINO / OWL-ViT), but
# essentially free.
OBJECTS = [
    "bench", "chair", "sofa", "table", "desk", "tree", "backpack", "handbag",
    "dog", "bicycle", "car", "umbrella", "door", "window", "plant", "stairs",
    "sunglasses",
]
OBJECT_SCORE_THRESHOLD = 0.5  # same sigmoid convention as CLOTHING_TYPES

# ---------- Garment segmentation (indexer/garment_analysis.py) ----------
# Replaces the old "MediaPipe pose landmarks -> proportional torso/leg crop"
# approach with real per-pixel garment masks from a human-parsing model, so
# color extraction no longer mixes in background/skin/other-garment pixels
# (the #1 accuracy bottleneck identified during the v1 eval run -- e.g. a
# blue shirt getting misread as gray because the crop box was ~40% shirt,
# 60% background).
#
# mattmdjaga/segformer_b2_clothes: a SegFormer-B2 model fine-tuned on the ATR
# human-parsing dataset. ~28M parameters (~110MB download), CPU inference is
# roughly 0.3-1.5s/image at the resolution used below -- an order of
# magnitude lighter than a general object detector like Grounding DINO/OWL-
# ViT while still giving labelled, pixel-accurate regions for the garment
# classes that matter here. See indexer/garment_analysis.py for the full
# rationale, the fallback path, and documented weaknesses (small/thin
# classes like "Belt" and "Scarf" are the least reliable, per the model's
# own published per-class accuracy).
SEGMENTATION_MODEL_NAME = os.environ.get("SEGMENTATION_MODEL_NAME", "mattmdjaga/segformer_b2_clothes")
# Set FASHION_USE_SEGMENTATION=0 to force the legacy MediaPipe-pose pipeline
# (e.g. very RAM-constrained machine, or --legacy-color on the CLI). The
# segmentation branch degrades to this same legacy path automatically and
# per-image if the model can't be loaded/downloaded or produces no usable
# mask for a given photo -- this flag is only for opting out proactively.
USE_SEGMENTATION = os.environ.get("FASHION_USE_SEGMENTATION", "1") != "0"
# Images are downscaled to this max side length before the segmentation
# forward pass purely for CPU latency -- garment color is a low-frequency
# signal, so this loses essentially no accuracy while cutting runtime a lot
# versus running at full photo resolution.
SEGMENTATION_MAX_SIDE = 384
# A label must cover at least this fraction of the (resized) image's pixels
# to be trusted -- filters out single stray misclassified pixels, especially
# important for the model's weakest classes (belt/scarf).
SEGMENTATION_MIN_MASK_FRAC = 0.012

# SegFormer-B2-clothes label id -> name (from the model's own config, kept
# here as a documented fallback in case a future checkpoint swap doesn't
# expose id2label the same way -- garment_analysis.py prefers reading it
# live off the loaded model whenever possible).
SEGMENTATION_LABELS = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe",
    10: "Right-shoe", 11: "Face", 12: "Left-leg", 13: "Right-leg",
    14: "Left-arm", 15: "Right-arm", 16: "Bag", 17: "Scarf",
}
# Which labels we extract a dominant *color* for (skin/hair/background/leg-
# skin labels are excluded on purpose -- they aren't garments).
GARMENT_COLOR_LABELS = ["Upper-clothes", "Pants", "Skirt", "Dress", "Belt", "Bag", "Scarf", "Hat"]
# Maps a SegFormer garment label to the CLOTHING_TYPES vocab word it implies
# (used to enrich clothing_tags at index time, same "union, don't replace"
# philosophy as the existing caption-word union in retriever/search.py).
GARMENT_LABEL_TO_CLOTHING_TYPE = {
    "Skirt": "skirt", "Dress": "dress", "Belt": "belt", "Scarf": "scarf", "Hat": "hat",
}
# Maps a SegFormer accessory label to the OBJECTS vocab word it implies.
GARMENT_LABEL_TO_OBJECT = {"Bag": "handbag", "Sunglasses": "sunglasses"}

# ---------- Retrieval hyperparameters ----------
STAGE1_TOP_K = 100   # initial semantic (FAISS) search
STAGE2_TOP_K = 30    # after attribute filtering
STAGE3_TOP_K = 10    # after ITM re-ranking -> final candidate pool
FINAL_K_DEFAULT = 5

# Weights for the final composite score. Must sum to 1.0.
# llm_verify's weight is automatically redistributed to the other three when
# no GEMINI_API_KEY is configured (see retriever/search.py::_normalise_weights).
SCORE_WEIGHTS = {
    "semantic": 0.35,
    "attribute": 0.30,
    "itm": 0.25,
    "llm_verify": 0.10,
}

LOW_CONFIDENCE_THRESHOLD = 0.45

# Partial credit given to the "scene" attribute when the wanted environment
# isn't the image's single best-scoring scene but IS among its next
# highest-scoring guesses (see config.TOPK_SCENE_STYLE and
# retriever/search.py::_scene_similarity). Scenes don't have a natural
# ordering the way STYLE_ORDER does for formality, so this is a flat partial
# credit rather than a distance-based one.
SCENE_SOFT_MATCH_CREDIT = 0.5

# ---------- Stage 2 candidate selection ("hard negative mining") ----------
# Instead of only keeping the top STAGE2_TOP_K candidates by a *blended*
# semantic+attribute score (which can silently drop a candidate that's a
# near-perfect attribute match but a middling semantic match, or vice
# versa), we keep the union of the top-N purely by semantic score AND the
# top-N purely by attribute score, so both signals get a chance to surface a
# candidate the other one would have buried. Cheap: this only changes which
# stage-1 rows we look at, no extra model inference.
STAGE2_HARD_NEGATIVE_TOPN = 20

# ---------- LLM (optional, free-tier) ----------
# Used ONLY for: (a) a query-parsing fallback for phrasing the rule-based
# parser misses, and (b) final candidate verification. Both are OPTIONAL --
# the system is fully functional with GEMINI_API_KEY unset, using the
# rule-based parser + BLIP-ITM score instead.
#
# Free-tier daily quota is the real constraint here, not capability -- as of
# mid-2026 Google's *full* Flash models sit at a much lower free daily
# request cap than the Flash-Lite variants (this is exactly the 429
# RESOURCE_EXHAUSTED behaviour seen with a stock "gemini-2.5-flash" key), so
# a "-lite" model is the safer default for a project meant to run entirely
# on the free tier. Verify current numbers at
# https://ai.google.dev/gemini-api/docs/rate-limits before relying on this
# for anything beyond development -- Google adjusts these without much
# notice. See common/llm_cache.py and MAX_LLM_VERIFY_CANDIDATES below for
# how this project also minimises the number of calls it makes in the first
# place, independent of which model you point it at.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
# Only the top N stage-3 candidates (by pre-LLM blended score) are sent to
# the LLM for verification, instead of all of STAGE3_TOP_K -- verifying every
# candidate provided little marginal value (the top few are where LLM
# judgment actually changes the ranking) while burning quota fast.
MAX_LLM_VERIFY_CANDIDATES = 3
