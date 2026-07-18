"""
Tiny disk-backed cache for Gemini calls (query parsing + candidate
verification).

Why this exists (concrete bug found in the v1 eval run):
    LLM verification was being called on every stage-3 candidate (up to
    STAGE3_TOP_K=10) for every query, and again on reflection retries -- that
    is up to ~20+ Gemini calls for a single 5-query eval run. The Gemini free
    tier's *daily* request quota for a given model is commonly in the
    low tens (the eval log this project shipped with hit
    "RESOURCE_EXHAUSTED ... quotaValue: 20" after only a handful of queries).
    Two independent fixes are applied together:
      1. retriever/search.py now only sends the top MAX_LLM_VERIFY_CANDIDATES
         candidates to the LLM instead of all of stage 3 (see config.py), and
         reuses cached scores across the reflection retry instead of
         re-verifying the same candidates twice.
      2. This cache: identical (prompt, model) pairs are never sent twice,
         even across separate `python demo.py` runs -- e.g. re-running
         `--eval` twice in a row during development costs zero extra quota
         the second time.

Format: one JSON object per line in data/index/llm_cache.jsonl, keyed by a
hash of (kind, model, prompt) so parser-cache and verify-cache entries can
safely share one file without colliding. This is a cache, not a database --
safe to delete at any time; a missing/corrupt file just means "cold cache",
never a crash (mirrors the same "never hard-fail on auxiliary state" pattern
used by the indexer's checkpoint files).
"""
import hashlib
import json
import os
import threading

from config import INDEX_DIR

_CACHE_PATH = os.path.join(INDEX_DIR, "llm_cache.jsonl")
_lock = threading.Lock()
_mem_cache = None  # lazily loaded dict, shared for the life of the process


def _key(kind: str, model: str, prompt: str) -> str:
    h = hashlib.sha256(f"{kind}::{model}::{prompt}".encode("utf-8")).hexdigest()
    return h


def _load():
    global _mem_cache
    if _mem_cache is not None:
        return _mem_cache
    cache = {}
    if os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        cache[rec["key"]] = rec["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue  # tolerate a truncated/corrupt line, skip it
        except OSError:
            pass  # cold start is fine -- this is only a speed/quota optimisation
    _mem_cache = cache
    return cache


def get(kind: str, model: str, prompt: str):
    """Returns the cached value, or None on a cache miss."""
    cache = _load()
    return cache.get(_key(kind, model, prompt))


def put(kind: str, model: str, prompt: str, value):
    """Stores a value and appends it to disk immediately (same
    flush+fsync-on-write pattern as the indexer's checkpoint files, so a
    crash right after a Gemini call doesn't lose the quota we just spent)."""
    cache = _load()
    k = _key(kind, model, prompt)
    cache[k] = value
    os.makedirs(INDEX_DIR, exist_ok=True)
    with _lock:
        with open(_CACHE_PATH, "a") as f:
            f.write(json.dumps({"key": k, "value": value}) + "\n")
            f.flush()
            os.fsync(f.fileno())
