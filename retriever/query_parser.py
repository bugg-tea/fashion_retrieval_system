"""
Convert a free-text query into a structured attribute dict, restricted to the
SAME controlled vocabularies used at index time (COLORS, CLOTHING_TYPES,
ENVIRONMENTS, STYLES). This restriction is what keeps the LLM path from
hallucinating: any value the LLM returns that isn't in the controlled
vocabulary is discarded, never trusted blindly.

Two parsing strategies are always both attempted, then merged:
  1. Rule-based keyword/phrase matching -- fast, deterministic, zero cost,
     works fully offline.
  2. Optional LLM parse via Gemini's free tier (only if GEMINI_API_KEY is
     set) -- catches paraphrases the keyword matcher misses (e.g. "throwing
     on a button-up" -> clothing_type="button-down shirt"). Validated against
     the same controlled vocab before being trusted. Uses the `google-genai`
     SDK (the actively-maintained successor to the now-deprecated
     `google-generativeai` package) and is cached on disk via
     common/llm_cache.py, so re-parsing the exact same query text twice
     (e.g. re-running `python demo.py --eval` during development) costs zero
     extra free-tier quota the second time.

Results are unioned rather than one overriding the other: if either method
found "park", environment="park" is used. We have no ground truth to decide
which extractor is "more correct" per query, so silently letting a
non-deterministic LLM output override a deterministic rule-based match would
itself be an unjustified guess -- unioning is the conservative choice.
"""
import json
import re

from config import COLORS, CLOTHING_TYPES, ENVIRONMENTS, STYLES, OBJECTS, GEMINI_API_KEY, GEMINI_MODEL
from common import llm_cache


def _rule_based(query: str) -> dict:
    q = query.lower()
    found = {"colors": [], "clothing_types": [], "environment": None, "style": None, "objects": []}

    for obj in OBJECTS:
        if re.search(rf"\b{re.escape(obj)}\b", q):
            found["objects"].append(obj)

    for color in COLORS:
        if re.search(rf"\b{re.escape(color)}\b", q):
            found["colors"].append(color)

    # Generic words that are the last token of a multi-word clothing type but
    # are too vague on their own to imply that specific type (e.g. "top" should
    # not, by itself, mean "tank top" -- it could be any upper-body garment).
    _GENERIC_LAST_WORDS = {"top", "wear", "outfit", "shirt"}
    for cloth in CLOTHING_TYPES:
        words = cloth.split()
        last_word = words[-1]
        full_match = re.search(rf"\b{re.escape(cloth)}\b", q)
        loose_match = (
            len(words) > 1
            and last_word not in _GENERIC_LAST_WORDS
            and re.search(rf"\b{re.escape(last_word)}\b", q)
        )
        if full_match or loose_match:
            found["clothing_types"].append(cloth)

    for env in ENVIRONMENTS:
        if env in q:
            found["environment"] = env
            break
    if found["environment"] is None and any(w in q for w in ["street", "city", "urban", "sidewalk"]):
        found["environment"] = "urban street"

    for style in STYLES:
        if style in q:
            found["style"] = style
            break
    if found["style"] is None:
        if "business" in q:
            found["style"] = "business casual"
        elif any(w in q for w in ["formal", "professional", "office wear"]):
            found["style"] = "formal"
        elif any(w in q for w in ["casual", "weekend", "relaxed"]):
            found["style"] = "casual"
        elif "streetwear" in q or "street style" in q:
            found["style"] = "streetwear"

    return found


def _llm_based(query: str):
    if not GEMINI_API_KEY:
        return None
    prompt = f"""Extract fashion search attributes from this query as JSON only, no prose, no markdown fences.
Query: "{query}"

Allowed values (use ONLY these, or null/[] if nothing applies -- never invent new values):
colors: {list(COLORS.keys())}
clothing_types: {CLOTHING_TYPES}
environment (single value): {ENVIRONMENTS}
style (single value): {STYLES}
objects: {OBJECTS}

Return strict JSON: {{"colors": [...], "clothing_types": [...], "environment": "..." or null, "style": "..." or null, "objects": [...]}}"""

    cached = llm_cache.get("parse", GEMINI_MODEL, prompt)
    if cached is not None:
        data = cached
    else:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                # response_mime_type="application/json" makes Gemini return
                # valid JSON directly instead of relying on prompt-following
                # alone + a manual ```json fence strip (the old approach) --
                # one less way for this to silently break.
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
            )
            data = json.loads(resp.text)
            llm_cache.put("parse", GEMINI_MODEL, prompt, data)
        except Exception as e:
            print(f"[query_parser] LLM parse skipped due to error: {e}")
            return None

    try:
        # Validate against controlled vocab -- discard anything hallucinated.
        return {
            "colors": [c for c in data.get("colors", []) if c in COLORS],
            "clothing_types": [c for c in data.get("clothing_types", []) if c in CLOTHING_TYPES],
            "environment": data.get("environment") if data.get("environment") in ENVIRONMENTS else None,
            "style": data.get("style") if data.get("style") in STYLES else None,
            "objects": [o for o in data.get("objects", []) if o in OBJECTS],
        }
    except Exception as e:
        print(f"[query_parser] LLM parse skipped due to error: {e}")
        return None


def parse_query(query: str) -> dict:
    rule = _rule_based(query)
    llm = _llm_based(query)

    return {
        "colors": sorted(set(rule["colors"]) | set((llm or {}).get("colors", []))),
        "clothing_types": sorted(set(rule["clothing_types"]) | set((llm or {}).get("clothing_types", []))),
        "environment": rule["environment"] or (llm or {}).get("environment"),
        "style": rule["style"] or (llm or {}).get("style"),
        "objects": sorted(set(rule["objects"]) | set((llm or {}).get("objects", []))),
        "raw_query": query,
        "used_llm": llm is not None,
    }
