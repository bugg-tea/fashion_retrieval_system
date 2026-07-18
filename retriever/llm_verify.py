"""
Optional final-stage LLM verification -- NOT retrieval, NOT RAG. It only grades
candidates that stage 1-3 already retrieved deterministically. Disabled
automatically when GEMINI_API_KEY is unset; its weight is then redistributed
to the model-based scores (see retriever/search.py::_normalise_weights), so
the system is fully functional -- just slightly less precise -- without it.

Uses the `google-genai` SDK (the actively-maintained successor to the now
deprecated `google-generativeai` package). Every (query, caption, attributes)
combination is cached on disk (common/llm_cache.py) and retriever/search.py
additionally only sends the top few stage-3 candidates here at all (see
config.MAX_LLM_VERIFY_CANDIDATES) -- both exist specifically to make this
sustainable on Gemini's free tier, whose *daily* request quota for a given
model can be quite low (see config.py's GEMINI_MODEL comment).

Hallucination guardrails:
  - The LLM never sees the raw image, only the caption + structured attributes
    we already extracted deterministically -- it can't invent new visual facts,
    only judge the ones we give it.
  - Output is a bounded integer score (0-100) plus one short reason, parsed
    with a strict JSON check (response_mime_type="application/json" -- Gemini
    is asked to guarantee valid JSON directly rather than us stripping
    markdown fences after the fact). A malformed response is treated as
    "unavailable" for that candidate (returns None), never silently defaulted
    to a made-up number.
"""
import json

from config import GEMINI_API_KEY, GEMINI_MODEL
from common import llm_cache


def is_enabled():
    return bool(GEMINI_API_KEY)


def verify_candidate(query: str, caption: str, attributes: dict):
    if not GEMINI_API_KEY:
        return None

    prompt = f"""You are grading whether an image matches a search query.
You are NOT shown the image -- only its extracted description below. Judge
strictly based on this description; do not assume unstated details, and do
not penalise attributes the query didn't ask about.

Query: "{query}"
Image caption: "{caption}"
Detected attributes: {json.dumps(attributes)}

Return strict JSON only, no markdown fences: {{"score": <integer 0-100>, "reason": "<one short sentence>"}}"""

    cached = llm_cache.get("verify", GEMINI_MODEL, prompt)
    if cached is not None:
        return cached

    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
        )
        data = json.loads(resp.text)
        score = max(0, min(100, int(data["score"])))
        result = {"score": score / 100.0, "reason": data.get("reason", "")}
        llm_cache.put("verify", GEMINI_MODEL, prompt, result)
        return result
    except Exception:
        return None  # verification unavailable for this candidate -- caller handles gracefully
