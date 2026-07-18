"""
Real end-to-end test: runs actual queries against your ALREADY-BUILT index
(data/index/clip.index + metadata.jsonl), using the real CLIP/BLIP/ITM models
-- and real Gemini LLM verification too, if GEMINI_API_KEY is set in your
environment/.env. This is different from tests/smoke_test.py, which only
exercises the pipeline plumbing with mock (fake) embeddings.

Unlike smoke_test.py, this does NOT build its own throwaway index and does
NOT delete anything -- it queries whatever index you already built with
`python -m indexer.build_index`, so build that first.

Run:
    python -m tests.e2e_test
    python -m tests.e2e_test --k 5          # ask for more results per query
    python -m tests.e2e_test -v             # print full result details

Exit code is 0 if every query returns a well-formed result, non-zero if any
query errors out or the index is missing.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FAISS_INDEX_PATH, METADATA_PATH, GEMINI_API_KEY  # noqa: E402

QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

REQUIRED_RESULT_KEYS = {
    "image_path", "confidence", "semantic_score", "attribute_score",
    "itm_score", "llm_score", "caption", "scene", "style",
    "upper_color", "lower_color", "garment_colors", "objects", "explanation",
}


def _check_result_shape(query, result):
    assert "results" in result and isinstance(result["results"], list), \
        f"[{query}] missing/invalid 'results' list"
    assert "confidence" in result, f"[{query}] missing top-level confidence"
    assert "parsed_attributes" in result, f"[{query}] missing parsed_attributes"
    for r in result["results"]:
        missing = REQUIRED_RESULT_KEYS - r.keys()
        assert not missing, f"[{query}] result missing keys: {missing}"
        assert os.path.exists(r["image_path"]), \
            f"[{query}] result points at a missing file: {r['image_path']}"
        assert 0.0 <= r["confidence"] <= 1.0, f"[{query}] confidence out of range: {r['confidence']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="results per query")
    ap.add_argument("-v", "--verbose", action="store_true", help="print full result details")
    args = ap.parse_args()

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print(f"ERROR: no built index found at {FAISS_INDEX_PATH} / {METADATA_PATH}.")
        print("Run `python -m indexer.build_index --data_dir data/images` first.")
        sys.exit(1)

    llm_on = bool(GEMINI_API_KEY)
    print(f"Index found. LLM verification: {'ENABLED (GEMINI_API_KEY set)' if llm_on else 'disabled (no GEMINI_API_KEY)'}")

    print("Loading real CLIP/BLIP models (this downloads/loads weights, may take a bit) ...")
    t_load = time.time()
    from retriever.search import Retriever
    retriever = Retriever(mock=False)
    print(f"Models loaded in {time.time() - t_load:.1f}s\n")

    any_llm_score_seen = False
    total_t = 0.0
    for q in QUERIES:
        t0 = time.time()
        result = retriever.search(q, top_k=args.k)
        dt = time.time() - t0
        total_t += dt

        _check_result_shape(q, result)

        top = result["results"][0] if result["results"] else None
        warn = " <-- LOW CONFIDENCE" if result["low_confidence_warning"] else ""
        print(f"'{q}'")
        print(f"  -> {len(result['results'])} results in {dt:.1f}s, "
              f"top confidence={result['confidence']}{warn}")
        if top:
            print(f"     top: {top['image_path']}  caption: {top['caption']!r}")
            if top["llm_score"] is not None:
                any_llm_score_seen = True
                print(f"     llm_score={top['llm_score']}")

        if args.verbose:
            print(json.dumps(result, indent=2))
        print()

    if llm_on:
        assert any_llm_score_seen, (
            "GEMINI_API_KEY is set but no query returned an llm_score -- "
            "check retriever/llm_verify.py / your API key / network access."
        )

    print(f"All {len(QUERIES)} queries completed OK in {total_t:.1f}s total "
          f"({total_t / len(QUERIES):.1f}s/query avg).")
    print("End-to-end test passed.")


if __name__ == "__main__":
    main()
