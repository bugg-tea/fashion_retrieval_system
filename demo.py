"""
CLI entry point for running a query against the built index.

Examples:
    python demo.py "A person in a bright yellow raincoat."
    python demo.py "Professional business attire inside a modern office." --k 10
    python demo.py --eval                 # run all 5 assignment evaluation queries
    python demo.py --eval --mock          # same, but with fake embeddings (no models needed)
"""
import argparse
import json

from retriever.search import Retriever

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def pretty_print(result):
    print(f"\n{'=' * 70}\nQuery: {result['query']}")
    print(f"Parsed attributes: {result['parsed_attributes']}")
    warn = " <-- LOW CONFIDENCE" if result["low_confidence_warning"] else ""
    print(f"Top-result confidence: {result['confidence']}{warn}")
    if result.get("reflection_triggered"):
        print("(reflection loop re-weighted attributes to improve this result)")

    for i, r in enumerate(result["results"], 1):
        print(f"\n  #{i} {r['image_path']}  confidence={r['confidence']}")
        print(f"     semantic={r['semantic_score']}  attribute={r['attribute_score']}  "
              f"itm={r['itm_score']}  llm={r['llm_score']}")
        print(f"     caption: {r['caption']}")
        print(f"     scene={r['scene']}  style={r['style']}  upper={r['upper_color']}  lower={r['lower_color']}"
              + (f"  objects={r['objects']}" if r.get("objects") else "")
              + (f"  garments={r['garment_colors']}" if r.get("garment_colors") else ""))
        for cat, info in r["explanation"].items():
            mark = "OK " if info["matched"] else "MISS"
            print(f"     [{mark}] {cat}: wanted={info['wanted']} found={info['found']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval", action="store_true", help="Run all 5 assignment evaluation queries")
    ap.add_argument("--mock", action="store_true", help="Use fake embeddings (no model downloads)")
    ap.add_argument("--json", action="store_true", help="Print raw JSON instead of formatted text")
    args = ap.parse_args()

    if not args.query and not args.eval:
        ap.error("Provide a query string, or use --eval to run all assignment evaluation queries.")

    retriever = Retriever(mock=args.mock)
    queries = EVAL_QUERIES if args.eval else [args.query]

    for q in queries:
        result = retriever.search(q, top_k=args.k)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            pretty_print(result)
