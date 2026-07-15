from __future__ import annotations

import argparse
import json
from dataclasses import replace

from .config import Settings
from .engine import EvidenceGraphEngine


def main() -> None:
    parser = argparse.ArgumentParser(prog="deepread", description="Budgeted hierarchical evidence retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ask = subparsers.add_parser("ask", help="Ask a question over a local corpus")
    ask.add_argument("question")
    ask.add_argument("--corpus", default="data/sample_corpus")
    ask.add_argument("--trace", default="traces/latest.json")
    ask.add_argument("--provider", choices=["auto", "offline", "openai"], default=None)
    ask.add_argument("--model", default=None)
    ask.add_argument("--retrieval-mode", choices=["bm25", "embeddings", "hybrid"])
    ask.add_argument("--model-rerank", action=argparse.BooleanOptionalAction, default=None)
    ask.add_argument("--reader-mode", choices=["flat", "hierarchical"])
    ask.add_argument("--flat-top-k", type=int)
    ask.add_argument("--supervisor-mode", choices=["single-pass", "bounded"])
    ask.add_argument("--max-search-rounds", type=int)
    ask.add_argument("--seed", type=int)
    inspect = subparsers.add_parser("inspect", help="Show corpus statistics")
    inspect.add_argument("--corpus", default="data/sample_corpus")
    args = parser.parse_args()
    settings = Settings.from_env()
    if getattr(args, "provider", None):
        settings = replace(settings, provider=args.provider)
    if getattr(args, "model", None):
        settings = replace(settings, openai_model=args.model)
    overrides = {
        "retrieval_mode": getattr(args, "retrieval_mode", None),
        "enable_model_rerank": getattr(args, "model_rerank", None),
        "reader_mode": getattr(args, "reader_mode", None),
        "flat_top_k": getattr(args, "flat_top_k", None),
        "supervisor_mode": (
            args.supervisor_mode.replace("-", "_")
            if getattr(args, "supervisor_mode", None)
            else None
        ),
        "max_search_rounds": getattr(args, "max_search_rounds", None),
        "random_seed": getattr(args, "seed", None),
    }
    settings = replace(settings, **{key: value for key, value in overrides.items() if value is not None})
    if args.command == "inspect":
        settings = replace(settings, provider="offline")
    engine = EvidenceGraphEngine(args.corpus, settings)
    if args.command == "ask":
        answer = engine.ask(args.question, args.trace)
        print(answer.text)
        print(f"\nCoverage: {answer.coverage:.0%} | Stop: {answer.stop_reason}")
        print(f"Read tokens: {answer.trace.read_tokens} | Citation tokens: {answer.trace.citation_tokens} | API tokens: {answer.trace.api_total_tokens}")
        if answer.trace.estimated_api_cost_usd is not None:
            print(f"Estimated API cost: ${answer.trace.estimated_api_cost_usd:.6f}")
        for index, citation in enumerate(answer.citations, 1):
            print(f"[{index}] {citation.title} — {citation.section}")
    else:
        print(json.dumps(engine.corpus_stats(), indent=2))


if __name__ == "__main__":
    main()
