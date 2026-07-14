from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from deepread.config import Settings
from deepread.engine import EvidenceGraphEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/sample_corpus")
    parser.add_argument("--queries", default="benchmark/queries.jsonl")
    parser.add_argument("--output", default="benchmark/results/latest.json")
    parser.add_argument("--provider", choices=["offline", "openai"], default="offline")
    args = parser.parse_args()
    engine = EvidenceGraphEngine(args.corpus, replace(Settings.from_env(), provider=args.provider))
    rows = []
    for line in Path(args.queries).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        started = time.perf_counter()
        answer = engine.ask(item["question"])
        cited_docs = {citation.document_id for citation in answer.citations}
        expected = set(item.get("expected_document_ids", []))
        rows.append({"id": item["id"], "coverage": answer.coverage, "read_tokens": answer.trace.read_tokens, "citation_tokens": answer.trace.citation_tokens, "api_tokens": answer.trace.api_total_tokens, "estimated_api_cost_usd": answer.trace.estimated_api_cost_usd, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "citation_hit": bool(cited_docs & expected) if expected else None, "stop_reason": answer.stop_reason})
    summary = {"queries": len(rows), "mean_coverage": statistics.mean(r["coverage"] for r in rows), "mean_read_tokens": statistics.mean(r["read_tokens"] for r in rows), "mean_citation_tokens": statistics.mean(r["citation_tokens"] for r in rows), "mean_api_tokens": statistics.mean(r["api_tokens"] for r in rows), "mean_latency_ms": statistics.mean(r["latency_ms"] for r in rows), "rows": rows}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
