from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_METRICS = (
    "answer_f1",
    "evidence_f1",
    "retrieval_recall_at_20",
    "mrr_at_10",
    "api_total_tokens",
)
LOWER_IS_BETTER = {
    "api_input_tokens",
    "api_output_tokens",
    "api_total_tokens",
    "citation_tokens",
    "evidence_tokens",
    "latency_ms",
    "read_tokens",
    "rounds",
}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a percentile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    metrics: list[str] | tuple[str, ...] = DEFAULT_METRICS,
    *,
    iterations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    baseline_rows = {
        row["question_id"]: row
        for row in baseline.get("rows", [])
        if row.get("status") == "ok"
    }
    candidate_rows = {
        row["question_id"]: row
        for row in candidate.get("rows", [])
        if row.get("status") == "ok"
    }
    paired_ids = sorted(set(baseline_rows) & set(candidate_rows))
    if not paired_ids:
        raise ValueError("the result files contain no completed questions in common")
    rng = random.Random(seed)
    comparisons: dict[str, Any] = {}
    for metric in metrics:
        pairs = [
            (float(baseline_rows[qid][metric]), float(candidate_rows[qid][metric]))
            for qid in paired_ids
            if baseline_rows[qid].get(metric) is not None
            and candidate_rows[qid].get(metric) is not None
        ]
        if not pairs:
            comparisons[metric] = None
            continue
        deltas = [right - left for left, right in pairs]
        bootstrap = [
            mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(iterations)
        ]
        direction = "lower" if metric in LOWER_IS_BETTER else "higher"
        comparisons[metric] = {
            "pairs": len(pairs),
            "baseline_mean": mean(left for left, _ in pairs),
            "candidate_mean": mean(right for _, right in pairs),
            "mean_delta": mean(deltas),
            "ci_95": [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)],
            "direction": direction,
            "improvement_rate": sum(
                delta < 0 if direction == "lower" else delta > 0 for delta in deltas
            )
            / len(deltas),
            "tie_rate": sum(delta == 0 for delta in deltas) / len(deltas),
        }
    return {
        "paired_questions": len(paired_ids),
        "bootstrap_iterations": iterations,
        "random_seed": seed,
        "metrics": comparisons,
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "| Metric | Pairs | Baseline | Candidate | Delta | 95% paired bootstrap CI | Improvement rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for metric, values in result["metrics"].items():
        if values is None:
            lines.append(f"| {metric} | 0 | — | — | — | — | — |")
            continue
        lower, upper = values["ci_95"]
        lines.append(
            f"| {metric} | {values['pairs']} | {values['baseline_mean']:.4f} | "
            f"{values['candidate_mean']:.4f} | {values['mean_delta']:+.4f} | "
            f"[{lower:+.4f}, {upper:+.4f}] | {values['improvement_rate']:.1%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Compare two DeepRead result files pairwise")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--metric", action="append", dest="metrics")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    args = parser.parse_args(argv)
    result = compare_results(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
        args.metrics or DEFAULT_METRICS,
        iterations=args.iterations,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2) if args.format == "json" else _markdown(result))
    return result


if __name__ == "__main__":
    main()
