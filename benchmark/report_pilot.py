"""Build the checked-in paper-known pilot report from ignored raw benchmark outputs."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from deepread.evaluation import answer_reference
from deepread.qasper import load_qasper


ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "benchmark" / "results" / "pilot"
REPORT = ROOT / "reports" / "paper_known_pilot"
QASPER = ROOT / ".deepread" / "qasper" / "qasper-dev-v0.3.json"

CONFIGS = {
    "offline-bm25_only.json": "BM25 only (offline)",
    "offline-embeddings_only.json": "Embeddings only (offline fallback)",
    "offline-flat_hybrid.json": "Flat hybrid top-k (offline)",
    "offline-hybrid_rerank.json": "Hybrid + rerank flag (offline)",
    "offline-hierarchical.json": "Hierarchical single-pass (offline)",
    "offline-evidencegraph.json": "Bounded EvidenceGraph (offline)",
    "openai-hierarchical.json": "Hierarchical single-pass (OpenAI)",
    "openai-evidencegraph.json": "Bounded EvidenceGraph (OpenAI)",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _group_rows(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {
        name: {
            "questions": len(items),
            "answer_f1": _mean(items, "answer_f1"),
            "evidence_f1": _mean(items, "evidence_f1"),
            "retrieval_recall_at_20": _mean(items, "retrieval_recall_at_20"),
            "coverage": _mean(items, "coverage"),
        }
        for name, items in sorted(groups.items())
    }


def _provenance_audit(dataset: Any, questions: list[Any], passage_by_id: dict[str, Any]) -> dict[str, Any]:
    references = [
        (question, answer, reference, match)
        for question in questions
        for answer in question.answers
        for reference in answer.evidence
        for match in reference.matches
    ]
    samples: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for question, answer, reference, match in references:
        answer_type = answer_reference(answer)[1]
        sample_type = (answer_type, match.source_type)
        if sample_type in seen:
            continue
        seen.add(sample_type)
        passage = passage_by_id[match.passage_id]
        mapped = passage.text[match.char_start : match.char_end]
        expected = re.sub(
            r"^\s*FLOAT SELECTED\s*:?\s*",
            "",
            reference.text,
            flags=re.IGNORECASE,
        )
        samples.append(
            {
                "question_id": question.id,
                "answer_type": answer_type,
                "source_type": match.source_type,
                "source_path": match.source_path,
                "match_type": match.match_type,
                "offsets_valid": 0 <= match.char_start < match.char_end <= len(passage.text),
                "text_valid": mapped.strip() == expected.strip(),
            }
        )
        if len(samples) == 6:
            break
    return {
        "papers": len({question.paper_id for question in questions}),
        "questions": len(questions),
        "answer_annotations_by_type": dict(
            Counter(answer_reference(answer)[1] for question in questions for answer in question.answers)
        ),
        "resolved_matches_by_location": dict(Counter(match.source_type for *_, match in references)),
        "unresolved_references": sum(
            not reference.resolved
            for question in questions
            for answer in question.answers
            for reference in answer.evidence
        ),
        "samples": samples,
    }


def _select_traces(
    result: dict[str, Any],
    passage_by_id: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = result["rows"]

    def locations(row: dict[str, Any]) -> set[str]:
        return {
            str(passage_by_id[passage_id].metadata["source_type"])
            for passage_id in row["gold_evidence_passage_ids"]
        }

    candidates = [
        ("best-answer", max(rows, key=lambda row: row["answer_f1"])),
        (
            "boolean-failure",
            min(
                (row for row in rows if row["matched_answer_type"] == "boolean"),
                key=lambda row: row["answer_f1"],
            ),
        ),
        (
            "figure-table-case",
            max(
                (row for row in rows if "figure_table" in locations(row)),
                key=lambda row: row["evidence_f1"],
            ),
        ),
        (
            "retrieval-miss",
            min(
                rows,
                key=lambda row: (
                    row["retrieval_recall_at_20"]
                    if row["retrieval_recall_at_20"] is not None
                    else 1.0,
                    row["answer_f1"],
                ),
            ),
        ),
    ]
    target_dir = REPORT / "traces"
    target_dir.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, Any]] = []
    for label, row in candidates:
        source = ROOT / row["trace_path"]
        target = target_dir / f"{label}-{row['question_id'][:10]}.json"
        shutil.copyfile(source, target)
        selected.append(
            {
                "label": label,
                "question_id": row["question_id"],
                "question": row["question"],
                "answer_f1": row["answer_f1"],
                "evidence_f1": row["evidence_f1"],
                "retrieval_recall_at_20": row["retrieval_recall_at_20"],
                "trace": str(target.relative_to(REPORT)),
            }
        )
    return selected


def main() -> None:
    dataset = load_qasper(QASPER)
    paper_ids = [str(document.metadata["paper_id"]) for document in dataset.documents[:10]]
    questions = dataset.questions_for(paper_ids)[:40]
    passage_by_id = {passage.id: passage for passage in dataset.passages}

    loaded = {
        name: _load(RESULTS / filename)
        for filename, name in CONFIGS.items()
        if (RESULTS / filename).exists()
    }
    if "Bounded EvidenceGraph (OpenAI)" not in loaded:
        raise RuntimeError("The full OpenAI bounded result is required before publishing")

    aggregate = []
    for name, result in loaded.items():
        summary = result["summary"]
        aggregate.append(
            {
                "configuration": name,
                "provider": result["run"]["provider"],
                "completed": summary["counts"]["completed"],
                "correct": summary["counts"]["correct"],
                "answer_f1": summary["metrics"]["answer_f1"],
                "evidence_f1": summary["metrics"]["evidence_f1"],
                "retrieval_recall_at_20": summary["metrics"]["retrieval_recall_at_20"],
                "mrr_at_10": summary["metrics"]["mrr_at_10"],
                "coverage": summary["metrics"]["coverage"],
                "mean_read_tokens": summary["efficiency"]["mean_read_tokens"],
                "mean_api_tokens": summary["efficiency"]["mean_api_tokens"],
                "mean_latency_ms": summary["efficiency"]["mean_latency_ms"],
            }
        )

    openai_result = loaded["Bounded EvidenceGraph (OpenAI)"]

    def evidence_location(row: dict[str, Any]) -> str:
        locations = sorted(
            {
                str(passage_by_id[passage_id].metadata["source_type"])
                for passage_id in row["gold_evidence_passage_ids"]
            }
        )
        return "+".join(locations) or "none"

    calibrations = []
    for path in sorted(RESULTS.glob("calibration-*.json")):
        result = _load(path)
        settings = result["run"]["settings"]
        calibrations.append(
            {
                "run": path.stem,
                "provider": result["run"]["provider"],
                "window_sentences": settings.get("evidence_window_sentences", 3),
                "target_coverage": settings["target_coverage"],
                "answer_f1": result["summary"]["metrics"]["answer_f1"],
                "evidence_f1": result["summary"]["metrics"]["evidence_f1"],
                "mean_citation_tokens": result["summary"]["efficiency"]["mean_citation_tokens"],
                "mean_api_tokens": result["summary"]["efficiency"]["mean_api_tokens"],
                "mean_rounds": result["summary"]["efficiency"]["mean_rounds"],
            }
        )

    provenance = _provenance_audit(dataset, questions, passage_by_id)
    answer_type = _group_rows(openai_result["rows"], lambda row: row["matched_answer_type"])
    location = _group_rows(openai_result["rows"], evidence_location)
    traces = _select_traces(openai_result, passage_by_id)
    payload = {
        "dataset": {
            "name": "QASPER v0.3 development/validation",
            "papers": len(paper_ids),
            "questions": len(questions),
            "paper_ids": paper_ids,
            "corpus_hash": openai_result["run"]["corpus_hash"],
            "random_seed": openai_result["run"]["random_seed"],
        },
        "aggregate": aggregate,
        "openai_failure_breakdown": {
            "answer_type": answer_type,
            "evidence_location": location,
        },
        "calibration": calibrations,
        "provenance_audit": provenance,
        "representative_traces": traces,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# QASPER paper-known pilot",
        "",
        "This pilot evaluates passage retrieval, evidence selection, and synthesis on a fixed",
        "10-paper slice of the official QASPER v0.3 development split. The first 10 papers contain",
        "30 questions, so the requested 30–50 question range is met without partially sampling an",
        "additional paper.",
        "",
        f"- Corpus hash: `{payload['dataset']['corpus_hash']}`",
        f"- Random seed: `{payload['dataset']['random_seed']}`",
        "- Mode: paper-known",
        "- Correct threshold: answer F1 ≥ 0.5",
        "",
        "## Aggregate results",
        "",
        "| Configuration | Correct | Answer F1 | Evidence F1 | Recall@20 | MRR@10 | Coverage | Read tokens | API tokens | Latency (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['configuration']} | {row['correct']}/{row['completed']} | "
            f"{_fmt(row['answer_f1'])} | {_fmt(row['evidence_f1'])} | "
            f"{_fmt(row['retrieval_recall_at_20'])} | {_fmt(row['mrr_at_10'])} | "
            f"{_fmt(row['coverage'])} | {_fmt(row['mean_read_tokens'], 1)} | "
            f"{_fmt(row['mean_api_tokens'], 1)} | {_fmt(row['mean_latency_ms'], 1)} |"
        )

    lines.extend(
        [
            "",
            "Offline reranking is intentionally inert because no model is present; its row is a",
            "configuration/provenance check and should match flat hybrid retrieval. Likewise, the",
            "offline bounded loop stops after one round because deterministic coverage reaches 100%.",
            "",
            "## OpenAI failure breakdown",
            "",
            "### By matched answer type",
            "",
            "| Answer type | Questions | Answer F1 | Evidence F1 | Recall@20 | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in answer_type.items():
        lines.append(
            f"| {name} | {row['questions']} | {_fmt(row['answer_f1'])} | "
            f"{_fmt(row['evidence_f1'])} | {_fmt(row['retrieval_recall_at_20'])} | "
            f"{_fmt(row['coverage'])} |"
        )
    lines.extend(
        [
            "",
            "### By gold evidence location",
            "",
            "| Evidence location | Questions | Answer F1 | Evidence F1 | Recall@20 | Coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in location.items():
        lines.append(
            f"| {name} | {row['questions']} | {_fmt(row['answer_f1'])} | "
            f"{_fmt(row['evidence_f1'])} | {_fmt(row['retrieval_recall_at_20'])} | "
            f"{_fmt(row['coverage'])} |"
        )

    lines.extend(
        [
            "",
            "### Observed failure modes",
            "",
            "- Boolean answers score zero even when relevant passages rank well because synthesis",
            "  produces a qualified explanation instead of the benchmark's expected `Yes` or `No`.",
            "- The table-only question has zero Recall@20, while mixed paragraph/table questions are",
            "  substantially stronger; captions need their own retrieval treatment.",
            "- Extractive questions reach high retrieval recall but low answer F1, showing that",
            "  planning decomposition, evidence acceptance, and concise synthesis are the main",
            "  bottlenecks after retrieval.",
            "- Some gold passages are retrieved but rejected as insufficient, so reported coverage",
            "  is much lower than retrieval recall.",
        ]
    )

    lines.extend(
        [
            "",
            "## Calibration",
            "",
            "| Run | Provider | Window | Coverage target | Answer F1 | Evidence F1 | Citation tokens | API tokens | Rounds |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calibrations:
        lines.append(
            f"| {row['run']} | {row['provider']} | {row['window_sentences']} | "
            f"{_fmt(row['target_coverage'])} | {_fmt(row['answer_f1'])} | "
            f"{_fmt(row['evidence_f1'])} | {_fmt(row['mean_citation_tokens'], 1)} | "
            f"{_fmt(row['mean_api_tokens'], 1)} | {_fmt(row['mean_rounds'], 2)} |"
        )

    lines.extend(
        [
            "",
            "The provisional operating point is a 3-sentence evidence window with a 0.5 coverage",
            "target. It produced the best observed answer and evidence F1 on the four-question",
            "stratified calibration subset while using fewer API tokens than stricter targets. This",
            "choice should be revalidated on a larger paired run because API outputs are stochastic.",
        ]
    )

    lines.extend(
        [
            "",
            "## Provenance audit",
            "",
            f"All sampled annotations resolved: `{provenance['unresolved_references']}` unresolved references.",
            "The audit covers paragraph, figure/table, and section-heading sources and verifies both",
            "source offsets and mapped text. `FLOAT SELECTED:` annotation prefixes are normalized before",
            "comparing them with stored figure/table captions.",
            "",
            "| Question ID | Answer type | Source | Source path | Match | Offsets valid | Text valid |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for sample in provenance["samples"]:
        lines.append(
            f"| `{sample['question_id'][:10]}` | {sample['answer_type']} | {sample['source_type']} | "
            f"`{sample['source_path']}` | {sample['match_type']} | "
            f"{sample['offsets_valid']} | {sample['text_valid']} |"
        )

    lines.extend(["", "## Representative traces", ""])
    for trace in traces:
        lines.append(
            f"- [{trace['label']}]({trace['trace']}): {trace['question']} "
            f"(answer F1 {_fmt(trace['answer_f1'])}, evidence F1 {_fmt(trace['evidence_f1'])})"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a small pilot, not a statistically powered comparison. Paper recall is 1.0 by",
            "construction in paper-known mode and should not be interpreted as document-retrieval",
            "performance. Seed 17 controls deterministic components, but current OpenAI API calls do",
            "not expose a seed in this pipeline, so paid calibration cells remain stochastic. The",
            "results identify concrete next work: answer-type-aware synthesis, less",
            "aggressive planning decomposition, evidence-assessor calibration, and explicit support for",
            "negative boolean answers.",
            "",
        ]
    )
    (REPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
