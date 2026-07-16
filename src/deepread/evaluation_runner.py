from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import Settings
from .engine import EvidenceGraphEngine
from .evaluation import aggregate_results, evaluate_qasper_answer
from .models import CorpusBundle
from .qasper import QasperDataset, QasperQuestion, load_qasper, load_qasper_hf


SCHEMA_VERSION = 4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepRead answers, evidence, retrieval, citations, and efficiency on QASPER."
    )
    parser.add_argument("--qasper-path", type=Path, help="Official QASPER JSON or JSONL file")
    parser.add_argument("--split", default="validation", help="Hugging Face split")
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face cache directory")
    parser.add_argument("--mode", choices=["paper-known", "corpus-wide"], default="paper-known")
    parser.add_argument("--provider", choices=["offline", "openai"], default="offline")
    parser.add_argument("--retrieval-mode", choices=["bm25", "embeddings", "hybrid"])
    parser.add_argument(
        "--model-rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable model-based reranking independently of retrieval",
    )
    parser.add_argument("--reader-mode", choices=["flat", "hierarchical"])
    parser.add_argument("--flat-top-k", type=int)
    parser.add_argument("--evidence-window-sentences", type=int)
    parser.add_argument("--evidence-candidates-per-requirement", type=int)
    parser.add_argument("--evidence-support-threshold", type=float)
    parser.add_argument("--supervisor-mode", choices=["single-pass", "bounded"])
    parser.add_argument(
        "--answer-policy",
        choices=["grounded", "benchmark"],
        default="benchmark",
        help="Use benchmark forced-choice outputs or grounded abstention",
    )
    parser.add_argument("--max-search-rounds", type=int)
    parser.add_argument("--model-rerank-top-k", type=int)
    parser.add_argument("--model-rerank-rescue-per-requirement", type=int)
    parser.add_argument("--model-rerank-max-chars", type=int)
    parser.add_argument("--target-coverage", type=float)
    parser.add_argument("--diversity-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--paper-id", action="append", dest="paper_ids")
    parser.add_argument("--question-id", action="append", dest="question_ids")
    parser.add_argument("--max-papers", type=int, default=10, help="Use 0 for every paper")
    parser.add_argument("--max-questions", type=int, default=50, help="Use 0 for every question")
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--correct-threshold", type=float, default=0.5)
    parser.add_argument("--text-evidence-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark/results/qasper.json"))
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume completed questions when the run fingerprint matches",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def _load_dataset(args: argparse.Namespace) -> tuple[QasperDataset, str]:
    if args.qasper_path is not None:
        return load_qasper(args.qasper_path), str(args.qasper_path)
    dataset = load_qasper_hf(
        args.split,
        cache_dir=args.cache_dir,
        revision=args.revision,
    )
    return dataset, f"huggingface:allenai/qasper:{args.split}:{args.revision or 'default'}"


def _select_questions(
    dataset: QasperDataset,
    args: argparse.Namespace,
) -> tuple[list[str], list[QasperQuestion]]:
    available_papers = [str(document.metadata["paper_id"]) for document in dataset.documents]
    if args.paper_ids:
        missing = sorted(set(args.paper_ids) - set(available_papers))
        if missing:
            raise ValueError(f"Unknown paper ids: {', '.join(missing)}")
        paper_ids = list(dict.fromkeys(args.paper_ids))
    else:
        paper_ids = available_papers[: args.max_papers or None]

    questions = dataset.questions_for(paper_ids)
    if args.question_ids:
        by_id = {question.id: question for question in questions}
        missing = sorted(set(args.question_ids) - set(by_id))
        if missing:
            raise ValueError(f"Unknown question ids in selected papers: {', '.join(missing)}")
        questions = [by_id[question_id] for question_id in dict.fromkeys(args.question_ids)]
    questions = questions[: args.max_questions or None]
    if not paper_ids or not questions:
        raise ValueError("The selected QASPER slice contains no questions")
    return paper_ids, questions


def _fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dataset_signature(
    corpus: CorpusBundle,
    questions: list[QasperQuestion],
) -> str:
    digest = hashlib.sha256()
    for collection in (corpus.documents, corpus.passages, questions):
        for item in collection:
            digest.update(json.dumps(asdict(item), sort_keys=True).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _corpus_hash(corpus: CorpusBundle) -> str:
    """Hash only corpus content and provenance, independent of selected questions."""
    digest = hashlib.sha256()
    for collection in (corpus.documents, corpus.passages):
        for item in collection:
            digest.update(json.dumps(asdict(item), sort_keys=True).encode())
            digest.update(b"\0")
    return digest.hexdigest()


def _trace_name(question_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", question_id).strip("-") or "question"
    digest = hashlib.sha1(question_id.encode()).hexdigest()[:8]
    return f"{slug[:80]}-{digest}.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _result_payload(
    run: dict[str, Any],
    questions: list[QasperQuestion],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    retrieval_k: int,
    ranking_k: int,
) -> dict[str, Any]:
    rows = [rows_by_id[question.id] for question in questions if question.id in rows_by_id]
    return {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "summary": aggregate_results(rows, retrieval_k=retrieval_k, ranking_k=ranking_k),
        "rows": rows,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_papers < 0 or args.max_questions < 0:
        raise ValueError("max-papers and max-questions must be non-negative")
    if args.retrieval_k < 1 or args.ranking_k < 1:
        raise ValueError("retrieval-k and ranking-k must be positive")
    if not 0 <= args.correct_threshold <= 1:
        raise ValueError("correct-threshold must be between 0 and 1")
    if args.flat_top_k is not None and args.flat_top_k < 1:
        raise ValueError("flat-top-k must be positive")
    if args.max_search_rounds is not None and args.max_search_rounds < 1:
        raise ValueError("max-search-rounds must be positive")
    if args.evidence_window_sentences is not None and args.evidence_window_sentences < 1:
        raise ValueError("evidence-window-sentences must be positive")
    if (
        args.evidence_candidates_per_requirement is not None
        and args.evidence_candidates_per_requirement < 1
    ):
        raise ValueError("evidence-candidates-per-requirement must be positive")
    if (
        args.evidence_support_threshold is not None
        and not 0 <= args.evidence_support_threshold <= 1
    ):
        raise ValueError("evidence-support-threshold must be between 0 and 1")
    if args.model_rerank_top_k is not None and args.model_rerank_top_k < 1:
        raise ValueError("model-rerank-top-k must be positive")
    if (
        args.model_rerank_rescue_per_requirement is not None
        and args.model_rerank_rescue_per_requirement < 0
    ):
        raise ValueError("model-rerank-rescue-per-requirement must be non-negative")
    if args.model_rerank_max_chars is not None and args.model_rerank_max_chars < 100:
        raise ValueError("model-rerank-max-chars must be at least 100")
    if args.target_coverage is not None and not 0 < args.target_coverage <= 1:
        raise ValueError("target-coverage must be greater than 0 and at most 1")
    if args.diversity_weight is not None and not 0 <= args.diversity_weight <= 1:
        raise ValueError("diversity-weight must be between 0 and 1")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    _validate_args(args)
    dataset, source = _load_dataset(args)
    paper_ids, questions = _select_questions(dataset, args)
    base_settings = Settings.from_env()
    setting_overrides: dict[str, Any] = {
        "provider": args.provider,
        "answer_policy": args.answer_policy,
    }
    if args.retrieval_mode is not None:
        setting_overrides["retrieval_mode"] = args.retrieval_mode
    if args.model_rerank is not None:
        setting_overrides["enable_model_rerank"] = args.model_rerank
    if args.reader_mode is not None:
        setting_overrides["reader_mode"] = args.reader_mode
    if args.flat_top_k is not None:
        setting_overrides["flat_top_k"] = args.flat_top_k
    if args.evidence_window_sentences is not None:
        setting_overrides["evidence_window_sentences"] = args.evidence_window_sentences
    if args.evidence_candidates_per_requirement is not None:
        setting_overrides["evidence_candidates_per_requirement"] = (
            args.evidence_candidates_per_requirement
        )
    if args.evidence_support_threshold is not None:
        setting_overrides["evidence_support_threshold"] = args.evidence_support_threshold
    if args.supervisor_mode is not None:
        setting_overrides["supervisor_mode"] = args.supervisor_mode.replace("-", "_")
    if args.max_search_rounds is not None:
        setting_overrides["max_search_rounds"] = args.max_search_rounds
    if args.model_rerank_top_k is not None:
        setting_overrides["model_rerank_top_k"] = args.model_rerank_top_k
    if args.model_rerank_rescue_per_requirement is not None:
        setting_overrides["model_rerank_rescue_per_requirement"] = (
            args.model_rerank_rescue_per_requirement
        )
    if args.model_rerank_max_chars is not None:
        setting_overrides["model_rerank_max_chars"] = args.model_rerank_max_chars
    if args.target_coverage is not None:
        setting_overrides["target_coverage"] = args.target_coverage
    if args.diversity_weight is not None:
        setting_overrides["redundancy_weight"] = args.diversity_weight
    if args.seed is not None:
        setting_overrides["random_seed"] = args.seed
    settings = replace(base_settings, **setting_overrides)
    random.seed(settings.random_seed)
    selected_corpus = dataset.corpus(paper_ids)
    dataset_signature = _dataset_signature(selected_corpus, questions)
    corpus_hash = _corpus_hash(selected_corpus)
    model_name = settings.openai_model if args.provider == "openai" else "offline-extractive"
    embedding_model_name = (
        settings.openai_embedding_model
        if args.provider == "openai"
        else "character-ngram-fallback"
    )
    result_configuration = {
        "dataset": "qasper",
        "dataset_split": args.split,
        "corpus_hash": corpus_hash,
        "random_seed": settings.random_seed,
        "provider": args.provider,
        "model_name": model_name,
        "embedding_model_name": embedding_model_name,
        "evaluation_mode": args.mode,
        "settings": asdict(settings),
    }
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": "qasper",
        "source": source,
        "split": args.split,
        "revision": args.revision,
        "mode": args.mode,
        "provider": args.provider,
        "paper_ids": paper_ids,
        "question_ids": [question.id for question in questions],
        "dataset_signature": dataset_signature,
        "corpus_hash": corpus_hash,
        "random_seed": settings.random_seed,
        "model_name": model_name,
        "embedding_model_name": embedding_model_name,
        "retrieval_k": args.retrieval_k,
        "ranking_k": args.ranking_k,
        "correct_threshold": args.correct_threshold,
        "text_evidence_only": args.text_evidence_only,
        "settings": asdict(settings),
        "result_configuration": result_configuration,
    }
    run = {**run_config, "fingerprint": _fingerprint(run_config)}
    rows_by_id: dict[str, dict[str, Any]] = {}
    if args.output.exists() and args.resume:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing_fingerprint = existing.get("run", {}).get("fingerprint")
        if existing_fingerprint != run["fingerprint"]:
            raise ValueError(
                "Existing results have a different run fingerprint; use --no-resume or a new output path"
            )
        rows_by_id = {row["question_id"]: row for row in existing.get("rows", [])}

    trace_dir = args.trace_dir or args.output.parent / f"{args.output.stem}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    passage_by_id = {passage.id: passage for passage in selected_corpus.passages}
    paper_engines: dict[str, EvidenceGraphEngine] = {}
    corpus_engine: EvidenceGraphEngine | None = None

    for index, question in enumerate(questions, 1):
        if rows_by_id.get(question.id, {}).get("status") == "ok":
            print(f"[{index}/{len(questions)}] resume {question.id}")
            continue
        print(f"[{index}/{len(questions)}] evaluate {question.id}")
        trace_path = trace_dir / _trace_name(question.id)
        started = time.perf_counter()
        try:
            if args.mode == "paper-known":
                engine = paper_engines.get(question.paper_id)
                if engine is None:
                    engine = EvidenceGraphEngine(dataset.corpus([question.paper_id]), settings)
                    paper_engines[question.paper_id] = engine
            else:
                if corpus_engine is None:
                    corpus_engine = EvidenceGraphEngine(selected_corpus, settings)
                engine = corpus_engine
            answer = engine.ask(question.question, trace_path)
            row = evaluate_qasper_answer(
                question,
                answer,
                passage_by_id,
                retrieval_k=args.retrieval_k,
                ranking_k=args.ranking_k,
                text_evidence_only=args.text_evidence_only,
                correct_threshold=args.correct_threshold,
            )
            row.update(
                {
                    "status": "ok",
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "trace_path": _display_path(trace_path),
                    "configuration": result_configuration,
                }
            )
        except Exception as exc:
            row = {
                "question_id": question.id,
                "paper_id": question.paper_id,
                "question": question.question,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "configuration": result_configuration,
            }
            rows_by_id[question.id] = row
            payload = _result_payload(
                run,
                questions,
                rows_by_id,
                retrieval_k=args.retrieval_k,
                ranking_k=args.ranking_k,
            )
            _atomic_write(args.output, payload)
            if args.fail_fast:
                raise
            continue
        rows_by_id[question.id] = row
        payload = _result_payload(
            run,
            questions,
            rows_by_id,
            retrieval_k=args.retrieval_k,
            ranking_k=args.ranking_k,
        )
        _atomic_write(args.output, payload)

    payload = _result_payload(
        run,
        questions,
        rows_by_id,
        retrieval_k=args.retrieval_k,
        ranking_k=args.ranking_k,
    )
    _atomic_write(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    return payload


if __name__ == "__main__":
    main()
