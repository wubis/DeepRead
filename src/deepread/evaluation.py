from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Any, Iterable

from .models import Answer, Evidence, Passage, ReadLevel
from .qasper import QasperAnswer, QasperEvidenceMatch, QasperQuestion


def normalize_answer(text: str) -> str:
    """Apply the official QASPER/SQuAD answer normalization."""

    lowered = text.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def strip_citation_markers(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def token_f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(prediction_tokens)
    recall = same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_reference(answer: QasperAnswer) -> tuple[str, str]:
    if answer.unanswerable:
        return "Unanswerable", "none"
    if answer.extractive_spans:
        return ", ".join(answer.extractive_spans), "extractive"
    if answer.free_form_answer:
        return answer.free_form_answer, "abstractive"
    if answer.yes_no is not None:
        return ("Yes" if answer.yes_no else "No"), "boolean"
    return "", "none"


def _set_prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    overlap = len(predicted & gold)
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return precision, recall, 2 * precision * recall / (precision + recall)


def _best_set_score(
    predicted: set[str],
    references: list[tuple[str, set[str]]],
) -> tuple[float, float, float, str | None]:
    if not references:
        references = [("", set())]
    scores = [(*_set_prf(predicted, gold), annotation_id) for annotation_id, gold in references]
    return max(scores, key=lambda item: (item[2], item[1], item[0]))


def _evidence_ids(answer: QasperAnswer, *, text_only: bool) -> set[str]:
    ids: set[str] = set()
    for reference in answer.evidence:
        for match in reference.matches:
            if text_only and match.source_type == "figure_table":
                continue
            ids.add(match.passage_id)
    return ids


def _highlighted_matches(answer: QasperAnswer) -> list[QasperEvidenceMatch]:
    matches: list[QasperEvidenceMatch] = []
    for reference in answer.highlighted_evidence:
        matches.extend(reference.matches)
    return matches


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return max(left_start, right_start) < min(left_end, right_end)


def _span_prf(
    citations: Iterable[Evidence],
    gold: list[QasperEvidenceMatch],
) -> tuple[float, float, float]:
    predicted = list(citations)
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 0.0, 0.0
    predicted_hits = {
        index
        for index, citation in enumerate(predicted)
        if any(
            citation.passage_id == match.passage_id
            and _ranges_overlap(
                citation.char_start, citation.char_end, match.char_start, match.char_end
            )
            for match in gold
        )
    }
    gold_hits = {
        index
        for index, match in enumerate(gold)
        if any(
            citation.passage_id == match.passage_id
            and _ranges_overlap(
                citation.char_start, citation.char_end, match.char_start, match.char_end
            )
            for citation in predicted
        )
    }
    precision = len(predicted_hits) / len(predicted) if predicted else 0.0
    recall = len(gold_hits) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _best_span_score(
    citations: Iterable[Evidence],
    answers: tuple[QasperAnswer, ...],
) -> tuple[float | None, float | None, float | None]:
    references = [_highlighted_matches(answer) for answer in answers]
    references = [matches for matches in references if matches]
    if not references:
        return None, None, None
    return max((_span_prf(citations, gold) for gold in references), key=lambda item: item[2])


def _unique_ranking(answer: Answer) -> list[str]:
    seen: set[str] = set()
    ranking: list[str] = []
    for hit in answer.trace.ranking:
        if hit.passage_id not in seen:
            seen.add(hit.passage_id)
            ranking.append(hit.passage_id)
    return ranking


def _retrieval_metrics(
    ranking: list[str],
    gold: set[str],
    *,
    retrieval_k: int,
    ranking_k: int,
) -> dict[str, float | None]:
    if not gold:
        return {
            f"retrieval_recall_at_{retrieval_k}": None,
            f"mrr_at_{ranking_k}": None,
            f"ndcg_at_{ranking_k}": None,
        }
    retrieved = ranking[:retrieval_k]
    recall = len(set(retrieved) & gold) / len(gold)
    first_relevant = next(
        (index for index, passage_id in enumerate(ranking[:ranking_k], 1) if passage_id in gold),
        None,
    )
    reciprocal_rank = 1 / first_relevant if first_relevant is not None else 0.0
    dcg = sum(
        1 / math.log2(index + 1)
        for index, passage_id in enumerate(ranking[:ranking_k], 1)
        if passage_id in gold
    )
    ideal_hits = min(len(gold), ranking_k)
    ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return {
        f"retrieval_recall_at_{retrieval_k}": recall,
        f"mrr_at_{ranking_k}": reciprocal_rank,
        f"ndcg_at_{ranking_k}": dcg / ideal_dcg if ideal_dcg else 0.0,
    }


def _best_retrieval_metrics(
    ranking: list[str],
    references: list[set[str]],
    *,
    retrieval_k: int,
    ranking_k: int,
) -> dict[str, float | None]:
    nonempty = [gold for gold in references if gold]
    if not nonempty:
        return _retrieval_metrics(
            ranking,
            set(),
            retrieval_k=retrieval_k,
            ranking_k=ranking_k,
        )
    scores = [
        _retrieval_metrics(
            ranking,
            gold,
            retrieval_k=retrieval_k,
            ranking_k=ranking_k,
        )
        for gold in nonempty
    ]
    return {key: max(score[key] for score in scores) for key in scores[0]}


def evaluate_qasper_answer(
    question: QasperQuestion,
    answer: Answer,
    passages: dict[str, Passage],
    *,
    retrieval_k: int = 20,
    ranking_k: int = 10,
    text_evidence_only: bool = False,
    correct_threshold: float = 0.5,
) -> dict[str, Any]:
    prediction = strip_citation_markers(answer.text)
    references = [
        (annotation.annotation_id, *answer_reference(annotation)) for annotation in question.answers
    ]
    if not references:
        references = [("", "", "none")]
    answer_scores = [
        (
            token_f1_score(prediction, reference_text),
            float(normalize_answer(prediction) == normalize_answer(reference_text)),
            annotation_id,
            answer_type,
        )
        for annotation_id, reference_text, answer_type in references
    ]
    answer_f1, answer_exact, answer_annotation_id, matched_answer_type = max(
        answer_scores, key=lambda item: (item[0], item[1])
    )

    evidence_references = [
        (annotation.annotation_id, _evidence_ids(annotation, text_only=text_evidence_only))
        for annotation in question.answers
    ]
    cited_ids = {citation.passage_id for citation in answer.citations}
    evidence_precision, evidence_recall, evidence_f1, evidence_annotation_id = _best_set_score(
        cited_ids, evidence_references
    )
    read_evidence_ids = {evidence.passage_id for evidence in answer.trace.evidence}
    read_precision, read_recall, read_f1, _ = _best_set_score(
        read_evidence_ids, evidence_references
    )
    gold_union = set().union(*(ids for _, ids in evidence_references))
    ranking = _unique_ranking(answer)
    retrieved_ids = set(ranking[:retrieval_k])
    conditional_selection_scores = [
        len(read_evidence_ids & gold_ids & retrieved_ids) / len(gold_ids & retrieved_ids)
        for _, gold_ids in evidence_references
        if gold_ids & retrieved_ids
    ]
    selection_recall_given_retrieval = (
        max(conditional_selection_scores) if conditional_selection_scores else None
    )
    retrieval = _best_retrieval_metrics(
        ranking,
        [ids for _, ids in evidence_references],
        retrieval_k=retrieval_k,
        ranking_k=ranking_k,
    )
    highlighted_precision, highlighted_recall, highlighted_f1 = _best_span_score(
        answer.citations, question.answers
    )

    boolean_answers = {
        normalize_answer(reference_text)
        for _, reference_text, answer_type in references
        if answer_type == "boolean"
    }
    has_unanswerable = any(answer_type == "none" for _, _, answer_type in references)
    consensus_unanswerable = all(answer_type == "none" for _, _, answer_type in references)
    answerability_disagreement = has_unanswerable and not consensus_unanswerable
    predicted_normalized = normalize_answer(prediction)
    yes_no_accuracy = float(predicted_normalized in boolean_answers) if boolean_answers else None
    unanswerable_accuracy = (
        float(predicted_normalized == normalize_answer("Unanswerable"))
        if consensus_unanswerable
        else None
    )
    unanswerable_annotation_match = (
        float(predicted_normalized == normalize_answer("Unanswerable"))
        if has_unanswerable
        else None
    )
    relevant_document_ids = {
        passage.document_id for passage_id, passage in passages.items() if passage_id in gold_union
    }
    ranked_document_ids = [
        passages[passage_id].document_id
        for passage_id in ranking[:retrieval_k]
        if passage_id in passages
    ]
    paper_recall = (
        float(bool(set(ranked_document_ids) & relevant_document_ids))
        if relevant_document_ids
        else None
    )
    full_document_reads = sum(
        decision.selected and decision.level == ReadLevel.DOCUMENT
        for decision in answer.trace.reads
    )

    return {
        "question_id": question.id,
        "paper_id": question.paper_id,
        "question": question.question,
        "predicted_answer": prediction,
        "reference_answers": [
            {
                "annotation_id": annotation_id,
                "text": reference_text,
                "type": answer_type,
            }
            for annotation_id, reference_text, answer_type in references
        ],
        "answer_f1": answer_f1,
        "answer_exact_match": answer_exact,
        "matched_answer_type": matched_answer_type,
        "matched_answer_annotation_id": answer_annotation_id,
        "yes_no_accuracy": yes_no_accuracy,
        "unanswerable_accuracy": unanswerable_accuracy,
        "unanswerable_annotation_match": unanswerable_annotation_match,
        "answerability_disagreement": float(answerability_disagreement),
        "evidence_precision": evidence_precision,
        "evidence_recall": evidence_recall,
        "evidence_f1": evidence_f1,
        "read_evidence_precision": read_precision,
        "read_evidence_recall": read_recall,
        "read_evidence_f1": read_f1,
        "selection_recall_given_retrieval": selection_recall_given_retrieval,
        "matched_evidence_annotation_id": evidence_annotation_id,
        "citation_precision": evidence_precision,
        "citation_recall": evidence_recall,
        "citation_f1": evidence_f1,
        "highlighted_citation_precision": highlighted_precision,
        "highlighted_citation_recall": highlighted_recall,
        "highlighted_citation_f1": highlighted_f1,
        **retrieval,
        f"paper_recall_at_{retrieval_k}": paper_recall,
        "gold_evidence_passage_ids": sorted(gold_union),
        "citation_passage_ids": sorted(cited_ids),
        "read_evidence_passage_ids": sorted(read_evidence_ids),
        "ranked_passage_ids": ranking[:retrieval_k],
        "coverage": answer.coverage,
        "rounds": answer.trace.rounds,
        "full_document_reads": full_document_reads,
        "full_document_open": float(full_document_reads > 0),
        "stop_reason": answer.stop_reason,
        "read_tokens": answer.trace.read_tokens,
        "evidence_tokens": answer.trace.evidence_tokens,
        "citation_tokens": answer.trace.citation_tokens,
        "api_input_tokens": answer.trace.api_input_tokens,
        "api_output_tokens": answer.trace.api_output_tokens,
        "api_total_tokens": answer.trace.api_total_tokens,
        "estimated_api_cost_usd": answer.trace.estimated_api_cost_usd,
        "api_tokens_by_operation": {
            operation: sum(
                event.get("total_tokens", 0) or 0
                for event in answer.trace.api_calls
                if event.get("operation") == operation
            )
            for operation in sorted(
                {
                    event.get("operation")
                    for event in answer.trace.api_calls
                    if event.get("operation") is not None
                }
            )
        },
        "correct": answer_f1 >= correct_threshold,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def aggregate_results(
    rows: list[dict[str, Any]],
    *,
    retrieval_k: int = 20,
    ranking_k: int = 10,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    errors = [row for row in rows if row.get("status") == "error"]
    metric_keys = [
        "answer_f1",
        "answer_exact_match",
        "yes_no_accuracy",
        "unanswerable_accuracy",
        "evidence_precision",
        "evidence_recall",
        "evidence_f1",
        "read_evidence_precision",
        "read_evidence_recall",
        "read_evidence_f1",
        "selection_recall_given_retrieval",
        "citation_precision",
        "citation_recall",
        "citation_f1",
        "highlighted_citation_precision",
        "highlighted_citation_recall",
        "highlighted_citation_f1",
        f"retrieval_recall_at_{retrieval_k}",
        f"mrr_at_{ranking_k}",
        f"ndcg_at_{ranking_k}",
        f"paper_recall_at_{retrieval_k}",
        "coverage",
    ]
    answer_types = sorted({row["matched_answer_type"] for row in completed})
    correct = [row for row in completed if row.get("correct")]
    costs = [
        row["estimated_api_cost_usd"]
        for row in completed
        if row.get("estimated_api_cost_usd") is not None
    ]
    total_api_tokens = sum(row["api_total_tokens"] for row in completed)
    total_read_tokens = sum(row["read_tokens"] for row in completed)
    total_evidence_tokens = sum(row.get("evidence_tokens", 0) for row in completed)
    total_citation_tokens = sum(row["citation_tokens"] for row in completed)
    total_cost = sum(costs) if costs else None
    api_tokens_by_operation = {
        operation: sum(
            row.get("api_tokens_by_operation", {}).get(operation, 0) for row in completed
        )
        for operation in sorted(
            {
                operation
                for row in completed
                for operation in row.get("api_tokens_by_operation", {})
            }
        )
    }
    return {
        "counts": {
            "rows": len(rows),
            "completed": len(completed),
            "errors": len(errors),
            "papers": len({row["paper_id"] for row in completed}),
            "correct": len(correct),
            "yes_no_questions": sum(row.get("yes_no_accuracy") is not None for row in completed),
            "unanswerable_questions": sum(
                row.get("unanswerable_accuracy") is not None for row in completed
            ),
            "answerability_disagreements": int(
                sum(row.get("answerability_disagreement", 0) for row in completed)
            ),
        },
        "metrics": {
            **{key: _mean(completed, key) for key in metric_keys},
            "full_document_open_rate": _mean(completed, "full_document_open"),
            "answer_f1_by_type": {
                answer_type: _mean(
                    [row for row in completed if row["matched_answer_type"] == answer_type],
                    "answer_f1",
                )
                for answer_type in answer_types
            },
        },
        "efficiency": {
            "mean_latency_ms": _mean(completed, "latency_ms"),
            "mean_rounds": _mean(completed, "rounds"),
            "mean_read_tokens": _mean(completed, "read_tokens"),
            "mean_evidence_tokens": _mean(completed, "evidence_tokens"),
            "mean_citation_tokens": _mean(completed, "citation_tokens"),
            "mean_api_tokens": _mean(completed, "api_total_tokens"),
            "total_read_tokens": total_read_tokens,
            "total_evidence_tokens": total_evidence_tokens,
            "total_citation_tokens": total_citation_tokens,
            "total_api_tokens": total_api_tokens,
            "api_tokens_by_operation": api_tokens_by_operation,
            "total_estimated_api_cost_usd": total_cost,
            "read_tokens_per_correct_answer": (
                total_read_tokens / len(correct) if correct else None
            ),
            "citation_tokens_per_correct_answer": (
                total_citation_tokens / len(correct) if correct else None
            ),
            "api_tokens_per_correct_answer": total_api_tokens / len(correct) if correct else None,
            "estimated_cost_per_correct_answer_usd": (
                total_cost / len(correct) if total_cost is not None and correct else None
            ),
        },
        "stop_reasons": dict(Counter(row["stop_reason"] for row in completed)),
    }


__all__ = [
    "aggregate_results",
    "answer_reference",
    "evaluate_qasper_answer",
    "normalize_answer",
    "strip_citation_markers",
    "token_f1_score",
]
