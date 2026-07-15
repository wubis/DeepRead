import json
import math
import tempfile
import unittest
from pathlib import Path

from deepread.evaluation import (
    aggregate_results,
    evaluate_qasper_answer,
    normalize_answer,
    token_f1_score,
)
from deepread.evaluation_runner import main as run_benchmark
from deepread.models import Answer, Evidence, QueryTrace, ReadLevel, SearchHit
from deepread.qasper import adapt_qasper


PAPER = {
    "id": "paper-eval",
    "title": "Evaluation Paper",
    "abstract": "This paper evaluates retrieval.",
    "full_text": [
        {
            "section_name": "Results",
            "paragraphs": ["Results show a 10% improvement over the baseline."],
        }
    ],
    "figures_and_tables": [],
    "qas": [
        {
            "question_id": "question-eval",
            "question": "What improvement was reported?",
            "answers": [
                {
                    "annotation_id": "answer-extractive",
                    "answer": {
                        "unanswerable": False,
                        "extractive_spans": ["10% improvement"],
                        "free_form_answer": "",
                        "yes_no": None,
                        "evidence": ["Results show a 10% improvement over the baseline."],
                        "highlighted_evidence": ["10% improvement"],
                    },
                },
                {
                    "annotation_id": "answer-none",
                    "answer": {
                        "unanswerable": True,
                        "extractive_spans": [],
                        "free_form_answer": "",
                        "yes_no": None,
                        "evidence": [],
                        "highlighted_evidence": [],
                    },
                },
            ],
        }
    ],
}


class EvaluationMetricTests(unittest.TestCase):
    def test_official_answer_normalization(self):
        self.assertEqual(normalize_answer("The Model's Answer!"), "models answer")
        self.assertEqual(token_f1_score("A useful model", "the useful model"), 1.0)

    def test_question_metrics_use_best_annotation_and_gold_provenance(self):
        dataset = adapt_qasper([PAPER])
        question = dataset.questions[0]
        passage_by_id = {passage.id: passage for passage in dataset.passages}
        gold = next(
            passage
            for passage in dataset.passages
            if passage.metadata["source_type"] == "paragraph"
        )
        distractor = next(
            passage for passage in dataset.passages if passage.metadata["source_type"] == "abstract"
        )
        start = gold.text.index("10% improvement")
        citation = Evidence(
            "evidence-1",
            "requirement-1",
            gold.id,
            gold.document_id,
            gold.title,
            gold.section,
            gold.text[start : start + len("10% improvement")],
            1.0,
            ReadLevel.PASSAGE,
            3,
            char_start=start,
            char_end=start + len("10% improvement"),
            citation_tokens=3,
            source_metadata=gold.metadata,
        )
        trace = QueryTrace(question.question)
        trace.ranking = [
            SearchHit(distractor.id, distractor.document_id),
            SearchHit(gold.id, gold.document_id),
        ]
        trace.rounds = 1
        trace.read_tokens = 3
        trace.citation_tokens = 3
        trace.api_total_tokens = 100
        trace.estimated_api_cost_usd = 0.01
        answer = Answer(
            question.question,
            "10% improvement [1]",
            [citation],
            1.0,
            "target_coverage_reached",
            trace,
        )

        row = evaluate_qasper_answer(question, answer, passage_by_id)

        self.assertEqual(row["answer_f1"], 1.0)
        self.assertEqual(row["answer_exact_match"], 1.0)
        self.assertEqual(row["matched_answer_type"], "extractive")
        self.assertEqual(row["unanswerable_accuracy"], 0.0)
        self.assertEqual(row["evidence_f1"], 1.0)
        self.assertEqual(row["highlighted_citation_f1"], 1.0)
        self.assertEqual(row["retrieval_recall_at_20"], 1.0)
        self.assertEqual(row["mrr_at_10"], 0.5)
        self.assertAlmostEqual(row["ndcg_at_10"], 1 / math.log2(3))
        self.assertTrue(row["correct"])

        row.update({"status": "ok", "latency_ms": 12.0})
        summary = aggregate_results([row])
        self.assertEqual(summary["counts"]["correct"], 1)
        self.assertEqual(summary["metrics"]["answer_f1_by_type"]["extractive"], 1.0)
        self.assertEqual(summary["metrics"]["full_document_open_rate"], 0.0)
        self.assertEqual(summary["efficiency"]["read_tokens_per_correct_answer"], 3)
        self.assertEqual(summary["efficiency"]["citation_tokens_per_correct_answer"], 3)
        self.assertEqual(summary["efficiency"]["api_tokens_per_correct_answer"], 100)
        self.assertEqual(summary["efficiency"]["estimated_cost_per_correct_answer_usd"], 0.01)


class EvaluationRunnerTests(unittest.TestCase):
    def test_runner_checkpoints_traces_and_resumes(self):
        article = json.loads(
            json.dumps({key: value for key, value in PAPER.items() if key != "id"})
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_path = root / "qasper.json"
            output_path = root / "results.json"
            trace_dir = root / "traces"
            dataset_path.write_text(
                json.dumps({PAPER["id"]: article}),
                encoding="utf-8",
            )
            arguments = [
                "--qasper-path",
                str(dataset_path),
                "--max-papers",
                "1",
                "--max-questions",
                "1",
                "--output",
                str(output_path),
                "--trace-dir",
                str(trace_dir),
            ]

            first = run_benchmark(arguments)
            resumed = run_benchmark(arguments)

            self.assertEqual(first["schema_version"], 1)
            self.assertEqual(first["summary"]["counts"]["completed"], 1)
            self.assertEqual(len(first["rows"]), 1)
            self.assertEqual(len(resumed["rows"]), 1)
            self.assertTrue(output_path.exists())
            self.assertEqual(len(list(trace_dir.glob("*.json"))), 1)

            corpus_wide = run_benchmark(
                [
                    "--qasper-path",
                    str(dataset_path),
                    "--mode",
                    "corpus-wide",
                    "--max-papers",
                    "1",
                    "--max-questions",
                    "1",
                    "--output",
                    str(root / "corpus-wide.json"),
                    "--trace-dir",
                    str(root / "corpus-wide-traces"),
                ]
            )
            self.assertEqual(corpus_wide["summary"]["counts"]["completed"], 1)

            article["qas"][0]["question"] = "A changed question invalidates resume."
            dataset_path.write_text(
                json.dumps({PAPER["id"]: article}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different run fingerprint"):
                run_benchmark(arguments)


if __name__ == "__main__":
    unittest.main()
