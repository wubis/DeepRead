import json
import tempfile
import unittest
from pathlib import Path

from deepread.config import Settings
from deepread.engine import EvidenceGraphEngine
from deepread.qasper import adapt_qasper, load_qasper


ROW = {
    "id": "paper-1",
    "title": "A Provenance Test Paper",
    "abstract": "The paper studies evidence-aware retrieval.",
    "full_text": {
        "section_name": ["Introduction", "Results"],
        "paragraphs": [
            ["The first paragraph keeps its original\nwhitespace.", "A second paragraph."],
            ["Results show a 10% improvement over the baseline."],
        ],
    },
    "figures_and_tables": {
        "caption": ["Figure 1: Accuracy by retrieval method."],
        "file": ["figure-1.png"],
    },
    "qas": {
        "question_id": ["q-1", "q-2"],
        "question": ["What improvement was reported?", "What does Figure 1 show?"],
        "answers": [
            {
                "annotation_id": ["a-1", "a-2"],
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": ["10% improvement"],
                        "free_form_answer": "",
                        "yes_no": None,
                        "evidence": [
                            "Results show a 10% improvement over the baseline.",
                            "This annotation cannot be resolved.",
                            "Results",
                        ],
                        "highlighted_evidence": [
                            "10% improvement over the baseline",
                            "Results\nResults show a 10% improvement over the baseline.",
                        ],
                    },
                    {
                        "unanswerable": True,
                        "extractive_spans": [],
                        "free_form_answer": "",
                        "yes_no": None,
                        "evidence": [],
                        "highlighted_evidence": [],
                    },
                ],
            },
            {
                "annotation_id": ["a-3"],
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "free_form_answer": "Accuracy by retrieval method.",
                        "yes_no": None,
                        "evidence": ["FLOAT SELECTED: Figure 1: Accuracy by retrieval method."],
                        "highlighted_evidence": [],
                    }
                ],
            },
        ],
        "topic_background": ["familiar", "familiar"],
        "paper_read": [False, False],
        "search_query": ["evidence retrieval", "retrieval accuracy"],
    },
}


class QasperAdapterTests(unittest.TestCase):
    def test_adapter_preserves_source_units_and_maps_gold_evidence(self):
        dataset = adapt_qasper([ROW])

        self.assertEqual(len(dataset.documents), 1)
        self.assertEqual(len(dataset.passages), 7)
        paragraph = next(
            item
            for item in dataset.passages
            if item.metadata["source_path"] == "full_text.paragraphs[0][0]"
        )
        self.assertEqual(paragraph.text, "The first paragraph keeps its original\nwhitespace.")
        self.assertEqual(paragraph.metadata["section_index"], 0)
        self.assertEqual(paragraph.metadata["paragraph_index"], 0)
        self.assertEqual(paragraph.metadata["source_char_end"], len(paragraph.text))

        question = dataset.questions[0]
        self.assertEqual(question.id, "q-1")
        self.assertEqual(len(question.answers), 2)
        answer = question.answers[0]
        self.assertTrue(answer.evidence[0].resolved)
        self.assertFalse(answer.evidence[1].resolved)
        self.assertEqual(answer.evidence[2].matches[0].source_type, "section_heading")
        evidence_match = answer.evidence[0].matches[0]
        evidence_passage = next(
            item for item in dataset.passages if item.id == evidence_match.passage_id
        )
        self.assertEqual(
            evidence_passage.text[evidence_match.char_start : evidence_match.char_end],
            evidence_passage.text,
        )
        highlighted = answer.highlighted_evidence[0].matches[0]
        self.assertEqual(
            evidence_passage.text[highlighted.char_start : highlighted.char_end],
            "10% improvement over the baseline",
        )
        composite_types = {
            item.source_type for item in answer.highlighted_evidence[1].matches
        }
        self.assertEqual(composite_types, {"section_heading", "paragraph"})

        float_ref = dataset.questions[1].answers[0].evidence[0]
        self.assertEqual(float_ref.matches[0].source_type, "figure_table")
        self.assertEqual(float_ref.matches[0].match_type, "float_exact")

    def test_adapter_corpus_runs_through_the_engine(self):
        dataset = adapt_qasper([ROW])
        corpus = dataset.corpus(["paper-1"])
        engine = EvidenceGraphEngine(corpus, Settings(provider="offline"))

        answer = engine.ask("What improvement did the results show over the baseline?")

        self.assertTrue(answer.citations)
        self.assertTrue(
            all(item.source_metadata["dataset"] == "qasper" for item in answer.citations)
        )
        passage_by_id = {item.id: item for item in corpus.passages}
        self.assertTrue(
            all(
                passage_by_id[item.passage_id].metadata["dataset"] == "qasper"
                for item in answer.citations
            )
        )
        cited_passages = {item.passage_id for item in answer.citations}
        self.assertTrue(cited_passages & {item.id for item in corpus.passages})

    def test_loader_accepts_official_object_keyed_by_paper_id(self):
        row = {key: value for key, value in ROW.items() if key != "id"}
        row["full_text"] = [
            {"section_name": name, "paragraphs": paragraphs}
            for name, paragraphs in zip(
                ROW["full_text"]["section_name"], ROW["full_text"]["paragraphs"]
            )
        ]
        row["figures_and_tables"] = [
            {"caption": caption, "file": file}
            for caption, file in zip(
                ROW["figures_and_tables"]["caption"],
                ROW["figures_and_tables"]["file"],
            )
        ]
        row["qas"] = [
            {key: column[index] for key, column in ROW["qas"].items()}
            for index in range(len(ROW["qas"]["question_id"]))
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.json"
            path.write_text(json.dumps({"paper-from-key": row}), encoding="utf-8")

            dataset = load_qasper(path)

        self.assertEqual(dataset.documents[0].metadata["paper_id"], "paper-from-key")
        self.assertEqual(dataset.questions[0].paper_id, "paper-from-key")


if __name__ == "__main__":
    unittest.main()
