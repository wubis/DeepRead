import unittest
from pathlib import Path

from deepread.config import Settings
from deepread.corpus import load_corpus
from deepread.models import Requirement
from deepread.reader import HierarchicalReader
from deepread.retrieval import HybridRetriever


CORPUS = Path(__file__).parents[1] / "data" / "sample_corpus"


class ReaderTests(unittest.TestCase):
    def test_multiword_planner_phrases_match_document_tokens(self):
        documents, passages = load_corpus(CORPUS)
        retriever = HybridRetriever(passages, Settings())
        reader = HierarchicalReader(documents, passages)
        hit = next(hit for hit in retriever.search("wetland water storage floods") if retriever.by_id[hit.passage_id].section == "Flood regulation")
        requirement = Requirement("req_1", "task_1", "Evidence that wetlands temporarily store floodwater and slow downstream runoff", ["wetland water storage", "floodwater retention", "runoff attenuation"])
        decision, _ = reader.choose(hit, requirement, 12_000)
        self.assertTrue(decision.selected)
        self.assertGreater(decision.expected_gain, 0)
        passage, span, start, end = reader.evidence_span(hit.passage_id, decision.level, requirement)
        self.assertEqual(passage.section, "Flood regulation")
        self.assertIn("store rainfall", span)
        self.assertEqual(passage.text[start:end].strip(), span)


if __name__ == "__main__":
    unittest.main()
