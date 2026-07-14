import tempfile
import unittest
from pathlib import Path

from deepread.config import Settings
from deepread.engine import EvidenceGraphEngine


CORPUS = Path(__file__).parents[1] / "data" / "sample_corpus"


class EngineTests(unittest.TestCase):
    def test_end_to_end_answer_is_cited_and_traced(self):
        engine = EvidenceGraphEngine(CORPUS)
        answer = engine.ask("How do wetlands reduce floods and store carbon?")
        self.assertGreater(answer.coverage, 0)
        self.assertTrue(answer.citations)
        self.assertIn("[1]", answer.text)
        self.assertGreater(len(answer.text.split()), 8)
        self.assertEqual(answer.stop_reason, "target_coverage_reached")
        self.assertLessEqual(answer.trace.tokens_used, engine.settings.max_evidence_tokens)

    def test_hard_budget_is_respected(self):
        engine = EvidenceGraphEngine(CORPUS, Settings(max_evidence_tokens=3))
        answer = engine.ask("Explain hybrid retrieval")
        self.assertLessEqual(answer.trace.tokens_used, 3)

    def test_trace_can_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "trace.json"
            EvidenceGraphEngine(CORPUS).ask("What is reciprocal rank fusion?", target)
            self.assertTrue(target.exists())
            self.assertIn('"ranking"', target.read_text())

    def test_corpus_stats(self):
        stats = EvidenceGraphEngine(CORPUS).corpus_stats()
        self.assertEqual(stats["documents"], 3)
        self.assertGreaterEqual(stats["passages"], 8)


if __name__ == "__main__":
    unittest.main()
