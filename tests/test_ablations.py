import unittest
from pathlib import Path

from deepread.config import Settings
from deepread.corpus import load_corpus
from deepread.engine import EvidenceGraphEngine
from deepread.retrieval import HybridRetriever
from deepread.text import token_count


CORPUS = Path(__file__).parents[1] / "data" / "sample_corpus"


class RetrievalAblationTests(unittest.TestCase):
    def setUp(self):
        _, self.passages = load_corpus(CORPUS)
        self.dense_vectors = {passage.id: [1.0, 0.0] for passage in self.passages}

    def _search(self, mode):
        calls = []

        def embed(query):
            calls.append(query)
            return [1.0, 0.0]

        retriever = HybridRetriever(
            self.passages,
            Settings(retrieval_mode=mode),
            self.dense_vectors,
            embed,
        )
        return retriever.search("wetland carbon", 3), calls

    def test_retrieval_channels_execute_independently(self):
        bm25_hits, bm25_calls = self._search("bm25")
        dense_hits, dense_calls = self._search("embeddings")
        hybrid_hits, hybrid_calls = self._search("hybrid")

        self.assertFalse(bm25_calls)
        self.assertTrue(all(hit.retrievers == ["bm25"] for hit in bm25_hits))
        self.assertEqual(len(dense_calls), 1)
        self.assertTrue(all(hit.retrievers == ["dense"] for hit in dense_hits))
        self.assertEqual(len(hybrid_calls), 1)
        self.assertTrue(all(set(hit.retrievers) == {"bm25", "dense"} for hit in hybrid_hits))


class ReaderAndSupervisorAblationTests(unittest.TestCase):
    def test_flat_reader_opens_and_charges_full_top_k_passages(self):
        settings = Settings(
            provider="offline",
            reader_mode="flat",
            flat_top_k=2,
            supervisor_mode="single_pass",
        )
        engine = EvidenceGraphEngine(CORPUS, settings)
        answer = engine.ask(
            "How do wetlands reduce floods and store carbon?"
        )
        selected_reads = [decision for decision in answer.trace.reads if decision.selected]

        self.assertEqual(answer.trace.rounds, 1)
        self.assertLessEqual(len(selected_reads), settings.flat_top_k)
        self.assertTrue(selected_reads)
        self.assertTrue(all(decision.level.value == "passage" for decision in selected_reads))
        self.assertEqual(answer.trace.read_tokens, sum(item.token_cost for item in selected_reads))
        for decision in selected_reads:
            passage = engine.retriever.by_id[decision.passage_id]
            self.assertEqual(decision.token_cost, token_count(passage.text))

    def test_single_pass_and_bounded_supervisor_have_distinct_limits(self):
        question = "What does quasar zephyrium prove?"
        single = EvidenceGraphEngine(
            CORPUS,
            Settings(
                provider="offline",
                supervisor_mode="single_pass",
                max_search_rounds=3,
                target_coverage=1.1,
            ),
        ).ask(question)
        bounded = EvidenceGraphEngine(
            CORPUS,
            Settings(
                provider="offline",
                supervisor_mode="bounded",
                max_search_rounds=3,
                target_coverage=1.1,
            ),
        ).ask(question)

        self.assertEqual(single.trace.rounds, 1)
        self.assertEqual(single.stop_reason, "single_pass_complete")
        self.assertEqual(bounded.trace.rounds, 3)
        self.assertEqual(bounded.stop_reason, "max_search_rounds_reached")


if __name__ == "__main__":
    unittest.main()
