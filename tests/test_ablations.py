import unittest
from pathlib import Path

from deepread.config import Settings
from deepread.corpus import load_corpus
from deepread.engine import EvidenceGraphEngine
from deepread.models import Passage, SearchHit
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

    def test_figure_table_candidates_receive_structure_boost(self):
        passages = [
            Passage(
                "paragraph",
                "doc-1",
                "Study",
                "Results",
                "Language pairs and evaluation scores are reported.",
                0,
                {"source_type": "paragraph"},
            ),
            Passage(
                "table",
                "doc-2",
                "Study",
                "Figures and Tables",
                "Language pairs and evaluation scores are reported.",
                0,
                {"source_type": "figure_table"},
            ),
        ]
        retriever = HybridRetriever(passages, Settings(retrieval_mode="bm25"))

        hits = retriever.search("Which language pairs are in the results table?", 2)

        self.assertEqual(hits[0].passage_id, "table")
        self.assertIn("structure_boost", hits[0].retrievers)

    def test_diversification_penalizes_similar_passages_not_shared_document(self):
        passages = [
            Passage("first", "doc", "Study", "A", "alpha beta gamma", 0),
            Passage("duplicate", "doc", "Study", "A", "alpha beta gamma repeated", 1),
            Passage("distinct", "doc", "Study", "B", "delta epsilon zeta", 2),
        ]
        retriever = HybridRetriever(
            passages,
            Settings(retrieval_mode="bm25", redundancy_weight=0.5),
        )
        hits = [SearchHit(passage.id, passage.document_id, final_score=0.1) for passage in passages]

        diversified = retriever._diversify(hits, 3)

        self.assertEqual([hit.passage_id for hit in diversified[:2]], ["first", "distinct"])
        self.assertEqual(diversified[1].redundancy_penalty, 0.0)
        self.assertGreater(diversified[2].redundancy_penalty, 0.0)


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
        self.assertEqual(bounded.trace.rounds, 1)
        self.assertEqual(bounded.stop_reason, "no_evidence_progress")


if __name__ == "__main__":
    unittest.main()
