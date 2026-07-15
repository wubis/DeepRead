import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from deepread.config import Settings
from deepread.engine import EvidenceGraphEngine
from deepread.openai_provider import (
    EvidenceAssessment,
    EvidenceVerdict,
    GroundedAnswer,
    GroundedClaim,
    PlannedQuery,
    PlannedRequirement,
    PlannedTask,
    RerankResult,
    CandidateScore,
)


CORPUS = Path(__file__).parents[1] / "data" / "sample_corpus"


class FakeEmbeddings:
    def __init__(self, calls):
        self.calls = calls

    def create(self, model, input, encoding_format):
        self.calls.append(("embeddings", model, len(input)))
        vectors = []
        for text in input:
            lower = text.lower()
            vectors.append([float("wetland" in lower), float("climate" in lower or "carbon" in lower), float("retrieval" in lower)])
        return SimpleNamespace(data=[SimpleNamespace(embedding=vector) for vector in vectors], usage=SimpleNamespace(prompt_tokens=10, total_tokens=10))


class FakeResponses:
    def __init__(self, calls, mismatched_verdicts=False):
        self.calls = calls
        self.mismatched_verdicts = mismatched_verdicts

    def parse(self, model, reasoning, store, input, text_format):
        self.calls.append(("response", text_format.__name__, model, store))
        user = input[-1]["content"]
        if text_format is PlannedQuery:
            parsed = PlannedQuery(tasks=[
                PlannedTask(question="Wetland flood benefits", requirements=[
                    PlannedRequirement(description="How wetlands store floodwater", keywords=["wetland water storage", "floodwater retention"]),
                    PlannedRequirement(description="How wetlands slow downstream runoff", keywords=["runoff attenuation", "flood peaks"]),
                ]),
                PlannedTask(question="Wetland climate benefits", requirements=[
                    PlannedRequirement(description="How waterlogged wetland soils store carbon", keywords=["waterlogged soils", "carbon sequestration"]),
                    PlannedRequirement(description="Why peatland conservation avoids emissions", keywords=["peatlands", "avoided emissions"]),
                    PlannedRequirement(description="What methane tradeoffs limit wetland benefits", keywords=["methane emissions", "climate tradeoffs"]),
                ]),
            ])
        elif text_format is RerankResult:
            payload = json.loads(user)
            parsed = RerankResult(candidates=[CandidateScore(passage_id=item["passage_id"], relevance=0.95 if "Wetlands" in item["title"] else 0.1, requirement_ids=["req_1_1", "req_1_2"]) for item in payload["candidates"]])
        elif text_format is EvidenceAssessment:
            payload = json.loads(user)
            parsed = EvidenceAssessment(verdicts=[EvidenceVerdict(evidence_id=item["id"], requirement_id="wrong_requirement" if self.mismatched_verdicts else item["requirement_id"], verdict="supports", confidence=0.95) for item in payload["evidence"]])
        elif text_format is GroundedAnswer:
            payload = json.loads(user)
            parsed = GroundedAnswer(claims=[GroundedClaim(text="Wetlands provide supported flood and climate benefits.", evidence_ids=[item["id"] for item in payload["evidence"]], requirement_id="req_1_1", confidence=0.9)], unsupported_requirement_ids=[])
        else:
            raise AssertionError(text_format)
        usage = SimpleNamespace(input_tokens=50, output_tokens=20, total_tokens=70, input_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(output_parsed=parsed, usage=usage, model=model)


class FakeOpenAIClient:
    def __init__(self, mismatched_verdicts=False):
        self.calls = []
        self.embeddings = FakeEmbeddings(self.calls)
        self.responses = FakeResponses(self.calls, mismatched_verdicts)


class OpenAIEngineTests(unittest.TestCase):
    def test_openai_pipeline_is_bounded_grounded_and_traced(self):
        with tempfile.TemporaryDirectory() as directory:
            client = FakeOpenAIClient()
            settings = Settings(provider="openai", embedding_cache_dir=directory, model_input_cost_per_million=1.0, model_output_cost_per_million=2.0, embedding_cost_per_million=0.1)
            answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask("How do wetlands reduce floods and mitigate climate change?")
            operations = [event["operation"] for event in answer.trace.api_calls]
            self.assertEqual(answer.trace.provider, "openai")
            self.assertIn("embed_corpus", operations)
            self.assertIn("embed_query", operations)
            self.assertIn("plan", operations)
            self.assertIn("rerank", operations)
            self.assertIn("assess_evidence", operations)
            self.assertIn("synthesize", operations)
            self.assertEqual(answer.coverage, 1.0)
            self.assertTrue(answer.trace.claims)
            self.assertTrue(all(evidence_id.startswith("evidence_") for claim in answer.trace.claims for evidence_id in claim.evidence_ids))
            evidence_by_id = {item.id: item for item in answer.citations}
            self.assertTrue(all(evidence_by_id[evidence_id].requirement_id == claim.requirement_id for claim in answer.trace.claims for evidence_id in claim.evidence_ids))
            self.assertIn("[1]", answer.text)
            self.assertLessEqual(answer.trace.rounds, settings.max_search_rounds)
            requirements = [requirement for task in answer.trace.tasks for requirement in task.requirements]
            self.assertEqual(len(requirements), settings.max_requirements)
            self.assertEqual(len(answer.citations), settings.max_requirements)
            self.assertEqual({item.requirement_id for item in answer.citations}, {item.id for item in requirements})
            self.assertEqual(answer.trace.read_tokens, sum(item.token_cost for item in answer.citations))
            self.assertEqual(answer.trace.citation_tokens, sum(item.citation_tokens for item in answer.citations))
            self.assertGreater(answer.trace.api_total_tokens, 0)
            self.assertIsNotNone(answer.trace.estimated_api_cost_usd)
            self.assertTrue(all(item.read_level.value != "document" for item in answer.citations))
            self.assertTrue(all(0 <= item.char_start < item.char_end <= len(next(p.text for p in EvidenceGraphEngine(CORPUS).passages if p.id == item.passage_id)) for item in answer.citations))

    def test_mismatched_assessor_requirement_cannot_inflate_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(provider="openai", embedding_cache_dir=directory, max_search_rounds=1)
            answer = EvidenceGraphEngine(CORPUS, settings, openai_client=FakeOpenAIClient(mismatched_verdicts=True)).ask("How do wetlands reduce floods and mitigate climate change?")
            self.assertEqual(answer.coverage, 0.0)
            self.assertTrue(all(not item.supports and item.relation == "insufficient" for item in answer.citations))

    def test_bm25_only_with_reranking_disabled_avoids_embedding_and_rerank_calls(self):
        client = FakeOpenAIClient()
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            enable_model_rerank=False,
            max_search_rounds=1,
        )
        answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask(
            "How do wetlands reduce floods?"
        )
        operations = [event[1] if event[0] == "response" else event[0] for event in client.calls]

        self.assertNotIn("embeddings", operations)
        self.assertNotIn("RerankResult", operations)
        self.assertNotIn("embed_corpus", [event["operation"] for event in answer.trace.api_calls])
        self.assertNotIn("embed_query", [event["operation"] for event in answer.trace.api_calls])
        self.assertNotIn("rerank", [event["operation"] for event in answer.trace.api_calls])


if __name__ == "__main__":
    unittest.main()
