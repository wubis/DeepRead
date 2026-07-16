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
    GroundedBooleanAnswer,
    GroundedClaim,
    PlannedQuery,
    PlannedRequirement,
    PlannedTask,
    RerankResult,
    CandidateScore,
    OpenAIProvider,
)
from deepread.models import Passage, Requirement, SearchHit


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
    def __init__(self, calls, mismatched_verdicts=False, boolean_mode=None):
        self.calls = calls
        self.mismatched_verdicts = mismatched_verdicts
        self.boolean_mode = boolean_mode

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
            self.calls.append(
                (
                    "rerank_payload",
                    len(payload["candidates"]),
                    max(len(item["text"]) for item in payload["candidates"]),
                    [item["passage_id"] for item in payload["candidates"]],
                )
            )
            parsed = RerankResult(candidates=[CandidateScore(passage_id=item["passage_id"], relevance=0.95 if "Wetlands" in item["title"] else 0.1, requirement_ids=["req_1_1", "req_1_2"]) for item in payload["candidates"]])
        elif text_format is EvidenceAssessment:
            payload = json.loads(user)
            assert payload["question"]
            self.calls.append(("assessment_payload", len(payload["evidence"])))
            verdicts = []
            for index, item in enumerate(payload["evidence"]):
                verdict = "supports"
                answer_value = (
                    False if payload["expected_answer_type"] == "boolean" else None
                )
                if self.boolean_mode == "partial_no":
                    verdict = "partial"
                elif self.boolean_mode == "conflict":
                    verdict = "supports" if index == 0 else "challenges"
                    answer_value = index == 0
                verdicts.append(
                    EvidenceVerdict(
                        evidence_id=item["id"],
                        requirement_id=(
                            "wrong_requirement"
                            if self.mismatched_verdicts
                            else item["requirement_id"]
                        ),
                        verdict=verdict,
                        confidence=0.95,
                        answer_value=answer_value,
                    )
                )
            parsed = EvidenceAssessment(verdicts=verdicts)
        elif text_format in {GroundedAnswer, GroundedBooleanAnswer}:
            payload = json.loads(user)
            is_boolean = payload["expected_answer_type"] == "boolean"
            requirement_ids = list(
                dict.fromkeys(item["requirement_id"] for item in payload["evidence"])
            )
            claims = [
                GroundedClaim(
                    text="Wetlands provide supported flood and climate benefits.",
                    evidence_ids=[
                        item["id"]
                        for item in payload["evidence"]
                        if item["requirement_id"] == requirement_id
                    ],
                    requirement_id=requirement_id,
                    confidence=0.9,
                )
                for requirement_id in requirement_ids
            ]
            if text_format is GroundedBooleanAnswer:
                parsed = GroundedBooleanAnswer(
                    boolean_answer=False,
                    claims=claims,
                    unsupported_requirement_ids=[],
                )
            else:
                parsed = GroundedAnswer(
                    answer_type=payload["expected_answer_type"],
                    answer=(
                        "Unanswerable"
                        if is_boolean
                        else "Wetlands provide supported flood and climate benefits."
                    ),
                    unanswerable=is_boolean,
                    claims=[] if is_boolean else claims,
                    unsupported_requirement_ids=[],
                )
        else:
            raise AssertionError(text_format)
        usage = SimpleNamespace(input_tokens=50, output_tokens=20, total_tokens=70, input_tokens_details=SimpleNamespace(cached_tokens=0))
        return SimpleNamespace(output_parsed=parsed, usage=usage, model=model)


class FakeOpenAIClient:
    def __init__(self, mismatched_verdicts=False, boolean_mode=None):
        self.calls = []
        self.embeddings = FakeEmbeddings(self.calls)
        self.responses = FakeResponses(self.calls, mismatched_verdicts, boolean_mode)


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
            rerank_payload = next(event for event in client.calls if event[0] == "rerank_payload")
            self.assertLessEqual(rerank_payload[1], settings.model_rerank_top_k)
            self.assertLessEqual(rerank_payload[2], settings.model_rerank_max_chars)
            self.assertEqual(answer.coverage, 1.0)
            self.assertTrue(answer.trace.claims)
            self.assertTrue(all(evidence_id.startswith("evidence_") for claim in answer.trace.claims for evidence_id in claim.evidence_ids))
            evidence_by_id = {item.id: item for item in answer.citations}
            self.assertTrue(all(evidence_by_id[evidence_id].requirement_id == claim.requirement_id for claim in answer.trace.claims for evidence_id in claim.evidence_ids))
            self.assertIn("[1]", answer.text)
            self.assertLessEqual(answer.trace.rounds, settings.max_search_rounds)
            requirements = [requirement for task in answer.trace.tasks for requirement in task.requirements]
            self.assertEqual(len(requirements), settings.max_requirements)
            self.assertGreaterEqual(len(answer.citations), settings.max_requirements)
            self.assertEqual({item.requirement_id for item in answer.citations}, {item.id for item in requirements})
            self.assertEqual(answer.trace.read_tokens, sum(item.token_cost for item in answer.citations))
            self.assertEqual(
                answer.trace.evidence_tokens,
                sum(item.citation_tokens for item in answer.trace.evidence),
            )
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

    def test_atomic_question_is_reduced_to_one_requirement(self):
        client = FakeOpenAIClient()
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            answer_policy="benchmark",
        )
        engine = EvidenceGraphEngine(CORPUS, settings, openai_client=client)

        answer = engine.ask("Which dataset was used?")
        requirements = [item for task in answer.trace.tasks for item in task.requirements]

        self.assertEqual(len(requirements), 1)

    def test_boolean_answer_uses_exact_yes_no_shape(self):
        client = FakeOpenAIClient()
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            answer_policy="benchmark",
        )
        answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask(
            "Do wetlands eliminate all flood risk?"
        )

        self.assertRegex(answer.text, r"^No(?:\s+\[\d+\])*$")
        self.assertTrue(all(item.answer_value is False for item in answer.citations))

    def test_boolean_routing_rescues_a_context_candidate_below_cutoff(self):
        client = FakeOpenAIClient()
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            model_rerank_top_k=2,
            model_rerank_rescue_per_requirement=1,
        )
        passages = [
            Passage(f"p{i}", "doc", "Study", "Methods", f"unrelated candidate {i}", i)
            for i in range(3)
        ]
        passages.append(
            Passage(
                "context",
                "doc",
                "Study",
                "Annotations",
                "The authors manually annotated the evaluation dataset.",
                3,
            )
        )
        hits = [SearchHit(p.id, p.document_id, final_score=1.0 - index / 10) for index, p in enumerate(passages)]
        requirement = Requirement(
            "req_1_1",
            "task_1",
            "Whether a crowdsourcing platform was used for manual annotations",
            ["crowdsourcing platform", "manual annotations"],
            ["manual annotations", "annotation process"],
        )

        OpenAIProvider(settings, client).rerank(
            "did they use a crowdsourcing platform for manual annotations?",
            [requirement],
            hits,
            {passage.id: passage for passage in passages},
        )

        payload = next(event for event in client.calls if event[0] == "rerank_payload")
        self.assertEqual(payload[1], 3)
        self.assertIn("context", payload[3])

    def test_grounded_boolean_policy_can_abstain(self):
        client = FakeOpenAIClient()
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            answer_policy="grounded",
        )

        answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask(
            "Do wetlands eliminate all flood risk?"
        )

        self.assertEqual(answer.text, "Unanswerable")
        self.assertFalse(answer.citations)

    def test_consistent_partial_boolean_evidence_is_aggregated(self):
        client = FakeOpenAIClient(boolean_mode="partial_no")
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            answer_policy="benchmark",
            evidence_candidates_per_requirement=2,
        )

        answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask(
            "Do wetlands eliminate all flood risk?"
        )

        self.assertRegex(answer.text, r"^No(?:\s+\[\d+\])+$")
        self.assertGreaterEqual(len(answer.citations), 2)
        self.assertTrue(
            all(item.relation == "supports_aggregate" for item in answer.citations)
        )

    def test_boolean_counterevidence_can_ground_no(self):
        client = FakeOpenAIClient(boolean_mode="conflict")
        settings = Settings(
            provider="openai",
            retrieval_mode="bm25",
            answer_policy="benchmark",
            evidence_candidates_per_requirement=2,
        )

        answer = EvidenceGraphEngine(CORPUS, settings, openai_client=client).ask(
            "Do wetlands eliminate all flood risk?"
        )

        self.assertRegex(answer.text, r"^No(?:\s+\[\d+\])+$")
        self.assertTrue(all(item.answer_value is False for item in answer.citations))

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
