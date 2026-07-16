from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .models import Claim, Evidence, Passage, Requirement, SearchHit, TaskNode
from .planner import infer_answer_type, should_decompose
from .text import best_sentence_window, keywords, tokens


class PlannedRequirement(BaseModel):
    description: str
    keywords: list[str] = Field(min_length=1, max_length=12)
    routing_keywords: list[str] = Field(default_factory=list, max_length=8)


class PlannedTask(BaseModel):
    question: str
    requirements: list[PlannedRequirement] = Field(min_length=1, max_length=3)


class PlannedQuery(BaseModel):
    tasks: list[PlannedTask] = Field(min_length=1, max_length=3)


class CandidateScore(BaseModel):
    passage_id: str
    relevance: float = Field(ge=0, le=1)
    requirement_ids: list[str]
    rationale: str | None = None


class RerankResult(BaseModel):
    candidates: list[CandidateScore]


class EvidenceVerdict(BaseModel):
    evidence_id: str
    requirement_id: str
    verdict: Literal["supports", "partial", "challenges", "insufficient"]
    confidence: float = Field(ge=0, le=1)
    answer_value: bool | None = None


class EvidenceAssessment(BaseModel):
    verdicts: list[EvidenceVerdict]


class GroundedClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)
    requirement_id: str
    confidence: float = Field(ge=0, le=1)


class GroundedAnswer(BaseModel):
    answer_type: Literal["boolean", "extractive", "abstractive", "unanswerable"]
    answer: str
    boolean_answer: bool | None = None
    unanswerable: bool = False
    claims: list[GroundedClaim]
    unsupported_requirement_ids: list[str]


class GroundedBooleanAnswer(BaseModel):
    answer_type: Literal["boolean"] = "boolean"
    boolean_answer: bool
    claims: list[GroundedClaim] = Field(min_length=1)
    unsupported_requirement_ids: list[str]


class OpenAIProvider:
    """Bounded OpenAI operations with structured outputs and auditable usage."""

    def __init__(self, settings: Settings, client: Any | None = None):
        self.settings = settings
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("Install OpenAI support with: pip install -e '.[openai]'") from exc
            client = OpenAI()
        self.client = client
        self.events: list[dict[str, Any]] = []

    def _record(self, operation: str, response: Any, **extra: Any) -> None:
        usage = getattr(response, "usage", None)
        event: dict[str, Any] = {"operation": operation, "model": getattr(response, "model", None), **extra}
        if usage is not None:
            event["input_tokens"] = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
            event["output_tokens"] = getattr(usage, "output_tokens", None)
            event["total_tokens"] = getattr(usage, "total_tokens", None)
            details = getattr(usage, "input_tokens_details", None)
            if details is not None:
                event["cached_tokens"] = getattr(details, "cached_tokens", 0)
            input_tokens = event.get("input_tokens") or 0
            output_tokens = event.get("output_tokens") or 0
            if operation.startswith("embed") and self.settings.embedding_cost_per_million is not None:
                event["estimated_cost_usd"] = input_tokens * self.settings.embedding_cost_per_million / 1_000_000
            elif self.settings.model_input_cost_per_million is not None and self.settings.model_output_cost_per_million is not None:
                event["estimated_cost_usd"] = (input_tokens * self.settings.model_input_cost_per_million + output_tokens * self.settings.model_output_cost_per_million) / 1_000_000
        self.events.append({key: value for key, value in event.items() if value is not None})

    def drain_events(self) -> list[dict[str, Any]]:
        events, self.events = self.events, []
        return events

    def _parse(self, operation: str, schema: type[BaseModel], system: str, user: str) -> BaseModel:
        response = self.client.responses.parse(
            model=self.settings.openai_model,
            reasoning={"effort": self.settings.openai_reasoning_effort},
            store=self.settings.openai_store,
            input=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            text_format=schema,
        )
        self._record(operation, response)
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError(f"OpenAI returned no structured output for {operation}")
        return parsed

    def plan(self, question: str, max_tasks: int, max_requirements: int) -> list[TaskNode]:
        parsed = self._parse(
            "plan",
            PlannedQuery,
            "You plan evidence retrieval. Use exactly one task and one requirement for an atomic "
            "question. Decompose only when the question explicitly asks for multiple independent "
            "facts. Do not add requirements for resolving pronouns, defining requested items, or "
            "explaining methodology unless the question asks for them. Requirements must describe "
            "facts that source text can directly support. For a yes/no question, routing_keywords "
            "must name the underlying activity or procedure that relevant evidence would describe "
            "even when the answer is No. Do not put speculative answers, vendor names, or examples "
            "such as Mechanical Turk into routing_keywords unless they appear in the question. "
            "Do not answer the question.",
            f"Question: {question}\nHard limits: at most {max_tasks} tasks and at most "
            f"{max_requirements} total requirements. Prefer the smallest sufficient checklist.",
        )
        assert isinstance(parsed, PlannedQuery)
        if not should_decompose(question):
            planned = [
                requirement
                for task in parsed.tasks
                for requirement in task.requirements
            ]
            if not planned:
                return []
            question_terms = set(tokens(question))

            def relevance(requirement: PlannedRequirement) -> tuple[int, int]:
                requirement_terms = set(tokens(requirement.description)) | {
                    term
                    for phrase in requirement.keywords
                    for term in tokens(phrase)
                }
                overlap = len(question_terms & requirement_terms)
                return overlap, -len(requirement_terms)

            selected = max(planned, key=relevance)
            routing_keywords = (
                selected.routing_keywords[:8]
                if infer_answer_type(question) == "boolean"
                else []
            )
            return [
                TaskNode(
                    "task_1",
                    question,
                    [
                        Requirement(
                            "req_1_1",
                            "task_1",
                            selected.description,
                            selected.keywords[:12],
                            routing_keywords,
                        )
                    ],
                )
            ]
        tasks: list[TaskNode] = []
        remaining = max_requirements
        for task_index, task in enumerate(parsed.tasks[:max_tasks], 1):
            if remaining <= 0:
                break
            task_id = f"task_{task_index}"
            requirements = [
                Requirement(
                    f"req_{task_index}_{req_index}",
                    task_id,
                    requirement.description,
                    requirement.keywords[:12],
                    (
                        requirement.routing_keywords[:8]
                        if infer_answer_type(question) == "boolean"
                        else []
                    ),
                )
                for req_index, requirement in enumerate(task.requirements[:remaining], 1)
            ]
            if requirements:
                tasks.append(TaskNode(task_id, task.question, requirements))
                remaining -= len(requirements)
        return tasks

    def embed(self, texts: list[str], operation: str = "embed") -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 128):
            batch = texts[start : start + 128]
            response = self.client.embeddings.create(model=self.settings.openai_embedding_model, input=batch, encoding_format="float")
            self._record(operation, response, items=len(batch), embedding_model=self.settings.openai_embedding_model)
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embedding_index(self, passages: list[Passage]) -> dict[str, list[float]]:
        digest_input = self.settings.openai_embedding_model + "\n" + "\n".join(f"{p.id}:{p.title}:{p.section}:{p.text}" for p in passages)
        digest = hashlib.sha256(digest_input.encode()).hexdigest()[:20]
        cache = Path(self.settings.embedding_cache_dir) / f"{digest}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.events.append({"operation": "embed_corpus_cache_hit", "embedding_model": self.settings.openai_embedding_model, "items": len(payload)})
            return payload
        texts = [f"{p.title}\n{p.section}\n{p.text}" for p in passages]
        vectors = self.embed(texts, "embed_corpus")
        result = {passage.id: vector for passage, vector in zip(passages, vectors)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result), encoding="utf-8")
        return result

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query], "embed_query")[0]

    def rerank(self, query: str, requirements: list[Requirement], hits: list[SearchHit], passages: dict[str, Passage]) -> list[SearchHit]:
        rerank_hits = list(hits[: self.settings.model_rerank_top_k])
        rerank_ids = {hit.passage_id for hit in rerank_hits}
        for requirement in requirements:
            routing_terms = {
                term
                for phrase in requirement.routing_keywords
                for term in tokens(phrase)
            }
            if not routing_terms:
                continue
            rescue_candidates = sorted(
                (
                    (
                        len(
                            routing_terms
                            & set(
                                tokens(
                                    f"{passages[hit.passage_id].section} "
                                    f"{passages[hit.passage_id].text}"
                                )
                            )
                        )
                        / len(routing_terms),
                        index,
                        hit,
                    )
                    for index, hit in enumerate(hits)
                    if hit.passage_id not in rerank_ids
                ),
                key=lambda item: (-item[0], item[1]),
            )
            for score, _, hit in rescue_candidates[
                : self.settings.model_rerank_rescue_per_requirement
            ]:
                if score <= 0:
                    continue
                rerank_hits.append(hit)
                rerank_ids.add(hit.passage_id)
        needed = set(keywords(query, limit=20))

        def candidate_text(passage: Passage) -> str:
            span, _, _ = best_sentence_window(passage.text, needed, max_sentences=2)
            return span[: self.settings.model_rerank_max_chars]

        candidates = [
            {
                "passage_id": hit.passage_id,
                "title": passages[hit.passage_id].title,
                "section": passages[hit.passage_id].section,
                "source_type": passages[hit.passage_id].metadata.get("source_type"),
                "source_path": passages[hit.passage_id].metadata.get("source_path"),
                "text": candidate_text(passages[hit.passage_id]),
                "hybrid_score": hit.final_score,
            }
            for hit in rerank_hits
        ]
        parsed = self._parse(
            "rerank",
            RerankResult,
            "Score every candidate for whether it contains evidence for the original query and "
            "listed requirements. Treat figure/table captions as first-class evidence when the "
            "question asks about results, metrics, datasets, benchmarks, or compared systems. Use "
            "For yes/no questions, passages describing the underlying activity can be relevant "
            "even when the disputed feature is absent. Distinguish the authors' own method from "
            "related work they merely cite. Use only supplied candidate text. Return every "
            "passage_id exactly once. Set rationale to "
            "null; only scores and requirement IDs are needed.",
            json.dumps(
                {
                    "query": query,
                    "requirements": [
                        {
                            "id": r.id,
                            "description": r.description,
                            "routing_keywords": r.routing_keywords,
                        }
                        for r in requirements
                    ],
                    "candidates": candidates,
                }
            ),
        )
        assert isinstance(parsed, RerankResult)
        scores = {item.passage_id: item.relevance for item in parsed.candidates}
        for hit in rerank_hits:
            if hit.passage_id in scores:
                hit.final_score = 0.35 * hit.final_score + 0.65 * scores[hit.passage_id]
                if "model_rerank" not in hit.retrievers:
                    hit.retrievers.append("model_rerank")
        return sorted(hits, key=lambda hit: hit.final_score, reverse=True)

    def assess(
        self,
        question: str,
        requirements: list[Requirement],
        evidence: list[Evidence],
    ) -> dict[str, EvidenceVerdict]:
        parsed = self._parse(
            "assess_evidence",
            EvidenceAssessment,
            "Judge whether each span helps answer the original question and its assigned "
            "requirement. Use 'partial' when a span contributes useful facts but cannot satisfy "
            "the requirement by itself. A span need not contain the entire answer to be partial. "
            "For a yes/no question, answer_value carries the conclusion: true for Yes and false "
            "for No. Treat both supports and challenges verdicts with a non-null answer_value as "
            "potentially usable polarity evidence. A "
            "negative answer does not require the literal word 'No': a specific description of "
            "the relevant procedure can support No when it omits a mechanism that would normally "
            "be stated, such as a platform, dataset, or model. For boolean questions, set "
            "answer_value to true or false when the verdict is supports or partial, and null when "
            "it is insufficient. Distinguish the authors' own method and results from related work "
            "they only cite; cited prior work does not answer what 'they' did. Return every "
            "evidence_id exactly "
            "once with its unchanged "
            "requirement_id. Never use one verdict to cover another requirement.",
            json.dumps(
                {
                    "question": question,
                    "expected_answer_type": infer_answer_type(question),
                    "requirements": [
                        {"id": r.id, "description": r.description} for r in requirements
                    ],
                    "evidence": [
                        {
                            "id": e.id,
                            "requirement_id": e.requirement_id,
                            "title": e.title,
                            "section": e.section,
                            "text": e.text,
                        }
                        for e in evidence
                    ],
                }
            ),
        )
        assert isinstance(parsed, EvidenceAssessment)
        return {verdict.evidence_id: verdict for verdict in parsed.verdicts}

    def synthesize(self, question: str, requirements: list[Requirement], evidence: list[Evidence]) -> tuple[str, list[Claim]]:
        expected_answer_type = infer_answer_type(question)
        forced_boolean_contract = (
            expected_answer_type == "boolean" and self.settings.answer_policy == "benchmark"
        )
        schema: type[BaseModel] = (
            GroundedBooleanAnswer if forced_boolean_contract else GroundedAnswer
        )
        answer_contract = (
            "This is a forced-choice boolean question: choose Yes or No; do not return "
            "Unanswerable. A specific description of the relevant procedure can ground No when "
            "the asked mechanism is absent. Reconcile every polarity-tagged supporting span. For "
            "universal or superlative claims such as 'all', 'best', or 'among all', one valid "
            "counterexample makes the answer No. "
            if forced_boolean_contract
            else (
                "For a boolean question, set boolean_answer when the evidence establishes Yes or "
                "No; otherwise mark the answer unanswerable. "
                if expected_answer_type == "boolean"
                else ""
            )
        )
        parsed = self._parse(
            "synthesize",
            schema,
            "Answer the question directly and concisely using only supplied evidence. "
            + answer_contract
            + "For extractive "
            "questions, return only the requested names, values, datasets, methods, or short span "
            "when possible—not an explanatory sentence. Mark unanswerable only when the evidence "
            "cannot support an answer. Claims may cite only evidence marked supports=true; partial, "
            "insufficient, and challenging spans are context rather than grounding. Distinguish the "
            "authors' own method from prior work they cite. Every claim must cite exact evidence "
            "IDs. Do not add citation markers or a references section yourself.",
            json.dumps(
                {
                    "question": question,
                    "expected_answer_type": expected_answer_type,
                    "answer_policy": self.settings.answer_policy,
                    "requirements": [
                        {"id": r.id, "description": r.description} for r in requirements
                    ],
                    "evidence": [
                        {
                            "id": e.id,
                            "requirement_id": e.requirement_id,
                            "title": e.title,
                            "section": e.section,
                            "relation": e.relation,
                            "supports": e.supports,
                            "answer_value": e.answer_value,
                            "text": e.text,
                        }
                        for e in evidence
                    ],
                }
            ),
        )
        assert isinstance(parsed, (GroundedAnswer, GroundedBooleanAnswer))
        evidence_by_id = {item.id: item for item in evidence}
        valid_requirements = {item.id for item in requirements}
        claims: list[Claim] = []
        cited_ids: list[str] = []
        parsed_boolean_value = getattr(parsed, "boolean_answer", None)
        for item in parsed.claims:
            evidence_ids = [
                evidence_id
                for evidence_id in item.evidence_ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].requirement_id == item.requirement_id
                and evidence_by_id[evidence_id].supports
                and (
                    expected_answer_type != "boolean"
                    or evidence_by_id[evidence_id].answer_value == parsed_boolean_value
                )
            ]
            if not evidence_ids or item.requirement_id not in valid_requirements:
                continue
            cited_ids.extend(evidence_id for evidence_id in evidence_ids if evidence_id not in cited_ids)
            claims.append(Claim(f"claim_{len(claims)+1}", item.text, evidence_ids, item.requirement_id, item.confidence))
        if getattr(parsed, "unanswerable", False) or not claims:
            return "Unanswerable", claims
        if expected_answer_type == "boolean" and parsed.boolean_answer is not None:
            direct_answer = "Yes" if parsed.boolean_answer else "No"
        else:
            direct_answer = parsed.answer.strip()
        markers = "".join(f" [{index}]" for index, _ in enumerate(cited_ids, 1))
        return direct_answer.rstrip() + markers, claims
