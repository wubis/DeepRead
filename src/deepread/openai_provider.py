from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import Settings
from .models import Claim, Evidence, Passage, Requirement, SearchHit, TaskNode


class PlannedRequirement(BaseModel):
    description: str
    keywords: list[str] = Field(min_length=1, max_length=12)


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
    verdict: Literal["supports", "challenges", "insufficient"]
    confidence: float = Field(ge=0, le=1)


class EvidenceAssessment(BaseModel):
    verdicts: list[EvidenceVerdict]


class GroundedClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(min_length=1)
    requirement_id: str
    confidence: float = Field(ge=0, le=1)


class GroundedAnswer(BaseModel):
    claims: list[GroundedClaim]
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
            "You plan evidence retrieval. Decompose the question into independent answerable tasks and concrete evidence requirements. Requirements must describe facts that sources can support. Do not answer the question.",
            f"Question: {question}\nHard limits: at most {max_tasks} tasks and at most {max_requirements} total requirements. Prefer the smallest checklist that fully captures the question.",
        )
        assert isinstance(parsed, PlannedQuery)
        tasks: list[TaskNode] = []
        remaining = max_requirements
        for task_index, task in enumerate(parsed.tasks[:max_tasks], 1):
            if remaining <= 0:
                break
            task_id = f"task_{task_index}"
            requirements = [Requirement(f"req_{task_index}_{req_index}", task_id, requirement.description, requirement.keywords[:12]) for req_index, requirement in enumerate(task.requirements[:remaining], 1)]
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
        candidates = [{"passage_id": hit.passage_id, "title": passages[hit.passage_id].title, "section": passages[hit.passage_id].section, "text": passages[hit.passage_id].text[:1200], "hybrid_score": hit.final_score} for hit in hits]
        parsed = self._parse(
            "rerank",
            RerankResult,
            "Score every candidate for whether it contains evidence for the listed requirements. Use only the supplied candidate text. Return every passage_id exactly once. Set rationale to null; only scores and requirement IDs are needed.",
            json.dumps({"query": query, "requirements": [{"id": r.id, "description": r.description} for r in requirements], "candidates": candidates}),
        )
        assert isinstance(parsed, RerankResult)
        scores = {item.passage_id: item.relevance for item in parsed.candidates}
        for hit in hits:
            if hit.passage_id in scores:
                hit.final_score = 0.35 * hit.final_score + 0.65 * scores[hit.passage_id]
                if "model_rerank" not in hit.retrievers:
                    hit.retrievers.append("model_rerank")
        return sorted(hits, key=lambda hit: hit.final_score, reverse=True)

    def assess(self, requirements: list[Requirement], evidence: list[Evidence]) -> dict[str, EvidenceVerdict]:
        parsed = self._parse(
            "assess_evidence",
            EvidenceAssessment,
            "Judge each evidence span only against its assigned requirement_id. Be conservative and return every evidence_id exactly once with that same requirement_id. Never use one verdict to cover another requirement.",
            json.dumps({"requirements": [{"id": r.id, "description": r.description} for r in requirements], "evidence": [{"id": e.id, "requirement_id": e.requirement_id, "text": e.text} for e in evidence]}),
        )
        assert isinstance(parsed, EvidenceAssessment)
        return {verdict.evidence_id: verdict for verdict in parsed.verdicts}

    def synthesize(self, question: str, requirements: list[Requirement], evidence: list[Evidence]) -> tuple[str, list[Claim]]:
        parsed = self._parse(
            "synthesize",
            GroundedAnswer,
            "Write concise answer claims using only supplied evidence. Every claim must cite one or more exact evidence IDs. Do not add unsupported facts, citation markers, or a references section. Preserve uncertainty and source boundaries.",
            json.dumps({"question": question, "requirements": [{"id": r.id, "description": r.description} for r in requirements], "evidence": [{"id": e.id, "title": e.title, "section": e.section, "relation": e.relation, "text": e.text} for e in evidence]}),
        )
        assert isinstance(parsed, GroundedAnswer)
        evidence_by_id = {item.id: item for item in evidence}
        valid_requirements = {item.id for item in requirements}
        citation_order = {item.id: index for index, item in enumerate(evidence, 1)}
        claims: list[Claim] = []
        paragraphs: list[str] = []
        for item in parsed.claims:
            evidence_ids = [evidence_id for evidence_id in item.evidence_ids if evidence_id in evidence_by_id and evidence_by_id[evidence_id].requirement_id == item.requirement_id]
            if not evidence_ids or item.requirement_id not in valid_requirements:
                continue
            markers = "".join(f" [{citation_order[evidence_id]}]" for evidence_id in evidence_ids)
            paragraphs.append(item.text.rstrip() + markers)
            claims.append(Claim(f"claim_{len(claims)+1}", item.text, evidence_ids, item.requirement_id, item.confidence))
        for requirement_id in parsed.unsupported_requirement_ids:
            if requirement_id in valid_requirements:
                requirement = next(item for item in requirements if item.id == requirement_id)
                paragraphs.append(f"Evidence is insufficient for: {requirement.description}")
        return "\n\n".join(paragraphs), claims
