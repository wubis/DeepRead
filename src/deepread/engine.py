from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import Settings
from .corpus import load_corpus
from .memory import EvidenceMemory
from .models import Answer, CorpusBundle, Evidence, QueryTrace, ReadDecision, Requirement, SearchHit
from .planner import RuleBasedPlanner
from .reader import HierarchicalReader
from .retrieval import HybridRetriever
from .synthesis import ExtractiveSynthesizer
from .text import token_count


class EvidenceGraphEngine:
    def __init__(
        self,
        corpus_path: str | Path | CorpusBundle,
        settings: Settings | None = None,
        db_path: str | Path | None = None,
        openai_client: Any | None = None,
    ):
        self.settings = settings or Settings.from_env()
        if isinstance(corpus_path, CorpusBundle):
            self.documents = list(corpus_path.documents)
            self.passages = list(corpus_path.passages)
        else:
            self.documents, self.passages = load_corpus(corpus_path)
        if not self.passages:
            raise ValueError(f"No passages found in corpus {corpus_path}")
        requested = self.settings.provider.lower()
        if requested not in {"auto", "offline", "openai"}:
            raise ValueError("provider must be one of: auto, offline, openai")
        self.provider_name = "openai" if requested == "openai" or (requested == "auto" and bool(os.getenv("OPENAI_API_KEY"))) else "offline"
        self.openai = None
        if self.provider_name == "openai":
            if openai_client is None and not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is required when provider='openai'")
            from .openai_provider import OpenAIProvider

            self.openai = OpenAIProvider(self.settings, openai_client)
            if self.settings.retrieval_mode == "bm25":
                dense_vectors = None
                query_embedder = None
            else:
                dense_vectors = self.openai.embedding_index(self.passages)
                query_embedder = self.openai.embed_query
            self.retriever = HybridRetriever(self.passages, self.settings, dense_vectors, query_embedder)
            self.planner = None
        else:
            self.planner = RuleBasedPlanner(self.settings.max_tasks)
            self.retriever = HybridRetriever(self.passages, self.settings)
        self.reader = HierarchicalReader(self.documents, self.passages)
        self.memory = EvidenceMemory(db_path)
        self.synthesizer = ExtractiveSynthesizer()

    def _add_evidence(
        self,
        trace: QueryTrace,
        hit: SearchHit,
        decision: ReadDecision,
        requirement: Requirement,
    ) -> None:
        passage, span, char_start, char_end = self.reader.evidence_span(
            hit.passage_id,
            decision.level,
            requirement,
        )
        citation_tokens = token_count(span)
        evidence = Evidence(
            id=f"evidence_{len(trace.evidence)+1}",
            requirement_id=requirement.id,
            passage_id=passage.id,
            document_id=passage.document_id,
            title=passage.title,
            section=passage.section,
            text=span,
            score=hit.final_score,
            read_level=decision.level,
            token_cost=decision.token_cost,
            char_start=char_start,
            char_end=char_end,
            citation_tokens=citation_tokens,
            source_metadata=dict(passage.metadata),
        )
        trace.evidence.append(evidence)
        trace.citation_tokens += citation_tokens
        self.memory.add_evidence(evidence)

    def _read_flat(
        self,
        trace: QueryTrace,
        hits: list[SearchHit],
        requirements: list[Requirement],
        covered: set[str],
        seen_reads: set[tuple[str, str]],
    ) -> None:
        """Open the first k ranked passages and charge each full passage once."""
        for hit in hits[: self.settings.flat_top_k]:
            read_key = (hit.passage_id, "__flat__")
            if read_key in seen_reads:
                continue
            remaining = self.settings.max_evidence_tokens - trace.read_tokens
            candidates = [
                self.reader.choose_passage(hit, requirement, remaining)
                for requirement in requirements
                if requirement.id not in covered
            ]
            if not candidates:
                continue
            decision = max(candidates, key=lambda item: item.expected_gain)
            seen_reads.add(read_key)
            if decision.token_cost > remaining:
                decision.selected = False
                decision.reason = "budget_exhausted"
                trace.reads.append(decision)
                continue
            # A flat reader opens the passage even when it yields no usable evidence.
            decision.selected = True
            trace.reads.append(decision)
            trace.tokens_used += decision.token_cost
            trace.read_tokens += decision.token_cost
            if decision.expected_gain <= 0:
                continue
            requirement = next(item for item in requirements if item.id == decision.requirement_id)
            if any(
                item.passage_id == hit.passage_id and item.requirement_id == requirement.id
                for item in trace.evidence
            ):
                continue
            self._add_evidence(trace, hit, decision, requirement)

    def _capture_api_events(self, trace: QueryTrace) -> None:
        if self.openai is not None:
            events = self.openai.drain_events()
            trace.api_calls.extend(events)
            trace.api_input_tokens += sum(event.get("input_tokens", 0) or 0 for event in events)
            trace.api_output_tokens += sum(event.get("output_tokens", 0) or 0 for event in events)
            trace.api_total_tokens += sum(event.get("total_tokens", 0) or 0 for event in events)
            costs = [event["estimated_cost_usd"] for event in events if "estimated_cost_usd" in event]
            if costs:
                trace.estimated_api_cost_usd = (trace.estimated_api_cost_usd or 0.0) + sum(costs)

    def ask(self, question: str, trace_path: str | Path | None = None) -> Answer:
        trace = QueryTrace(question)
        trace.provider = self.provider_name
        if self.openai is not None:
            trace.tasks = self.openai.plan(question, self.settings.max_tasks, self.settings.max_requirements)
        else:
            assert self.planner is not None
            trace.tasks = self.planner.plan(question)
        self._capture_api_events(trace)
        requirements = [requirement for task in trace.tasks for requirement in task.requirements]
        covered: set[str] = set()
        seen_reads: set[tuple[str, str]] = set()
        query = question

        max_rounds = 1 if self.settings.supervisor_mode == "single_pass" else self.settings.max_search_rounds
        for round_number in range(1, max_rounds + 1):
            trace.rounds = round_number
            retrieval_k = (
                max(self.settings.rerank_top_k, self.settings.flat_top_k)
                if self.settings.reader_mode == "flat"
                else self.settings.rerank_top_k
            )
            hits = self.retriever.search(query, retrieval_k)
            self._capture_api_events(trace)
            if self.openai is not None and self.settings.enable_model_rerank:
                hits = self.openai.rerank(query, requirements, hits, self.retriever.by_id)
                self._capture_api_events(trace)
            trace.searches.append({"round": round_number, "query": query, "hit_count": len(hits)})
            trace.ranking.extend(hits)
            if self.settings.reader_mode == "flat":
                self._read_flat(trace, hits, requirements, covered, seen_reads)
            else:
                for requirement in requirements:
                    if requirement.id in covered:
                        continue
                    options: list[tuple[Any, ReadDecision, str]] = []
                    for hit in hits:
                        if (hit.passage_id, requirement.id) in seen_reads:
                            continue
                        remaining = self.settings.max_evidence_tokens - trace.read_tokens
                        decision, _ = self.reader.choose(hit, requirement, remaining)
                        if not decision.selected:
                            trace.reads.append(decision)
                            continue
                        passage, _, _, _ = self.reader.evidence_span(
                            hit.passage_id,
                            decision.level,
                            requirement,
                        )
                        if any(
                            item.passage_id == passage.id and item.requirement_id == requirement.id
                            for item in trace.evidence
                        ):
                            decision.selected = False
                            decision.reason = "duplicate_requirement_span"
                            trace.reads.append(decision)
                            continue
                        options.append((hit, decision, passage.id))
                    if not options:
                        continue
                    options.sort(key=lambda option: option[1].utility, reverse=True)
                    hit, decision, evidence_passage_id = options[0]
                    for _, rejected, _ in options[1:]:
                        rejected.selected = False
                        rejected.reason = "lower_utility_than_selected"
                    trace.reads.extend(option[1] for option in options)
                    seen_reads.add((hit.passage_id, requirement.id))
                    seen_reads.add((evidence_passage_id, requirement.id))
                    trace.tokens_used += decision.token_cost
                    trace.read_tokens += decision.token_cost
                    self._add_evidence(trace, hit, decision, requirement)
            if self.openai is not None and self.settings.enable_evidence_assessment and trace.evidence:
                verdicts = self.openai.assess(requirements, trace.evidence)
                self._capture_api_events(trace)
                covered = set()
                for evidence in trace.evidence:
                    verdict = verdicts.get(evidence.id)
                    verdict_matches = bool(verdict and verdict.requirement_id == evidence.requirement_id)
                    evidence.supports = bool(verdict_matches and verdict.verdict == "supports")
                    evidence.relation = verdict.verdict if verdict_matches else "insufficient"
                    if evidence.supports:
                        covered.add(evidence.requirement_id)
            else:
                covered.update(evidence.requirement_id for evidence in trace.evidence if evidence.supports)
            trace.coverage = len(covered) / max(1, len(requirements))
            if trace.coverage >= self.settings.target_coverage:
                trace.stop_reason = "target_coverage_reached"
                break
            if trace.read_tokens >= self.settings.max_evidence_tokens:
                trace.stop_reason = "evidence_token_budget_exhausted"
                break
            missing = [r for r in requirements if r.id not in covered]
            query = " ".join(word for requirement in missing for word in requirement.keywords) or question
        else:
            trace.stop_reason = (
                "single_pass_complete"
                if self.settings.supervisor_mode == "single_pass"
                else "max_search_rounds_reached"
            )

        if self.openai is not None:
            answer_text, claims = self.openai.synthesize(question, requirements, trace.evidence)
            self._capture_api_events(trace)
        else:
            answer_text, claims = self.synthesizer.synthesize(requirements, trace.evidence)
        trace.claims = claims
        for claim in claims:
            self.memory.add_claim(claim)
        answer = Answer(question, answer_text, trace.evidence, trace.coverage, trace.stop_reason, trace)
        if trace_path:
            target = Path(trace_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(answer.to_dict(), indent=2), encoding="utf-8")
        return answer

    def corpus_stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "passages": len(self.passages)}
