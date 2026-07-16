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
from .planner import infer_answer_type
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
        self.reader = HierarchicalReader(
            self.documents,
            self.passages,
            self.settings.evidence_window_sentences,
        )
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
        trace.evidence_tokens += citation_tokens
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
        previous_hit_ids: set[str] = set()
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
            current_hit_ids = {hit.passage_id for hit in hits}
            new_hit_count = len(current_hit_ids - previous_hit_ids)
            overlap = (
                len(current_hit_ids & previous_hit_ids) / max(1, len(current_hit_ids | previous_hit_ids))
                if previous_hit_ids
                else 0.0
            )
            trace.searches.append(
                {
                    "round": round_number,
                    "query": query,
                    "hit_count": len(hits),
                    "new_hit_count": new_hit_count,
                    "candidate_overlap": overlap,
                }
            )
            if round_number > 1 and not current_hit_ids - previous_hit_ids:
                trace.stop_reason = "no_new_retrieval_candidates"
                break
            previous_hit_ids = current_hit_ids
            if self.openai is not None and self.settings.enable_model_rerank:
                hits = self.openai.rerank(query, requirements, hits, self.retriever.by_id)
                self._capture_api_events(trace)
            trace.ranking.extend(hits)
            evidence_count_before = len(trace.evidence)
            covered_before = set(covered)
            supported_count_before = sum(item.supports for item in trace.evidence)
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
                    selected_options: list[tuple[Any, ReadDecision, str]] = []
                    selected_passage_ids: set[str] = set()
                    remaining = self.settings.max_evidence_tokens - trace.read_tokens
                    for option in options:
                        _, decision, evidence_passage_id = option
                        if evidence_passage_id in selected_passage_ids:
                            decision.selected = False
                            decision.reason = "duplicate_requirement_span"
                        elif len(selected_options) >= self.settings.evidence_candidates_per_requirement:
                            decision.selected = False
                            decision.reason = "lower_utility_than_selected"
                        elif decision.token_cost > remaining:
                            decision.selected = False
                            decision.reason = "budget_exhausted"
                        else:
                            selected_options.append(option)
                            selected_passage_ids.add(evidence_passage_id)
                            remaining -= decision.token_cost
                    trace.reads.extend(option[1] for option in options)
                    for hit, decision, evidence_passage_id in selected_options:
                        seen_reads.add((hit.passage_id, requirement.id))
                        seen_reads.add((evidence_passage_id, requirement.id))
                        trace.tokens_used += decision.token_cost
                        trace.read_tokens += decision.token_cost
                        self._add_evidence(trace, hit, decision, requirement)
            new_evidence = trace.evidence[evidence_count_before:]
            if (
                self.openai is not None
                and self.settings.enable_evidence_assessment
                and new_evidence
            ):
                verdicts = self.openai.assess(question, requirements, new_evidence)
                self._capture_api_events(trace)
                for evidence in new_evidence:
                    verdict = verdicts.get(evidence.id)
                    verdict_matches = bool(verdict and verdict.requirement_id == evidence.requirement_id)
                    evidence.relation = verdict.verdict if verdict_matches else "insufficient"
                    evidence.answer_value = verdict.answer_value if verdict_matches else None
                    evidence.assessment_confidence = (
                        verdict.confidence if verdict_matches else 0.0
                    )
                    if infer_answer_type(question) == "boolean":
                        evidence.supports = bool(
                            verdict_matches
                            and verdict.verdict in {"supports", "challenges"}
                            and verdict.answer_value is not None
                            and verdict.confidence >= self.settings.evidence_support_threshold
                        )
                    else:
                        evidence.supports = bool(
                            verdict_matches
                            and verdict.verdict == "supports"
                            and verdict.confidence >= self.settings.evidence_support_threshold
                        )
                if infer_answer_type(question) == "boolean":
                    for requirement in requirements:
                        partials = [
                            evidence
                            for evidence in trace.evidence
                            if evidence.requirement_id == requirement.id
                            and evidence.relation == "partial"
                            and evidence.answer_value is not None
                            and evidence.assessment_confidence
                            >= self.settings.evidence_support_threshold
                        ]
                        for answer_value in (False, True):
                            consistent = [
                                evidence
                                for evidence in partials
                                if evidence.answer_value is answer_value
                            ]
                            detailed = [
                                evidence
                                for evidence in consistent
                                if evidence.section.strip().lower()
                                not in {"abstract", "introduction"}
                            ]
                            if len(consistent) >= 2 and detailed:
                                for evidence in detailed:
                                    evidence.supports = True
                                    evidence.relation = "supports_aggregate"
            covered = {
                evidence.requirement_id for evidence in trace.evidence if evidence.supports
            }
            trace.coverage = len(covered) / max(1, len(requirements))
            if trace.coverage >= self.settings.target_coverage:
                trace.stop_reason = "target_coverage_reached"
                break
            if trace.read_tokens >= self.settings.max_evidence_tokens:
                trace.stop_reason = "evidence_token_budget_exhausted"
                break
            supported_count = sum(item.supports for item in trace.evidence)
            if (
                round_number > 1
                and supported_count == supported_count_before
                and covered == covered_before
            ):
                trace.stop_reason = "no_supported_evidence_progress"
                break
            if (
                round_number < max_rounds
                and len(trace.evidence) == evidence_count_before
                and covered == covered_before
            ):
                trace.stop_reason = "no_evidence_progress"
                break
            missing = [r for r in requirements if r.id not in covered]
            query = " ".join(
                word
                for requirement in missing
                for word in (requirement.routing_keywords + requirement.keywords)
            ) or question
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
            answer_text, claims = self.synthesizer.synthesize(
                question,
                requirements,
                trace.evidence,
                self.settings.answer_policy,
            )
        trace.claims = claims
        for claim in claims:
            self.memory.add_claim(claim)
        cited_evidence_ids = {
            evidence_id for claim in claims for evidence_id in claim.evidence_ids
        }
        citations = [
            evidence for evidence in trace.evidence if evidence.id in cited_evidence_ids
        ]
        trace.citation_tokens = sum(evidence.citation_tokens for evidence in citations)
        answer = Answer(question, answer_text, citations, trace.coverage, trace.stop_reason, trace)
        if trace_path:
            target = Path(trace_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(answer.to_dict(), indent=2), encoding="utf-8")
        return answer

    def corpus_stats(self) -> dict[str, int]:
        return {"documents": len(self.documents), "passages": len(self.passages)}
