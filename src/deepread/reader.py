from __future__ import annotations

import math

from .models import Document, Passage, ReadDecision, ReadLevel, Requirement, SearchHit
from .text import best_sentence_window, requirement_terms, token_count, tokens


class HierarchicalReader:
    def __init__(
        self,
        documents: list[Document],
        passages: list[Passage],
        evidence_window_sentences: int = 3,
    ):
        self.documents = {d.id: d for d in documents}
        self.passages = {p.id: p for p in passages}
        self.evidence_window_sentences = evidence_window_sentences
        self.by_document: dict[str, list[Passage]] = {}
        for passage in passages:
            self.by_document.setdefault(passage.document_id, []).append(passage)

    def views(self, passage_id: str) -> dict[ReadLevel, str]:
        p = self.passages[passage_id]
        d = self.documents[p.document_id]
        section_text = " ".join(d.sections.get(p.section, []))
        full = " ".join(text for section in d.sections.values() for text in section)
        return {ReadLevel.TITLE: d.title, ReadLevel.SUMMARY: d.summary, ReadLevel.SECTION: section_text, ReadLevel.PASSAGE: p.text, ReadLevel.DOCUMENT: full}

    def choose(self, hit: SearchHit, requirement: Requirement, remaining_tokens: int) -> tuple[ReadDecision, str]:
        p = self.passages[hit.passage_id]
        views = self.views(hit.passage_id)
        needed = requirement_terms(requirement.description, requirement.keywords)
        routing_needed = {
            term
            for phrase in requirement.routing_keywords
            for term in tokens(phrase)
        }
        best: tuple[float, ReadLevel, str, float, int] | None = None
        # Titles are useful routing signals but are never sufficient factual evidence.
        # Compare substantive views, preferring coverage before cost efficiency.
        for level in (ReadLevel.SUMMARY, ReadLevel.SECTION, ReadLevel.PASSAGE, ReadLevel.DOCUMENT):
            text = views[level]
            cost = token_count(text)
            if level == ReadLevel.SUMMARY:
                # A summary can route the reader, but producing a verifiable citation
                # also requires opening the underlying candidate passage.
                cost += token_count(p.text)
            routing_terms = set(tokens(text))
            if level in (ReadLevel.SECTION, ReadLevel.PASSAGE, ReadLevel.DOCUMENT):
                routing_terms |= set(tokens(p.section))
            requirement_coverage = len(needed & routing_terms) / max(1, len(needed))
            routing_coverage = (
                len(routing_needed & routing_terms) / len(routing_needed)
                if routing_needed
                else requirement_coverage
            )
            overlap = (
                0.70 * routing_coverage + 0.30 * requirement_coverage
                if routing_needed
                else requirement_coverage
            )
            depth_bonus = {ReadLevel.TITLE: .70, ReadLevel.SUMMARY: .85, ReadLevel.SECTION: .95, ReadLevel.PASSAGE: 1.0, ReadLevel.DOCUMENT: .70}[level]
            gain = overlap * max(hit.final_score, 0.001) * depth_bonus
            # Sublinear cost discount prevents tiny headings from beating a much more
            # informative passage solely because they contain very few tokens.
            utility = gain / math.sqrt(max(1, cost))
            if cost <= remaining_tokens and (best is None or utility > best[0]):
                best = (utility, level, text, gain, cost)
        if best is None:
            return ReadDecision(hit.passage_id, ReadLevel.TITLE, requirement.id, 0, 0, 0, False, "budget_exhausted"), ""
        _, level, text, gain, cost = best
        utility = gain / math.sqrt(max(1, cost))
        selected = gain > 0
        reason = "highest_expected_coverage_gain_per_token" if selected else "no_requirement_overlap"
        return ReadDecision(hit.passage_id, level, requirement.id, gain, cost, utility, selected, reason), text

    def choose_passage(
        self,
        hit: SearchHit,
        requirement: Requirement,
        remaining_tokens: int,
    ) -> ReadDecision:
        """Score a conventional flat read while charging the full passage cost."""
        passage = self.passages[hit.passage_id]
        cost = token_count(passage.text)
        if cost > remaining_tokens:
            return ReadDecision(
                hit.passage_id,
                ReadLevel.PASSAGE,
                requirement.id,
                0,
                cost,
                0,
                False,
                "budget_exhausted",
            )
        needed = requirement_terms(requirement.description, requirement.keywords)
        passage_terms = set(tokens(f"{passage.section} {passage.text}"))
        overlap = len(needed & passage_terms) / max(1, len(needed))
        gain = overlap * max(hit.final_score, 0.001)
        utility = gain / max(1, cost)
        return ReadDecision(
            hit.passage_id,
            ReadLevel.PASSAGE,
            requirement.id,
            gain,
            cost,
            utility,
            gain > 0,
            "flat_top_k_read" if gain > 0 else "flat_top_k_read_no_requirement_overlap",
        )

    def evidence_span(self, passage_id: str, level: ReadLevel, requirement: Requirement) -> tuple[Passage, str, int, int]:
        """Refine a broad read into one requirement-specific passage and sentence window."""
        source = self.passages[passage_id]
        if level == ReadLevel.DOCUMENT:
            candidates = self.by_document[source.document_id]
        elif level == ReadLevel.SECTION:
            candidates = [passage for passage in self.by_document[source.document_id] if passage.section == source.section]
        else:
            candidates = [source]
        needed = requirement_terms(requirement.description, requirement.keywords)
        routing_needed = {
            term
            for phrase in requirement.routing_keywords
            for term in tokens(phrase)
        }

        def passage_score(passage: Passage) -> tuple[float, float, int]:
            text_terms = set(tokens(f"{passage.section} {passage.text}"))
            coverage = len(needed & text_terms) / max(1, len(needed))
            routing_coverage = (
                len(routing_needed & text_terms) / len(routing_needed)
                if routing_needed
                else coverage
            )
            weighted_coverage = (
                0.70 * routing_coverage + 0.30 * coverage
                if routing_needed
                else coverage
            )
            return weighted_coverage, coverage, -passage.ordinal

        selected = max(candidates, key=passage_score)
        span, start, end = best_sentence_window(
            selected.text,
            needed,
            max_sentences=self.evidence_window_sentences,
        )
        return selected, span, start, end
