from __future__ import annotations

from collections import defaultdict

from .models import Claim, Evidence, Requirement
from .planner import infer_answer_type
from .text import keywords, split_sentences, tokens


BOOLEAN_ROUTING_WORDS = {
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "is",
    "reach",
    "they",
    "use",
    "used",
    "using",
    "was",
    "were",
    "will",
    "would",
}
NEGATION_TERMS = {"cannot", "no", "not", "never", "neither", "without"}


class ExtractiveSynthesizer:
    """Grounded no-LLM synthesis: selects evidence sentences and binds citations."""

    def synthesize(
        self,
        question: str,
        requirements: list[Requirement],
        evidence: list[Evidence],
        answer_policy: str = "grounded",
    ) -> tuple[str, list[Claim]]:
        by_requirement: defaultdict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            by_requirement[item.requirement_id].append(item)
        lines: list[str] = []
        claims: list[Claim] = []
        if not evidence:
            return "Unanswerable", []
        if infer_answer_type(question) == "boolean":
            best = max(evidence, key=lambda item: item.score)
            evidence_terms = set(tokens(best.text))
            focus = set(keywords(question, limit=20)) - BOOLEAN_ROUTING_WORDS
            supported = bool(focus) and focus <= evidence_terms
            if not supported and answer_policy == "grounded":
                return "Unanswerable", []
            answer = "No" if evidence_terms & NEGATION_TERMS or not supported else "Yes"
            claim = Claim("claim_1", answer, [best.id], best.requirement_id, 0.65)
            return f"{answer} [1]", [claim]
        for requirement in requirements:
            candidates = by_requirement[requirement.id]
            if not candidates:
                lines.append(f"Evidence is insufficient for: {requirement.description}")
                continue
            best = max(candidates, key=lambda item: item.score)
            sentences = split_sentences(best.text)
            sentence = max(sentences or [best.text], key=lambda s: len(set(tokens(s)) & set(requirement.keywords)))
            marker = len(claims) + 1
            lines.append(f"{sentence} [{marker}]")
            claims.append(Claim(f"claim_{len(claims)+1}", sentence, [best.id], requirement.id, min(0.99, 0.55 + best.score * 10)))
        return "\n\n".join(lines), claims
