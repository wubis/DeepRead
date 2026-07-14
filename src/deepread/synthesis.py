from __future__ import annotations

from collections import defaultdict

from .models import Claim, Evidence, Requirement
from .text import split_sentences, tokens


class ExtractiveSynthesizer:
    """Grounded no-LLM synthesis: selects evidence sentences and binds citations."""

    def synthesize(self, requirements: list[Requirement], evidence: list[Evidence]) -> tuple[str, list[Claim]]:
        by_requirement: defaultdict[str, list[Evidence]] = defaultdict(list)
        for item in evidence:
            by_requirement[item.requirement_id].append(item)
        lines: list[str] = []
        claims: list[Claim] = []
        citation_number = {item.id: index for index, item in enumerate(evidence, 1)}
        for requirement in requirements:
            candidates = by_requirement[requirement.id]
            if not candidates:
                lines.append(f"Evidence is insufficient for: {requirement.description}")
                continue
            best = max(candidates, key=lambda item: item.score)
            sentences = split_sentences(best.text)
            sentence = max(sentences or [best.text], key=lambda s: len(set(tokens(s)) & set(requirement.keywords)))
            marker = citation_number[best.id]
            lines.append(f"{sentence} [{marker}]")
            claims.append(Claim(f"claim_{len(claims)+1}", sentence, [best.id], requirement.id, min(0.99, 0.55 + best.score * 10)))
        return "\n\n".join(lines), claims

