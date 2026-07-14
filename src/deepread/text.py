from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "what", "when",
    "where", "which", "who", "why", "with", "would", "should", "could", "can", "than",
}


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def keywords(text: str, limit: int = 10) -> list[str]:
    counts = Counter(t for t in tokens(text) if t not in STOPWORDS and len(t) > 2)
    return [word for word, _ in counts.most_common(limit)]


def requirement_terms(description: str, phrases: list[str]) -> set[str]:
    """Normalize model phrases into the same token space used by the reader."""
    phrase_terms = {term for phrase in phrases for term in tokens(phrase)}
    description_terms = set(keywords(description, limit=20))
    return {term for term in phrase_terms | description_terms if term not in STOPWORDS and len(term) > 2}


def token_count(text: str) -> int:
    return max(1, math.ceil(len(tokens(text)) * 1.3))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def best_sentence_window(text: str, needed: set[str], max_sentences: int = 3) -> tuple[str, int, int]:
    """Return the shortest contiguous sentence window with the best term coverage."""
    spans = [(match.start(), match.end(), match.group().strip()) for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", text) if match.group().strip()]
    if not spans:
        return text, 0, len(text)
    best: tuple[int, int, int, int] | None = None
    for start_index in range(len(spans)):
        for size in range(1, min(max_sentences, len(spans) - start_index) + 1):
            end_index = start_index + size - 1
            start, end = spans[start_index][0], spans[end_index][1]
            coverage = len(needed & set(tokens(text[start:end])))
            candidate = (coverage, -size, -start, end)
            if best is None or candidate > best:
                best = candidate
                best_start, best_end = start, end
    if best is None or best[0] == 0:
        return text, 0, len(text)
    return text[best_start:best_end].strip(), best_start, best_end


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    common = set(left) & set(right)
    dot = sum(left[k] * right[k] for k in common)
    a = math.sqrt(sum(v * v for v in left.values()))
    b = math.sqrt(sum(v * v for v in right.values()))
    return dot / (a * b) if a and b else 0.0


def char_ngrams(text: str, n: int = 3) -> dict[str, float]:
    compact = " ".join(tokens(text))
    counts = Counter(compact[i : i + n] for i in range(max(0, len(compact) - n + 1)))
    return {key: float(value) for key, value in counts.items()}
