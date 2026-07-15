from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable

from .config import Settings
from .models import Passage, SearchHit
from .text import char_ngrams, cosine, tokens


class HybridRetriever:
    """BM25-style lexical + dependency-free character-vector retrieval."""

    def __init__(
        self,
        passages: list[Passage],
        settings: Settings,
        dense_vectors: dict[str, list[float]] | None = None,
        query_embedder: Callable[[str], list[float]] | None = None,
    ):
        self.passages = passages
        self.settings = settings
        self.by_id = {p.id: p for p in passages}
        self.terms = {p.id: tokens(f"{p.title} {p.section} {p.text}") for p in passages}
        self.doc_freq = Counter(term for terms in self.terms.values() for term in set(terms))
        self.avg_len = sum(map(len, self.terms.values())) / max(1, len(self.terms))
        self.dense_vectors = dense_vectors
        self.query_embedder = query_embedder
        needs_dense = settings.retrieval_mode in {"embeddings", "hybrid"}
        self.vectors = (
            None
            if dense_vectors or not needs_dense
            else {p.id: char_ngrams(f"{p.title} {p.section} {p.text}") for p in passages}
        )

    @staticmethod
    def _vector_cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _bm25(self, query: str, passage_id: str) -> float:
        terms = self.terms[passage_id]
        counts = Counter(terms)
        score = 0.0
        for term in tokens(query):
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (len(self.passages) - df + 0.5) / (df + 0.5))
            tf = counts.get(term, 0)
            score += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * len(terms) / max(1, self.avg_len)))
        return score

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        k = top_k or self.settings.rerank_top_k
        use_lexical = self.settings.retrieval_mode in {"bm25", "hybrid"}
        use_dense = self.settings.retrieval_mode in {"embeddings", "hybrid"}
        lexical = (
            sorted(
                ((p.id, self._bm25(query, p.id)) for p in self.passages),
                key=lambda x: x[1],
                reverse=True,
            )
            if use_lexical
            else []
        )
        if not use_dense:
            dense: list[tuple[str, float]] = []
        elif self.dense_vectors is not None and self.query_embedder is not None:
            query_vector = self.query_embedder(query)
            dense = sorted(
                (
                    (p.id, self._vector_cosine(query_vector, self.dense_vectors[p.id]))
                    for p in self.passages
                ),
                key=lambda x: x[1],
                reverse=True,
            )
        else:
            query_vector = char_ngrams(query)
            assert self.vectors is not None
            dense = sorted(
                ((p.id, cosine(query_vector, self.vectors[p.id])) for p in self.passages),
                key=lambda x: x[1],
                reverse=True,
            )
        lexical_rank = {pid: (rank, score) for rank, (pid, score) in enumerate(lexical[: self.settings.search_top_k], 1)}
        dense_rank = {pid: (rank, score) for rank, (pid, score) in enumerate(dense[: self.settings.search_top_k], 1)}
        hits: dict[str, SearchHit] = {}
        for pid in set(lexical_rank) | set(dense_rank):
            p = self.by_id[pid]
            hit = SearchHit(pid, p.document_id)
            if pid in lexical_rank:
                hit.bm25_rank, hit.lexical_score = lexical_rank[pid]
                hit.rrf_score += self.settings.lexical_weight / (self.settings.rrf_k + hit.bm25_rank)
                hit.retrievers.append("bm25")
            if pid in dense_rank:
                hit.dense_rank, hit.dense_score = dense_rank[pid]
                hit.rrf_score += self.settings.dense_weight / (self.settings.rrf_k + hit.dense_rank)
                hit.retrievers.append("dense")
            hit.metadata_score = 1.0 if any(t in tokens(p.title) for t in tokens(query)) else 0.0
            hit.final_score = hit.rrf_score + self.settings.metadata_weight * hit.metadata_score
            hits[pid] = hit
        ranked = sorted(hits.values(), key=lambda h: h.final_score, reverse=True)
        selected: list[SearchHit] = []
        seen_docs: defaultdict[str, int] = defaultdict(int)
        for hit in ranked:
            hit.redundancy_penalty = self.settings.redundancy_weight * max(0, seen_docs[hit.document_id] - 1)
            hit.final_score -= hit.redundancy_penalty
            selected.append(hit)
            seen_docs[hit.document_id] += 1
        return sorted(selected, key=lambda h: h.final_score, reverse=True)[:k]
