from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable

from .config import Settings
from .models import Passage, SearchHit
from .text import char_ngrams, cosine, keywords, tokens


STRUCTURED_QUERY_TERMS = {
    "benchmark",
    "benchmarks",
    "figure",
    "figures",
    "table",
    "tables",
}
FLOAT_REFERENCE = re.compile(
    r"\b(table|figure|fig\.?)[\s\u00a0]+([0-9]+[a-z]?)\b",
    re.I,
)


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
        self.content_terms = {
            p.id: set(tokens(f"{p.section} {p.text}")) for p in passages
        }
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

    def _passage_similarity(self, left_id: str, right_id: str) -> float:
        left = self.content_terms[left_id]
        right = self.content_terms[right_id]
        union = left | right
        return len(left & right) / len(union) if union else 0.0

    def _diversify(self, ranked: list[SearchHit], k: int) -> list[SearchHit]:
        """Apply bounded MMR-style passage diversity without penalizing document identity."""
        if not ranked or self.settings.redundancy_weight <= 0:
            return ranked[:k]
        base_scores = {hit.passage_id: hit.final_score for hit in ranked}
        score_values = list(base_scores.values())
        score_scale = max(
            max(score_values) - min(score_values),
            abs(max(score_values)) * 0.25,
            0.001,
        )
        remaining = list(ranked)
        selected: list[SearchHit] = []
        while remaining and len(selected) < k:
            best: SearchHit | None = None
            best_score = float("-inf")
            best_penalty = 0.0
            for hit in remaining:
                similarity = max(
                    (
                        self._passage_similarity(hit.passage_id, prior.passage_id)
                        for prior in selected
                    ),
                    default=0.0,
                )
                penalty = self.settings.redundancy_weight * similarity * score_scale
                diversified_score = base_scores[hit.passage_id] - penalty
                if diversified_score > best_score:
                    best = hit
                    best_score = diversified_score
                    best_penalty = penalty
            assert best is not None
            best.redundancy_penalty = best_penalty
            best.final_score = best_score
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _float_labels(text: str) -> set[str]:
        labels = set()
        for match in FLOAT_REFERENCE.finditer(text):
            kind = "figure" if match.group(1).lower().startswith("fig") else "table"
            labels.add(f"{kind}:{match.group(2).lower()}")
        return labels

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        k = top_k or self.settings.rerank_top_k
        use_lexical = self.settings.retrieval_mode in {"bm25", "hybrid"}
        use_dense = self.settings.retrieval_mode in {"embeddings", "hybrid"}
        query_terms = set(tokens(query))
        use_structure = bool(query_terms & STRUCTURED_QUERY_TERMS)
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
        structured: list[tuple[str, float]] = []
        if use_structure:
            floats_by_label: defaultdict[str, list[str]] = defaultdict(list)
            for passage in self.passages:
                if passage.metadata.get("source_type") != "figure_table":
                    continue
                for label in self._float_labels(passage.text):
                    floats_by_label[label].append(passage.id)

            # Use the prose ranking as a graph edge into the specific floats it cites.
            # This avoids promoting every caption merely because the question says "table".
            reference_scores: defaultdict[str, float] = defaultdict(float)
            anchors = lexical[:10] + dense[:10]
            for anchor_rank, (passage_id, _) in enumerate(anchors, 1):
                passage = self.by_id[passage_id]
                if passage.metadata.get("source_type") == "figure_table":
                    continue
                for label in self._float_labels(passage.text):
                    for float_id in floats_by_label[label]:
                        reference_scores[float_id] += 1.0 / anchor_rank

            # Captions with actual lexical overlap remain eligible even when prose does
            # not cite them. Zero-overlap captions are admitted only through a reference.
            for passage in self.passages:
                if passage.metadata.get("source_type") != "figure_table":
                    continue
                lexical_score = self._bm25(" ".join(keywords(query, limit=20)), passage.id)
                score = reference_scores[passage.id] + lexical_score
                if score > 0:
                    structured.append((passage.id, score))
            structured.sort(key=lambda item: item[1], reverse=True)
        lexical_rank = {pid: (rank, score) for rank, (pid, score) in enumerate(lexical[: self.settings.search_top_k], 1)}
        dense_rank = {pid: (rank, score) for rank, (pid, score) in enumerate(dense[: self.settings.search_top_k], 1)}
        structure_rank = {
            pid: (rank, score)
            for rank, (pid, score) in enumerate(structured[: self.settings.search_top_k], 1)
        }
        hits: dict[str, SearchHit] = {}
        for pid in set(lexical_rank) | set(dense_rank) | set(structure_rank):
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
            title_match = bool(query_terms & set(tokens(p.title)))
            hit.metadata_score = 1.0 if title_match else 0.0
            if pid in structure_rank:
                hit.structure_rank, hit.structure_score = structure_rank[pid]
                hit.rrf_score += self.settings.structure_weight / (
                    self.settings.rrf_k + hit.structure_rank
                )
                hit.retrievers.append("structure_boost")
            hit.final_score = hit.rrf_score + self.settings.metadata_weight * hit.metadata_score
            hits[pid] = hit
        ranked = sorted(
            hits.values(),
            key=lambda hit: (
                -hit.final_score,
                self.by_id[hit.passage_id].ordinal,
                hit.passage_id,
            ),
        )
        return self._diversify(ranked, k)
