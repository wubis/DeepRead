import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    max_tasks: int = 3
    max_requirements: int = 4
    max_search_rounds: int = 2
    retrieval_mode: str = "hybrid"
    reader_mode: str = "hierarchical"
    flat_top_k: int = 5
    evidence_window_sentences: int = 3
    evidence_candidates_per_requirement: int = 2
    evidence_support_threshold: float = 0.60
    supervisor_mode: str = "bounded"
    answer_policy: str = "grounded"
    random_seed: int = 0
    search_top_k: int = 75
    rerank_top_k: int = 20
    model_rerank_top_k: int = 8
    model_rerank_rescue_per_requirement: int = 1
    model_rerank_max_chars: int = 700
    max_evidence_tokens: int = 12_000
    target_coverage: float = 0.50
    rrf_k: int = 60
    lexical_weight: float = 0.55
    dense_weight: float = 0.45
    metadata_weight: float = 0.05
    structure_weight: float = 0.05
    redundancy_weight: float = 0.0
    provider: str = "auto"
    openai_model: str = "gpt-5.6-terra"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_reasoning_effort: str = "low"
    openai_store: bool = False
    enable_model_rerank: bool = True
    enable_evidence_assessment: bool = True
    embedding_cache_dir: str = ".deepread/embeddings"
    model_input_cost_per_million: float | None = None
    model_output_cost_per_million: float | None = None
    embedding_cost_per_million: float | None = None

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"bm25", "embeddings", "hybrid"}:
            raise ValueError("retrieval_mode must be one of: bm25, embeddings, hybrid")
        if self.reader_mode not in {"flat", "hierarchical"}:
            raise ValueError("reader_mode must be one of: flat, hierarchical")
        if self.supervisor_mode not in {"single_pass", "bounded"}:
            raise ValueError("supervisor_mode must be one of: single_pass, bounded")
        if self.answer_policy not in {"grounded", "benchmark"}:
            raise ValueError("answer_policy must be one of: grounded, benchmark")
        if self.max_search_rounds < 1:
            raise ValueError("max_search_rounds must be at least 1")
        if self.flat_top_k < 1:
            raise ValueError("flat_top_k must be at least 1")
        if self.evidence_window_sentences < 1:
            raise ValueError("evidence_window_sentences must be at least 1")
        if self.evidence_candidates_per_requirement < 1:
            raise ValueError("evidence_candidates_per_requirement must be at least 1")
        if not 0 <= self.evidence_support_threshold <= 1:
            raise ValueError("evidence_support_threshold must be between 0 and 1")
        if self.model_rerank_top_k < 1:
            raise ValueError("model_rerank_top_k must be at least 1")
        if self.model_rerank_rescue_per_requirement < 0:
            raise ValueError("model_rerank_rescue_per_requirement must be non-negative")
        if self.model_rerank_max_chars < 100:
            raise ValueError("model_rerank_max_chars must be at least 100")
        if not 0 <= self.redundancy_weight <= 1:
            raise ValueError("redundancy_weight must be between 0 and 1")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("DEEPREAD_PROVIDER", "auto"),
            openai_model=os.getenv("DEEPREAD_OPENAI_MODEL", "gpt-5.6-terra"),
            openai_embedding_model=os.getenv("DEEPREAD_EMBEDDING_MODEL", "text-embedding-3-small"),
            openai_reasoning_effort=os.getenv("DEEPREAD_REASONING_EFFORT", "low"),
            openai_store=os.getenv("DEEPREAD_OPENAI_STORE", "false").lower() == "true",
            enable_model_rerank=os.getenv("DEEPREAD_MODEL_RERANK", "true").lower() == "true",
            enable_evidence_assessment=os.getenv("DEEPREAD_EVIDENCE_ASSESSMENT", "true").lower() == "true",
            embedding_cache_dir=os.getenv("DEEPREAD_EMBEDDING_CACHE", ".deepread/embeddings"),
            max_requirements=int(os.getenv("DEEPREAD_MAX_REQUIREMENTS", "4")),
            max_search_rounds=int(os.getenv("DEEPREAD_MAX_SEARCH_ROUNDS", "2")),
            retrieval_mode=os.getenv("DEEPREAD_RETRIEVAL_MODE", "hybrid"),
            reader_mode=os.getenv("DEEPREAD_READER_MODE", "hierarchical"),
            flat_top_k=int(os.getenv("DEEPREAD_FLAT_TOP_K", "5")),
            evidence_window_sentences=int(
                os.getenv("DEEPREAD_EVIDENCE_WINDOW_SENTENCES", "3")
            ),
            evidence_candidates_per_requirement=int(
                os.getenv("DEEPREAD_EVIDENCE_CANDIDATES_PER_REQUIREMENT", "2")
            ),
            evidence_support_threshold=float(
                os.getenv("DEEPREAD_EVIDENCE_SUPPORT_THRESHOLD", "0.60")
            ),
            supervisor_mode=os.getenv("DEEPREAD_SUPERVISOR_MODE", "bounded").replace("-", "_"),
            answer_policy=os.getenv("DEEPREAD_ANSWER_POLICY", "grounded"),
            random_seed=int(os.getenv("DEEPREAD_RANDOM_SEED", "0")),
            target_coverage=float(os.getenv("DEEPREAD_TARGET_COVERAGE", "0.50")),
            redundancy_weight=float(os.getenv("DEEPREAD_DIVERSITY_WEIGHT", "0.0")),
            model_rerank_top_k=int(os.getenv("DEEPREAD_MODEL_RERANK_TOP_K", "8")),
            model_rerank_rescue_per_requirement=int(
                os.getenv("DEEPREAD_MODEL_RERANK_RESCUE_PER_REQUIREMENT", "1")
            ),
            model_rerank_max_chars=int(os.getenv("DEEPREAD_MODEL_RERANK_MAX_CHARS", "700")),
            model_input_cost_per_million=_optional_float("DEEPREAD_MODEL_INPUT_COST_PER_MILLION"),
            model_output_cost_per_million=_optional_float("DEEPREAD_MODEL_OUTPUT_COST_PER_MILLION"),
            embedding_cost_per_million=_optional_float("DEEPREAD_EMBEDDING_COST_PER_MILLION"),
        )


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None
