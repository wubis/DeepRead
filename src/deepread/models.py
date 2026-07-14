from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ReadLevel(str, Enum):
    TITLE = "title"
    SUMMARY = "summary"
    SECTION = "section"
    PASSAGE = "passage"
    DOCUMENT = "document"


@dataclass(slots=True)
class Passage:
    id: str
    document_id: str
    title: str
    section: str
    text: str
    ordinal: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    id: str
    title: str
    summary: str
    sections: dict[str, list[str]]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Requirement:
    id: str
    task_id: str
    description: str
    keywords: list[str]


@dataclass(slots=True)
class TaskNode:
    id: str
    question: str
    requirements: list[Requirement]


@dataclass(slots=True)
class SearchHit:
    passage_id: str
    document_id: str
    bm25_rank: int | None = None
    dense_rank: int | None = None
    lexical_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    metadata_score: float = 0.0
    redundancy_penalty: float = 0.0
    final_score: float = 0.0
    retrievers: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Evidence:
    id: str
    requirement_id: str
    passage_id: str
    document_id: str
    title: str
    section: str
    text: str
    score: float
    read_level: ReadLevel
    token_cost: int
    supports: bool = True
    relation: str = "supports"
    char_start: int = 0
    char_end: int = 0
    citation_tokens: int = 0


@dataclass(slots=True)
class Claim:
    id: str
    text: str
    evidence_ids: list[str]
    requirement_id: str
    confidence: float


@dataclass(slots=True)
class ReadDecision:
    passage_id: str
    level: ReadLevel
    requirement_id: str
    expected_gain: float
    token_cost: int
    utility: float
    selected: bool
    reason: str


@dataclass(slots=True)
class QueryTrace:
    question: str
    tasks: list[TaskNode] = field(default_factory=list)
    searches: list[dict[str, Any]] = field(default_factory=list)
    ranking: list[SearchHit] = field(default_factory=list)
    reads: list[ReadDecision] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    rounds: int = 0
    tokens_used: int = 0
    coverage: float = 0.0
    stop_reason: str = ""
    provider: str = "offline"
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    read_tokens: int = 0
    citation_tokens: int = 0
    api_input_tokens: int = 0
    api_output_tokens: int = 0
    api_total_tokens: int = 0
    estimated_api_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Answer:
    question: str
    text: str
    citations: list[Evidence]
    coverage: float
    stop_reason: str
    trace: QueryTrace

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
