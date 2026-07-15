from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

from .corpus import stable_id
from .models import CorpusBundle, Document, Passage


@dataclass(frozen=True, slots=True)
class QasperEvidenceMatch:
    passage_id: str
    char_start: int
    char_end: int
    source_type: str
    source_path: str
    match_type: str


@dataclass(frozen=True, slots=True)
class QasperEvidenceRef:
    """A gold evidence string and every source location to which it maps."""

    text: str
    matches: tuple[QasperEvidenceMatch, ...] = ()

    @property
    def resolved(self) -> bool:
        return bool(self.matches)


@dataclass(frozen=True, slots=True)
class QasperAnswer:
    annotation_id: str
    unanswerable: bool
    extractive_spans: tuple[str, ...]
    free_form_answer: str | None
    yes_no: bool | None
    evidence: tuple[QasperEvidenceRef, ...]
    highlighted_evidence: tuple[QasperEvidenceRef, ...]


@dataclass(frozen=True, slots=True)
class QasperQuestion:
    id: str
    paper_id: str
    question: str
    answers: tuple[QasperAnswer, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QasperDataset:
    documents: list[Document]
    passages: list[Passage]
    questions: list[QasperQuestion]

    def corpus(self, paper_ids: Iterable[str] | None = None) -> CorpusBundle:
        """Return runtime corpus records, optionally restricted to selected papers."""

        if paper_ids is None:
            return CorpusBundle(list(self.documents), list(self.passages))
        selected = set(paper_ids)
        documents = [item for item in self.documents if item.metadata["paper_id"] in selected]
        document_ids = {item.id for item in documents}
        passages = [item for item in self.passages if item.document_id in document_ids]
        return CorpusBundle(documents, passages)

    def questions_for(self, paper_ids: Iterable[str]) -> list[QasperQuestion]:
        selected = set(paper_ids)
        return [item for item in self.questions if item.paper_id in selected]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _records(value: Any, anchors: tuple[str, ...]) -> list[dict[str, Any]]:
    """Normalize list-of-records and Arrow-style dict-of-lists structures."""

    if value is None:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if not isinstance(value, Mapping):
        return []
    anchor = next((key for key in anchors if isinstance(value.get(key), list)), None)
    if anchor is None:
        return [dict(value)]
    size = len(value[anchor])
    rows: list[dict[str, Any]] = []
    for index in range(size):
        row: dict[str, Any] = {}
        for key, column in value.items():
            row[key] = column[index] if isinstance(column, list) and len(column) == size else column
        rows.append(row)
    return rows


def _nonempty_strings(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in _as_list(value) if item is not None and str(item).strip())


@lru_cache(maxsize=32_768)
def _normalised_with_offsets(text: str) -> tuple[str, tuple[int, ...]]:
    normalised: list[str] = []
    offsets: list[int] = []
    for match_index, match in enumerate(re.finditer(r"\S+", text)):
        if match_index:
            normalised.append(" ")
            offsets.append(match.start() - 1 if match.start() else 0)
        for offset, character in enumerate(match.group()):
            normalised.append(character)
            offsets.append(match.start() + offset)
    return "".join(normalised), tuple(offsets)


def _span_in_text(haystack: str, needle: str) -> tuple[int, int] | None:
    direct = haystack.find(needle)
    if direct >= 0:
        return direct, direct + len(needle)
    normalised_haystack, offsets = _normalised_with_offsets(haystack)
    normalised_needle, _ = _normalised_with_offsets(needle)
    if not normalised_needle:
        return None
    start = normalised_haystack.find(normalised_needle)
    if start < 0:
        return None
    end = start + len(normalised_needle)
    return offsets[start], offsets[end - 1] + 1


def _match_evidence(text: str, passages: list[Passage]) -> QasperEvidenceRef:
    float_match = re.match(r"^\s*FLOAT SELECTED\s*:?\s*(.*)$", text, flags=re.IGNORECASE)
    query = float_match.group(1) if float_match and float_match.group(1).strip() else text
    candidates = passages
    if float_match:
        candidates = [item for item in passages if item.metadata["source_type"] == "figure_table"]

    exact: list[QasperEvidenceMatch] = []
    substring: list[QasperEvidenceMatch] = []
    composite: list[QasperEvidenceMatch] = []
    normalised_query, _ = _normalised_with_offsets(query)
    for passage in candidates:
        normalised_passage, _ = _normalised_with_offsets(passage.text)
        source_type = str(passage.metadata["source_type"])
        source_path = str(passage.metadata["source_path"])
        if normalised_passage == normalised_query:
            exact.append(
                QasperEvidenceMatch(
                    passage.id,
                    0,
                    len(passage.text),
                    source_type,
                    source_path,
                    "float_exact" if float_match else "exact",
                )
            )
            continue
        span = _span_in_text(passage.text, query)
        if span is not None:
            substring.append(
                QasperEvidenceMatch(
                    passage.id,
                    span[0],
                    span[1],
                    source_type,
                    source_path,
                    "float_substring" if float_match else "substring",
                )
            )
            continue
        reverse_span = _span_in_text(query, passage.text)
        heading_prefix = source_type == "section_heading" and normalised_query.startswith(
            f"{normalised_passage} "
        )
        if reverse_span is not None and (len(normalised_passage) >= 20 or heading_prefix):
            composite.append(
                QasperEvidenceMatch(
                    passage.id,
                    0,
                    len(passage.text),
                    source_type,
                    source_path,
                    "composite_source",
                )
            )
    return QasperEvidenceRef(text, tuple(exact or substring or composite))


def _answer_records(value: Any) -> list[dict[str, Any]]:
    records = _records(value, ("annotation_id", "answer"))
    expanded: list[dict[str, Any]] = []
    for record in records:
        annotation_ids = _as_list(record.get("annotation_id"))
        answers = _as_list(record.get("answer"))
        if len(annotation_ids) > 1 and len(annotation_ids) == len(answers):
            expanded.extend(
                {"annotation_id": annotation_id, "answer": answer}
                for annotation_id, answer in zip(annotation_ids, answers)
            )
        else:
            expanded.append(record)
    return expanded


def _adapt_answer(record: Mapping[str, Any], passages: list[Passage]) -> QasperAnswer:
    payload = record.get("answer", record)
    if not isinstance(payload, Mapping):
        payload = {}
    extractive = _nonempty_strings(payload.get("extractive_spans"))
    free_form_raw = payload.get("free_form_answer")
    free_form = str(free_form_raw).strip() if free_form_raw is not None else ""
    unanswerable = bool(payload.get("unanswerable", False))
    yes_no_raw = payload.get("yes_no")
    yes_no = None
    if not unanswerable and not extractive and not free_form and isinstance(yes_no_raw, bool):
        yes_no = yes_no_raw
    evidence = tuple(
        _match_evidence(text, passages) for text in _nonempty_strings(payload.get("evidence"))
    )
    highlighted = tuple(
        _match_evidence(text, passages)
        for text in _nonempty_strings(payload.get("highlighted_evidence"))
    )
    return QasperAnswer(
        annotation_id=str(record.get("annotation_id") or ""),
        unanswerable=unanswerable,
        extractive_spans=extractive,
        free_form_answer=free_form or None,
        yes_no=yes_no,
        evidence=evidence,
        highlighted_evidence=highlighted,
    )


def _paper_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError("QASPER data must be a JSON object, array, or JSONL sequence")
    if isinstance(payload.get("data"), list):
        return [dict(item) for item in payload["data"] if isinstance(item, Mapping)]
    if "full_text" in payload and "title" in payload:
        return [dict(payload)]
    rows: list[dict[str, Any]] = []
    for paper_id, value in payload.items():
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        row.setdefault("id", paper_id)
        rows.append(row)
    return rows


def adapt_qasper(rows: Iterable[Mapping[str, Any]]) -> QasperDataset:
    """Convert QASPER rows without merging or re-chunking annotated source units."""

    documents: list[Document] = []
    passages: list[Passage] = []
    questions: list[QasperQuestion] = []
    for raw_row in rows:
        row = dict(raw_row)
        paper_id = str(
            row.get("id") or row.get("paper_id") or row.get("article_id") or ""
        ).strip()
        if not paper_id:
            raise ValueError("Every QASPER paper must have an id")
        document_id = stable_id("doc", f"qasper:{paper_id}")
        title = str(row.get("title") or paper_id)
        abstract = str(row.get("abstract") or "")
        document_metadata = {
            "dataset": "qasper",
            "paper_id": paper_id,
            "source_path": f"qasper/{paper_id}",
        }
        sections: dict[str, list[str]] = {}
        paper_passages: list[Passage] = []

        def add_passage(
            text: str,
            section: str,
            source_type: str,
            source_path: str,
            **metadata: Any,
        ) -> None:
            if not text.strip():
                return
            ordinal = len(paper_passages)
            passage_metadata = {
                **document_metadata,
                "source_type": source_type,
                "source_path": source_path,
                "source_char_start": 0,
                "source_char_end": len(text),
                **metadata,
            }
            sections.setdefault(section, []).append(text)
            paper_passages.append(
                Passage(
                    stable_id("passage", f"qasper:{paper_id}:{source_path}"),
                    document_id,
                    title,
                    section,
                    text,
                    ordinal,
                    passage_metadata,
                )
            )

        add_passage(abstract, "Abstract", "abstract", "abstract")
        full_text = _records(row.get("full_text"), ("section_name", "paragraphs"))
        for section_index, item in enumerate(full_text):
            name = item.get("section_name")
            section = str(name).strip() if name is not None and str(name).strip() else (
                f"Untitled Section {section_index + 1}"
            )
            sections.setdefault(section, [])
            if name is not None and str(name).strip():
                add_passage(
                    str(name).strip(),
                    section,
                    "section_heading",
                    f"full_text.section_name[{section_index}]",
                    section_index=section_index,
                )
            for paragraph_index, paragraph in enumerate(_as_list(item.get("paragraphs"))):
                if paragraph is None:
                    continue
                add_passage(
                    str(paragraph),
                    section,
                    "paragraph",
                    f"full_text.paragraphs[{section_index}][{paragraph_index}]",
                    section_index=section_index,
                    paragraph_index=paragraph_index,
                )

        floats = _records(row.get("figures_and_tables"), ("caption", "file"))
        for float_index, item in enumerate(floats):
            caption = str(item.get("caption") or "")
            add_passage(
                caption,
                "Figures and Tables",
                "figure_table",
                f"figures_and_tables[{float_index}]",
                float_index=float_index,
                file=item.get("file"),
            )

        document = Document(document_id, title, abstract, sections, document_metadata)
        documents.append(document)
        passages.extend(paper_passages)

        qas = _records(row.get("qas"), ("question_id", "question"))
        for question_index, item in enumerate(qas):
            question_id = str(item.get("question_id") or f"{paper_id}:{question_index}")
            answers = tuple(
                _adapt_answer(answer, paper_passages)
                for answer in _answer_records(item.get("answers"))
            )
            metadata = {
                key: item.get(key)
                for key in (
                    "nlp_background",
                    "topic_background",
                    "paper_read",
                    "search_query",
                )
                if key in item
            }
            questions.append(
                QasperQuestion(
                    question_id,
                    paper_id,
                    str(item.get("question") or ""),
                    answers,
                    metadata,
                )
            )
    return QasperDataset(documents, passages, questions)


def load_qasper(path: str | Path) -> QasperDataset:
    """Load official JSON, JSONL, or an object keyed by QASPER paper id."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        return adapt_qasper(row for row in rows if isinstance(row, Mapping))
    payload = json.loads(source.read_text(encoding="utf-8"))
    return adapt_qasper(_paper_records(payload))


def load_qasper_hf(
    split: str = "validation",
    *,
    cache_dir: str | Path | None = None,
    revision: str | None = None,
) -> QasperDataset:
    """Load QASPER from Hugging Face with the optional ``eval`` dependency."""

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install QASPER support with: pip install -e '.[eval]'") from exc
    kwargs: dict[str, Any] = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    dataset = load_dataset("allenai/qasper", **kwargs)
    return adapt_qasper(dataset)


__all__ = [
    "QasperAnswer",
    "QasperDataset",
    "QasperEvidenceMatch",
    "QasperEvidenceRef",
    "QasperQuestion",
    "adapt_qasper",
    "load_qasper",
    "load_qasper_hf",
]
