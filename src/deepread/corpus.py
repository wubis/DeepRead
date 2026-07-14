from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from .models import Document, Passage
from .text import split_sentences


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha1(value.encode()).hexdigest()[:10]}"


def _passage_chunks(text: str, max_words: int = 120) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    for sentence in split_sentences(text):
        if current and len(" ".join(current).split()) + len(sentence.split()) > max_words:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks or ([text.strip()] if text.strip() else [])


def parse_markdown(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title = next((line[2:].strip() for line in lines if line.startswith("# ")), path.stem)
    sections: dict[str, list[str]] = {}
    section = "Overview"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        text = " ".join(line.strip() for line in buffer if line.strip())
        if text:
            sections.setdefault(section, []).extend(_passage_chunks(text))
        buffer = []

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            flush()
            heading = re.sub(r"^#{1,3}\s+", "", line).strip()
            if not line.startswith("# ") or heading != title:
                section = heading
        else:
            buffer.append(line)
    flush()
    first = next((p for values in sections.values() for p in values), "")
    summary = " ".join(split_sentences(first)[:2])
    return Document(stable_id("doc", str(path)), title, summary, sections, {"path": str(path)})


def parse_text(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    title = next((line.strip() for line in raw.splitlines() if line.strip()), path.stem)
    chunks = _passage_chunks(raw)
    summary = " ".join(split_sentences(chunks[0])[:2]) if chunks else ""
    return Document(stable_id("doc", str(path)), title, summary, {"Body": chunks}, {"path": str(path)})


def load_corpus(path: str | Path) -> tuple[list[Document], list[Passage]]:
    root = Path(path)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"})
    documents: list[Document] = []
    passages: list[Passage] = []
    for file in files:
        document = parse_markdown(file) if file.suffix.lower() == ".md" else parse_text(file)
        documents.append(document)
        ordinal = 0
        for section, chunks in document.sections.items():
            for chunk in chunks:
                passages.append(Passage(stable_id("passage", f"{document.id}:{ordinal}:{chunk}"), document.id, document.title, section, chunk, ordinal, document.metadata))
                ordinal += 1
    return documents, passages


def save_index(path: str | Path, documents: list[Document], passages: list[Passage]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"documents": [asdict(d) for d in documents], "passages": [asdict(p) for p in passages]}, indent=2), encoding="utf-8")
