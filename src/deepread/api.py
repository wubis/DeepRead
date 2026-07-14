from __future__ import annotations

import os
from functools import lru_cache

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install the app dependencies with: pip install -e '.[app]'") from exc

from .engine import EvidenceGraphEngine
from .corpus import load_corpus


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    trace_path: str | None = None


@lru_cache
def engine() -> EvidenceGraphEngine:
    return EvidenceGraphEngine(os.getenv("DEEPREAD_CORPUS", "data/sample_corpus"), db_path=os.getenv("DEEPREAD_DB", ".deepread/evidence.db"))


app = FastAPI(title="DeepRead EvidenceGraph", version="0.1.0")


@app.get("/health")
def health() -> dict[str, object]:
    corpus = os.getenv("DEEPREAD_CORPUS", "data/sample_corpus")
    documents, passages = load_corpus(corpus)
    requested = os.getenv("DEEPREAD_PROVIDER", "auto")
    provider = "openai" if requested == "openai" or (requested == "auto" and bool(os.getenv("OPENAI_API_KEY"))) else "offline"
    return {"status": "ok", "provider": provider, "documents": len(documents), "passages": len(passages)}


@app.post("/v1/query")
def query(request: QueryRequest) -> dict[str, object]:
    try:
        return engine().ask(request.question, request.trace_path).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
