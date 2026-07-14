from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .models import Claim, Evidence


class EvidenceMemory:
    """Query-scoped claim/provenance graph with optional SQLite persistence."""

    def __init__(self, db_path: str | Path | None = None):
        self.evidence: dict[str, Evidence] = {}
        self.claims: dict[str, Claim] = {}
        self.edges: list[tuple[str, str, str]] = []
        self.db_path = Path(db_path) if db_path else None
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialize()

    def _connect(self) -> sqlite3.Connection:
        assert self.db_path
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS evidence (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS claims (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS edges (source TEXT, target TEXT, relation TEXT);
            """)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence[evidence.id] = evidence
        self.edges.append((evidence.requirement_id, evidence.id, "supported_by"))
        if self.db_path:
            with self._connect() as connection:
                connection.execute("INSERT OR REPLACE INTO evidence VALUES (?, ?)", (evidence.id, json.dumps(asdict(evidence))))
                connection.execute("INSERT INTO edges VALUES (?, ?, ?)", (evidence.requirement_id, evidence.id, "supported_by"))

    def add_claim(self, claim: Claim) -> None:
        self.claims[claim.id] = claim
        for evidence_id in claim.evidence_ids:
            self.edges.append((claim.id, evidence_id, "supports"))
        if self.db_path:
            with self._connect() as connection:
                connection.execute("INSERT OR REPLACE INTO claims VALUES (?, ?)", (claim.id, json.dumps(asdict(claim))))
                connection.executemany("INSERT INTO edges VALUES (?, ?, 'supports')", [(claim.id, eid) for eid in claim.evidence_ids])

    def graph(self) -> dict[str, object]:
        return {"evidence": [asdict(e) for e in self.evidence.values()], "claims": [asdict(c) for c in self.claims.values()], "edges": self.edges}

