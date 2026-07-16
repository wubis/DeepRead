from __future__ import annotations

import re

from .models import Requirement, TaskNode
from .text import keywords


BOOLEAN_START = re.compile(
    r"^\s*(?:did|do|does|is|are|was|were|can|could|would|will|has|have|had)\b",
    flags=re.IGNORECASE,
)
COMPOUND_MARKER = re.compile(
    r"(?:;|\b(?:and|versus|vs\.)\b|\b(?:compare|difference|trade-?offs?|both)\b)",
    flags=re.IGNORECASE,
)


def infer_answer_type(question: str) -> str:
    """Infer the response shape needed by concise QA synthesis."""
    if BOOLEAN_START.search(question):
        return "boolean"
    if re.search(r"^\s*(?:what|which|who|where|when|how many|how much)\b", question, re.I):
        return "extractive"
    return "abstractive"


def should_decompose(question: str) -> bool:
    """Reserve multi-requirement plans for explicitly compound questions."""
    return bool(COMPOUND_MARKER.search(question))


class RuleBasedPlanner:
    """Deterministic planner used by the no-model MVP."""

    def __init__(self, max_tasks: int = 3):
        self.max_tasks = max_tasks

    def plan(self, question: str) -> list[TaskNode]:
        parts = [p.strip(" ,") for p in re.split(r"\?|;|\b(?:and also|and|versus|vs\.)\b", question, flags=re.I) if p.strip(" ,")]
        if len(parts) == 1 and re.search(r"\b(compare|difference|trade-?offs?)\b", question, re.I):
            parts = [question]
        tasks: list[TaskNode] = []
        for index, part in enumerate(parts[: self.max_tasks], 1):
            task_id = f"task_{index}"
            kws = keywords(part)
            requirement = Requirement(f"req_{index}_1", task_id, part, kws)
            tasks.append(TaskNode(task_id, part, [requirement]))
        return tasks or [TaskNode("task_1", question, [Requirement("req_1_1", "task_1", question, keywords(question))])]
