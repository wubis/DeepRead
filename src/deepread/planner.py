from __future__ import annotations

import re

from .models import Requirement, TaskNode
from .text import keywords


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
            requirement = Requirement(f"req_{index}_1", task_id, f"Find evidence that answers: {part}", kws)
            tasks.append(TaskNode(task_id, part, [requirement]))
        return tasks or [TaskNode("task_1", question, [Requirement("req_1_1", "task_1", question, keywords(question))])]
