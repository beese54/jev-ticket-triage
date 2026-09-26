"""Task specs: which questions are asked per dataset, shared by every runner."""

from dataclasses import dataclass
from typing import Literal

from triage import labels

FieldKind = Literal["choice", "score", "signal"]


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind  # choice: pick one; score: ordered levels; signal: yes/no
    instruction: str
    options: dict[str, str]  # label -> description (ordered low->high for score)
    scored: bool = True  # False for demo-only signals without gold labels


@dataclass(frozen=True)
class Task:
    dataset: str
    fields: tuple[Field, ...]

    def state(self, row: dict) -> str:
        if self.dataset == "banking77":
            return row["text"]
        subject = row.get("subject") or ""
        return f"Subject: {subject}\n\n{row['body']}" if subject else row["body"]

    def gold(self, row: dict) -> dict[str, str]:
        if self.dataset == "banking77":
            return {"intent": row["intent"]}
        return {"queue": row["queue"], "priority": row["priority"], "type": row["type"]}


BANKING77 = Task(
    dataset="banking77",
    fields=(
        Field(
            "intent", "choice", labels.BANKING77_INSTRUCTION, labels.BANKING77_INTENTS
        ),
    ),
)

TICKETS = Task(
    dataset="tickets",
    fields=(
        Field("queue", "choice", labels.QUEUE_INSTRUCTION, labels.QUEUES),
        Field("priority", "score", labels.PRIORITY_INSTRUCTION, labels.PRIORITIES),
        Field("type", "choice", labels.TYPE_INSTRUCTION, labels.TYPES),
        *(
            Field(name, "signal", desc, {}, scored=False)
            for name, desc in labels.SIGNALS.items()
        ),
    ),
)

TASKS = {t.dataset: t for t in (BANKING77, TICKETS)}
