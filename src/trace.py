"""Per-turn trace records emitted as single JSON lines."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def question_hash(question: str) -> str:
    """Stable short hash of a question, so repeats are detectable without storing it."""
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TraceRecord:
    """One query turn.

    Deliberately holds question_len and question_hash only, never the question
    text, so that no PII can reach disk (FR-G5, NFR-3).
    """

    template_id: str
    question_len: int
    question_hash: str
    trace_id: str = ""
    stages_skipped: list[str] = field(default_factory=list)
    stage_ms: dict[str, int] = field(default_factory=dict)
    n_hits: int = 0
    hits: list[dict[str, Any]] = field(default_factory=list)
    cited_chunk_id: str | None = None
    citation_url: str | None = None
    last_updated: str | None = None
    sentence_count: int = 0
    truncated_by_cap: bool = False
    guardrails: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    at: str = field(default_factory=utc_now)

    @classmethod
    def build(cls, question: str, template_id: str) -> TraceRecord:
        digest = question_hash(question)
        return cls(
            template_id=template_id,
            question_len=len(question),
            question_hash=digest,
            trace_id=digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def emit(record: TraceRecord) -> None:
    """Write a trace record to stdout as one JSON line."""
    sys.stdout.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log_pii_blocked(pattern: str) -> None:
    """Log a PII block. Records the pattern name only, never the input."""
    sys.stdout.write(
        json.dumps({"pii_blocked": True, "pattern": pattern, "at": utc_now()}, ensure_ascii=False)
        + "\n"
    )
    sys.stdout.flush()