"""Shared dataclasses.

Kept free of ML imports so guardrails and tests stay importable without
sentence-transformers, chromadb, or groq installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

DocType = Literal[
    "scheme_page",
    "sid",
    "kim",
    "factsheet",
    "efactsheet",
    "downloads",
    "homepage",
    "third_party",
]
Scheme = Literal["large_cap", "flexi_cap", "elss", "midcap", "amc_wide"]
Plan = Literal["direct", "regular", "n_a"]
BlockKind = Literal["heading", "text", "list", "table"]
ExtractFlag = Literal["ok", "fetch_failed", "suspected_client_rendered", "extract_failed"]
TemplateId = Literal[
    "FACTUAL",
    "NOT_FOUND",
    "NO_RETRIEVAL",
    "REFUSE_ADVICE",
    "REFUSE_PERFORMANCE",
    "PII_BLOCKED",
]


@dataclass(frozen=True)
class Source:
    """One row of sources.csv."""

    source_id: str
    scheme: Scheme
    plan: Plan
    doc_type: DocType
    url: str
    enabled: bool
    notes: str = ""

    @property
    def is_pdf(self) -> bool:
        return self.url.lower().split("?")[0].endswith(".pdf")

    @property
    def filename(self) -> str:
        """Filename used under data/raw/<source_id>/, derived from the URL path."""
        from urllib.parse import unquote, urlparse

        path = urlparse(self.url).path
        name = unquote(path.rsplit("/", 1)[-1]).strip()
        if not name:
            return f"{self.source_id}.pdf" if self.is_pdf else f"{self.source_id}.html"
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
        suffix = Path(safe).suffix.lower()
        if self.is_pdf and suffix != ".pdf":
            safe = f"{safe}.pdf"
        elif not self.is_pdf and suffix not in (".html", ".htm", ".xhtml", ".pdf"):
            safe = f"{safe}.html"
        return safe


@dataclass
class FetchResult:
    """Outcome of one HTTP GET."""

    source_id: str
    url: str
    status: str
    http_status: int | None = None
    byte_size: int = 0
    content_hash: str = ""
    fetched_at: str = ""
    local_path: str = ""
    error: str = ""


@dataclass
class Block:
    """A structured unit of a document, before normalization and chunking."""

    kind: BlockKind
    text: str
    heading_path: list[str] = field(default_factory=list)
    page_num: int | None = None
    table_rows: list[list[str]] | None = None
    font_size: float | None = None

    @property
    def heading_str(self) -> str:
        return " > ".join(self.heading_path)


@dataclass
class Chunk:
    """A retrievable unit, as written to artifacts/chunks.json."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]

    @property
    def char_len(self) -> int:
        return len(self.text)


@dataclass
class Hit:
    """One retrieved chunk with its similarity score."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float

    @property
    def source_url(self) -> str:
        return str(self.metadata.get("source_url", ""))

    @property
    def source_date(self) -> str | None:
        value = self.metadata.get("source_date")
        return str(value) if value else None


@dataclass
class ScreenResult:
    """Verdict from the pre-LLM input guardrail."""

    verdict: Literal["FACTUAL", "ADVICE", "PERFORMANCE", "PII"]
    matched: list[str] = field(default_factory=list)
    escalated: bool = False
    pii_pattern: str | None = None

    @property
    def is_factual(self) -> bool:
        return self.verdict == "FACTUAL"


@dataclass
class VerifyResult:
    """Verdict from the post-generation output guardrail."""

    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)


@dataclass
class AnswerEnvelope:
    """The only object the UI consumes."""

    answer: str
    template_id: TemplateId
    citation_url: str | None = None
    source_title: str | None = None
    chunk_id: str | None = None
    last_updated: str | None = None
    hits: list[Hit] = field(default_factory=list)
    latency_ms: dict[str, int] = field(default_factory=dict)
    guardrail_trace: dict[str, Any] = field(default_factory=dict)
    raw_reply: str = ""


@dataclass
class IngestReport:
    """Per-source status collected during the fetch and extract stages."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def by_flag(self, flag: str) -> list[dict[str, Any]]:
        return [r for r in self.rows if r.get("flag") == flag]