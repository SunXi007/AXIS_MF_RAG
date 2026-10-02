"""Loading and validation of sources.csv."""

from __future__ import annotations

import csv
from pathlib import Path

import config
from src.models import Source


class SourceValidationError(ValueError):
    """Raised when sources.csv violates the manifest contract."""


def _require(value: str, field: str, source_id: str) -> str:
    text = (value or "").strip()
    if not text:
        raise SourceValidationError(f"{source_id}: '{field}' is required")
    return text


def load_sources(path: Path | None = None, *, enabled_only: bool = False) -> list[Source]:
    """Parse sources.csv into Source objects, validating every row.

    Raises SourceValidationError on a duplicate id, an unknown enum value,
    a malformed URL, or a missing required field.
    """
    csv_path = Path(path) if path else config.SOURCES_CSV
    if not csv_path.exists():
        raise SourceValidationError(f"sources.csv not found at {csv_path}")

    sources: list[Source] = []
    seen: set[str] = set()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_id", "scheme", "plan", "doc_type", "url", "enabled"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SourceValidationError(f"sources.csv missing columns: {sorted(missing)}")

        for line_no, row in enumerate(reader, start=2):
            source_id = _require(row["source_id"], "source_id", f"line {line_no}")
            if source_id in seen:
                raise SourceValidationError(f"duplicate source_id '{source_id}' on line {line_no}")
            seen.add(source_id)

            scheme = _require(row["scheme"], "scheme", source_id)
            if scheme not in config.SCHEMES:
                raise SourceValidationError(f"{source_id}: unknown scheme '{scheme}'")

            plan = _require(row["plan"], "plan", source_id)
            if plan not in config.PLANS:
                raise SourceValidationError(f"{source_id}: unknown plan '{plan}'")

            doc_type = _require(row["doc_type"], "doc_type", source_id)
            if doc_type not in config.DOC_TYPES:
                raise SourceValidationError(f"{source_id}: unknown doc_type '{doc_type}'")

            url = _require(row["url"], "url", source_id)
            if not url.startswith(("http://", "https://")):
                raise SourceValidationError(f"{source_id}: url must be http(s), got '{url}'")

            enabled = _require(row["enabled"], "enabled", source_id).lower()
            if enabled not in ("true", "false"):
                raise SourceValidationError(f"{source_id}: enabled must be true/false, got '{enabled}'")

            sources.append(
                Source(
                    source_id=source_id,
                    scheme=scheme,
                    plan=plan,
                    doc_type=doc_type,
                    url=url,
                    enabled=enabled == "true",
                    notes=(row.get("notes") or "").strip(),
                )
            )

    if not sources:
        raise SourceValidationError("sources.csv contains no data rows")

    return [s for s in sources if s.enabled] if enabled_only else sources


def excluded_sources(path: Path | None = None) -> list[Source]:
    """Sources present in the manifest but deliberately not fetched (decision D5)."""
    return [s for s in load_sources(path) if not s.enabled]


def counts_by_scheme(sources: list[Source]) -> dict[str, int]:
    """Number of enabled sources per scheme."""
    counts: dict[str, int] = {}
    for source in sources:
        counts[source.scheme] = counts.get(source.scheme, 0) + 1
    return counts