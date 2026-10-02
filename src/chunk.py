"""Stage 2b - structure-aware chunking.

Implements the three tiers of `architecture.md` §4.5 and PRD §9.2. This runs
before any embedding exists (FR-I5): bad chunks degrade retrieval silently, so
the chunk set is written to disk and reviewed as a human artifact first.

Reads  : data/processed/<source_id>.normalized.json
Writes : artifacts/chunks.json, artifacts/chunks.txt
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import config
from src.models import Block, Chunk, Source
from src.normalize import blocks_path, load_blocks
from src.sources import load_sources
from src.state import read_state
from src.trace import utc_now

SCHEME_TITLES = {
    "large_cap": "Axis Large Cap Fund",
    "flexi_cap": "Axis Flexi Cap Fund",
    "elss": "Axis ELSS Tax Saver Fund",
    "midcap": "Axis Midcap Fund",
    "amc_wide": "Axis Mutual Fund",
}
PLAN_TITLES = {"direct": "Direct plan", "regular": "Regular plan", "n_a": ""}

# Header cells that add no meaning as a plan or column qualifier.
GENERIC_HEADERS = frozenset(
    {
        "value",
        "values",
        "amount",
        "amounts",
        "details",
        "particulars",
        "description",
        "remarks",
        "v",
        "",
    }
)

# PDF headers frequently read "SID"/"SAI"/"SIDE" as part of a real sentence.
SID_STANDALONE_RE = re.compile(r"\b(SID|SAI|SIDE)\b$", re.IGNORECASE)

MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
)


def scheme_title(scheme: str) -> str:
    return SCHEME_TITLES.get(scheme, scheme.replace("_", " ").title())


def document_title(source: Source) -> str:
    """Breadcrumb that makes every chunk self-describing (PRD §9.2)."""
    title = scheme_title(source.scheme)
    plan = PLAN_TITLES.get(source.plan, "")
    return f"{title} - {plan}" if plan else title


def _short(cell: str) -> str:
    return re.sub(r"\s+", " ", (cell or "").strip())


def _is_filler(cell: str) -> bool:
    text = _short(cell)
    return not text or text in {":", "-", "--", "|", "n.a.", "na", "nan"}


def _is_generic_header(cell: str) -> bool:
    return _short(cell).lower() in GENERIC_HEADERS


# ------------------------------------------------------------------ flattening


def _expand_label(row_label: str, column_header: str) -> str:
    """Rule of §4.5.1: an empty first-column cell borrows its column header."""
    if _short(row_label):
        return _short(row_label)
    return _short(column_header)


def _has_number(value: str) -> bool:
    return bool(re.search(r"\d", value or ""))


def _looks_numeric(value: str) -> bool:
    text = _short(value)
    return bool(text) and not re.search(r"[A-Za-z]{3,}", text)


def _detect_header_depth(rows: list[list[str]], width: int, limit: int = 2) -> int:
    """Count leading rows that read as header levels.

    A factsheet often carries a two-level header (a merged span above a pair of
    sub-columns). Treating the second level as data would fabricate pairs such as
    `Individuals/HUF (Option): Others`, so it is promoted into the header.
    """
    depth = 0
    for row in rows[:-1]:
        if depth >= limit:
            break
        cells = [_short(c) for c in row[:width]]
        if not any(cells):
            break
        if any(_has_number(c) for c in cells):
            break
        depth += 1
    return depth


def _join_header_levels(levels: list[list[str]], column: int) -> str:
    parts = [_short(level[column]) for level in levels if column < len(level)]
    return " ".join(p for p in parts if p and not _is_generic_header(p))


def _row_lines(label: str, values: list[tuple[str, str]], all_headers_blank: bool) -> list[str]:
    """Render one data row as `Label (Qualifier): value` lines.

    A wide row whose value columns have no headers keeps its values on one
    line joined by pipes, because inventing per-column labels for columns such
    as `13.74% | 12.72% | 12.13%` would fabricate structure that is not there.
    """
    label = _short(label)
    if all_headers_blank:
        joined = " | ".join(_short(v) for _, v in values if _short(v))
        return [f"{label}: {joined}"] if label and joined else []

    lines: list[str] = []
    grouped: list[str] = []
    for header, value in values:
        value = _short(value)
        if not value:
            continue
        qualifier = _short(header)
        if not qualifier:
            grouped.append(value)
            continue
        qualified = _expand_label(label, qualifier)
        lines.append(f"{_qualify(qualified, qualifier)}: {value}")
    if grouped and lines:
        lines.append(f"{label} (other values): " + " | ".join(grouped))
    elif grouped:
        lines.append(f"{label}: " + " | ".join(grouped))
    return lines


def _qualify(label: str, qualifier: str) -> str:
    """Append the column qualifier, without nesting redundant parentheses."""
    if not qualifier or qualifier.lower() in label.lower():
        return label
    if len(label) > 60:
        # A prose label already identifies itself; the qualifier would only add noise.
        return label
    if "(" in qualifier:
        # e.g. "IDCW (Per Unit) Cum IDCW" - avoid `label (a (b))`.
        return f"{label} - {qualifier}"
    return f"{label} ({qualifier})"


def flatten_table(rows: list[list[str]], heading_path: str) -> str:
    """Serialize a table to `Label (Qualifier): value` lines (§4.5.1).

    The matrix
        | | Direct Growth | Regular Growth |
        | Expense ratio | 0.95% | 1.95% |
    becomes the two unambiguous lines
        Expense ratio (Direct Growth): 0.95%
        Expense ratio (Regular Growth): 1.95%
    which is the mechanism that puts plan identity inside the embedded text.
    """
    cleaned = [[_short(cell) for cell in row] for row in rows if any(_short(c) for c in row)]
    if not cleaned:
        return _short(heading_path)
    # PDFs often render a pair as `Name of Mutual Fund | : | Axis Mutual Fund`.
    # Dropping the colon column recovers the intended two-column shape.
    cleaned = [[cell for i, cell in enumerate(row) if len(row) < 3 or cell != ":"] for row in cleaned]

    title = _short(heading_path)
    lines: list[str] = [title] if title else []

    width = max(len(row) for row in cleaned)
    padded_all = [row + [""] * (width - len(row)) for row in cleaned]

    if len(cleaned) == 1 or width <= 2:
        for row in padded_all:
            body = row[1:]
            if not any(not _is_filler(c) for c in body):
                if not _is_filler(row[0]):
                    lines.append(f"{_short(row[0])}")
                continue
            lines.extend(_row_lines(row[0], [("", c) for c in body], True))
        return "\n".join(lines)

    depth = _detect_header_depth(padded_all, width)
    levels = padded_all[:depth]
    headers = [_join_header_levels(levels, column) for column in range(width)]

    for row in padded_all[depth:]:
        label_index = next((i for i, cell in enumerate(row) if not _is_filler(cell)), 0)
        label = row[label_index]
        pairs = [(headers[i], row[i]) for i in range(label_index + 1, width) if not _is_filler(row[i])]
        if not pairs:
            if not _is_filler(label):
                lines.append(f"{_short(label)}")
            continue
        # A row with no numeric value is a caption or sub-header, not a record.
        if not any(_has_number(value) for _, value in pairs) and not _has_number(label):
            lines.append(" | ".join(_short(c) for c in row if not _is_filler(c)))
            continue
        all_blank = all(_is_generic_header(h) for h, _ in pairs)
        lines.extend(_row_lines(label, pairs, all_blank))

    return "\n".join(lines)


# --------------------------------------------------------------------- tiering


def _normalize_cell(value: str) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    return SID_STANDALONE_RE.sub("", text).strip(" .:-") if text.lower() in {"sid", "sai", "side"} else text


def _join_body(blocks: list[Block]) -> str:
    parts = []
    for block in blocks:
        if block.kind == "list":
            parts.append(f"- {_normalize_cell(block.text)}")
        elif block.text:
            parts.append(_normalize_cell(block.text))
    return "\n\n".join(p for p in parts if p)


def _sections(blocks: list[Block]) -> list[tuple[list[str], list[Block]]]:
    """Group prose blocks into heading-bounded sections.

    A table is a boundary, not part of a section: Tier 1 emits it atomically, so
    leaving it inside the section would either bridge prose across the table or
    force the whole section to be skipped. Headings open a new section, and the
    blocks before any heading form one untitled section.
    """
    sections: list[tuple[list[str], list[Block]]] = []
    current_path: list[str] = []
    current: list[Block] = []

    def flush() -> None:
        nonlocal current
        if current:
            sections.append((list(current_path), current))
            current = []

    for block in blocks:
        if block.kind == "heading":
            flush()
            current_path = list(block.heading_path) or [block.text]
            continue
        if block.kind == "table":
            flush()
            continue
        current.append(block)
    flush()
    return sections


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == " ":
        return text.split(" ")
    return re.split(separator, text)


def _split_units(text: str, separator: str) -> list[tuple[str, str]]:
    """Split into (unit, glue-after) pairs, keeping the literal separator text.

    The glue must be preserved verbatim: re-joining with the *pattern* instead of
    the matched text would write `(?<=[.!?])\\s+` into the chunk body.
    """
    if separator == " ":
        parts = text.split(" ")
        units = [(part, " ") for part in parts[:-1]]
        if parts:
            units.append((parts[-1], ""))
        return units

    parts = re.split(f"({separator})", text)
    units: list[tuple[str, str]] = []
    index = 0
    while index < len(parts):
        unit = parts[index]
        glue = parts[index + 1] if index + 1 < len(parts) else ""
        if unit:
            units.append((unit, glue))
        index += 2
    return units


def recursive_split(
    text: str,
    max_chars: int,
    overlap: int,
    separators: Iterable[str] | None = None,
) -> list[str]:
    """Window `text` to `max_chars`, preferring the highest-quality separator.

    Windows are packed on whole separator-delimited units, so no sentence or
    table row is ever cut in half. Overlap is then applied by moving whole
    trailing units across a boundary rather than by slicing raw characters,
    which keeps the overlap aligned to a real break.
    """
    separators = [s for s in (separators if separators is not None else config.RECURSIVE_SEPARATORS) if s]
    if not text.strip():
        return []
    if len(text) <= max_chars:
        return [text]

    for index, separator in enumerate(separators):
        units = _split_units(text, separator)
        if len(units) <= 1:
            continue

        windows: list[str] = []
        buffer = ""
        for unit, glue in units:
            if len(buffer) + len(unit) <= max_chars:
                buffer += unit + glue
                continue
            if buffer:
                windows.append(buffer)
            if len(unit) > max_chars:
                windows.extend(recursive_split(unit, max_chars, overlap, separators[index + 1 :]))
                buffer = ""
                continue
            buffer = unit + glue
        if buffer:
            windows.append(buffer)

        if windows:
            return _apply_overlap(windows, max_chars, overlap)

    return _hard_window(text, max_chars, overlap)


def _hard_window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Character stride used only when no separator yields a usable boundary."""
    step = max(1, max_chars - overlap)
    windows = [text[start : start + max_chars] for start in range(0, len(text), step)]
    return [window for window in windows if window.strip()]


def _tail_units(window: str, overlap: int) -> str:
    """The trailing whole units of `window` that fit inside `overlap` characters.

    The leading separator is dropped: the carry is re-attached in front of the
    window that continues the same text, so re-inserting it would leave the new
    chunk opening with a stray `. `.
    """
    for separator in ("\n\n", "\n", ". ", "; ", " "):
        index = window.rfind(separator)
        if index <= 0:
            continue
        tail = window[index + len(separator) :]
        if 0 < len(tail) <= overlap:
            return tail
    return ""


def _apply_overlap(windows: list[str], max_chars: int, overlap: int) -> list[str]:
    """Share whole units across a boundary so context survives both windows."""
    if overlap <= 0 or len(windows) < 2:
        return windows
    out = [windows[0]]
    for window in windows[1:]:
        previous = out[-1]
        if len(previous) + len(window) <= max_chars:
            out[-1:1] = [previous + window]
            continue
        carry = _tail_units(previous, overlap)
        if carry and len(carry) + len(window) <= max_chars:
            out[-1] = previous[: len(previous) - len(carry)]
            out.append(carry + window)
        else:
            out.append(window)
    return out


def _prefix(breadcrumb: list[str], body: str) -> str:
    return "\n".join(breadcrumb + [body.rstrip()]) if breadcrumb else body.rstrip()


def _make_chunk(
    text: str,
    *,
    breadcrumb: list[str],
    source: Source,
    index: int,
    is_table: bool,
    page_num: int | None,
    fetched_at: str,
    source_date: str | None,
    table_part: int = 1,
    table_parts: int = 1,
) -> Chunk:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    section_path = " > ".join(breadcrumb)
    return Chunk(
        chunk_id=f"{source.source_id}-c{index:03d}",
        text=text,
        metadata={
            "source_id": source.source_id,
            "source_url": source.url,
            "source_title": document_title(source),
            "doc_type": source.doc_type,
            "scheme": source.scheme,
            "plan": source.plan,
            "page_num": page_num,
            "section_path": section_path,
            "source_date": source_date,
            "fetched_at": fetched_at,
            "content_hash": digest,
            "chunk_index": index,
            "is_table": is_table,
            "char_len": len(text),
            "table_part": table_part,
            "table_parts": table_parts,
        },
    )


def merge_short_chunks(chunks: list[Chunk], min_chars: int, max_chars: int) -> list[Chunk]:
    """Fold undersized chunks into their neighbour, trying forward before backward.

    A table chunk is atomic (Tier 1), so nothing is ever merged into it, and a
    chunk is joined rather than dropped, so no content can be lost.
    """
    out: list[Chunk] = []
    for chunk in chunks:
        if len(chunk.text) < min_chars and out and not chunk.metadata["is_table"]:
            previous = out[-1]
            if not previous.metadata["is_table"] and len(previous.text) + len(chunk.text) <= max_chars:
                out[-1] = _rejoined(previous, chunk)
                continue
        out.append(chunk)

    # Second pass: a short chunk that could not fit backwards may fit forwards.
    index = 0
    while index < len(out) - 1:
        current = out[index]
        following = out[index + 1]
        if (
            len(current.text) < min_chars
            and not current.metadata["is_table"]
            and not following.metadata["is_table"]
            and len(current.text) + len(following.text) <= max_chars
        ):
            out[index] = _rejoined(current, following)
            del out[index + 1]
            continue
        index += 1

    for position, chunk in enumerate(out):
        chunk.metadata["chunk_index"] = position
        chunk.metadata["char_len"] = len(chunk.text)
    return out


def _rejoined(first: Chunk, second: Chunk) -> Chunk:
    """Concatenate two prose chunks, keeping the first one's identity."""
    return _make_chunk(
        f"{first.text}\n\n{second.text}".rstrip(),
        breadcrumb=_breadcrumb_of(first),
        source=_source_of(first),
        index=first.metadata["chunk_index"],
        is_table=False,
        page_num=first.metadata.get("page_num"),
        fetched_at=first.metadata.get("fetched_at") or "",
        source_date=first.metadata.get("source_date"),
        table_part=1,
        table_parts=1,
    )


def _breadcrumb_of(chunk: Chunk) -> list[str]:
    section = chunk.metadata.get("section_path") or ""
    return [part for part in section.split(" > ") if part]


def _source_of(chunk: Chunk) -> Source:
    return Source(
        source_id=chunk.metadata["source_id"],
        scheme=chunk.metadata["scheme"],
        plan=chunk.metadata["plan"],
        doc_type=chunk.metadata["doc_type"],
        url=chunk.metadata["source_url"],
        enabled=True,
    )


def _split_table_rows(flattened: str, budget: int) -> list[str]:
    """Group a flattened table into whole-row chunks of at most `budget` characters.

    `flatten_table` emits one line per table row, each already carrying its own
    label and values, so grouping by line boundary cannot orphan a label from its
    value. No header is repeated: the column headers are already fused into each
    row's label, and the breadcrumb prepended later re-identifies every part.
    """
    lines = [line for line in flattened.splitlines() if line.strip()]
    if not lines:
        return []
    if len(flattened) <= budget:
        return [flattened]

    parts: list[str] = []
    current: list[str] = []
    used = 0
    for row in lines:
        cost = len(row) + (1 if current else 0)
        if current and used + cost > budget:
            parts.append("\n".join(current))
            current, used = [], 0
            cost = len(row)
        current.append(row)
        used += cost
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_document(
    blocks: list[Block],
    source: Source,
    cfg: Any = None,
    *,
    fetched_at: str = "",
    source_date: str | None = None,
) -> list[Chunk]:
    """Tier 1 tables, Tier 2 heading-bounded sections, Tier 3 character windows."""
    cfg = cfg if cfg is not None else config
    max_chars = cfg.MAX_CHUNK_CHARS
    min_chars = cfg.MIN_CHUNK_CHARS
    overlap = cfg.CHUNK_OVERLAP
    doc_breadcrumb = [document_title(source)]

    drafts: list[tuple[str, list[str], bool, int | None, tuple[int, int]]] = []

    # Tier 1 - a table is never split. `flatten_table` has already fused every
    # label with its value into one line, so a row boundary is a line boundary
    # and a row can never be cut in half. A table that is still too large for the
    # embedder's window is emitted as several chunks of whole rows instead of one
    # oversized chunk that MiniLM would silently truncate.
    for block in blocks:
        if block.kind != "table" or not block.table_rows:
            continue
        flattened = flatten_table(block.table_rows, "")
        if not flattened.strip():
            continue
        breadcrumb = doc_breadcrumb + list(block.heading_path)
        overhead = len(_prefix(breadcrumb, ""))
        groups = _split_table_rows(flattened, max_chars - overhead)
        for position, group in enumerate(groups, start=1):
            drafts.append(
                (
                    _prefix(breadcrumb, group),
                    breadcrumb,
                    True,
                    block.page_num,
                    (position, len(groups)),
                )
            )

    # Tier 2 and Tier 3.
    for heading_path, section in _sections(blocks):
        body = _join_body(section)
        if not body.strip():
            continue
        breadcrumb = doc_breadcrumb + heading_path
        overhead = len(_prefix(breadcrumb, ""))
        if len(body) <= max_chars - overhead:
            drafts.append(
                (_prefix(breadcrumb, body), breadcrumb, False, _first_page(section), (1, 1))
            )
            continue
        # The breadcrumb is prepended after splitting, so it must come out of the
        # window budget or every chunk overshoots MAX_CHUNK_CHARS.
        budget = max(min_chars, max_chars - overhead)
        for window in recursive_split(body, budget, overlap, cfg.RECURSIVE_SEPARATORS):
            if window.strip():
                drafts.append(
                    (_prefix(breadcrumb, window), breadcrumb, False, _first_page(section), (1, 1))
                )

    chunks = [
        _make_chunk(
            text,
            breadcrumb=breadcrumb,
            source=source,
            index=index,
            is_table=is_table,
            page_num=page_num,
            fetched_at=fetched_at,
            source_date=source_date,
            table_part=part,
            table_parts=total,
        )
        for index, (text, breadcrumb, is_table, page_num, (part, total)) in enumerate(drafts)
    ]
    return merge_short_chunks(chunks, min_chars, max_chars)


def _first_page(blocks: list[Block]) -> int | None:
    for block in blocks:
        if block.page_num is not None:
            return block.page_num
    return None


# ------------------------------------------------------------------- artifacts


def source_date_for(source: Source) -> str | None:
    """Read a document date from the filename, driving `Last updated from sources:`."""
    name = source.filename.lower()
    pattern = re.compile(rf"\b({MONTHS})[-_ ]?(\d{{4}})\b")
    match = pattern.search(name)
    if not match:
        return None
    return f"{match.group(1).title()} {match.group(2)}"


def quality_summary(chunks: list[Chunk]) -> str:
    """The A-2 block appended to chunks.txt so the gate needs no script."""
    lengths = [chunk.char_len for chunk in chunks]
    lines = [
        "=" * 80,
        "QUALITY SUMMARY  (PRD 9.5 criteria, reported by src/chunk.py)",
        "=" * 80,
        f"total chunks          : {len(chunks)}",
    ]

    if not lengths:
        return "\n".join(lines + ["no chunks produced"]) + "\n"

    lines += ["", "-- chunks per scheme", ""]
    for scheme in config.SCHEMES:
        count = sum(1 for c in chunks if c.metadata["scheme"] == scheme)
        if count:
            lines.append(f"  {scheme:<12} {count:>6}")

    lines += ["", "-- chunks per plan", ""]
    for plan in config.PLANS:
        count = sum(1 for c in chunks if c.metadata["plan"] == plan)
        if count:
            lines.append(f"  {plan:<12} {count:>6}")

    lines += ["", "-- chunks per doc type", ""]
    for doc_type in sorted({c.metadata["doc_type"] for c in chunks}):
        count = sum(1 for c in chunks if c.metadata["doc_type"] == doc_type)
        lines.append(f"  {doc_type:<12} {count:>6}")

    median = statistics.median(lengths)
    in_band = sum(1 for n in lengths if config.MIN_CHUNK_CHARS <= n <= config.MAX_CHUNK_CHARS)
    outside = len(lengths) - in_band
    target_band = sum(1 for n in lengths if 400 <= n <= 900)
    lines += [
        "",
        "-- criterion 4: chunk length distribution",
        "",
        f"  median length                : {median:.0f} chars   (target 400-900: "
        f"{'PASS' if 400 <= median <= 900 else 'REVIEW'})",
        f"  min / max                    : {min(lengths)} / {max(lengths)}",
        f"  within [MIN, MAX]            : {in_band} ({in_band / len(lengths) * 100:.1f}%)",
        f"  within [400, 900]            : {target_band} ({target_band / len(lengths) * 100:.1f}%)",
        f"  outside [MIN, MAX]           : {outside} ({outside / len(lengths) * 100:.1f}%, target < 5%)",
        f"  below MIN (atomic tables)    : {sum(1 for n in lengths if n < config.MIN_CHUNK_CHARS)}",
        "",
        "  histogram (100-char buckets):",
    ]
    buckets: dict[int, int] = {}
    for length in lengths:
        buckets[length // 100 * 100] = buckets.get(length // 100 * 100, 0) + 1
    for start in sorted(buckets):
        lines.append(f"    {start:>5}-{start + 99:<5} {'#' * min(60, buckets[start])} {buckets[start]}")

    lines += [
        "",
        "-- criterion 1: chunks with no identifiable scheme and plan",
        "",
    ]
    unidentified = [c for c in chunks if not identifies_scheme_plan(c)]
    pct = (len(chunks) - len(unidentified)) / len(chunks) * 100
    lines.append(f"  identifiable                 : {len(chunks) - len(unidentified)} ({pct:.1f}%, target >= 90%)")
    lines.append(f"  unidentified                 : {len(unidentified)}")
    for chunk in unidentified[:10]:
        lines.append(f"    {chunk.chunk_id}  {chunk.metadata['section_path'][:70]}")

    table_chunks = sum(1 for c in chunks if c.metadata["is_table"])
    lines += [
        "",
        "-- structural checks",
        "",
        f"  table chunks (atomic)        : {table_chunks}",
        f"  max chunk length             : {max(lengths)} (cap {config.MAX_CHUNK_CHARS})",
        f"  sources covered              : {len({c.metadata['source_id'] for c in chunks})}",
        "",
    ]
    return "\n".join(lines) + "\n"


def identifies_scheme_plan(chunk: Chunk) -> bool:
    """Criterion 1: the chunk text names its scheme, or is explicitly amc_wide."""
    metadata = chunk.metadata
    if metadata["scheme"] == "amc_wide":
        return True
    text = chunk.text
    title = scheme_title(metadata["scheme"])
    if title.lower() in text.lower():
        return True
    alt = title.replace("Axis ", "").lower()
    return alt in text.lower()


def write_chunks_json(chunks: list[Chunk], path: Path | None = None) -> Path:
    destination = Path(path) if path else config.CHUNKS_JSON
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": utc_now(),
        "params": {
            "max_chunk_chars": config.MAX_CHUNK_CHARS,
            "min_chunk_chars": config.MIN_CHUNK_CHARS,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "table_overlap": config.TABLE_OVERLAP,
            "separators": config.RECURSIVE_SEPARATORS,
            "embed_model": config.EMBED_MODEL,
        },
        "chunk_count": len(chunks),
        "chunks": [{"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata} for c in chunks],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def write_chunks_txt(chunks: list[Chunk], path: Path | None = None) -> Path:
    """Write the human-review artifact, with the quality summary appended (A-2)."""
    destination = Path(path) if path else config.CHUNKS_TXT
    destination.parent.mkdir(parents=True, exist_ok=True)
    rule = "=" * 80
    dash = "-" * 80
    parts: list[str] = []
    for chunk in chunks:
        m = chunk.metadata
        parts.append(
            "\n".join(
                [
                    rule,
                    f"CHUNK {chunk.chunk_id}  |  index {m['chunk_index']}  |  {m['char_len']} chars"
                    f"  |  table={str(bool(m['is_table'])).lower()}",
                    f"SECTION   {m['section_path']}",
                    f"SCHEME    {m['scheme']:<14} PLAN  {m['plan']:<8} DOCTYPE  {m['doc_type']}",
                    f"SOURCE    {m['source_url']}",
                    f"DATE      source_date={m['source_date'] or '<unset>'}"
                    f"   fetched_at={m['fetched_at'] or '<unset>'}"
                    f"   hash={m['content_hash'][:12]}",
                    dash,
                    chunk.text,
                ]
            )
        )
    parts.append(quality_summary(chunks))
    destination.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return destination


def load_all_normalized(sources: list[Source]) -> dict[str, list[Block]]:
    found: dict[str, list[Block]] = {}
    for source in sources:
        path = blocks_path(source, normalized=True)
        if not path.exists():
            path = blocks_path(source, normalized=False)
        if path.exists():
            found[source.source_id] = load_blocks(path)[1]
    return found


def chunk_all_sources(sources: list[Source]) -> list[Chunk]:
    """Chunk every enabled source and write both chunk artifacts.

    Shared by `python -m src.chunk` and `src.ingest`, so the orchestrator cannot
    drift from the standalone stage.
    """
    state = read_state().get("sources", {})
    block_map = load_all_normalized(sources)
    chunks: list[Chunk] = []
    for source in sources:
        blocks = block_map.get(source.source_id)
        if not blocks:
            print(f"  {'skipped':>26} {source.source_id} no normalized blocks")
            continue
        produced = chunk_document(
            blocks,
            source,
            config,
            fetched_at=str(state.get(source.source_id, {}).get("fetched_at") or ""),
            source_date=source_date_for(source),
        )
        chunks.extend(produced)
        tables = sum(1 for c in produced if c.metadata["is_table"])
        print(
            f"  {'ok':>26} {source.source_id} {source.scheme:<11} "
            f"{len(produced):>5} chunks {tables:>4} tables"
        )
    if not chunks:
        raise RuntimeError(
            "chunking produced 0 chunks; refusing to overwrite "
            f"{config.CHUNKS_JSON} with an empty corpus"
        )
    json_path = write_chunks_json(chunks)
    txt_path = write_chunks_txt(chunks)
    print(f"\n{len(chunks)} chunks written to {json_path}")
    print(f"human-review artifact written to {txt_path}")
    return chunks


def main() -> int:
    config.ensure_dirs()
    sources = load_sources(enabled_only=True)
    print(f"chunking {len(sources)} sources")
    chunk_all_sources(sources)
    print("\nreview artifacts/chunks.txt before starting P3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())