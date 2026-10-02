"""Stage 2a - normalization of extracted blocks.

Applies the six rules of `architecture.md` §4.4 in order. Each rule is
idempotent, so re-running the stage on already-normalized output is a no-op.

Reads  : data/processed/<source_id>.blocks.json
Writes : data/processed/<source_id>.normalized.json

The extraction artifact is deliberately left untouched: it is the forensic
record of what the raw document contained, and normalization is a separate,
re-runnable stage.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import config
from src.models import Block, Source
from src.sources import load_sources
from src.trace import utc_now

# Rule 4: a line must both recur across pages and look like furniture.
FURNITURE_RE = re.compile(r"page\s*\d+|axis mutual fund|www\.|sid$|^\d+$", re.IGNORECASE)

# Page numbering and bare page numbers differ on every page, so they never recur
# verbatim. Compare a canonical key instead of the raw line.
PAGE_NUMBER_RE = re.compile(r"^page\s*\d+(\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)
BARE_NUMBER_RE = re.compile(r"^\d+$")

# Rule 5: HTML regions stripped before extraction, re-checked here on the text
# that survived, because extract.py and the DOM differ between doc types.
BOILERPLATE_RE = re.compile(
    r"(skip to main content|request call back|request a callback|click here|read more"
    r"|cookie|consent banner|breadcrumb|share (this|on) (page|facebook|twitter|linkedin)"
    r"|sign in|log ?in|register now|open account|disclaimer: this|all rights reserved"
    r"|privacy policy|terms (and|&) conditions|do not sell my personal information"
    r"|download (our|the) (mobile )?app|invest now)",
    re.IGNORECASE,
)

ORPHAN_RE = re.compile(r"^[\d\s.,:;/()\[\]%+\-*#₹$€£]*$")

# Axis UI chrome that survives extraction but carries no fund fact. The tab bar
# is a run of capitalised words with no sentence punctuation.
NAV_TAB_RE = re.compile(r"^(?:[A-Z][A-Za-z0-9']{2,20}[ ]){3,}[A-Z][A-Za-z0-9']{2,20}$")
UI_CHROME_RE = re.compile(
    r"^(show \d* ?more|view all( \w+)?|loading pdf viewer|invest now|popular choice"
    r"|similar funds|get in touch|apply now|html|updated as on:?|as on:?)\s*\.{0,3}$",
    re.IGNORECASE,
)

# Table-of-contents dot leaders: "Axis Midcap Fund ....... 14". Long runs of dots are
# never facts, and a single such line can run to thousands of characters.
DOT_LEADER_RE = re.compile(r"\.{20,}")

# A block whose real content is shorter than this is treated as pure boilerplate.
BOILERPLATE_MIN_REMAIN = 25

# Ranking tables end each row with a call-to-action button rendered as a cell.
# `Axis Multicap Fund Equity Popular Choice` is a real fund name, so only the
# trailing button label is removed and never a name that merely contains it.
CTA_CELL_RE = re.compile(r"\s*invest\s+now\s*$", re.IGNORECASE)


def normalize_blocks(blocks: list[Block]) -> list[Block]:
    """Apply §4.4 rules 1-6 in order and return the cleaned blocks."""
    out = _rule_unicode(list(blocks))
    out = _rule_dehyphenate(out)
    out = _rule_whitespace(out)
    out = _rule_pdf_furniture(out)
    out = _rule_boilerplate(out)
    return _rule_orphan_lines(out)


# --------------------------------------------------------------------- rules


def _rule_unicode(blocks: list[Block]) -> list[Block]:
    """NFC, NBSP to space, soft hyphen removed."""
    out = []
    for block in blocks:
        text = unicodedata.normalize("NFC", block.text or "")
        text = text.replace("\u00a0", " ").replace("\u00ad", "")
        rows = None
        if block.table_rows:
            rows = [
                [CTA_CELL_RE.sub("", unicodedata.normalize("NFC", (c or "").replace("\u00a0", " ").replace("\u00ad", "")))
                 for c in row]
                for row in block.table_rows
            ]
        heading = [unicodedata.normalize("NFC", h).replace("\u00a0", " ") for h in block.heading_path]
        out.append(replace(block, text=text, table_rows=rows, heading_path=heading))
    return out


def _rule_dehyphenate(blocks: list[Block]) -> list[Block]:
    """Join words broken across a line or block boundary by a trailing hyphen.

    PDF extraction yields one block per line, so the `(\\w)-\\n(\\w)` pattern of
    §4.4 rule 2 has to be applied across consecutive blocks, not just inside one.
    """
    out: list[Block] = []
    for block in blocks:
        if out and block.kind in ("text", "list") and block.page_num == out[-1].page_num:
            previous = out[-1]
            if (
                previous.kind in ("text", "list")
                and previous.text.endswith("-")
                and block.text[:1].isalnum()
            ):
                out[-1] = replace(
                    previous, text=previous.text[:-1] + block.text
                )
                continue
        out.append(block)
    return out


def _rule_whitespace(blocks: list[Block]) -> list[Block]:
    """Collapse 3+ blank lines to 2, and runs of spaces or tabs to one space."""
    out = []
    for block in blocks:
        text = re.sub(r"[ \t]+", " ", block.text or "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        out.append(replace(block, text=text.strip("\n")))
    return out


def _pages_from_blocks(blocks: list[Block]) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.page_num is None:
            continue
        pages.setdefault(block.page_num, []).append(block.text or "")
    return {page: "\n".join(lines) for page, lines in pages.items()}


def _furniture_key(line: str) -> str:
    """Canonical key so `Page 1` .. `Page 9` count as the same recurring line.

    The key is itself furniture-shaped so FURNITURE_RE still recognises it.
    """
    if PAGE_NUMBER_RE.match(line):
        return "page 0"
    if BARE_NUMBER_RE.match(line):
        return "0"
    return line


def _repeated_furniture_keys(pages: dict[int, str], threshold: float) -> set[str]:
    """Furniture keys that recur on more than `threshold` of the pages."""
    if len(pages) < 2:
        return set()
    seen: Counter[str] = Counter()
    for text in pages.values():
        for line in {_furniture_key(stripped) for stripped in (raw.strip() for raw in text.splitlines()) if stripped}:
            seen[line] += 1
    limit = len(pages) * threshold
    return {key for key, count in seen.items() if count > limit and FURNITURE_RE.search(key)}


def _repeated_furniture_lines(pages: dict[int, str], threshold: float) -> set[str]:
    """Lines that recur on more than `threshold` of pages and match §4.4 rule 4."""
    return _repeated_furniture_keys(pages, threshold)


def drop_repeated_pdf_lines(pages: dict[int, str], threshold: float = 0.5) -> dict[int, str]:
    """Remove recurring PDF page furniture such as headers, footers and page numbers."""
    doomed = _repeated_furniture_keys(pages, threshold)
    if not doomed:
        return dict(pages)
    cleaned: dict[int, str] = {}
    for page, text in pages.items():
        kept = [
            raw
            for raw in text.splitlines()
            if raw.strip() and _furniture_key(raw.strip()) not in doomed
        ]
        cleaned[page] = "\n".join(kept)
    return cleaned


def _rule_pdf_furniture(blocks: list[Block]) -> list[Block]:
    """Drop blocks whose entire text is recurring PDF furniture."""
    pages = _pages_from_blocks(blocks)
    doomed = _repeated_furniture_keys(pages, threshold=0.5)
    if not doomed:
        return blocks
    return [
        block
        for block in blocks
        if not (
            block.page_num is not None
            and block.text.strip()
            and _furniture_key(block.text.strip()) in doomed
        )
    ]


def _strip_boilerplate_text(text: str) -> str:
    """Remove furniture from inside a block, keeping the facts that share the block.

    Extracted HTML often packs a CTA into the same element as real facts, e.g.
    `Exit load NIL Tax implication Click here to view the tax implication.`
    Dropping the whole block would silently discard the exit load.
    """
    kept_lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            # Blank lines are structural. Tier 3 splitting relies on "\n\n" to
            # avoid run-ons such as `...Axis Large Cap FundFAQ: ...`.
            kept_lines.append("")
            continue
        if DOT_LEADER_RE.search(stripped):
            # Keep a dot-leader line only if it also carries real prose.
            residue = DOT_LEADER_RE.sub(" ", stripped)
            residue = re.sub(r"[\s.]{2,}", " ", residue).strip()
            if len(residue) < BOILERPLATE_MIN_REMAIN:
                continue
            kept_lines.append(residue)
            continue
        if BOILERPLATE_RE.search(stripped) and len(stripped) < BOILERPLATE_MIN_REMAIN:
            continue
        cleaned = BOILERPLATE_RE.sub(" ", stripped)
        if cleaned != stripped:
            cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:|,")
        if cleaned:
            kept_lines.append(cleaned)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()


def _rule_boilerplate(blocks: list[Block]) -> list[Block]:
    """Drop nav, footer, cookie, breadcrumb and CTA residue without losing facts."""
    out: list[Block] = []
    for block in blocks:
        if BOILERPLATE_RE.search(" ".join(block.heading_path)):
            continue
        if block.kind in ("heading", "text", "list"):
            raw = (block.text or "").strip()
            if raw and (NAV_TAB_RE.match(raw) or UI_CHROME_RE.match(raw)):
                continue
            if block.kind == "text" and raw.lower().startswith("popular choice"):
                continue
            stripped = _strip_boilerplate_text(block.text)
            if not stripped and (block.text or "").strip():
                continue
            if len(stripped) < BOILERPLATE_MIN_REMAIN and (block.text or "").strip() != stripped:
                # Nothing but furniture survived, so the whole block was furniture.
                if BOILERPLATE_RE.search(block.text or ""):
                    continue
            out.append(replace(block, text=stripped))
            continue
        out.append(block)
    return out


def _rule_orphan_lines(blocks: list[Block]) -> list[Block]:
    """Drop blocks that are only numbers or punctuation."""
    out = []
    for block in blocks:
        text = (block.text or "").strip()
        if text and ORPHAN_RE.match(text) and not any(ch.isalpha() for ch in text):
            continue
        out.append(block)
    return out


# ------------------------------------------------------------------ artifacts


def blocks_path(source: Source, normalized: bool = True) -> Path:
    suffix = ".normalized.json" if normalized else ".blocks.json"
    return config.PROCESSED_DIR / f"{source.source_id}{suffix}"


def load_blocks(path: Path) -> tuple[dict, list[Block]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    blocks = [
        Block(
            kind=row["kind"],
            text=row["text"],
            heading_path=list(row.get("heading_path") or []),
            page_num=row.get("page_num"),
            table_rows=row.get("table_rows"),
            font_size=row.get("font_size"),
        )
        for row in data["blocks"]
    ]
    return data, blocks


def write_normalized(source: Source, blocks: list[Block]) -> Path:
    destination = blocks_path(source, normalized=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_id": source.source_id,
        "scheme": source.scheme,
        "plan": source.plan,
        "doc_type": source.doc_type,
        "url": source.url,
        "normalized_at": utc_now(),
        "block_count": len(blocks),
        "char_count": sum(len(b.text) for b in blocks if b.text),
        "table_count": sum(1 for b in blocks if b.kind == "table"),
        "page_count": max((b.page_num or 0 for b in blocks), default=0),
        "blocks": [asdict(b) for b in blocks],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def normalize_source(source: Source) -> tuple[int, int] | None:
    """Normalize one source, returning (before, after) block counts."""
    raw = blocks_path(source, normalized=False)
    if not raw.exists():
        return None
    _, blocks = load_blocks(raw)
    cleaned = normalize_blocks(blocks)
    write_normalized(source, cleaned)
    return len(blocks), len(cleaned)


def main() -> int:
    config.ensure_dirs()
    sources = load_sources(enabled_only=True)

    print(f"normalizing {len(sources)} sources")
    total_before = total_after = 0
    skipped: list[str] = []

    for source in sources:
        result = normalize_source(source)
        if result is None:
            skipped.append(source.source_id)
            continue
        before, after = result
        total_before += before
        total_after += after
        removed = before - after
        note = f"{removed:>4} removed" if removed else "     --"
        print(f"  {source.source_id} {source.scheme:<11} {before:>6} -> {after:<6} blocks  {note}")

    print(
        f"\n{total_before} -> {total_after} blocks "
        f"({total_before - total_after} removed across {total_after and len(sources) - len(skipped)} sources)"
    )
    if skipped:
        print(f"skipped (no extraction artifact): {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def iter_blocks(path: Path) -> Iterable[Block]:
    _, blocks = load_blocks(path)
    return blocks