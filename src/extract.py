"""Stage 3 of ingestion: HTML/PDF to structured Block lists.

Tables and heading paths are preserved here because the chunker (P2) depends
on them; a fee table that loses its rows cannot be recovered downstream.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import config
from src.models import Block, Source
from src.sources import load_sources
from src.trace import utc_now

HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
BOILERPLATE_SELECTORS = (
    "nav",
    "footer",
    "header",
    "aside",
    "[role=navigation]",
    "[class*=cookie]",
    "[class*=consent]",
    "[class*=breadcrumb]",
    "[class*=cta]",
    "[class*=footer]",
    "[class*=header]",
    "[class*=sidebar]",
    "[class*=popup]",
    "[class*=modal]",
    "[id*=cookie]",
)
NUMBERED_HEADING = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+\S")
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
MIN_HEADING_CHARS = 2
MAX_HEADING_CHARS = 120


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", (text or "")).strip()


def _is_meaningful(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if not re.search(r"[A-Za-z0-9]", stripped):
        return False
    return True


# --------------------------------------------------------------------------- html


def _strip_boilerplate(soup) -> None:
    for selector in BOILERPLATE_SELECTORS:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            continue


def _table_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([_clean(cell.get_text(" ", strip=True)) for cell in cells])
    width = max((len(row) for row in rows), default=0)
    if len(rows) >= MIN_TABLE_ROWS and width >= MIN_TABLE_COLS:
        return [row + [""] * (width - len(row)) for row in rows]
    return []


SKIP_TAGS = frozenset(
    {"script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed"}
)
MAX_BLOCK_CHARS = 4000


def _walk_html(node, heading_path: list[str], blocks: list[Block]) -> None:
    from bs4 import NavigableString, Tag

    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if _is_meaningful(text) and len(text) <= MAX_BLOCK_CHARS:
                blocks.append(Block(kind="text", text=text, heading_path=list(heading_path)))
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name.lower()

        if name in SKIP_TAGS:
            continue

        if name in HEADING_TAGS:
            text = _clean(child.get_text(" ", strip=True))
            if _is_meaningful(text) and len(text) <= MAX_HEADING_CHARS:
                level = int(name[1])
                depth = max(1, min(level - 1, 3))
                blocks.append(Block(kind="heading", text=text, heading_path=list(heading_path)))
                blocks[-1].heading_path = list(heading_path) + [text]
                heading_path = blocks[-1].heading_path[-depth:]
                continue

        if name == "table":
            rows = _table_rows(child)
            if rows:
                blocks.append(
                    Block(kind="table", text="", heading_path=list(heading_path), table_rows=rows)
                )
                continue

        if name in ("ul", "ol"):
            for item in child.find_all("li", recursive=False):
                text = _clean(item.get_text(" ", strip=True))
                if _is_meaningful(text) and len(text) <= MAX_BLOCK_CHARS:
                    blocks.append(Block(kind="list", text=text, heading_path=list(heading_path)))
            continue

        if name in ("p", "div", "section", "article", "blockquote", "td"):
            nested_tables = child.find_all("table")
            if nested_tables:
                _walk_html(child, heading_path, blocks)
                continue
            text = _clean(child.get_text(" ", strip=True))
            if _is_meaningful(text) and len(text) <= MAX_BLOCK_CHARS:
                blocks.append(Block(kind="text", text=text, heading_path=list(heading_path)))
                continue

        _walk_html(child, heading_path, blocks)


def extract_html(path: Path, source: Source | None = None) -> list[Block]:
    """Parse HTML into blocks.

    The visible DOM is parsed for prose, and the embedded RSC payload and
    JSON-LD FAQ block are parsed for facts. On Axis scheme pages the facts
    exist only in the payload, so the DOM pass alone yields no fee data.
    """
    from bs4 import BeautifulSoup

    from src.embedded import (
        card_blocks,
        extract_faq_pairs,
        extract_facts,
        fact_rows,
        tax_rows,
    )

    html = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[Block] = []

    if source is not None:
        scheme = _titleise(source.scheme)
        plan = {"direct": "Direct", "regular": "Regular"}.get(source.plan, "All plans")
        heading = [f"{scheme} - {plan}"]

        facts = extract_facts(html)
        if facts:
            rows = fact_rows(facts, scheme, plan)
            if rows:
                blocks.append(
                    Block(kind="table", text="", heading_path=heading + ["Key facts"], table_rows=rows)
                )

        rows = tax_rows(html)
        if rows:
            blocks.append(
                Block(
                    kind="table",
                    text="",
                    heading_path=heading + ["Tax implications"],
                    table_rows=rows,
                )
            )

        if source.doc_type in PROCEDURAL_DOC_TYPES:
            for path_heading, text in card_blocks(html, heading + ["Services and downloads"]):
                blocks.append(Block(kind="text", text=text, heading_path=path_heading))

        for question, answer in extract_faq_pairs(html):
            blocks.append(
                Block(
                    kind="text",
                    text=f"FAQ: {question} {answer}",
                    heading_path=heading + ["Frequently asked questions"],
                )
            )

    prose = _extract_dom_blocks(html)
    if not blocks and not prose:
        raise ValueError("no text or embedded facts recovered")
    return blocks + prose


def _titleise(scheme: str) -> str:
    names = {
        "large_cap": "Axis Large Cap Fund",
        "flexi_cap": "Axis Flexi Cap Fund",
        "elss": "Axis ELSS Tax Saver Fund",
        "midcap": "Axis Midcap Fund",
        "amc_wide": "Axis Mutual Fund",
    }
    return names.get(scheme, scheme.replace("_", " ").title())


def _extract_dom_blocks(html: str) -> list[Block]:
    """Walk the source DOM, which preserves headings and tables.

    trafilatura is deliberately not the primary path: its XML output collapses
    tables and headings into a single paragraph, and both are load-bearing for
    the chunker. It is used only when the DOM walk yields nothing.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    _strip_boilerplate(soup)
    blocks: list[Block] = []
    _walk_html(soup, [], blocks)
    if blocks:
        return blocks

    try:
        import trafilatura

        xml = trafilatura.extract(
            html, output_format="xml", include_comments=False, include_tables=True
        )
    except Exception:
        xml = None

    if not xml:
        return []

    fallback: list[Block] = []
    try:
        soup = BeautifulSoup(xml, "lxml")
        _strip_boilerplate(soup)
        _walk_html(soup, [], fallback)
    except Exception:
        return []
    return fallback


# ---------------------------------------------------------------------------- pdf


def _pdf_heading_level(text: str, font_size: float, body_size: float) -> int | None:
    if not _is_meaningful(text) or len(text) > MAX_HEADING_CHARS:
        return None
    if NUMBERED_HEADING.match(text):
        return 2
    if font_size >= body_size * 1.15:
        return 1
    if text.isupper() and len(text) > 3:
        return 2
    return None


def _pdf_table_rows(page) -> list[list[str]]:
    try:
        found = page.find_tables()
    except Exception:
        return []

    rows: list[list[str]] = []
    for table in getattr(found, "tables", []) or []:
        try:
            extracted = table.extract()
        except Exception:
            continue
        if not extracted:
            continue
        cleaned = [[_clean(cell or "") for cell in row] for row in extracted if any(row)]
        width = max((len(row) for row in cleaned), default=0)
        if len(cleaned) >= MIN_TABLE_ROWS and width >= MIN_TABLE_COLS:
            rows.append([row + [""] * (width - len(row)) for row in cleaned])
    return rows


def extract_pdf(path: Path) -> list[Block]:
    """Parse a PDF page by page, detecting headings by font size and numbering."""
    import pymupdf

    blocks: list[Block] = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            try:
                text_dict = page.get_text("dict")
            except Exception:
                continue

            sizes: list[float] = []
            for line in text_dict.get("blocks", []):
                for line_item in line.get("lines", []):
                    for span in line_item.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(round(float(span.get("size", 0)), 1))
            body_size = _mode(sizes) or 10.0

            heading_path: list[str] = []

            for line_block in text_dict.get("blocks", []):
                if line_block.get("type") != 0:
                    continue

                spans = [
                    span
                    for line in line_block.get("lines", [])
                    for span in line.get("spans", [])
                    if span.get("text", "").strip()
                ]
                if not spans:
                    continue

                text = _clean("".join(span.get("text", "") for span in spans))
                font_size = max(float(span.get("size", 0)) for span in spans)

                level = _pdf_heading_level(text, font_size, body_size)
                if level is not None:
                    heading_path = (heading_path + [text])[-level:]
                    blocks.append(
                        Block(
                            kind="heading",
                            text=text,
                            heading_path=list(heading_path),
                            page_num=page_number,
                            font_size=font_size,
                        )
                    )
                    continue

                if _is_meaningful(text):
                    blocks.append(
                        Block(kind="text", text=text, heading_path=list(heading_path), page_num=page_number)
                    )

            for rows in _pdf_table_rows(page):
                blocks.append(
                    Block(
                        kind="table",
                        text="",
                        heading_path=list(heading_path),
                        page_num=page_number,
                        table_rows=rows,
                    )
                )

    return blocks


def _mode(values: list[float]) -> float | None:
    if not values:
        return None
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


# ----------------------------------------------------------------------- entry


def extract_blocks(path: Path, doc_type: str, source: Source | None = None) -> list[Block]:
    """Dispatch on the file signature, falling back to doc_type."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.read_bytes()[:5] == b"%PDF-":
        return extract_pdf(path)
    if doc_type in ("sid", "kim", "factsheet"):
        return extract_pdf(path)
    return extract_html(path, source)


def write_blocks_json(source: Source, blocks: list[Block], raw_result_status: str = "ok") -> Path:
    """Persist blocks to data/processed/<source_id>.blocks.json."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = config.PROCESSED_DIR / f"{source.source_id}.blocks.json"
    payload = {
        "source_id": source.source_id,
        "scheme": source.scheme,
        "plan": source.plan,
        "doc_type": source.doc_type,
        "url": source.url,
        "extracted_at": utc_now(),
        "fetch_status": raw_result_status,
        "block_count": len(blocks),
        "char_count": sum(len(b.text) for b in blocks if b.text),
        "table_count": sum(1 for b in blocks if b.kind == "table"),
        "page_count": max((b.page_num or 0 for b in blocks), default=0),
        "blocks": [asdict(b) for b in blocks],
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


FACT_MARKERS = (
    "expense ratio",
    "exit load",
    "benchmark",
    "minimum sip",
    "minimum investment",
    "minimum lump",
    "riskometer",
    "lock in",
    "lock-in",
)

SCHEME_PAGE_TYPES = ("scheme_page", "efactsheet", "homepage", "downloads")

# Cards are site-wide furniture, meaningful only where procedural help is the point.
PROCEDURAL_DOC_TYPES = ("downloads", "homepage")


def _char_count(blocks: list[Block]) -> int:
    return sum(len(b.text) for b in blocks if b.text)


def _facts_present(blocks: list[Block]) -> bool:
    haystack = " ".join(b.text.lower() for b in blocks if b.text)
    rows = " ".join(
        " ".join(cell.lower() for cell in row)
        for b in blocks
        if b.table_rows
        for row in b.table_rows
    )
    combined = haystack + " " + rows
    return any(marker in combined for marker in FACT_MARKERS)


def classify(source: Source, blocks: list[Block], fetch_status: str) -> str:
    """Choose an extract flag.

    A page that yielded recognised facts is never flagged, however short it is:
    a concise fee summary is legitimate content, whereas a long page with no
    facts at all is the client-rendered shell described by risk R1.
    """
    if fetch_status != "ok":
        return fetch_status
    if not blocks:
        return "extract_failed"
    if _facts_present(blocks):
        return "ok"
    thin = _char_count(blocks) < config.MIN_EXTRACTED_CHARS
    if not any(b.kind == "table" for b in blocks):
        if source.doc_type in SCHEME_PAGE_TYPES or thin:
            return "suspected_client_rendered"
    return "ok"


def main() -> int:
    config.ensure_dirs()
    sources = load_sources(enabled_only=True)

    fetch_rows = {r["source_id"]: r for r in build_fetch_rows(sources)}
    print(f"extracting {len(sources)} sources")

    rows: list[dict] = []
    for source in sources:
        fetch_info = fetch_rows.get(source.source_id, {})
        raw = config.RAW_DIR / source.source_id / source.filename

        if fetch_info.get("status") != "ok" or not raw.exists():
            flag = fetch_info.get("status", "fetch_failed")
            detail = fetch_info.get("error", "not fetched")
            print(f"  [{flag:>26}] {source.source_id} {source.scheme:<11} {detail}")
            rows.append(
                {
                    "source_id": source.source_id,
                    "scheme": source.scheme,
                    "plan": source.plan,
                    "doc_type": source.doc_type,
                    "url": source.url,
                    "flag": flag,
                    "http_status": fetch_info.get("http_status", ""),
                    "byte_size": fetch_info.get("byte_size", ""),
                    "error": detail,
                    "extracted_chars": 0,
                }
            )
            continue

        try:
            blocks = extract_blocks(raw, source.doc_type, source)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            print(f"  [{'extract_failed':>26}] {source.source_id} {source.scheme:<11} {detail}")
            rows.append(
                {
                    "source_id": source.source_id,
                    "scheme": source.scheme,
                    "plan": source.plan,
                    "doc_type": source.doc_type,
                    "url": source.url,
                    "flag": "extract_failed",
                    "http_status": fetch_info.get("http_status", ""),
                    "byte_size": fetch_info.get("byte_size", ""),
                    "error": detail,
                    "extracted_chars": 0,
                }
            )
            continue

        write_blocks_json(source, blocks)
        chars = _char_count(blocks)
        tables = sum(1 for b in blocks if b.kind == "table")
        flag = classify(source, blocks, "ok")
        print(
            f"  [{flag:>26}] {source.source_id} {source.scheme:<11} "
            f"{len(blocks):>5} blocks {chars:>7} chars {tables:>3} tables"
        )
        rows.append(
            {
                "source_id": source.source_id,
                "scheme": source.scheme,
                "plan": source.plan,
                "doc_type": source.doc_type,
                "url": source.url,
                "flag": flag,
                "http_status": fetch_info.get("http_status", ""),
                "byte_size": fetch_info.get("byte_size", ""),
                "extracted_chars": chars,
                "block_count": len(blocks),
                "table_count": tables,
            }
        )

    from src.report import write_report

    path = write_report(rows)
    print(f"\nreport written to {path}")
    return 0


def build_fetch_rows(sources: list[Source]) -> list[dict]:
    """Read fetch state from disk so extraction can run standalone."""
    from src.state import read_state

    state = read_state()
    out: list[dict] = []
    for source in sources:
        record = state.get("sources", {}).get(source.source_id)
        if record is None:
            path = config.RAW_DIR / source.source_id / source.filename
            record = {
                "status": "ok" if path.exists() and path.stat().st_size else "fetch_failed",
                "http_status": None,
                "byte_size": path.stat().st_size if path.exists() else 0,
                "error": "not fetched" if not path.exists() else "",
            }
        out.append({"source_id": source.source_id, **record})
    return out


if __name__ == "__main__":
    raise SystemExit(main())