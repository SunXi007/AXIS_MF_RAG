"""Tests for HTML and PDF extraction to Block lists."""

from __future__ import annotations

import json

import pymupdf
import pytest

from src.extract import classify, extract_blocks, extract_html, extract_pdf
from src.models import Source

SCHEME_SOURCE = Source(
    source_id="s01",
    scheme="large_cap",
    plan="direct",
    doc_type="scheme_page",
    url="https://example.com/fund/direct",
    enabled=True,
)

HTML_WITH_TABLE = """
<html><body>
  <nav>Home About Contact</nav>
  <h1>Axis Large Cap Fund</h1>
  <h2>Direct Growth</h2>
  <table>
    <tr><th>Fact</th><th>Value</th></tr>
    <tr><td>Expense ratio</td><td>0.92%</td></tr>
    <tr><td>Exit load</td><td>Nil</td></tr>
  </table>
  <h2>Performance</h2>
  <p>Benchmark is the BSE 100 TRI index.</p>
  <ul><li>Capital appreciation over the long term.</li></ul>
  <footer>© Axis Mutual Fund. All rights reserved.</footer>
</body></html>
"""

PDF_TEXT = (
    "4. Fees and Expenses\n"
    "4.1 Expense ratio\n"
    "The expense ratio of the scheme is 1.95% for the regular plan.\n"
    "4.2 Exit load\n"
    "Exit load is 1.00% if units are redeemed within 365 days.\n"
)


def write_pdf(path, pages_text: list[str]) -> None:
    document = pymupdf.open()
    for text in pages_text:
        page = document.new_page()
        page.insert_text((72, 100), text, fontsize=11)
    document.save(path)
    document.close()


def test_html_table_becomes_a_table_block(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    blocks = extract_html(path, SCHEME_SOURCE)
    tables = [b for b in blocks if b.kind == "table"]
    assert tables, "fee table was lost"
    assert "Expense ratio" in " ".join(" ".join(r) for r in tables[0].table_rows)


def test_html_headings_become_heading_blocks(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    blocks = extract_html(path, SCHEME_SOURCE)
    headings = [b.text for b in blocks if b.kind == "heading"]
    assert "Axis Large Cap Fund" in headings
    assert "Direct Growth" in headings


def test_html_nav_and_footer_are_stripped(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    blocks = extract_html(path, SCHEME_SOURCE)
    text = " ".join(b.text for b in blocks if b.text)
    assert "All rights reserved" not in text


def test_html_prose_is_captured(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    blocks = extract_html(path, SCHEME_SOURCE)
    text = " ".join(b.text for b in blocks if b.text)
    assert "BSE 100 TRI" in text


def test_pdf_extracts_text_and_page_numbers(tmp_path):
    path = tmp_path / "sid.pdf"
    write_pdf(path, [PDF_TEXT, "5. Risk profile\nVery high risk."])
    blocks = extract_pdf(path)
    text = " ".join(b.text for b in blocks if b.text)
    assert "1.95%" in text
    assert "Very high risk" in text
    pages = {b.page_num for b in blocks}
    assert pages == {1, 2}


def test_pdf_dispatch_uses_magic_bytes(tmp_path):
    path = tmp_path / "doc.pdf"
    write_pdf(path, [PDF_TEXT])
    assert any(b.text and "1.95%" in b.text for b in extract_blocks(path, "scheme_page"))


def test_html_dispatch_uses_extension(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    assert extract_blocks(path, "scheme_page", SCHEME_SOURCE)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_blocks(tmp_path / "absent.html", "scheme_page")


def test_empty_document_raises(tmp_path):
    path = tmp_path / "empty.html"
    path.write_text("<html><body></body></html>", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_html(path, SCHEME_SOURCE)


def test_orphan_numeric_lines_are_dropped(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><p>Expense ratio is 0.92%.</p><p>1</p><p>2</p><p>3</p></body></html>",
        encoding="utf-8",
    )
    blocks = extract_html(path, SCHEME_SOURCE)
    bare = [b.text for b in blocks if b.text and b.text.strip() in {"1", "2", "3"}]
    assert bare == []


def test_classify_flags_scheme_page_without_facts():
    source = SCHEME_SOURCE
    from src.models import Block

    blocks = [Block(kind="text", text="Welcome to Axis Mutual Fund. Please explore.")]
    assert classify(source, blocks, "ok") == "suspected_client_rendered"


def test_classify_passes_scheme_page_with_facts():
    from src.models import Block

    blocks = [Block(kind="text", text="The expense ratio is 0.92% and exit load is nil.")]
    assert classify(SCHEME_SOURCE, blocks, "ok") == "ok"


def test_classify_propagates_fetch_failure():
    from src.models import Block

    blocks = [Block(kind="text", text="Expense ratio 0.92%")]
    assert classify(SCHEME_SOURCE, blocks, "fetch_failed") == "fetch_failed"


def test_classify_flags_empty_block_list():
    assert classify(SCHEME_SOURCE, [], "ok") == "extract_failed"


def test_classify_accepts_table_only_pdf_facts():
    from src.models import Block

    blocks = [
        Block(
            kind="table",
            text="",
            table_rows=[["Exit load", "1.00%"], ["Minimum SIP", "500"]],
        )
    ]
    sid = Source(
        source_id="s03",
        scheme="large_cap",
        plan="n_a",
        doc_type="sid",
        url="https://example.com/SID.pdf",
        enabled=True,
    )
    assert classify(sid, blocks, "ok") == "ok"


def test_blocks_json_round_trips(tmp_path, monkeypatch):
    import config
    from src.extract import write_blocks_json

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    path = tmp_path / "page.html"
    path.write_text(HTML_WITH_TABLE, encoding="utf-8")
    blocks = extract_html(path, SCHEME_SOURCE)
    destination = write_blocks_json(SCHEME_SOURCE, blocks)
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["source_id"] == "s01"
    assert data["block_count"] == len(blocks)
    assert data["table_count"] >= 1
    assert data["scheme"] == "large_cap"