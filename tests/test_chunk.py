"""Tests for the three-tier chunker and the table flattener."""

from __future__ import annotations

import json
import re

import pytest

import config
from src.chunk import (
    chunk_document,
    flatten_table,
    identifies_scheme_plan,
    merge_short_chunks,
    quality_summary,
    recursive_split,
    write_chunks_json,
    write_chunks_txt,
)
from src.models import Block, Chunk, Source

DIRECT = Source(
    source_id="s04",
    scheme="flexi_cap",
    plan="direct",
    doc_type="scheme_page",
    url="https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-dg/direct",
    enabled=True,
)
REGULAR = Source(
    source_id="s05",
    scheme="flexi_cap",
    plan="regular",
    doc_type="scheme_page",
    url="https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/rg/regular",
    enabled=True,
)


def make_chunk(text: str, index: int = 0, is_table: bool = False, source: Source = DIRECT) -> Chunk:
    return Chunk(
        chunk_id=f"{source.source_id}-c{index:03d}",
        text=text,
        metadata={
            "source_id": source.source_id,
            "source_url": source.url,
            "source_title": "Axis Flexi Cap Fund - Direct plan",
            "doc_type": source.doc_type,
            "scheme": source.scheme,
            "plan": source.plan,
            "page_num": None,
            "section_path": "Axis Flexi Cap Fund - Direct plan",
            "source_date": None,
            "fetched_at": "2026-10-02T09:14:22Z",
            "content_hash": "0" * 64,
            "chunk_index": index,
            "is_table": is_table,
            "char_len": len(text),
        },
    )


# --------------------------------------------------------------- flatten_table


def test_flatten_table_expands_plan_columns():
    rows = [["", "Direct Growth", "Regular Growth"], ["Expense ratio", "0.95%", "1.95%"]]
    text = flatten_table(rows, "Fees")
    lines = text.splitlines()
    assert lines[0] == "Fees"
    assert "Expense ratio (Direct Growth): 0.95%" in lines
    assert "Expense ratio (Regular Growth): 1.95%" in lines


def test_flatten_table_keeps_plan_identity_separable():
    """The G2 mechanism: a Direct query must not match the Regular value."""
    rows = [["", "Direct Growth", "Regular Growth"], ["Expense ratio", "0.95%", "1.95%"]]
    lines = flatten_table(rows, "Fees").splitlines()
    direct = [line for line in lines if "Direct Growth" in line]
    regular = [line for line in lines if "Regular Growth" in line]
    assert len(direct) == 1 and len(regular) == 1
    assert "0.95%" in direct[0] and "1.95%" not in direct[0]
    assert "1.95%" in regular[0] and "0.95%" not in regular[0]


def test_flatten_table_two_column_pairs():
    rows = [["Exit load", "Nil"], ["Minimum SIP", "Rs. 500"]]
    text = flatten_table(rows, "")
    assert "Exit load: Nil" in text
    assert "Minimum SIP: Rs. 500" in text


def test_flatten_table_single_row_has_no_header():
    assert flatten_table([["Exit load", "Nil"]], "Fees").splitlines()[-1] == "Exit load: Nil"


def test_flatten_table_without_headers_joins_values():
    rows = [["Since Inception", "13.74%", "12.72%", "12.13%"], ["1 Year", "3.21%", "1.99%", "-0.35%"]]
    lines = flatten_table(rows, "Returns").splitlines()
    assert lines[1] == "Since Inception: 13.74% | 12.72% | 12.13%"


def test_flatten_table_promotes_second_header_level():
    """A merged span above sub-columns must not become `label (col): value` pairs."""
    rows = [
        ["Record Date", "Option", "IDCW (Per Unit)", "NAV (Per Unit)", ""],
        ["Individuals/HUF", "Others", "Cum IDCW", "Ex- IDCW", ""],
        ["2026-01-20", "Dividend", "1.97", "1.97", ""],
    ]
    text = flatten_table(rows, "IDCW")
    assert "Individuals/HUF (Option): Others" not in text
    assert "2026-01-20 (Option Others): Dividend" in text


def test_flatten_table_does_not_nest_parentheses():
    rows = [["Fund", "IDCW (Per Unit)", "NAV"], ["Axis Large Cap", "1.97", "27.08"]]
    text = flatten_table(rows, "IDCW")
    assert "Axis Large Cap (IDCW (Per Unit)): 1.97" not in text
    assert "Axis Large Cap - IDCW (Per Unit): 1.97" in text


def test_flatten_table_empty_rows_yields_heading_only():
    assert flatten_table([["", ""], ["", ""]], "Fees") == "Fees"


def test_flatten_table_collapses_newlines_in_cells():
    rows = [["Name of Mutual Fund", ":", "Axis Mutual Fund"]]
    text = flatten_table(rows, "About")
    assert text.splitlines()[-1] == "Name of Mutual Fund: Axis Mutual Fund"
    assert "\n" not in text.splitlines()[-1]


# ------------------------------------------------------------------ tier rules


def test_tier1_table_is_one_atomic_chunk():
    blocks = [Block(kind="table", text="", heading_path=["Fees"], table_rows=[["Exit load", "Nil"]])]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) == 1
    assert chunks[0].metadata["is_table"] is True


def test_tier1_table_is_never_split_even_when_huge():
    """A row is never cut in half. An oversized table becomes parts of whole rows."""
    rows = [[f"Row {i}", f"value {i}"] for i in range(400)]
    blocks = [Block(kind="table", text="", heading_path=["NAV"], table_rows=rows)]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) > 1
    assert all(c.metadata["is_table"] for c in chunks)
    for chunk in chunks:
        assert len(chunk.text) <= config.MAX_CHUNK_CHARS
        for line in chunk.text.splitlines():
            assert line.strip().startswith(("Axis Flexi Cap Fund - Direct plan", "NAV", "Row "))
    # every row survives exactly once, in order, and is never cut mid-row
    emitted = [
        line
        for chunk in chunks
        for line in chunk.text.splitlines()
        if line.startswith("Row ")
    ]
    assert emitted == [f"Row {i}: value {i}" for i in range(400)]


def test_oversized_table_parts_are_reidentified_by_breadcrumb():
    rows = [[f"Row {i}", "x" * 40] for i in range(60)]
    blocks = [Block(kind="table", text="", heading_path=["NAV"], table_rows=rows)]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Axis Flexi Cap Fund - Direct plan\nNAV\n")
    parts = [c.metadata["table_part"] for c in chunks]
    totals = [c.metadata["table_parts"] for c in chunks]
    assert parts == list(range(1, len(chunks) + 1))
    assert totals == [len(chunks)] * len(chunks)


def test_small_table_is_one_part():
    rows = [["Exit load", "Nil"], ["Minimum SIP", "500"]]
    blocks = [Block(kind="table", text="", heading_path=["Fees"], table_rows=rows)]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) == 1
    assert (chunks[0].metadata["table_part"], chunks[0].metadata["table_parts"]) == (1, 1)


def test_prose_chunks_report_single_table_part():
    blocks = [Block(kind="text", text="A fact about the fund.", heading_path=["F"])]
    chunk = chunk_document(blocks, DIRECT, config)[0]
    assert chunk.metadata["table_part"] == 1 and chunk.metadata["table_parts"] == 1


def test_tier2_prepends_heading_breadcrumb():
    body = "The fund invests in large cap stocks with a minimum investment horizon."
    blocks = [
        Block(kind="heading", text="Objective", heading_path=["Objective"]),
        Block(kind="text", text=body, heading_path=["Objective"]),
    ]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("Axis Flexi Cap Fund - Direct plan\nObjective\n")
    assert body in chunks[0].text


def test_tier2_short_section_is_not_split():
    blocks = [Block(kind="text", text="Short note.", heading_path=["Notes"])]
    assert len(chunk_document(blocks, DIRECT, config)) == 1


def test_tier3_splits_over_long_section():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    blocks = [Block(kind="text", text=body, heading_path=["Long"])]
    chunks = chunk_document(blocks, DIRECT, config)
    assert len(chunks) > 1
    assert all(len(c.text) <= config.MAX_CHUNK_CHARS for c in chunks)


def test_tier3_respects_max_chunk_chars_including_breadcrumb():
    body = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(30))
    blocks = [Block(kind="text", text=body, heading_path=["Deeply Nested Heading"])]
    for chunk in chunk_document(blocks, DIRECT, config):
        assert len(chunk.text) <= config.MAX_CHUNK_CHARS


def test_tier3_preserves_all_content():
    sentences = [f"Sentence number {i} carries a fact." for i in range(60)]
    body = " ".join(sentences)
    blocks = [Block(kind="text", text=body, heading_path=["Facts"])]
    chunks = chunk_document(blocks, DIRECT, config)
    joined = " ".join(c.text for c in chunks)
    for sentence in sentences:
        assert sentence in joined


def test_recursive_split_does_not_leak_the_separator_pattern():
    body = " ".join(f"Fact {i} with some text." for i in range(120))
    for window in recursive_split(body, 300, 50, config.RECURSIVE_SEPARATORS):
        assert "(?<=" not in window
        assert "\\s+" not in window


def test_recursive_split_respects_max_chars():
    text = " ".join(f"word{i}" for i in range(500))
    for window in recursive_split(text, 200, 40, config.RECURSIVE_SEPARATORS):
        assert len(window) <= 200


def test_recursive_split_keeps_structural_separator_between_blocks():
    body = "First block ends here.\n\nSecond block starts here."
    windows = recursive_split(body, len(body) - 5, 0, config.RECURSIVE_SEPARATORS)
    assert any("First block ends here.\n\n" in w for w in windows)


def test_recursive_split_does_not_run_blocks_together():
    """Adjacent blocks must never concatenate into `...FundFAQ: ...`."""
    body = ("You can now start investing in Axis Large Cap Fund\n\n"
            "FAQ: Who should invest in Large Cap Fund? Equity markets.\n\n"
            + " ".join(f"Filler sentence {i} with extra words." for i in range(40)))
    windows = recursive_split(body, 400, 150, config.RECURSIVE_SEPARATORS)
    for window in windows:
        assert "FundFAQ" not in window
        assert not re.search(r"[a-z]{3}(?:FAQ|Exit|Expense)", window)


def test_recursive_split_applies_overlap():
    text = "\n\n".join(f"Paragraph {i} with enough text to matter." for i in range(40))
    windows = recursive_split(text, 300, 150, config.RECURSIVE_SEPARATORS)
    assert len(windows) > 1
    assert any(len(set(windows[i].split()) & set(windows[i + 1].split())) > 0 for i in range(len(windows) - 1))


def test_recursive_split_on_unbreakable_text_falls_back_to_characters():
    text = "x" * 1000
    windows = recursive_split(text, 300, 50, config.RECURSIVE_SEPARATORS)
    assert all(len(w) <= 300 for w in windows)
    assert "".join(windows).count("x") >= 1000


def test_recursive_split_empty_text():
    assert recursive_split("   ", 300, 50, config.RECURSIVE_SEPARATORS) == []


# --------------------------------------------------------------------- merging


def test_merge_folds_short_chunk_into_neighbour():
    chunks = [make_chunk("A" * 500, 0), make_chunk("tiny", 1)]
    merged = merge_short_chunks(chunks, config.MIN_CHUNK_CHARS, config.MAX_CHUNK_CHARS)
    assert len(merged) == 1
    assert "tiny" in merged[0].text


def test_merge_prefers_forward_when_backward_would_overflow():
    chunks = [make_chunk("A" * 890, 0), make_chunk("tiny", 1), make_chunk("B" * 400, 2)]
    merged = merge_short_chunks(chunks, config.MIN_CHUNK_CHARS, config.MAX_CHUNK_CHARS)
    assert len(merged) == 2
    assert "tiny" in merged[0].text


def test_merge_never_merges_into_a_table_chunk():
    chunks = [make_chunk("Label: value" * 8, 0, is_table=True), make_chunk("tiny", 1)]
    merged = merge_short_chunks(chunks, config.MIN_CHUNK_CHARS, config.MAX_CHUNK_CHARS)
    assert len(merged) == 2
    assert merged[0].metadata["is_table"] is True
    assert len(merged[0].text) < config.MIN_CHUNK_CHARS


def test_merge_keeps_every_chunk_index_contiguous():
    chunks = [make_chunk("A" * 300, i) for i in range(5)]
    merged = merge_short_chunks(chunks, config.MIN_CHUNK_CHARS, config.MAX_CHUNK_CHARS)
    assert [c.metadata["chunk_index"] for c in merged] == list(range(len(merged)))


def test_merge_does_not_lose_content():
    parts = [f"block{i} " + "x" * 200 for i in range(6)]
    chunks = [make_chunk(p, i) for i, p in enumerate(parts)]
    merged = merge_short_chunks(chunks, 400, 2000)
    joined = " ".join(c.text for c in merged)
    for part in parts:
        assert part in joined


# ----------------------------------------------------------------- metadata


def test_chunk_metadata_carries_every_required_field():
    blocks = [Block(kind="text", text="Expense ratio is 0.85%.", heading_path=["Fees"])]
    chunk = chunk_document(blocks, DIRECT, config)[0]
    assert chunk.chunk_id == "s04-c000"
    for field in (
        "source_id",
        "source_url",
        "source_title",
        "doc_type",
        "scheme",
        "plan",
        "section_path",
        "content_hash",
        "chunk_index",
        "is_table",
        "char_len",
        "page_num",
        "source_date",
        "fetched_at",
    ):
        assert field in chunk.metadata, field
    assert chunk.chunk_id.startswith("s04-c")
    assert chunk.metadata["source_url"] == DIRECT.url


def test_chunk_ids_are_zero_padded_and_ordered():
    blocks = [Block(kind="text", text=f"Fact {i}.", heading_path=["F"]) for i in range(4)]
    chunks = chunk_document(blocks, DIRECT, config)
    assert [c.chunk_id for c in chunks] == [f"s04-c{i:03d}" for i in range(len(chunks))]


def test_direct_and_regular_are_not_merged():
    direct = chunk_document(
        [Block(kind="text", text="Expense ratio 0.85%.", heading_path=["Fees"])], DIRECT, config
    )
    regular = chunk_document(
        [Block(kind="text", text="Expense ratio 1.84%.", heading_path=["Fees"])], REGULAR, config
    )
    assert direct[0].text != regular[0].text
    assert "Direct plan" in direct[0].text and "Regular plan" in regular[0].text


def test_every_chunk_identifies_scheme_and_plan():
    for scheme, plan in (
        ("large_cap", "direct"),
        ("flexi_cap", "regular"),
        ("elss", "direct"),
        ("midcap", "direct"),
    ):
        source = Source("s99", scheme, plan, "scheme_page", "https://example.com", True)
        chunk = chunk_document([Block(kind="text", text="Some fact.", heading_path=["F"])], source, config)[0]
        assert identifies_scheme_plan(chunk), (scheme, plan)


def test_content_hash_is_stable():
    blocks = [Block(kind="text", text="Stable text.", heading_path=["F"])]
    first = chunk_document(blocks, DIRECT, config)[0]
    second = chunk_document(blocks, DIRECT, config)[0]
    assert first.metadata["content_hash"] == second.metadata["content_hash"]


def test_page_number_is_carried_from_pdf_blocks():
    blocks = [Block(kind="text", text="Page text.", heading_path=["H"], page_num=7)]
    assert chunk_document(blocks, DIRECT, config)[0].metadata["page_num"] == 7


# ------------------------------------------------------------------ artifacts


def test_write_chunks_json_shape(tmp_path):
    chunks = chunk_document([Block(kind="text", text="Fact.", heading_path=["F"])], DIRECT, config)
    path = write_chunks_json(chunks, tmp_path / "chunks.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["chunk_count"] == 1
    assert data["params"]["max_chunk_chars"] == config.MAX_CHUNK_CHARS
    assert data["chunks"][0]["chunk_id"] == "s04-c000"


def test_write_chunks_txt_is_human_readable(tmp_path):
    chunks = chunk_document([Block(kind="text", text="Fact.", heading_path=["F"])], DIRECT, config)
    path = write_chunks_txt(chunks, tmp_path / "chunks.txt")
    text = path.read_text(encoding="utf-8")
    assert "CHUNK s04-c000" in text
    assert "SECTION" in text and "SCHEME" in text and "SOURCE" in text
    assert "QUALITY SUMMARY" in text
    assert DIRECT.url in text


def test_quality_summary_reports_scheme_plan_and_length():
    chunks = chunk_document([Block(kind="text", text="Fact.", heading_path=["F"])], DIRECT, config)
    summary = quality_summary(chunks)
    assert "chunks per scheme" in summary
    assert "flexi_cap" in summary
    assert "chunks per plan" in summary
    assert "criterion 4" in summary
    assert "criterion 1" in summary
    assert "median length" in summary


def test_quality_summary_handles_empty_input():
    assert "no chunks produced" in quality_summary([])


def test_page_num_is_null_for_html_chunks():
    chunk = chunk_document([Block(kind="text", text="Fact.", heading_path=["F"])], DIRECT, config)[0]
    assert chunk.metadata["page_num"] is None


@pytest.mark.parametrize("plan", ["direct", "regular"])
def test_plan_appears_in_chunk_text(plan):
    source = Source("s98", "large_cap", plan, "scheme_page", "https://example.com", True)
    chunk = chunk_document([Block(kind="text", text="NAV is 65.82.", heading_path=["F"])], source, config)[0]
    assert plan.title() in chunk.text