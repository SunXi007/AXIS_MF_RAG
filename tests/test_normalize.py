"""Tests for the six normalization rules of architecture.md 4.4."""

from __future__ import annotations

import json

import pytest

from src.models import Block
from src.normalize import (
    blocks_path,
    drop_repeated_pdf_lines,
    normalize_blocks,
    write_normalized,
)
from src.models import Source

SOURCE = Source(
    source_id="s01",
    scheme="large_cap",
    plan="direct",
    doc_type="scheme_page",
    url="https://example.com/fund",
    enabled=True,
)


def text_block(text: str, heading: list[str] | None = None, page: int | None = None) -> Block:
    return Block(kind="text", text=text, heading_path=heading or [], page_num=page)


# ------------------------------------------------------------------ rule 1-3


def test_unicode_is_nfc_normalized():
    decomposed = "Cafe\u0301"          # e + combining acute
    composed = "Caf\u00e9"
    out = normalize_blocks([text_block(decomposed)])
    assert out[0].text == composed


def test_nbsp_becomes_space_and_soft_hyphen_disappears():
    out = normalize_blocks([text_block("Minimum\u00a0SIP\u00ad amount")])
    assert out[0].text == "Minimum SIP amount"


def test_dehyphenation_joins_words_across_blocks():
    blocks = [text_block("tax-", page=3), text_block("efficient", page=3), text_block("plain", page=3)]
    out = normalize_blocks(blocks)
    assert out[0].text == "taxefficient"
    assert out[1].text == "plain"


def test_dehyphenation_does_not_cross_pages():
    blocks = [text_block("tax-", page=3), text_block("efficient", page=4)]
    assert len(normalize_blocks(blocks)) == 2


def test_blank_lines_collapse_to_two():
    out = normalize_blocks([text_block("a\n\n\n\n\nb")])
    assert out[0].text == "a\n\nb"


def test_runs_of_spaces_collapse():
    out = normalize_blocks([text_block("Expense   ratio \t\t 0.92")])
    assert out[0].text == "Expense ratio 0.92"


# -------------------------------------------------------------------- rule 4


def test_drop_repeated_pdf_lines_removes_page_numbers():
    pages = {p: f"Axis Large Cap Fund\ncontent {p}\nPage {p}" for p in range(1, 5)}
    cleaned = drop_repeated_pdf_lines(pages)
    assert "Page 1" not in cleaned[1]
    assert "content 1" in cleaned[1]


def test_drop_repeated_pdf_lines_removes_www_furniture():
    pages = {p: "www.axismf.com\nreal body" for p in range(1, 4)}
    cleaned = drop_repeated_pdf_lines(pages)
    assert "www.axismf.com" not in cleaned[1]
    assert "real body" in cleaned[1]


def test_drop_repeated_pdf_lines_keeps_unique_lines():
    pages = {p: "unique line for page " + str(p) for p in range(1, 4)}
    assert drop_repeated_pdf_lines(pages) == pages


def test_drop_repeated_pdf_lines_keeps_recurring_content_without_furniture_pattern():
    pages = {p: "The fund invests in large cap stocks." for p in range(1, 4)}
    cleaned = drop_repeated_pdf_lines(pages)
    assert "The fund invests in large cap stocks." in cleaned[1]


def test_drop_repeated_pdf_lines_needs_more_than_one_page():
    assert drop_repeated_pdf_lines({1: "Page 1"}) == {1: "Page 1"}


def test_pdf_furniture_removed_from_block_stream():
    blocks = [text_block("Axis Mutual Fund", page=p) for p in range(1, 5)]
    blocks += [text_block("Body content", page=p) for p in range(1, 5)]
    out = normalize_blocks(blocks)
    assert not any(b.text == "Axis Mutual Fund" for b in out)
    assert sum(1 for b in out if b.text == "Body content") == 4


# -------------------------------------------------------------------- rule 5


def test_cookie_banner_dropped():
    out = normalize_blocks([text_block("Cookie consent banner text")])
    assert out == []


def test_breadcrumb_heading_drops_whole_section():
    blocks = [
        Block(kind="text", text="body", heading_path=["Breadcrumb", "Axis"]),
        Block(kind="text", text="real", heading_path=["Overview"]),
    ]
    out = normalize_blocks(blocks)
    assert len(out) == 1 and out[0].text == "real"


def test_cta_phrase_removed_but_facts_kept():
    out = normalize_blocks(
        [text_block("Exit load NIL Tax implication Click here to view the tax implication. Minimum SIP 500")]
    )
    assert len(out) == 1
    assert "Exit load NIL" in out[0].text
    assert "Minimum SIP 500" in out[0].text
    assert "Click here" not in out[0].text


def test_nav_tab_bar_dropped():
    out = normalize_blocks([text_block("Overview Performance Fund Quants Portfolio IDCW Outlook")])
    assert out == []


@pytest.mark.parametrize("junk", ["Show 5 more", "Loading PDF viewer...", "Popular Choice", "Invest Now"])
def test_axis_ui_chrome_dropped(junk):
    assert normalize_blocks([text_block(junk)]) == []


def test_popular_choice_block_dropped():
    out = normalize_blocks([text_block("Popular Choice Returns: 21.25 % ( 3Y ) NAV 20.15 Sharpe 0.94")])
    assert out == []


def test_dot_leader_index_lines_dropped():
    out = normalize_blocks([text_block("Axis Midcap Fund " + "." * 120 + " 14")])
    assert out == []


def test_trailing_punctuation_preserved_when_nothing_removed():
    out = normalize_blocks([text_block("The NAV shall be calculated as shown below:")])
    assert out[0].text.endswith("below:")


# -------------------------------------------------------------------- rule 6


@pytest.mark.parametrize("orphan", ["12", "1,234", "3.5", "50.00", "100", "-", "()"])
def test_numeric_or_punctuation_orphans_dropped(orphan):
    assert normalize_blocks([text_block(orphan)]) == []


def test_orphan_with_letters_kept():
    out = normalize_blocks([text_block("N/A for this scheme")])
    assert len(out) == 1


def test_table_rows_are_preserved_through_normalization():
    blocks = [Block(kind="table", text="", heading_path=["Fees"], table_rows=[["Exit\u00a0load", "Nil"]])]
    out = normalize_blocks(blocks)
    assert out[0].table_rows == [["Exit load", "Nil"]]


def test_table_block_text_stays_empty():
    blocks = [Block(kind="table", text="", heading_path=["Fees"], table_rows=[["a", "b"]])]
    assert normalize_blocks(blocks)[0].text == ""


# --------------------------------------------------------------- idempotency


def test_normalization_is_idempotent():
    blocks = [
        text_block("Exit\u00a0load\u00ad NIL", heading=["Fees"]),
        text_block("tax-", page=2),
        text_block("efficient scheme", page=2),
        text_block("Overview Performance Fund Quants Portfolio IDCW Outlook"),
        text_block("42"),
        Block(kind="table", text="", heading_path=["Fees"], table_rows=[["Exit load", "Nil"]]),
    ]
    once = normalize_blocks(blocks)
    twice = normalize_blocks(once)
    assert [(b.kind, b.text, b.table_rows) for b in once] == [
        (b.kind, b.text, b.table_rows) for b in twice
    ]


def test_facts_survive_normalization():
    blocks = [text_block("Expense ratio 0.92% and exit load is nil for the direct plan")]
    assert "Expense ratio 0.92%" in normalize_blocks(blocks)[0].text


# ---------------------------------------------------------------- artifacts


def test_write_normalized_round_trips(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
    blocks = [text_block("Expense ratio 0.92%"), Block(kind="table", text="", table_rows=[["a", "b"]])]
    path = write_normalized(SOURCE, blocks)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["source_id"] == "s01"
    assert data["block_count"] == 2
    assert data["table_count"] == 1
    assert blocks_path(SOURCE).name == "s01.normalized.json"
    assert blocks_path(SOURCE, normalized=False).name == "s01.blocks.json"