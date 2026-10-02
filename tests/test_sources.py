"""Tests for sources.csv loading and validation."""

from __future__ import annotations

import pytest

import config
from src.models import Source
from src.sources import (
    SourceValidationError,
    counts_by_scheme,
    excluded_sources,
    load_sources,
)

HEADER = "source_id,scheme,plan,doc_type,url,enabled,notes\n"
GOOD = "s01,large_cap,direct,scheme_page,https://example.com/a,true,note\n"


def write_csv(tmp_path, body: str):
    path = tmp_path / "sources.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_manifest_loads_19_rows():
    assert len(load_sources()) == 19


def test_manifest_enables_18_sources():
    assert len(load_sources(enabled_only=True)) == 18


def test_indmoney_source_is_excluded_per_d5():
    excluded = excluded_sources()
    assert len(excluded) == 1
    assert excluded[0].source_id == "s15"
    assert "indmoney" in excluded[0].url
    assert excluded[0].enabled is False


def test_all_four_schemes_have_enabled_sources():
    counts = counts_by_scheme(load_sources(enabled_only=True))
    for scheme in ("large_cap", "flexi_cap", "elss", "midcap"):
        assert counts.get(scheme, 0) > 0, f"{scheme} has no enabled source"


def test_both_plan_variants_present_for_core_schemes():
    enabled = load_sources(enabled_only=True)
    for scheme in ("large_cap", "flexi_cap", "elss"):
        plans = {s.plan for s in enabled if s.scheme == scheme}
        assert "direct" in plans and "regular" in plans, f"{scheme} missing a plan variant"


def test_urls_match_prd_inventory_verbatim():
    urls = {s.url for s in load_sources()}
    assert "https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-dg/direct" in urls
    assert (
        "https://www.axismf.com/cms/sites/default/files/pdf-factsheets/"
        "20190204018-Flexi%20Cap%20Fund%20(November%202024)%20DP-Leaflet.pdf" in urls
    )
    assert (
        "https://transact.axismf.com/cms/sites/default/files/pdf-factsheets/"
        "Axis%20Fund%20Factsheet%20March%202026.pdf" in urls
    )


def test_duplicate_source_id_rejected(tmp_path):
    path = write_csv(tmp_path, GOOD + GOOD)
    with pytest.raises(SourceValidationError, match="duplicate source_id"):
        load_sources(path)


def test_unknown_scheme_rejected(tmp_path):
    path = write_csv(tmp_path, "s01,bad_scheme,direct,scheme_page,https://e.com/a,true,\n")
    with pytest.raises(SourceValidationError, match="unknown scheme"):
        load_sources(path)


def test_unknown_plan_rejected(tmp_path):
    path = write_csv(tmp_path, "s01,elss,holographic,scheme_page,https://e.com/a,true,\n")
    with pytest.raises(SourceValidationError, match="unknown plan"):
        load_sources(path)


def test_unknown_doc_type_rejected(tmp_path):
    path = write_csv(tmp_path, "s01,elss,direct,podcast,https://e.com/a,true,\n")
    with pytest.raises(SourceValidationError, match="unknown doc_type"):
        load_sources(path)


def test_non_http_url_rejected(tmp_path):
    path = write_csv(tmp_path, "s01,elss,direct,scheme_page,ftp://e.com/a,true,\n")
    with pytest.raises(SourceValidationError, match="http"):
        load_sources(path)


def test_bad_enabled_value_rejected(tmp_path):
    path = write_csv(tmp_path, "s01,elss,direct,scheme_page,https://e.com/a,maybe,\n")
    with pytest.raises(SourceValidationError, match="enabled"):
        load_sources(path)


def test_missing_column_rejected(tmp_path):
    path = tmp_path / "sources.csv"
    path.write_text("source_id,scheme\ns01,elss\n", encoding="utf-8")
    with pytest.raises(SourceValidationError, match="missing columns"):
        load_sources(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(SourceValidationError, match="not found"):
        load_sources(tmp_path / "absent.csv")


def test_pdf_filename_derived_and_url_decoded():
    source = Source(
        source_id="s03",
        scheme="large_cap",
        plan="n_a",
        doc_type="sid",
        url="https://e.com/files/Axis%20Bluechip%20Fund%20-%20SID.pdf",
        enabled=True,
    )
    assert source.is_pdf is True
    assert source.filename == "Axis_Bluechip_Fund_-_SID.pdf"


def test_html_filename_falls_back_when_path_has_no_name():
    source = Source(
        source_id="s16",
        scheme="amc_wide",
        plan="n_a",
        doc_type="homepage",
        url="https://www.axismf.com/",
        enabled=True,
    )
    assert source.is_pdf is False
    assert source.filename == "s16.html"


def test_every_doc_type_in_manifest_is_known():
    for source in load_sources():
        assert source.doc_type in config.DOC_TYPES
        assert source.scheme in config.SCHEMES
        assert source.plan in config.PLANS


def test_every_source_derives_a_usable_filename():
    for source in load_sources():
        name = source.filename
        assert name
        assert name.startswith(source.source_id) or "." in name
        assert " " not in name
        if source.is_pdf:
            assert name.lower().endswith(".pdf")
        else:
            assert not name.endswith(".pdf") or "xhtml" not in name