"""Tests for embedded payload extraction.

Fixtures are synthetic but structurally identical to the real Axis pages:
a Next.js RSC flight payload plus a JSON-LD FAQ block.
"""

from __future__ import annotations

import json

import pytest

from src.embedded import (
    card_blocks,
    extract_faq_pairs,
    extract_facts,
    fact_rows,
    procedural_cards,
    strip_markup,
    tax_rows,
    usable,
)

SCHEME = {
    "schemeName": "Axis Large Cap Fund",
    "planType": "Direct",
    "optionName": "Growth",
    "planCode": "DG",
    "schemeCode": "EF",
    "expenseRatio": "0.92",
    "expenseRatioDate": "2026-10-01",
    "exitLoad": "No Exit Load after 1 week",
    "benchmark": "BSE 100 TRI",
    "riskType": "Very High",
    "minAmount": "100.0",
    "sip": "100.0",
    "schemeDescription": (
        "An open ended equity linked saving scheme with a statutory lock in of 3 years"
    ),
    "minimumInvestment": {
        "lumpsum": 500,
        "sip": "500.0",
        "sipDaily": "500.0",
        "additionalInvestment": 500,
    },
    "riskometerDetails": {
        "riskDescription": "<ul><li>Capital appreciation over long term.</li></ul>",
        "benchmark": "NIFTY 500 TRI",
        "benchmarkRiskStatus": "The risk of the benchmark is Very High",
    },
}


def rsc_page(payload: dict, extra_scripts: str = "") -> str:
    flight = "1:" + json.dumps(payload)
    call = "self.__next_f.push([1," + json.dumps(flight) + "])"
    return f"<html><body><script>{call}</script>{extra_scripts}</body></html>"


FAQ_SCRIPT = (
    '<script id="json-ld-faqsSchema" type="application/ld+json">'
    + json.dumps(
        {
            "@type": "FAQPage",
            "mainEntity": [
                {
                    "name": "What is a Large Cap Fund?",
                    "@type": "Question",
                    "acceptedAnswer": {"text": "An open ended equity scheme."},
                },
                {
                    "name": "Is there a lock-in?",
                    "@type": "Question",
                    "acceptedAnswer": {"text": "No lock-in for this scheme."},
                },
            ],
        }
    )
    + "</script>"
)


def test_usable_rejects_placeholders():
    assert usable("0.92") is True
    assert usable("N/A") is False
    assert usable("null") is False
    assert usable("") is False
    assert usable("   ") is False
    assert usable({"a": 1}) is False
    assert usable(None) is False


def test_strip_markup_flattens_lists():
    assert strip_markup("<ul><li>alpha</li><li>beta</li></ul>") == "alpha | beta"
    assert strip_markup("plain text") == "plain text"


def test_extracts_expense_ratio_and_exit_load():
    facts = extract_facts(rsc_page(SCHEME))
    assert facts["Expense ratio"] == "0.92"
    assert facts["Exit load"] == "No Exit Load after 1 week"


def test_extracts_benchmark_and_riskometer():
    facts = extract_facts(rsc_page(SCHEME))
    assert facts["Benchmark"] == "BSE 100 TRI"
    assert facts["Riskometer category"] == "Very High"
    assert facts["Benchmark risk status"] == "The risk of the benchmark is Very High"


def test_extracts_minimum_investment_rows():
    facts = extract_facts(rsc_page(SCHEME))
    assert facts["Lump sum"] == "500"
    assert facts["SIP"] == "500.0"
    assert facts["Daily SIP"] == "500.0"


def test_extracts_scheme_description_containing_lockin():
    facts = extract_facts(rsc_page(SCHEME))
    assert "lock in of 3 years" in facts["Scheme description"]


def test_risk_description_markup_is_flattened():
    facts = extract_facts(rsc_page(SCHEME))
    assert facts["Risk profile"] == "Capital appreciation over long term."


def test_facts_are_empty_when_page_has_no_payload():
    assert extract_facts("<html><body><p>nothing here</p></body></html>") == {}


def test_faq_pairs_are_extracted():
    pairs = extract_faq_pairs(rsc_page({}, FAQ_SCRIPT))
    assert len(pairs) == 2
    assert pairs[0][0] == "What is a Large Cap Fund?"
    assert pairs[0][1] == "An open ended equity scheme."


def test_faq_pairs_empty_when_block_absent():
    assert extract_faq_pairs("<html></html>") == []


def test_tax_rows_are_extracted():
    payload = dict(SCHEME)
    payload["taxImplication"] = [
        {
            "title": "Returns are taxed at 15% if you redeem",
            "description": "After 1 year you pay LTCG of 10%",
        }
    ]
    rows = tax_rows(rsc_page(payload))
    assert rows == [["Returns are taxed at 15% if you redeem", "After 1 year you pay LTCG of 10%"]]


def test_tax_rows_empty_when_absent():
    assert tax_rows(rsc_page(SCHEME)) == []


def test_fact_rows_are_two_column_with_plan_header():
    facts = extract_facts(rsc_page(SCHEME))
    rows = fact_rows(facts, "Axis Large Cap Fund", "Direct")
    assert rows[0][0] == "Axis Large Cap Fund (Direct plan) - key facts"
    assert rows[0][1] == ""
    labels = {row[0] for row in rows[1:]}
    assert "Expense ratio" in labels
    assert all(len(row) == 2 for row in rows)


def test_fact_rows_empty_without_facts():
    assert fact_rows({}, "x", "Direct") == []


def test_procedural_cards_are_extracted():
    payload = {
        "optionsList": ["Our Services", "Account Statement"],
        "services": [
            {
                "title": "Account Statement",
                "description": (
                    "Just enter your Folio Number or PAN Number and get your mutual fund "
                    "account statement mailed to your email id registered with us."
                ),
                "link": "/servicecenter/accountstatement/accountstatements",
                "postLogin": False,
            }
        ],
    }
    cards = procedural_cards(rsc_page(payload))
    assert len(cards) == 1
    title, description, link = cards[0]
    assert title == "Account Statement"
    assert "account statement" in description
    assert link == "/servicecenter/accountstatement/accountstatements"


def test_card_blocks_carry_heading_and_link():
    payload = {
        "services": [
            {
                "title": "Account Statement",
                "description": "Get your account statement by email.",
                "link": "/servicecenter/accountstatement/accountstatements",
                "icon": "https://x/y.svg",
            }
        ]
    }
    blocks = card_blocks(rsc_page(payload), ["Axis Mutual Fund"])
    assert len(blocks) == 1
    heading, text = blocks[0]
    assert heading[0] == "Axis Mutual Fund"
    assert heading[-1] == "Account Statement"
    assert text.startswith("Account Statement: Get your account statement by email.")
    assert "Link: /servicecenter/accountstatement" in text


def test_non_card_dicts_are_not_treated_as_cards():
    payload = {"fundManagers": [{"name": "Mr. X", "description": "Head - Equity"}]}
    assert procedural_cards(rsc_page(payload)) == []


def test_duplicate_cards_are_deduplicated():
    card = {
        "title": "Account Statement",
        "description": "Get your account statement by email.",
        "link": "/x",
        "icon": "https://x/y.svg",
    }
    assert len(procedural_cards(rsc_page({"a": [card], "b": [card]}))) == 1


@pytest.mark.parametrize("broken", ["<html>", "<script>not json</script>", ""])
def test_malformed_html_does_not_raise(broken):
    assert extract_facts(broken) == {}
    assert extract_faq_pairs(broken) == []
    assert procedural_cards(broken) == []