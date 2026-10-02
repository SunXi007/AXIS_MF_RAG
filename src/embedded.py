"""Extraction of facts embedded in page payloads.

Axis scheme pages are client-rendered: the visible DOM that trafilatura sees
carries almost no facts, while the numbers live in a Next.js RSC flight
payload and in JSON-LD blocks. Without this module the corpus would contain
prose but no expense ratio, no exit load, no minimum SIP.

Emitted fact rows are two-column, so the P2 chunker's Tier-1 table handling
flattens them into the 'Field: Value' form the prompt needs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from bs4 import BeautifulSoup

SCRIPT_BODY = re.compile(r"<script[^>]*>(.*?)</script>", re.S | re.I)
JSONLD_FAQ = re.compile(
    r'<script[^>]*id="json-ld-faqsSchema"[^>]*>(.*?)</script>', re.S | re.I
)
FLIGHT_CALL = re.compile(r"^[^(]*\((.*)\)[;\s]*$", re.S)
RSC_ROW = re.compile(r"^[0-9a-f]{1,6}:(.*)$", re.M)
MAX_FLIGHT_LENGTH = 8_000_000
MAX_WALK_DEPTH = 60

FLAT_FACTS: dict[str, str] = {
    "expenseRatio": "Expense ratio",
    "exitLoad": "Exit load",
    "entryLoad": "Entry load",
    "benchmark": "Benchmark",
    "riskType": "Riskometer category",
    "schemeName": "Scheme name",
    "schemeCode": "Scheme code",
    "planType": "Plan type",
    "optionName": "Option",
    "category": "Category",
    "fundType": "Fund type",
    "aum": "AUM",
    "nav": "NAV",
    "minAmount": "Minimum lump sum amount",
    "minAmountDaily": "Minimum daily amount",
    "sip": "Minimum SIP",
    "dateUpdated": "Source updated on",
    "investmentHorizon": "Investment horizon",
    "schemeOpenDate": "Scheme open date",
    "sinceInceptionDate": "Date of inception",
}

DESCRIPTION_FACTS: dict[str, str] = {
    "schemeDescription": "Scheme description",
    "schemeDisclaimer": "Scheme objective",
    "expenseRatioDate": "Expense ratio as on",
}

NESTED_FACTS: dict[str, tuple[str, dict[str, str]]] = {
    "minimumInvestment": (
        "Minimum investment",
        {
            "lumpsum": "Lump sum",
            "sip": "SIP",
            "sipDaily": "Daily SIP",
            "additionalInvestment": "Additional investment",
        },
    ),
    "riskometerDetails": (
        "Riskometer",
        {
            "benchmark": "Benchmark index",
            "benchmarkRiskStatus": "Benchmark risk status",
            "riskDescription": "Risk profile",
        },
    ),
}

PLACEHOLDER = {"", "null", "undefined", "none", "n/a", "-", "--"}


def strip_markup(value: str) -> str:
    """Reduce an HTML-bearing value such as '<ul><li>a</li></ul>' to plain text."""
    if "<" in value and ">" in value:
        text = BeautifulSoup(value, "lxml").get_text(" | ", strip=True)
    else:
        text = value
    text = re.sub(r"\s*\|\s*", " | ", text).strip(" |")
    return re.sub(r"[ \t]+", " ", text)


def usable(value: Any) -> bool:
    """Reject placeholders and empty scalars so they never reach the corpus."""
    if value is None or isinstance(value, (dict, list)):
        return False
    text = str(value).strip()
    if text.lower() in PLACEHOLDER:
        return False
    return bool(text)


def _flight_strings(html: str) -> list[str]:
    out: list[str] = []
    for body in SCRIPT_BODY.findall(html):
        if len(body) > MAX_FLIGHT_LENGTH:
            continue
        if "expenseRatio" not in body and "riskometerDetails" not in body:
            continue
        call = FLIGHT_CALL.match(body.strip())
        if not call:
            continue
        try:
            outer = json.loads(call.group(1))
        except Exception:
            continue
        if isinstance(outer, list) and len(outer) > 1 and isinstance(outer[1], str):
            out.append(outer[1])
    return out


def _all_flight_strings(html: str) -> list[str]:
    """Every RSC flight payload on the page, including ones with no fund facts."""
    out: list[str] = []
    for body in SCRIPT_BODY.findall(html):
        if len(body) > MAX_FLIGHT_LENGTH:
            continue
        if "__next_f" not in body:
            continue
        call = FLIGHT_CALL.match(body.strip())
        if not call:
            continue
        try:
            outer = json.loads(call.group(1))
        except Exception:
            continue
        if isinstance(outer, list) and len(outer) > 1 and isinstance(outer[1], str):
            out.append(outer[1])
    return out


def _rows(flight: str) -> list[Any]:
    nodes: list[Any] = []
    matched = RSC_ROW.findall(flight)
    if not matched:
        matched = [flight]
    for row in matched:
        row = row.strip()
        if not row or row in ("[]", "{}"):
            continue
        try:
            nodes.append(json.loads(row))
        except Exception:
            continue
    return nodes


def extract_faq_pairs(html: str) -> list[tuple[str, str]]:
    """Pull question and answer pairs from the JSON-LD FAQ block."""
    match = JSONLD_FAQ.search(html)
    if not match:
        return []
    try:
        data = json.loads(match.group(1).strip())
    except Exception:
        return []

    pairs: list[tuple[str, str]] = []
    entities = data.get("mainEntity", []) if isinstance(data, dict) else []
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        question = str(entity.get("name", "")).strip()
        answer = entity.get("acceptedAnswer", {})
        if not isinstance(answer, dict):
            continue
        text = str(answer.get("text", "")).strip()
        if question and text:
            pairs.append((strip_markup(question), strip_markup(text)))
    return pairs


def _collect(node: Any, facts: dict[str, str], depth: int = 0) -> None:
    if depth > MAX_WALK_DEPTH:
        return

    if isinstance(node, dict):
        for key, mapping in NESTED_FACTS.items():
            nested = node.get(key)
            if not isinstance(nested, dict):
                continue
            group = mapping[0]
            for sub_key, label in mapping[1].items():
                value = nested.get(sub_key)
                if usable(value):
                    facts[label] = strip_markup(str(value))
            if group and key == "minimumInvestment":
                facts.setdefault("Minimum investment", "")
        for key, value in node.items():
            if key in FLAT_FACTS and usable(value):
                facts.setdefault(FLAT_FACTS[key], strip_markup(str(value)))
            elif key in DESCRIPTION_FACTS and usable(value):
                facts.setdefault(DESCRIPTION_FACTS[key], strip_markup(str(value)))
        for value in node.values():
            _collect(value, facts, depth + 1)

    elif isinstance(node, list):
        for item in node:
            _collect(item, facts, depth + 1)


def extract_facts(html: str) -> dict[str, str]:
    """Collect plan-specific facts from every RSC payload in the page."""
    facts: dict[str, str] = {}
    for flight in _flight_strings(html):
        for node in _rows(flight):
            _collect(node, facts)
    return {k: v for k, v in facts.items() if v}


def tax_rows(html: str) -> list[list[str]]:
    """Tax implication rows, which appear as a titled list rather than a table."""
    for flight in _flight_strings(html):
        for node in _rows(flight):
            stack = [node]
            while stack:
                current = stack.pop()
                if isinstance(current, dict):
                    entries = current.get("taxImplication")
                    if isinstance(entries, list) and entries:
                        rows: list[list[str]] = []
                        for entry in entries:
                            if not isinstance(entry, dict):
                                continue
                            title = str(entry.get("title", "")).strip()
                            description = strip_markup(str(entry.get("description", "")))
                            if title and description:
                                rows.append([title, description])
                        if rows:
                            return rows
                    stack.extend(current.values())
                elif isinstance(current, list):
                    stack.extend(current)
    return []


def fact_rows(facts: dict[str, str], scheme: str, plan: str) -> list[list[str]]:
    """Shape facts into two-column rows with a plan-scoped header."""
    header = f"{scheme} ({plan} plan) - key facts"
    rows: list[list[str]] = [["Fact", "Value"]]
    for label, value in facts.items():
        rows.append([label, value])
    if len(rows) < 2:
        return []
    rows[0] = [header, ""]
    return rows


CARD_KEYS = ("title", "description")
CARD_CONTEXT = ("link", "linkOne", "linkTwo", "postLogin", "webpIcon", "icon", "mobileEnabled")


def _is_card(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    if not all(isinstance(node.get(k), str) and node[k].strip() for k in CARD_KEYS):
        return False
    return any(k in node for k in CARD_CONTEXT)


def _walk_all(node: Any, depth: int = 0):
    if depth > MAX_WALK_DEPTH:
        return
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_all(value, depth + 1)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_all(item, depth + 1)


def procedural_cards(html: str) -> list[tuple[str, str, str]]:
    """Collect service cards of the form title / description / link.

    The downloads and services pages carry their entire value in these cards
    rather than in rendered markup, so without this the corpus has no way to
    answer a 'how do I download my statement' question.
    """
    cards: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    payloads = _all_flight_strings(html) or [html]
    for flight in payloads:
        for node in (_rows(flight) if flight is not html else [html]):
            for candidate in _walk_all(node):
                if not _is_card(candidate):
                    continue
                title = strip_markup(str(candidate["title"])).strip()
                description = strip_markup(str(candidate["description"])).strip()
                link = str(candidate.get("link") or candidate.get("linkOne") or "").strip()
                if not title or not description:
                    continue
                key = (title, description)
                if key in seen:
                    continue
                seen.add(key)
                cards.append((title, description, link))

    return cards


def card_blocks(html: str, heading_prefix: list[str]) -> list[tuple[list[str], str]]:
    """Shape procedural cards into (heading_path, text) pairs for the chunker."""
    out: list[tuple[list[str], str]] = []
    for title, description, link in procedural_cards(html):
        text = f"{title}: {description}"
        if link:
            text = f"{text} Link: {link}"
        out.append((list(heading_prefix) + [title], text))
    return out