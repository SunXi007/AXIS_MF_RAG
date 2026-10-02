"""Response templates, the prompt contract, and the disclaimer.

Single source of truth for every fixed user-facing string (FR-O4): the UI, the
README and `disclaimer.md` all import from here so they cannot drift apart.

Imports only `config`, per `architecture.md` §3.2.
"""

from __future__ import annotations

from typing import Any

from src.models import TemplateId

DISCLAIMER = (
    "Facts-only. No investment advice.\n"
    "Answers are generated only from official Axis Mutual Fund public pages and "
    "statutory documents.\n"
    "Mutual fund investments are subject to market risks. "
    "Read all scheme related documents carefully."
)

AMC_DOWNLOADS_URL = "https://www.axismf.com/downloads"
FACTSHEET_URL = (
    "https://www.axismf.com/1/5/1423/1484/1487/2872/4561/"
    "Axis_Fund_Factsheet_July_2026_2412c4ee93.pdf"
)

TEMPLATES: dict[str, str] = {
    "REFUSE_ADVICE": (
        "I can share facts about these schemes, but I don't give investment advice. "
        "For guidance on choosing a scheme, please speak with a SEBI-registered "
        "investment adviser. You can review scheme facts here: {downloads_url}"
    ),
    "REFUSE_PERFORMANCE": (
        "I don't calculate or compare returns. The official monthly factsheet has the "
        "official performance figures: {factsheet_url}"
    ),
    "NOT_FOUND": (
        "I couldn't find that in the official sources I'm using (Axis Mutual Fund — "
        "Large Cap, Flexi Cap, ELSS Tax Saver, Midcap). Try asking about expense "
        "ratio, exit load, minimum SIP, ELSS lock-in, benchmark, riskometer, or how "
        "to download a statement: {downloads_url}"
    ),
    "NO_RETRIEVAL": "I don't have that in the sources I use. — Facts-only. No investment advice.",
    "PII_BLOCKED": (
        "For your security I can't process personal identifiers. Please remove account "
        "numbers, PAN, Aadhaar, OTPs, email addresses, or phone numbers. "
        "— Facts-only. No investment advice."
    ),
}

TEMPLATE_IDS: tuple[TemplateId, ...] = ("FACTUAL", *TEMPLATES)

PROMPT = """You are a mutual fund FAQ assistant for Axis Mutual Fund.

STRICT RULES
1. Answer ONLY from the CONTEXT below. If the answer is not in the context, reply exactly:
   NOT_FOUND
2. Maximum 3 sentences. No bullet lists. No tables.
3. Do NOT give investment advice, recommendations, opinions, or suitability guidance.
4. Do NOT state, compute, or compare returns, NAV, or performance. If asked, reply exactly: NO_PERF
5. Do NOT write any URL or link. Citations are added automatically after your reply.
6. Copy every number, rate, and date exactly as written in the CONTEXT. Never round, convert,
   reformat, or infer a figure. Omit rather than guess.

CONTEXT
{context}"""


def render(template_id: TemplateId, **kwargs: Any) -> str:
    """Render one of the six template ids.

    `FACTUAL` has no fixed wording: the answer text is the template, so the caller
    passes it as `answer`. Unknown ids raise, because silently rendering a blank
    string would look like a working refusal in the UI.
    """
    if template_id == "FACTUAL":
        return str(kwargs.get("answer", "")).strip()
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown template_id {template_id!r}; expected one of {TEMPLATE_IDS}")
    fields = {
        "downloads_url": AMC_DOWNLOADS_URL,
        "factsheet_url": FACTSHEET_URL,
        **kwargs,
    }
    return TEMPLATES[template_id].format(**fields)