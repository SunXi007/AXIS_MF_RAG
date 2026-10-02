"""Input guardrails: refuse before anything is embedded, searched or generated.

Pure functions with no network and no LLM call, so every refusal path is unit
testable offline (`architecture.md` §7.1).

Order is fixed by `architecture.md` §5.2 and matters:

1. PII first and unconditionally, so no identifier can reach a model (FR-G4, R12).
   Within PII the patterns run strongest-signal first: `pan` and `email` are
   unambiguous, `account` and `phone` are context-gated, and `aadhaar` runs last
   because a bare 12-digit run is indistinguishable from an account number. A
   context word beats the digit-shape guess, so `a/c no 123456789012` is reported
   as an account rather than an Aadhaar.
2. Advice before performance, because "should I buy the fund that performed best"
   is advice-shaped and refusing it as a performance question would imply we would
   otherwise have answered it (A-3).
3. Anything ambiguous goes to Stage 1 rule 4, the LLM classifier, which lands in
   P5. Until then an ambiguous trigger is reported as `escalated` rather than
   refused, so recall is never lost to a false positive (A-5).

Pulled forward from P6, where `implementation.md` line 442 schedules input
screening, because the refusal behaviour was required before retrieval could be
validated against it.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation

from src.models import Hit, ScreenResult, VerifyResult

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "account": re.compile(
        r"(?i)(?:a/?c|account|folio)\s*(?:no|number)?\b[^0-9\n]{0,20}\d{9,18}"
    ),
    "otp": re.compile(r"(?i)(?:otp|verification code|pin)\s*[:.#]?\s*\d{4,6}"),
    "phone": re.compile(r"(?:\+?91[\s-]?)?[6-9](?:[\s-]?\d){9}"),
    "aadhaar": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}

ADVICE_TRIGGERS = [
    r"\bshould (?:i|we)\b",
    r"\bis it (?:safe|worth|good|bad)\b",
    r"\bworth investing\b",
    r"\bwhich (?:is|one is) better\b",
    r"\brecommend\b",
    r"\bsuitable for\b",
    r"\bgood time to\b",
    r"\b(?:allocate|allocation)\b",
    r"\bportfolio should\b",
    r"\bbest (?:fund|scheme|option)\b",
    r"\bshould i (?:buy|sell|switch|exit|redeem)\b",
    r"\bcan i (?:buy|sell|invest)\b",
    r"\bopinion\b",
    r"\btips?\b",
    r"\bpros and cons\b",
]

PERF_TRIGGERS = [
    r"\breturns?\b",
    r"\bCAGR\b",
    r"\bXIRR\b",
    r"\bNAV\b",
    r"\bAUM\b",
    r"\bperformance\b",
    r"\bbest performing\b",
    r"\boutperform\w*\b",
    r"\bprofit\b",
    r"\bhow much (?:did|has|will) .* (?:grow|gained|earned)\b",
    r"\bcompare returns?\b",
    r"\bexpect(?:ed)? (?:return|growth)\b",
]

_ADVICE_RE = re.compile("|".join(ADVICE_TRIGGERS), re.IGNORECASE)
_PERF_RE = re.compile("|".join(PERF_TRIGGERS), re.IGNORECASE)

# Triggers that are common enough in legitimate factual questions to need the
# classifier rather than an outright refusal. A-5 keeps these in the list on
# purpose: the escape valve protects recall, so the trigger list is not trimmed.
AMBIGUOUS_ADVICE = (r"\btips?\b", r"\bopinion\b", r"\bpros and cons\b")
AMBIGUOUS_PERF = (r"\breturns?\b", r"\bNAV\b", r"\bAUM\b", r"\bprofit\b")

_AMBIGUOUS_ADVICE_RE = re.compile("|".join(AMBIGUOUS_ADVICE), re.IGNORECASE)
_AMBIGUOUS_PERF_RE = re.compile("|".join(AMBIGUOUS_PERF), re.IGNORECASE)


def find_pii(text: str) -> str | None:
    """Return the name of the first PII pattern that matches, or None."""
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(text or ""):
            return name
    return None


def screen_input(text: str) -> ScreenResult:
    """Classify a question as FACTUAL, ADVICE, PERFORMANCE or PII.

    Never logs or returns the question text, so PII cannot escape through this
    function (FR-G5, NFR-3).
    """
    if find_pii(text):
        return ScreenResult(verdict="PII", escalated=False, pii_pattern=find_pii(text))

    advice = _ADVICE_RE.search(text or "")
    if advice:
        return ScreenResult(
            verdict="ADVICE",
            matched=[advice.group(0).lower()],
            escalated=bool(_AMBIGUOUS_ADVICE_RE.match(advice.group(0))),
        )

    performance = _PERF_RE.search(text or "")
    if performance:
        return ScreenResult(
            verdict="PERFORMANCE",
            matched=[performance.group(0).lower()],
            escalated=bool(_AMBIGUOUS_PERF_RE.match(performance.group(0))),
        )

    return ScreenResult(verdict="FACTUAL")


def template_for(screen: ScreenResult) -> str | None:
    """Map a screen verdict to the template that must be rendered for it."""
    if not screen.is_factual:
        return {
            "ADVICE": "REFUSE_ADVICE",
            "PERFORMANCE": "REFUSE_PERFORMANCE",
            "PII": "PII_BLOCKED",
        }[screen.verdict]
    return None


OUTPUT_ADVICE_PHRASES = [
    r"\byou should\b",
    r"\bi recommend\b",
    r"\bi would suggest\b",
    r"\bsuitable for you\b",
    r"\bconsider investing\b",
    r"\bi advise\b",
    r"\bwe recommend\b",
    r"\bis a good option for you\b",
]

OUTPUT_PERF_CLAIMS = [
    r"\breturns?\b",
    r"\bCAGR\b",
    r"\bXIRR\b",
    r"\boutperform\w*\b",
    r"\bexpected (?:return|growth)\b",
    r"\bwill grow to\b",
    r"\bprofit\b",
]

_ADVICE_OUTPUT_RE = re.compile("|".join(OUTPUT_ADVICE_PHRASES), re.IGNORECASE)
_PERF_OUTPUT_RE = re.compile("|".join(OUTPUT_PERF_CLAIMS), re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NUMERIC_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")
_CURRENCY_RE = re.compile(r"(?i)\b(?:inr|rs\.?|rupees)\b|[₹$€£]")


def numeric_tokens(text: str) -> set[str]:
    """Return the canonical numeric tokens in a string.

    Check 4 of `architecture.md` §5.8 is the strictest rule in the system and
    fails on formatting, not on substance: `Rs. 500` must match `Rs 500`,
    `₹1,00,000` must match `Rs 100000`, and `1.00%` must match `1%`. So both
    sides are normalized to Unicode NFKC, stripped of currency markers, and
    reduced to a canonical decimal string before comparison.
    """
    normalized = _CURRENCY_RE.sub(" ", unicodedata.normalize("NFKC", text or ""))
    tokens: set[str] = set()
    for raw in _NUMERIC_RE.findall(normalized):
        is_percent = raw.endswith("%")
        core = raw.rstrip("%").replace(",", "")
        try:
            canonical = format(Decimal(core).normalize(), "f")
        except (InvalidOperation, ValueError):
            canonical = core
        tokens.add(canonical + ("%" if is_percent else ""))
    return tokens


def count_sentences(text: str) -> int:
    """Count sentences the same way `cap_sentences` splits them."""
    return len([part for part in _SENTENCE_SPLIT_RE.split((text or "").strip()) if part])


def verify_output(
    body: str,
    hits: list[Hit],
    template_id: str,
    citation_url: str | None = None,
    max_sentences: int = 3,
) -> VerifyResult:
    """Check a generated answer body against the §5.8 table.

    Runs on the answer body only, never on the injected citation or the
    `Last updated` line, which are server-generated and carry digits of their
    own (A-7). Any failure discards the answer and the caller renders the
    template named in `violations`.
    """
    text = (body or "").strip()
    checks: dict[str, bool] = {}
    violations: list[str] = []

    checks["non_empty"] = bool(text)
    if not text:
        violations.append("NOT_FOUND")

    sentences = count_sentences(text)
    checks["sentence_count"] = sentences <= max_sentences
    if not checks["sentence_count"]:
        violations.append("NOT_FOUND")

    has_advice = bool(_ADVICE_OUTPUT_RE.search(text))
    checks["no_advice"] = not has_advice
    if has_advice:
        violations.append("REFUSE_ADVICE")

    has_perf = bool(_PERF_OUTPUT_RE.search(text))
    checks["no_performance"] = not has_perf
    if has_perf:
        violations.append("REFUSE_PERFORMANCE")

    context_numbers: set[str] = set()
    for hit in hits:
        context_numbers |= numeric_tokens(hit.text)
    unsupported = numeric_tokens(text) - context_numbers
    checks["numerics_grounded"] = not unsupported
    if unsupported:
        violations.append("NOT_FOUND")

    allowed_urls = {hit.source_url for hit in hits if hit.source_url}
    checks["citation_trusted"] = citation_url is None or citation_url in allowed_urls
    if not checks["citation_trusted"]:
        violations.append("NOT_FOUND")

    checks["refusal_not_advice"] = not (template_id.startswith("REFUSE_") and has_advice)
    if not checks["refusal_not_advice"]:
        violations.append("REFUSE_ADVICE")

    return VerifyResult(ok=not violations, checks=checks, violations=list(dict.fromkeys(violations)))


def body_without_safeguards(answer: str, citation_url: str | None, last_updated: str) -> str:
    """Strip server-generated suffixes so only the model's own text is verified."""
    body = answer or ""
    for suffix in (citation_url, last_updated):
        if suffix:
            body = body.replace(suffix, " ")
    return body