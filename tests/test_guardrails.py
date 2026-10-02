"""Tests for input guardrails and response templates.

No network, no LLM, no collection: every refusal path is a pure function, which
is the seam `architecture.md` §7.1 relies on.
"""

from __future__ import annotations

import re

import pytest

from src import templates
from src.guardrails import find_pii, screen_input, template_for


class TestPiiPatterns:
    @pytest.mark.parametrize(
        "text,pattern",
        [
            ("My PAN is ABCDE1234F", "pan"),
            ("aadhaar 1234 5678 9012", "aadhaar"),
            ("a/c no 123456789012", "account"),
            ("my folio number is 9876543210", "account"),
            ("otp 483920 please", "otp"),
            ("write to sunil@example.com", "email"),
            ("call me on 9876543210", "phone"),
            ("+91 98765 43210", "phone"),
        ],
    )
    def test_detects_identifier(self, text: str, pattern: str) -> None:
        assert find_pii(text) == pattern

    @pytest.mark.parametrize(
        "text",
        [
            "What is the minimum investment of Rs 5000?",
            "What is the exit load after 365 days?",
            "Is the benchmark Nifty 100 TRI?",
            "What is the expense ratio of the Flexi Cap Fund?",
        ],
    )
    def test_allows_legitimate_numbers(self, text: str) -> None:
        assert find_pii(text) is None
        assert screen_input(text).verdict == "FACTUAL"

    def test_bare_digit_run_is_never_called_an_account(self) -> None:
        assert find_pii("123456789012345678") != "account"


class TestAdvicePrecedence:
    def test_advice_question_is_refused(self) -> None:
        result = screen_input("Should I invest in the Axis ELSS Tax Saver Fund?")
        assert result.verdict == "ADVICE"
        assert result.matched == ["should i"]
        assert template_for(result) == "REFUSE_ADVICE"

    def test_advice_beats_performance(self) -> None:
        result = screen_input("Should I buy the fund that performed best?")
        assert result.verdict == "ADVICE"

    @pytest.mark.parametrize(
        "text",
        [
            "Which is better, Direct or Regular?",
            "Is it safe to invest in the Midcap fund?",
            "Recommend a fund for me",
            "What is a good time to invest?",
            "How should I allocate my money?",
            "Should I buy the Flexi Cap Fund?",
            "Can I sell my ELSS units?",
            "Any tips for equity investing?",
        ],
    )
    def test_advice_triggers(self, text: str) -> None:
        assert screen_input(text).verdict == "ADVICE"

    def test_ambiguous_advice_is_escalated_not_silently_refused(self) -> None:
        assert screen_input("Any tips for a beginner?").escalated is True
        assert screen_input("Should I invest?").escalated is False


class TestPerformanceRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "What are the returns of the Large Cap Fund?",
            "What is the CAGR?",
            "What is the NAV?",
            "What is the AUM?",
            "Which fund outperformed?",
            "Compare returns of Direct and Regular",
        ],
    )
    def test_performance_triggers(self, text: str) -> None:
        result = screen_input(text)
        assert result.verdict == "PERFORMANCE"
        assert template_for(result) == "REFUSE_PERFORMANCE"

    def test_ambiguous_performance_is_escalated(self) -> None:
        assert screen_input("What is the NAV growth?").escalated is True


class TestFactualPassthrough:
    @pytest.mark.parametrize(
        "text",
        [
            "What is the exit load on the Axis Large Cap Regular Growth plan?",
            "What is the minimum SIP amount for the Flexi Cap Fund?",
            "What is the ELSS lock-in period?",
            "How do I download a statement?",
            "What is the expense ratio of the Midcap Direct plan?",
            "Who is the custodian of the ELSS Tax Saver Fund?",
        ],
    )
    def test_factual_questions_reach_retrieval(self, text: str) -> None:
        result = screen_input(text)
        assert result.verdict == "FACTUAL"
        assert template_for(result) is None


class TestPiiNeverLeaksText:
    def test_result_carries_no_question_text(self) -> None:
        secret = "ABCDE1234F"
        result = screen_input(f"My PAN is {secret}")
        assert result.verdict == "PII"
        assert result.pii_pattern == "pan"
        assert secret not in str(result)

    def test_pii_takes_precedence_over_advice(self) -> None:
        result = screen_input("Should I invest? My PAN is ABCDE1234F")
        assert result.verdict == "PII"
        assert template_for(result) == "PII_BLOCKED"

    def test_empty_input_is_factual(self) -> None:
        assert screen_input("").verdict == "FACTUAL"


class TestTemplates:
    def test_all_six_ids_render(self) -> None:
        for template_id in templates.TEMPLATE_IDS:
            rendered = templates.render(
                template_id, **({"answer": "Exit load is 1%."} if template_id == "FACTUAL" else {})
            )
            assert rendered.strip()

    def test_factual_renders_empty_without_an_answer(self) -> None:
        assert templates.render("FACTUAL") == ""

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(KeyError):
            templates.render("MADE_UP")

    def test_factual_passes_answer_through(self) -> None:
        assert templates.render("FACTUAL", answer="  Exit load is 1%.  ") == "Exit load is 1%."

    def test_refusals_carry_no_unsourced_numbers(self) -> None:
        for template_id in ("REFUSE_ADVICE", "REFUSE_PERFORMANCE", "NOT_FOUND"):
            body = re.sub(r"https?://\S+", "", templates.render(template_id))
            assert not re.search(r"\d", body)

    def test_not_found_names_the_schemes_in_scope(self) -> None:
        rendered = templates.render("NOT_FOUND")
        for scheme in ("Large Cap", "Flexi Cap", "ELSS Tax Saver", "Midcap"):
            assert scheme in rendered

    def test_advice_refusal_points_at_sebi_adviser(self) -> None:
        assert "SEBI-registered investment adviser" in templates.render("REFUSE_ADVICE")

    def test_performance_refusal_links_the_factsheet(self) -> None:
        assert templates.render("REFUSE_PERFORMANCE").endswith(templates.FACTSHEET_URL)

    def test_urls_can_be_overridden(self) -> None:
        assert "https://example.test/x" in templates.render(
            "REFUSE_ADVICE", downloads_url="https://example.test/x"
        )

    def test_disclaimer_matches_prd_verbatim(self) -> None:
        from pathlib import Path

        prd = (Path("docs") / "PRD.md").read_text(encoding="utf-8")
        for line in templates.DISCLAIMER.splitlines():
            assert line in prd

    def test_disclaimer_is_the_required_text(self) -> None:
        lines = templates.DISCLAIMER.splitlines()
        assert lines[0] == "Facts-only. No investment advice."
        assert len(lines) == 3


class TestPromptContract:
    def test_prompt_has_all_six_rules(self) -> None:
        for number in range(1, 7):
            assert f"{number}. " in templates.PROMPT

    def test_prompt_forbids_urls_and_unrounded_numbers(self) -> None:
        assert "Do NOT write any URL" in templates.PROMPT
        assert "Never round" in templates.PROMPT

    def test_prompt_requires_not_found_sentinel(self) -> None:
        assert "NOT_FOUND" in templates.PROMPT
        assert "NO_PERF" in templates.PROMPT

    def test_prompt_takes_a_context_placeholder(self) -> None:
        assert "{context}" in templates.PROMPT
        assert templates.PROMPT.format(context="X").count("{context}") == 0