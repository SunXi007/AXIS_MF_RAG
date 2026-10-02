"""Tests for generation, citation injection and the output verifier.

Every test uses a fake Groq client, so the suite runs offline and never needs a
key (`architecture.md` §7.1).
"""

from __future__ import annotations

from typing import Any

import pytest

import config
from src import generate, templates
from src.guardrails import numeric_tokens, verify_output
from src.models import Hit


class _Obj:
    """Mutable attribute bag standing in for a pydantic response object."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeStream:
    def __init__(self, text: str) -> None:
        self.text = text

    def __iter__(self):
        for piece in self.text.split(" "):
            yield _Obj(choices=[_Obj(delta=_Obj(content=piece + " "))])


class FakeClient:
    """Minimal stand-in for `groq.Groq` that replays a canned reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, **kwargs: Any) -> FakeStream:
        self.calls.append(kwargs)
        return FakeStream(self.reply)


def make_hit(chunk_id: str, text: str, score: float = 0.8, **meta: Any) -> Hit:
    base = {
        "source_url": f"https://axis.test/{chunk_id}",
        "source_title": "Axis Test",
        "scheme": "large_cap",
        "plan": "direct",
        "doc_type": "scheme_page",
        "source_id": chunk_id.split("-")[0],
        "source_date": "2026-06-30",
    }
    base.update(meta)
    return Hit(chunk_id=chunk_id, text=text, metadata=base, score=score)


HITS = [
    make_hit("s01-c017", "Exit Load NIL Entry Load N/A Minimum Investment Amount Rs 500"),
    make_hit("s02-c016", "Exit Load NIL Entry Load N/A Minimum SIP Rs 500"),
]


class TestNumericTokens:
    def test_currency_prefix_variants_match(self) -> None:
        assert numeric_tokens("Rs. 500") == numeric_tokens("Rs 500") == {"500"}

    def test_rupee_symbol_and_grouping(self) -> None:
        assert numeric_tokens("₹1,00,000") == numeric_tokens("Rs 100000") == {"100000"}

    def test_trailing_zeros_are_insignificant(self) -> None:
        assert numeric_tokens("1.00%") == numeric_tokens("1%") == {"1%"}

    def test_percent_is_distinct_from_bare_number(self) -> None:
        assert numeric_tokens("5%") != numeric_tokens("5")

    def test_unicode_normalization(self) -> None:
        assert numeric_tokens("Rs 500") == numeric_tokens("Rs 500")

    def test_no_digits_yields_empty_set(self) -> None:
        assert numeric_tokens("Exit Load NIL") == set()

    def test_date_digits_are_captured(self) -> None:
        assert numeric_tokens("as on 2026-06-30") == {"2026", "6", "30"}


class TestVerifyOutput:
    def test_grounded_answer_passes(self) -> None:
        result = verify_output("Exit Load NIL.", HITS, "FACTUAL", HITS[0].source_url)
        assert result.ok
        assert all(result.checks.values())

    def test_invented_number_is_caught(self) -> None:
        result = verify_output("The exit load is 7.5%.", HITS, "FACTUAL", HITS[0].source_url)
        assert not result.ok
        assert "NOT_FOUND" in result.violations

    def test_advice_phrasing_is_caught(self) -> None:
        result = verify_output("You should invest here.", HITS, "FACTUAL", HITS[0].source_url)
        assert not result.ok
        assert "REFUSE_ADVICE" in result.violations

    def test_performance_claim_is_caught(self) -> None:
        result = verify_output("The fund returns 15%.", HITS, "FACTUAL", HITS[0].source_url)
        assert not result.ok
        assert "REFUSE_PERFORMANCE" in result.violations

    def test_too_many_sentences_is_caught(self) -> None:
        text = "One. Two. Three. Four."
        result = verify_output(text, HITS, "FACTUAL", HITS[0].source_url)
        assert not result.ok
        assert not result.checks["sentence_count"]

    def test_uncited_url_is_caught(self) -> None:
        result = verify_output(
            "Exit Load NIL.", HITS, "FACTUAL", "https://evil.test/made-up"
        )
        assert not result.ok
        assert not result.checks["citation_trusted"]

    def test_no_context_at_all_fails_numerics(self) -> None:
        result = verify_output("Exit Load is 5%.", [], "FACTUAL", None)
        assert not result.ok
        assert not result.checks["numerics_grounded"]

    def test_number_free_answer_without_context_is_fine(self) -> None:
        assert verify_output("Exit Load NIL.", [], "FACTUAL", None).ok


class TestEmptyReply:
    def test_empty_model_reply_is_never_factual(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient(""))
        assert envelope.template_id == "NOT_FOUND"
        assert envelope.answer == templates.render("NOT_FOUND")
        assert envelope.citation_url is None

    def test_empty_body_fails_the_verifier(self) -> None:
        result = verify_output("   ", HITS, "FACTUAL", HITS[0].source_url)
        assert not result.ok
        assert not result.checks["non_empty"]


class TestReasoningEffort:
    def test_gpt_oss_models_get_a_pinned_effort(self) -> None:
        assert generate.reasoning_supports_effort("openai/gpt-oss-20b") is True

    def test_other_models_do_not(self) -> None:
        assert generate.reasoning_supports_effort("llama-3.1-8b-instant") is False

    def test_effort_is_sent_for_gpt_oss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "GROQ_MODEL", "openai/gpt-oss-20b")
        client = FakeClient("Exit Load NIL.")
        generate.call_model("q", "c", client)
        assert client.calls[0]["reasoning_effort"] == config.GROQ_REASONING_EFFORT

    def test_effort_is_omitted_for_other_models(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "GROQ_MODEL", "some-other-model")
        client = FakeClient("Exit Load NIL.")
        generate.call_model("q", "c", client)
        assert "reasoning_effort" not in client.calls[0]

    def test_rounding_a_context_number_is_caught(self) -> None:
        result = verify_output("The minimum is Rs 500.", HITS, "FACTUAL", HITS[0].source_url)
        assert result.ok
        bad = verify_output("The minimum is Rs 600.", HITS, "FACTUAL", HITS[0].source_url)
        assert not bad.ok


class TestCapSentences:
    def test_keeps_short_text_unchanged(self) -> None:
        assert generate.cap_sentences("Exit load is NIL.") == ("Exit load is NIL.", False)

    def test_truncates_to_three(self) -> None:
        text, truncated = generate.cap_sentences("One. Two. Three. Four. Five.")
        assert text == "One. Two. Three."
        assert truncated is True

    def test_truncation_restores_punctuation(self) -> None:
        text, _ = generate.cap_sentences("One. Two. Three. Four")
        assert text.endswith(".")

    def test_honours_explicit_limit(self) -> None:
        text, truncated = generate.cap_sentences("One. Two. Three.", max_sentences=2)
        assert text == "One. Two."
        assert truncated is True


class TestBuildContext:
    def test_numbers_and_labels_chunks(self) -> None:
        context = generate.build_context(HITS)
        assert "[1] SOURCE" in context
        assert "[2] SOURCE" in context
        assert "scheme=large_cap" in context

    def test_respects_char_cap_but_keeps_first_chunk(self) -> None:
        context = generate.build_context(HITS, max_chars=10)
        assert context.startswith("[1] SOURCE")
        assert "[2] SOURCE" not in context

    def test_cap_defaults_to_config(self) -> None:
        assert config.MAX_CONTEXT_CHARS == 12000


class TestCapLength:
    def test_short_text_is_untouched(self) -> None:
        assert generate.cap_length("Exit load is NIL.") == ("Exit load is NIL.", False)

    def test_long_run_on_is_trimmed(self) -> None:
        text, truncated = generate.cap_length("a, b, c, " * 200, max_chars=200)
        assert len(text) <= 203
        assert truncated is True

    def test_prefers_a_sentence_end(self) -> None:
        text, _ = generate.cap_length(
            "First sentence here. Second sentence here. Third one here.", max_chars=30
        )
        assert text == "First sentence here."

    def test_honours_explicit_limit(self) -> None:
        text, truncated = generate.cap_length("aaaa bbbb cccc", max_chars=5)
        assert len(text) <= 8
        assert truncated is True

    def test_config_bound_is_four_hundred_twenty(self) -> None:
        assert config.MAX_ANSWER_CHARS == 420

    def test_run_on_sentence_is_capped_end_to_end(self) -> None:
        run_on = "Up to 2.25% of daily net assets, " + "plus trustee, audit, " * 40
        envelope = generate.answer("q", HITS, client=FakeClient(run_on))
        assert len(envelope.answer) <= config.MAX_ANSWER_CHARS + 3


class TestCitation:
    def test_url_comes_from_metadata_not_the_model(self) -> None:
        client = FakeClient("Exit Load NIL.")
        envelope = generate.answer("q", HITS, client=client)
        assert envelope.citation_url == HITS[0].source_url
        assert "http" not in envelope.raw_reply

    def test_cites_best_hit_above_floor(self) -> None:
        hits = [make_hit("s01-c001", "x", score=0.10), make_hit("s02-c002", "y", score=0.90)]
        assert generate.pick_citation(hits).chunk_id == "s02-c002"

    def test_skips_hits_below_floor_without_url(self) -> None:
        hits = [make_hit("s01-c001", "x", score=0.99, source_url="")]
        assert generate.pick_citation(hits) is None

    def test_last_updated_uses_max_date(self) -> None:
        hits = [make_hit("s01-c001", "x", source_date="2026-01-01"),
                make_hit("s02-c002", "y", source_date="2026-06-30")]
        assert "2026-06-30" in generate.last_updated_line(hits)

    def test_missing_date_is_stated_not_invented(self) -> None:
        hits = [make_hit("s01-c001", "x", source_date=None)]
        assert "date not stated on page" in generate.last_updated_line(hits)


class TestSentinels:
    def test_not_found_sentinel_maps_to_template(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("NOT_FOUND"))
        assert envelope.template_id == "NOT_FOUND"
        assert envelope.answer == templates.render("NOT_FOUND")

    def test_no_perf_sentinel_maps_to_refusal(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("NO_PERF"))
        assert envelope.template_id == "REFUSE_PERFORMANCE"

    def test_sentinel_discards_citation(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("NOT_FOUND"))
        assert envelope.citation_url is None

    def test_empty_hits_short_circuits(self) -> None:
        client = FakeClient("should never be called")
        envelope = generate.answer("q", [], client=client)
        assert envelope.template_id == "NO_RETRIEVAL"
        assert client.calls == []


class TestVerifierIntegration:
    def test_fabricated_number_replaced_by_template(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("The exit load is 9.99%."))
        assert envelope.template_id == "NOT_FOUND"
        assert envelope.answer == templates.render("NOT_FOUND")
        assert envelope.citation_url is None

    def test_advice_replaced_by_refusal(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("You should invest here."))
        assert envelope.template_id == "REFUSE_ADVICE"

    def test_clean_answer_is_kept(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("Exit Load NIL."))
        assert envelope.template_id == "FACTUAL"
        assert envelope.answer == "Exit Load NIL."
        assert envelope.guardrail_trace["verifier"]["ok"] is True

    def test_truncation_is_recorded(self) -> None:
        client = FakeClient("Exit Load NIL. Entry Load N/A. Tax as applicable. Extra trailing.")
        envelope = generate.answer("q", HITS, client=client)
        assert envelope.guardrail_trace["truncated_by_cap"] is True

    def test_chunk_ids_are_recorded_for_trace(self) -> None:
        envelope = generate.answer("q", HITS, client=FakeClient("Exit Load NIL."))
        assert envelope.guardrail_trace["chunk_ids"] == ["s01-c017", "s02-c016"]


class TestGroqCall:
    def test_uses_locked_generation_params(self) -> None:
        client = FakeClient("Exit Load NIL.")
        generate.call_model("q", "CONTEXT", client)
        call = client.calls[0]
        assert call["model"] == config.GROQ_MODEL
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 220
        assert call["stream"] is True

    def test_system_prompt_carries_the_context(self) -> None:
        client = FakeClient("x")
        generate.call_model("q", "MARKER-CONTEXT", client)
        system = client.calls[0]["messages"][0]["content"]
        assert "MARKER-CONTEXT" in system
        assert "STRICT RULES" in system

    def test_question_is_the_user_message(self) -> None:
        client = FakeClient("x")
        generate.call_model("my question", "ctx", client)
        assert client.calls[0]["messages"][1]["content"] == "my question"

    def test_stream_is_reassembled(self) -> None:
        assert generate.call_model("q", "c", FakeClient("Exit Load NIL.")) == "Exit Load NIL."


class TestRespondShortCircuits:
    def test_advice_never_reaches_the_model(self) -> None:
        client = FakeClient("should not be called")
        envelope = generate.respond("Should I invest in the ELSS fund?", client=client)
        assert envelope.template_id == "REFUSE_ADVICE"
        assert client.calls == []
        assert envelope.guardrail_trace["stages_skipped"] == ["embed", "search", "generate"]

    def test_pii_never_reaches_the_model(self) -> None:
        client = FakeClient("should not be called")
        envelope = generate.respond("My PAN is ABCDE1234F", client=client)
        assert envelope.template_id == "PII_BLOCKED"
        assert client.calls == []

    def test_off_topic_question_is_refused(self) -> None:
        class Stub:
            fingerprint = generate.config.EMBED_MODEL and "stub"

        client = FakeClient("should not be called")
        envelope = generate.respond(
            "What is the weather in Mumbai tomorrow?",
            client=client,
            threshold=0.99,
            top_k=5,
        )
        assert envelope.template_id in {"NO_RETRIEVAL", "NOT_FOUND"}
        assert client.calls == []