"""Tests for the rolling conversation window and follow-up rewriting.

Pure functions only: no model, no collection, no network.
"""

from __future__ import annotations

import config
from src.memory import Conversation, Message, describe, is_followup, rewrite_followup
from src.models import Hit


class TestWindow:
    def test_keeps_only_the_last_ten(self) -> None:
        conversation = Conversation()
        for index in range(15):
            conversation.add("user", f"question {index}")
        assert len(conversation.messages) == config.MEMORY_MAX_MESSAGES == 10

    def test_drops_the_oldest_first(self) -> None:
        conversation = Conversation()
        for index in range(12):
            conversation.add("user", f"question {index}")
        texts = [message.text for message in conversation.messages]
        assert texts[0] == "question 2"
        assert texts[-1] == "question 11"

    def test_counts_assistant_turns_too(self) -> None:
        conversation = Conversation()
        for index in range(6):
            conversation.add("user", f"q{index}")
            conversation.add("assistant", f"a{index}")
        assert len(conversation.messages) == 10
        assert conversation.messages[-1].role == "assistant"

    def test_default_window_is_ten(self) -> None:
        assert Conversation().max_messages == 10

    def test_custom_window_is_respected(self) -> None:
        conversation = Conversation(max_messages=2)
        for index in range(5):
            conversation.add("user", f"q{index}")
        assert len(conversation.messages) == 2

    def test_clear_empties_the_window(self) -> None:
        conversation = Conversation()
        conversation.add("user", "q")
        conversation.clear()
        assert conversation.messages == []

    def test_blank_messages_are_not_stored(self) -> None:
        conversation = Conversation()
        conversation.add("user", "   ")
        assert conversation.messages == []


class TestSchemeTracking:
    def test_user_turn_resolves_scheme_and_plan(self) -> None:
        conversation = Conversation()
        message = conversation.add("user", "What is the exit load on the Flexi Cap Direct plan?")
        assert message.scheme == "flexi_cap"
        assert message.plan == "direct"

    def test_subject_is_human_readable(self) -> None:
        conversation = Conversation()
        message = conversation.add("user", "Flexi Cap Direct plan fees")
        assert message.subject == "Axis Flexi Cap Fund (Direct plan)"

    def test_plan_only_subject_uses_scheme_display(self) -> None:
        conversation = Conversation()
        message = conversation.add("user", "ELSS tax saver lock-in")
        assert message.subject == "Axis ELSS Tax Saver Fund"

    def test_assistant_turn_has_no_subject(self) -> None:
        conversation = Conversation()
        assert conversation.add("assistant", "Exit Load NIL.").subject is None

    def test_off_topic_turn_has_no_subject(self) -> None:
        conversation = Conversation()
        assert conversation.add("user", "How do I cook pasta?").subject is None


class TestFollowupDetection:
    def test_detects_dangling_pronouns(self) -> None:
        for question in (
            "what about its fees?",
            "what about the exit load?",
            "what are its limits?",
            "how much does it cost?",
            "is this eligible for deduction?",
            "and the exit load?",
        ):
            assert is_followup(question) is True

    def test_standalone_question_is_not_a_followup(self) -> None:
        assert is_followup("What is the exit load on the Flexi Cap Direct plan?") is False

    def test_question_naming_a_scheme_is_standalone(self) -> None:
        assert is_followup("Is this Flexi Cap fund's lock-in three years?") is False

    def test_a_resolved_alias_counts_as_standalone(self) -> None:
        # "that" dangles, but "80C" names the ELSS scheme, so the question can
        # stand on its own and must not be rewritten.
        assert is_followup("is that eligible for 80C?") is False

    def test_empty_is_not_a_followup(self) -> None:
        assert is_followup("") is False

    def test_question_with_no_reference_is_left_alone(self) -> None:
        # Documented limit: with no pronoun and no follow-up opener there is no
        # signal to resolve, so guessing a scheme would be worse than admitting
        # the question is ambiguous.
        assert is_followup("how long is the lock-in?") is False


class TestRewriteFollowup:
    def test_resolves_its_from_the_previous_turn(self) -> None:
        conversation = Conversation()
        conversation.add("user", "What is the exit load on the Axis Flexi Cap Direct plan?")
        assert rewrite_followup("what about its fees?", conversation) == (
            "Axis Flexi Cap Fund (Direct plan): what about its fees?"
        )

    def test_standalone_question_is_untouched(self) -> None:
        conversation = Conversation()
        conversation.add("user", "What is the exit load on the Flexi Cap Direct plan?")
        question = "What is the minimum SIP?"
        assert rewrite_followup(question, conversation) == question

    def test_no_conversation_is_untouched(self) -> None:
        assert rewrite_followup("what about its fees?", None) == "what about its fees?"

    def test_unresolvable_reference_is_untouched(self) -> None:
        conversation = Conversation()
        conversation.add("user", "How do I cook pasta?")
        assert rewrite_followup("what about its fees?", conversation) == "what about its fees?"

    def test_empty_history_is_untouched(self) -> None:
        assert rewrite_followup("what about its fees?", Conversation()) == "what about its fees?"

    def test_uses_the_most_recent_subject(self) -> None:
        conversation = Conversation()
        conversation.add("user", "Flexi Cap Direct plan fees")
        conversation.add("user", "Large Cap Regular plan fees")
        assert rewrite_followup("what about its exit load?", conversation).startswith(
            "Axis Large Cap Fund (Regular plan):"
        )

    def test_rewritten_question_names_a_scheme_for_the_embedder(self) -> None:
        conversation = Conversation()
        conversation.add("user", "What is the exit load on the ELSS tax saver fund?")
        rewritten = rewrite_followup("what about its fees?", conversation)
        from src.retrieve import infer_scheme

        assert infer_scheme(rewritten) == "elss"


class TestRecentSubject:
    def test_uses_only_user_turns(self) -> None:
        conversation = Conversation()
        conversation.add("assistant", "Axis Flexi Cap Fund (Direct plan)")
        assert conversation.recent_subject() is None

    def test_returns_none_without_history(self) -> None:
        assert Conversation().recent_subject() is None

    def test_skips_turns_with_no_subject(self) -> None:
        conversation = Conversation()
        conversation.add("user", "Flexi Cap Direct plan fees")
        conversation.add("user", "thanks")
        assert conversation.recent_subject() == "Axis Flexi Cap Fund (Direct plan)"


class TestDescribe:
    def test_reports_empty(self) -> None:
        assert describe(Conversation()) == "no history"
        assert describe(None) == "no history"

    def test_reports_the_window_size(self) -> None:
        conversation = Conversation()
        conversation.add("user", "q")
        assert describe(conversation) == "1/10 messages in memory"