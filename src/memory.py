"""Conversation memory and follow-up question resolution.

A follow-up like "what about its fees?" is unanswerable on its own: it contains
no scheme name, so the embedder has nothing to match against and retrieval falls
back to whole-corpus similarity. `rewrite_followup` resolves the dangling
reference to the subject of the most recent relevant turn and returns a
standalone question for retrieval.

The resolver is a pure function over stored messages. It never calls a model, so
every rewrite rule is unit-testable offline and a rewrite can never invent a
scheme name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import config
from src.retrieve import infer_plan, infer_scheme

SCHEME_DISPLAY = {
    "large_cap": "Axis Large Cap Fund",
    "flexi_cap": "Axis Flexi Cap Fund",
    "elss": "Axis ELSS Tax Saver Fund",
    "midcap": "Axis Midcap Fund",
    "amc_wide": "Axis Mutual Fund",
}

PLAN_DISPLAY = {
    "direct": "Direct plan",
    "regular": "Regular plan",
    "n_a": "",
}

DANGLING_REFERENCE = re.compile(
    r"\b(?:its|it|it's|this|that|these|those|they|them|their|theirs|he|she|her|his)\b",
    re.IGNORECASE,
)

# "what about the exit load?" has no pronoun at all, but the definite article is
# the same dangling reference, so follow-up openers count on their own.
FOLLOWUP_OPENER = re.compile(r"^\s*(?:and\s+)?(?:what|how)\s+about\b", re.IGNORECASE)

# "and the exit load?" elides the subject entirely, which in a chat transcript
# only reads as a continuation.
ELIDED_SUBJECT = re.compile(
    r"^\s*(?:and|also|so|okay\s+and|ok\s+and|and\s+then|then)\b", re.IGNORECASE
)


@dataclass
class Message:
    """One stored turn. `scheme` and `plan` are resolved when it is added."""

    role: str
    text: str
    scheme: str | None = None
    plan: str | None = None
    template_id: str | None = None

    @property
    def subject(self) -> str | None:
        """Readable referent for this turn, e.g. `Axis Flexi Cap Fund (Direct plan)`."""
        if not self.scheme and not self.plan:
            return None
        name = SCHEME_DISPLAY.get(self.scheme or "", "")
        if self.scheme and not name:
            return None
        if self.scheme and self.plan and PLAN_DISPLAY.get(self.plan):
            return f"{name} ({PLAN_DISPLAY[self.plan]})"
        return name or None


@dataclass
class Conversation:
    """Rolling window of the last `MEMORY_MAX_MESSAGES` turns."""

    messages: list[Message] = field(default_factory=list)
    max_messages: int | None = None

    def __post_init__(self) -> None:
        if self.max_messages is None:
            self.max_messages = config.MEMORY_MAX_MESSAGES

    def add(self, role: str, text: str, template_id: str | None = None) -> Message | None:
        """Append a turn, dropping the oldest once the window is full.

        Returns None for blank input so an empty REPL line cannot consume a slot
        in the window.
        """
        clean = (text or "").strip()
        if not clean:
            return None
        if role == "user":
            message = Message(
                role=role,
                text=clean,
                scheme=infer_scheme(clean),
                plan=infer_plan(clean),
                template_id=template_id,
            )
        else:
            message = Message(role=role, text=clean, template_id=template_id)
        self.messages.append(message)
        if self.max_messages is not None and len(self.messages) > self.max_messages:
            del self.messages[: len(self.messages) - self.max_messages]
        return message

    def user_turns(self) -> list[Message]:
        return [message for message in self.messages if message.role == "user"]

    def recent_subject(self, before: int | None = None) -> str | None:
        """Subject of the most recent user turn that named a scheme or plan."""
        turns = self.user_turns()
        if before is not None:
            turns = turns[:before]
        for message in reversed(turns):
            if message.subject:
                return message.subject
        return None

    def clear(self) -> None:
        self.messages.clear()

    def as_tuples(self) -> list[tuple[str, str]]:
        return [(message.role, message.text) for message in self.messages]


def is_followup(question: str) -> bool:
    """True when the question dangles off an earlier turn.

    A question that already names its scheme is standalone even if it contains
    "this" or "it", so scheme resolution is required before calling it a follow-up.
    """
    text = (question or "").strip()
    if not text or infer_scheme(text):
        return False
    return bool(
        DANGLING_REFERENCE.search(text)
        or FOLLOWUP_OPENER.search(text)
        or ELIDED_SUBJECT.search(text)
    )


def rewrite_followup(question: str, conversation: Conversation | None) -> str:
    """Return a standalone question for retrieval.

    Standalone questions pass through untouched. A follow-up with no resolvable
    subject also passes through, because guessing a scheme would be worse than
    admitting the question was ambiguous.
    """
    text = (question or "").strip()
    if conversation is None or not is_followup(text):
        return text
    subject = conversation.recent_subject()
    if not subject:
        return text
    return f"{subject}: {text}"


def describe(conversation: Conversation | None) -> str:
    """One-line summary of the window, for the CLI header."""
    if conversation is None or not conversation.messages:
        return "no history"
    limit = conversation.max_messages or config.MEMORY_MAX_MESSAGES
    return f"{len(conversation.messages)}/{limit} messages in memory"