"""Stage 3-5 - grounded generation with a server-injected citation.

The LLM is never trusted. It sees the system prompt plus numbered chunks, it may
only copy numbers verbatim, and it may not write a URL: the citation is injected
server-side from hit metadata (`architecture.md` §5.4-§5.6, AD-3). Anything that
fails `verify_output` is discarded and replaced by a template, so the worst case
is a refusal, never a fabricated figure.

    python -m src.generate "What is the exit load on the Axis Large Cap Regular plan?"
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Any, Sequence

import config
from src import templates
from src.embed import EmbedderProtocol, load_embedder
from src.guardrails import (
    body_without_safeguards,
    count_sentences,
    screen_input,
    template_for,
    verify_output,
)
from src.models import AnswerEnvelope, Hit, TemplateId
from src.memory import Conversation, describe, rewrite_followup
from src.retrieve import retrieve
from src.trace import TraceRecord, emit, log_pii_blocked, question_hash

NOT_FOUND_SENTINEL = "NOT_FOUND"
NO_PERF_SENTINEL = "NO_PERF"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def build_context(hits: Sequence[Hit], max_chars: int | None = None) -> str:
    """Render hits as a numbered, labelled block, capped at `MAX_CONTEXT_CHARS`.

    Chunk 1 is always kept even if it alone exceeds the cap, because dropping the
    best match would silently turn an answerable question into a refusal.
    """
    limit = config.MAX_CONTEXT_CHARS if max_chars is None else max_chars
    parts: list[str] = []
    used = 0
    for index, hit in enumerate(hits, start=1):
        meta = hit.metadata
        label = (
            f"[{index}] SOURCE  {meta.get('source_title') or meta.get('source_id') or ''}"
            f"  (scheme={meta.get('scheme')}, plan={meta.get('plan')},"
            f" doc_type={meta.get('doc_type')}, page={meta.get('page')})"
        )
        block = f"{label}\n{hit.text}"
        if parts and used + len(block) > limit:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def cap_sentences(text: str, max_sentences: int | None = None) -> tuple[str, bool]:
    """Keep at most `MAX_SENTENCES`, re-asserting trailing punctuation."""
    limit = config.MAX_SENTENCES if max_sentences is None else max_sentences
    cleaned = (text or "").strip()
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]
    if len(parts) <= limit:
        return cleaned, False
    kept = " ".join(parts[:limit]).strip()
    if kept and kept[-1] not in ".!?":
        kept += "."
    return kept, True


def cap_length(text: str, max_chars: int | None = None) -> tuple[str, bool]:
    """Trim to `MAX_ANSWER_CHARS`, preferring a sentence end, then a clause end.

    A sentence counter is not a length guard, so this backs it up: the cut is made
    at the last terminator inside the budget so the answer still reads as
    complete prose rather than stopping mid-word.
    """
    limit = config.MAX_ANSWER_CHARS if max_chars is None else max_chars
    clean = (text or "").strip()
    if len(clean) <= limit:
        return clean, False

    window = clean[:limit]
    for boundary in (". ", "? ", "! ", "; ", ", "):
        cut = window.rfind(boundary)
        if cut > limit // 3:
            return window[: cut + 1].strip(), True
    return window.strip().rstrip(",;") + "...", True


def groq_client() -> Any:
    """Build the production Groq client. Imported lazily so tests need no key."""
    if not config.key_present():
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")
    from groq import Groq

    return Groq(api_key=config.GROQ_API_KEY)


def reasoning_supports_effort(model: str) -> bool:
    """True for models that accept `reasoning_effort` (Groq's gpt-oss family)."""
    return "gpt-oss" in model


def call_model(question: str, context: str, client: Any) -> str:
    """Call Groq once and return the raw reply. Injectable for tests."""
    extra: dict[str, Any] = {}
    if reasoning_supports_effort(config.GROQ_MODEL):
        extra["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    stream = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": templates.PROMPT.format(context=context)},
            {"role": "user", "content": question},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        stream=True,
        **extra,
    )
    chunks: list[str] = []
    for event in stream:
        piece = event.choices[0].delta.content
        if piece:
            chunks.append(piece)
    return "".join(chunks).strip()


def pick_citation(hits: Sequence[Hit]) -> Hit | None:
    """Cite the best hit that clears `CITATION_SCORE_FLOOR` and has a URL."""
    for hit in hits:
        if hit.score >= config.CITATION_SCORE_FLOOR and hit.source_url:
            return hit
    for hit in hits:
        if hit.source_url:
            return hit
    return None


def last_updated_line(cited: Sequence[Hit]) -> str:
    """Render the freshness line, never inventing a date."""
    dates = [hit.source_date for hit in cited if hit.source_date]
    if dates:
        return f"Last updated from sources: {max(dates)}"
    return "Last updated from sources: date not stated on page"


def answer(
    question: str,
    hits: Sequence[Hit],
    cfg: Any | None = None,
    client: Any | None = None,
) -> AnswerEnvelope:
    """Generate, cap, cite and verify an answer. Assumes screening already passed."""
    envelope = AnswerEnvelope(
        answer="",
        template_id="FACTUAL",
        hits=list(hits),
        latency_ms={},
        guardrail_trace={},
    )

    if not hits:
        envelope.template_id = "NO_RETRIEVAL"
        envelope.answer = templates.render("NO_RETRIEVAL")
        return envelope

    started = time.perf_counter()
    context = build_context(hits)
    envelope.latency_ms["build_context"] = int((time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    raw = call_model(question, context, client if client is not None else groq_client())
    envelope.latency_ms["groq"] = int((time.perf_counter() - started) * 1000)
    envelope.raw_reply = raw

    stripped = raw.strip()
    if stripped == NOT_FOUND_SENTINEL or stripped.startswith(NOT_FOUND_SENTINEL):
        envelope.template_id = "NOT_FOUND"
        envelope.answer = templates.render("NOT_FOUND")
        envelope.guardrail_trace["sentinel"] = NOT_FOUND_SENTINEL
        return envelope
    if stripped == NO_PERF_SENTINEL or stripped.startswith(NO_PERF_SENTINEL):
        envelope.template_id = "REFUSE_PERFORMANCE"
        envelope.answer = templates.render("REFUSE_PERFORMANCE")
        envelope.guardrail_trace["sentinel"] = NO_PERF_SENTINEL
        return envelope

    body, truncated = cap_sentences(raw)
    body, length_truncated = cap_length(body)
    truncated = truncated or length_truncated
    envelope.cited_chunk_id = None
    cited = pick_citation(hits)
    if cited is not None:
        envelope.cited_chunk_id = cited.chunk_id
        envelope.citation_url = cited.source_url
        envelope.source_title = str(cited.metadata.get("source_title") or "")
        envelope.last_updated = last_updated_line([cited])

    envelope.answer = body
    envelope.guardrail_trace["truncated_by_cap"] = truncated
    envelope.guardrail_trace["sentence_count"] = count_sentences(body)
    envelope.guardrail_trace["chunk_ids"] = [hit.chunk_id for hit in hits]

    verified = verify_output(
        body_without_safeguards(body, envelope.citation_url, envelope.last_updated or ""),
        list(hits),
        "FACTUAL",
        envelope.citation_url,
    )
    envelope.guardrail_trace["verifier"] = {
        "ok": verified.ok,
        "checks": verified.checks,
        "violations": verified.violations,
    }
    if not verified.ok:
        envelope.template_id = verified.violations[0]
        envelope.answer = templates.render(envelope.template_id)
        envelope.citation_url = None
        envelope.cited_chunk_id = None

    return envelope


def respond(
    question: str,
    *,
    client: Any | None = None,
    embedder: EmbedderProtocol | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    conversation: Conversation | None = None,
    remember: bool = True,
) -> AnswerEnvelope:
    """Full pipeline: screen, resolve follow-ups, retrieve, generate, verify.

    Screening runs on the raw question so advice and PII triggers are judged on
    what the user actually typed. Retrieval runs on the rewritten standalone
    question, because a dangling "its" carries no scheme name for the embedder
    to match.
    """
    text = (question or "").strip()
    started = time.perf_counter()

    if conversation is not None and remember:
        conversation.add("user", text)

    screen = screen_input(text)
    trace: dict[str, Any] = {
        "verdict": screen.verdict,
        "matched": screen.matched,
        "escalated": screen.escalated,
        "pii_pattern": screen.pii_pattern,
    }

    retrieval_question = rewrite_followup(text, conversation)
    if retrieval_question != text:
        trace["rewritten_question"] = retrieval_question
        trace["follow_up"] = True
    else:
        trace["follow_up"] = False

    if not screen.is_factual:
        if screen.verdict == "PII":
            log_pii_blocked(screen.pii_pattern or "unknown")
        envelope = AnswerEnvelope(
            answer=templates.render(template_for(screen) or "NOT_FOUND"),
            template_id=template_for(screen) or "NOT_FOUND",
            latency_ms={"screen": int((time.perf_counter() - started) * 1000)},
            guardrail_trace={
                **trace,
                "stages_skipped": ["embed", "search", "generate"],
            },
        )
        if conversation is not None and remember:
            conversation.add("assistant", envelope.answer, envelope.template_id)
        return envelope

    started = time.perf_counter()
    hits = retrieve(retrieval_question, top_k=top_k, threshold=threshold, embedder=embedder)
    search_ms = int((time.perf_counter() - started) * 1000)

    envelope = answer(retrieval_question, hits, client=client)
    envelope.latency_ms = {"screen": 0, "search": search_ms, **envelope.latency_ms}
    envelope.guardrail_trace = {**trace, **envelope.guardrail_trace}
    envelope.guardrail_trace["question"] = text
    envelope.guardrail_trace["retrieval_question"] = retrieval_question

    if conversation is not None and remember:
        conversation.add("assistant", envelope.answer, envelope.template_id)
    return envelope


def _print_hits(hits: Sequence[Hit]) -> None:
    for rank, hit in enumerate(hits, start=1):
        meta = hit.metadata
        print(
            f"    {rank}. score {hit.score:.3f}  {hit.chunk_id}  "
            f"{meta.get('scheme')}/{meta.get('plan')}/{meta.get('doc_type')}"
        )
        print(f"       {hit.text[:400]}")


def _ask(
    question: str,
    conversation: Conversation | None,
    args: argparse.Namespace,
) -> AnswerEnvelope:
    """Run one turn and print it."""
    envelope = respond(
        question,
        top_k=args.top_k,
        threshold=args.threshold,
        conversation=conversation,
    )

    trace = envelope.guardrail_trace
    print(f"Q: {question}")
    if trace.get("follow_up"):
        print(f"   follow-up, retrieved as: {trace.get('retrieval_question')}")
    print(f"verdict: {trace.get('verdict')}")
    print(f"retrieved {len(envelope.hits)} chunk(s) in {envelope.latency_ms.get('search', 0)} ms")
    if args.show_chunks and envelope.hits:
        print("retrieved chunks:")
        _print_hits(envelope.hits)

    print("")
    print(f"template_id: {envelope.template_id}")
    print(f"A: {envelope.answer}")
    if envelope.citation_url:
        print(f"Source: {envelope.citation_url}")
        print(envelope.last_updated or "")
    print("")
    print(templates.DISCLAIMER)
    print("")
    return envelope


def _repl(args: argparse.Namespace) -> int:
    """Interactive session that keeps the rolling memory window."""
    conversation = Conversation()
    print(templates.DISCLAIMER)
    print("")
    print("Ask a question. Type /memory for the window, /reset to clear it, /quit to exit.")
    print("")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return 0
        if question == "/memory":
            print(f"{describe(conversation)}")
            for role, text in conversation.as_tuples():
                print(f"  {role}: {text[:90]}")
            print("")
            continue
        if question == "/reset":
            conversation.clear()
            print("memory cleared")
            continue
        _ask(question, conversation, args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.generate",
        description="Ask a question and see the chunks that supported the answer.",
    )
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        default=True,
        help="print every retrieved chunk (default)",
    )
    parser.add_argument("--no-chunks", dest="show_chunks", action="store_false")
    parser.add_argument("--chat", action="store_true", help="interactive session with memory")
    parser.add_argument("--trace", action="store_true", help="emit the JSON trace record")
    args = parser.parse_args(argv)

    if args.chat:
        return _repl(args)

    question = " ".join(args.question).strip()
    if not question:
        parser.error("provide a question, or use --chat for an interactive session")

    envelope = _ask(question, None, args)

    if args.trace:
        emit(
            TraceRecord(
                template_id=envelope.template_id,
                question_len=len(question),
                question_hash=question_hash(question),
                stages_skipped=list(envelope.guardrail_trace.get("stages_skipped", [])),
                stage_ms=envelope.latency_ms,
                n_hits=len(envelope.hits),
                hits=[
                    {"chunk_id": h.chunk_id, "score": h.score, **h.metadata} for h in envelope.hits
                ],
                cited_chunk_id=envelope.cited_chunk_id,
                citation_url=envelope.citation_url,
                last_updated=envelope.last_updated,
                sentence_count=envelope.guardrail_trace.get("sentence_count", 0),
                truncated_by_cap=bool(envelope.guardrail_trace.get("truncated_by_cap")),
                guardrails=envelope.guardrail_trace,
                model=config.GROQ_MODEL,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())