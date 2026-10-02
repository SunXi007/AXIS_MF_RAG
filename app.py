"""Streamlit UI for the Axis MF RAG chatbot (P6).

The UI is a thin shell over `src.generate.respond`, which is the only sanctioned
entry point: screening, follow-up rewriting, retrieval, generation, and output
verification all stay in the pipeline, so a browser cannot bypass a guardrail that
the CLI enforces.

Run with:  streamlit run app.py --server.port $PORT --server.address 0.0.0.0
"""

from __future__ import annotations

import streamlit as st

import config
from src.embed import load_embedder
from src.memory import Conversation, describe
from src.models import AnswerEnvelope
from src.retrieve import open_collection
from src.templates import DISCLAIMER

st.set_page_config(page_title="Axis MF Facts Chatbot", page_icon="chart", layout="centered")


@st.cache_resource(show_spinner="Loading the retrieval index...")
def get_embedder():
    """Load MiniLM once per server process, not once per message."""
    return load_embedder()


@st.cache_resource(show_spinner="Connecting to the vector store...")
def get_collection():
    """Open the Chroma collection once, to validate it before serving traffic."""
    return open_collection()


def check_prerequisites() -> tuple[bool, str]:
    """Report the first blocking problem with the deployment, if any."""
    if not config.key_present():
        return False, (
            "`GROQ_API_KEY` is not set. Add it in the Render dashboard under "
            "Environment, or put it in a local `.env` file. Never commit it."
        )
    if not config.CHROMA_DIR.exists():
        return False, (
            f"No vector store at `{config.CHROMA_DIR}`. The build step must run "
            "`python -m src.ingest` so the index is built during the build."
        )
    try:
        get_collection()
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the operator
        return False, (
            f"The vector store could not be opened: `{exc}`. Check that the build "
            "step completed and that `chroma/` still exists."
        )
    return True, ""


def render_sources(envelope: AnswerEnvelope) -> None:
    """Show which chunks were retrieved for this answer, with their scores."""
    if not envelope.hits:
        return
    with st.expander(f"Retrieved chunks ({len(envelope.hits)})"):
        for hit in envelope.hits:
            title = hit.metadata.get("scheme") or hit.metadata.get("source_id") or "chunk"
            st.markdown(f"**{hit.chunk_id}** - `{title}` - score `{hit.score}`")
            text = hit.text.strip()
            st.caption(text[:400] + ("..." if len(text) > 400 else ""))
            if hit.source_url:
                st.caption(hit.source_url)


def render_envelope(envelope: AnswerEnvelope) -> None:
    """Render one turn: answer, citation, provenance, and guardrail trace."""
    st.markdown(envelope.answer)

    if envelope.citation_url:
        label = envelope.source_title or "official Axis MF source"
        st.markdown(f"Source: [{label}]({envelope.citation_url})")
        updated = envelope.last_updated or "date not stated on page"
        st.caption(f"Last updated from sources: {updated}")

    trace = envelope.guardrail_trace
    if trace.get("rewritten_question"):
        st.info(f"Resolved your follow-up and searched for: {trace['rewritten_question']}")

    if envelope.template_id != "FACTUAL":
        with st.expander("Why this answer", expanded=True):
            st.write(f"verdict: `{trace.get('verdict')}`")
            if trace.get("matched"):
                st.write("matched: " + ", ".join(f"`{m}`" for m in trace["matched"]))
            if trace.get("pii_pattern"):
                st.write(f"redacted pattern: `{trace['pii_pattern']}`")

    render_sources(envelope)

    if envelope.latency_ms:
        st.caption("timing: " + ", ".join(f"{k} {v}ms" for k, v in envelope.latency_ms.items()))


def generate_answer(question: str) -> AnswerEnvelope:
    """Call the pipeline with the cached embedder and the live conversation."""
    from src import generate

    return generate.respond(
        question,
        embedder=get_embedder(),
        conversation=st.session_state.conversation,
    )


def main() -> None:
    st.title("Axis Mutual Fund facts chatbot")
    st.caption(DISCLAIMER)

    ok, problem = check_prerequisites()
    if not ok:
        st.error(problem)
        st.stop()

    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()
    if "chat" not in st.session_state:
        st.session_state.chat = []

    conversation: Conversation = st.session_state.conversation

    with st.sidebar:
        st.subheader("Session")
        st.write(describe(conversation))
        st.caption(f"model: `{config.GROQ_MODEL}`")
        st.caption(f"top_k `{config.TOP_K}` / threshold `{config.SIMILARITY_THRESHOLD}`")
        st.caption("Answers come only from official Axis MF pages and statutory documents.")
        if st.button("Clear conversation", use_container_width=True):
            conversation.clear()
            st.session_state.chat = []
            st.rerun()

    for turn in st.session_state.chat:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            render_envelope(turn["envelope"])

    question = st.chat_input("Ask about exit loads, fees, risk, or ELSS lock-in")
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the official sources..."):
                envelope = generate_answer(question)
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            st.error(f"The request failed: `{exc}`")
            st.caption("Check the server logs for the full traceback.")
            return
        render_envelope(envelope)

    st.session_state.chat.append({"question": question, "envelope": envelope})


main()