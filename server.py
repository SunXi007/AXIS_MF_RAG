"""FastAPI backend for the Axis MF RAG chatbot.

The HTTP layer is deliberately thin. It owns no retrieval, screening, or
generation logic: every answer is produced by `src.generate.respond`, the same
entry point the CLI uses, so a browser cannot bypass a guardrail that the CLI
enforces. The only state here is the rolling conversation window, kept in a
bounded in-process store because the pipeline already caps each window at
`MEMORY_MAX_MESSAGES`.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src.memory import Conversation, describe
from src.templates import DISCLAIMER

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Start the model warm-up in the background, without delaying the boot.

    Deliberately not awaited: Render needs the port open quickly to finish its
    health checks, and blocking here would make a cold deploy look dead.
    """
    threading.Thread(target=warm_pipeline, name="warm", daemon=True).start()
    yield


app = FastAPI(title="Axis MF RAG chatbot", docs_url=None, redoc_url=None, lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=32, max_length=32)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=32, max_length=32)


class SessionStore:
    """Bounded, TTL-evicting map of session id to rolling conversation.

    Unbounded session state is a memory leak on a 512 MB instance, so the oldest
    session is evicted past `max_sessions` and any idle session is dropped after
    `ttl_s`. Eviction is invisible to the client: the pipeline rewrites a
    dangling "its" only when a subject is in the window, so a cold session simply
    treats the next question as standalone.
    """

    def __init__(self, max_sessions: int, ttl_s: int) -> None:
        self._max = max_sessions
        self._ttl = ttl_s
        self._items: OrderedDict[str, tuple[float, Conversation]] = OrderedDict()

    def _evict(self) -> None:
        now = time.monotonic()
        stale = [key for key, (seen, _) in self._items.items() if now - seen > self._ttl]
        for key in stale:
            del self._items[key]
        while len(self._items) > self._max:
            self._items.popitem(last=False)

    def get(self, session_id: str) -> Conversation:
        self._evict()
        found = self._items.get(session_id)
        if found is None:
            conversation = Conversation()
            self._items[session_id] = (time.monotonic(), conversation)
            return conversation
        self._items[session_id] = (time.monotonic(), found[1])
        self._items.move_to_end(session_id)
        return found[1]

    def clear(self, session_id: str) -> None:
        self._items.pop(session_id, None)

    def __len__(self) -> int:
        self._evict()
        return len(self._items)


sessions = SessionStore(config.WEB_MAX_SESSIONS, config.WEB_SESSION_TTL_S)


def check_session_id(session_id: str) -> str:
    """Reject anything that is not a client-generated 32-char hex id."""
    if not SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="invalid session_id")
    return session_id


_pipeline: dict[str, Any] = {}
_pipeline_lock = threading.Lock()
_warm: dict[str, str] = {"state": "cold", "error": ""}


def get_pipeline() -> dict[str, Any]:
    """Load the embedder once per process and reuse it across requests.

    Loading MiniLM costs seconds and hundreds of megabytes, so it must not
    happen per request on a small instance. Opening the collection here also
    validates the index up front, so a bad build fails on the first request
    rather than mid-answer.

    The lock matters because a warm-up thread and the first user request race:
    without it, both would load a second copy of the model and a 512 MB
    instance would fall over.
    """
    if "embedder" in _pipeline:
        return _pipeline
    with _pipeline_lock:
        if "embedder" not in _pipeline:
            from src.embed import load_embedder
            from src.retrieve import open_collection

            _pipeline["embedder"] = load_embedder()
            _pipeline["collection"] = open_collection()
    return _pipeline


def warm_pipeline() -> None:
    """Force the heavy imports and the model onto the instance, off the request path.

    `Embedder` defers loading sentence-transformers and the MiniLM weights until
    the first `encode`. Deferring that to the first `/api/chat` means the first
    person to ask waits for it, on a 0.1 CPU instance, for tens of seconds. Doing
    it on a background thread at startup overlaps the cost with Render's health
    probes and with the user reading the page.

    Failures are recorded, not raised: a cold container should still start and
    serve the UI, and the first real request will retry and report the error.
    """
    if _warm["state"] == "ready":
        return
    _warm["state"] = "warming"
    try:
        get_pipeline()
        _warm["state"] = "ready"
        _warm["error"] = ""
    except Exception as exc:  # noqa: BLE001 - surfaced through /api/health
        _warm["state"] = "failed"
        _warm["error"] = str(exc)


def startup_report() -> tuple[bool, str]:
    """Report the first blocking deployment problem, if any."""
    if not config.key_present():
        return False, "GROQ_API_KEY is not set on the server."
    if not config.CHROMA_DIR.exists():
        return False, f"No vector store at {config.CHROMA_DIR}. The build must run src.ingest."
    return True, ""


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Liveness plus the facts an operator needs to debug a cold container."""
    ready, problem = startup_report()
    return {
        "ok": ready,
        "problem": problem,
        "model": config.GROQ_MODEL,
        "top_k": config.TOP_K,
        "threshold": config.SIMILARITY_THRESHOLD,
        "memory_window": config.MEMORY_MAX_MESSAGES,
        "sessions": len(sessions),
        "pipeline": _warm["state"],
        "pipeline_problem": _warm["error"],
    }


@app.get("/api/config")
def public_config() -> dict[str, Any]:
    """Everything the frontend needs to render shell chrome.

    The model name is deliberately absent: which LLM answers a question is an
    implementation detail, and surfacing it in the product invites users to
    judge a grounded answer by its vendor instead of its cited source. It stays
    on `/api/health` for operators.
    """
    from src.chunk import SCHEME_TITLES
    from src.sources import load_sources

    # Derive the "backed by" chips from sources.csv instead of hardcoding them,
    # so the UI can never advertise a document type the corpus does not hold.
    doc_types: dict[str, list[str]] = {}
    for source in load_sources():
        if not source.enabled:
            continue
        doc_types.setdefault(source.scheme, []).append(source.doc_type)

    categories = [
        {
            "scheme": entry["scheme"],
            "title": SCHEME_TITLES.get(entry["scheme"], entry["scheme"]),
            "category": entry["category"],
            "icon": entry["icon"],
            "tagline": entry["tagline"],
            "starters": list(entry["starters"]),
            "documents": sorted(set(doc_types.get(entry["scheme"], []))),
        }
        for entry in config.FUND_CATALOG
    ]

    return {
        "disclaimer": DISCLAIMER,
        "memory_window": config.MEMORY_MAX_MESSAGES,
        "categories": categories,
        "suggestions": [
            "What is the exit load on the Axis Large Cap Regular plan?",
            "What is the lock-in period for the Axis ELSS Tax Saver Fund?",
            "What is the risk profile of the Axis Flexi Cap Direct plan?",
            "What are the expense ratios of the Axis Midcap Direct plan?",
            "How long will the answer take to verify?",
        ],
    }


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Answer one question, keeping the rolling window for the session."""
    check_session_id(request.session_id)

    ready, problem = startup_report()
    if not ready:
        raise HTTPException(status_code=503, detail=problem)

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    conversation = sessions.get(request.session_id)

    try:
        pipeline = get_pipeline()
    except Exception as exc:  # noqa: BLE001 - surfaced to the client verbatim
        raise HTTPException(status_code=503, detail=f"pipeline unavailable: {exc}") from exc

    from src import generate

    try:
        envelope = generate.respond(
            question,
            embedder=pipeline["embedder"],
            conversation=conversation,
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not kill the worker
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    return {
        "answer": envelope.answer,
        "template_id": envelope.template_id,
        "citation_url": envelope.citation_url,
        "source_title": envelope.source_title,
        "chunk_id": envelope.chunk_id,
        "last_updated": envelope.last_updated,
        "rewritten_question": envelope.guardrail_trace.get("rewritten_question"),
        "verdict": envelope.guardrail_trace.get("verdict"),
        "matched": envelope.guardrail_trace.get("matched", []),
        "pii_pattern": envelope.guardrail_trace.get("pii_pattern"),
        "latency_ms": envelope.latency_ms,
        "memory": describe(conversation),
        "hits": [
            {
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "scheme": hit.metadata.get("scheme"),
                "source_id": hit.metadata.get("source_id"),
                "source_url": hit.source_url,
                "excerpt": hit.text.strip()[:280],
            }
            for hit in envelope.hits
        ],
    }


@app.post("/api/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    """Clear the session's rolling window."""
    check_session_id(request.session_id)
    sessions.clear(request.session_id)
    return {"ok": True, "memory": "no history"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")