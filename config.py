"""Single source of truth for every tunable in the project.

No other module may hard-code a path, threshold, model id, or size.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent

load_dotenv(ROOT_DIR / ".env")

RAW_DIR = Path(os.getenv("AXIS_RAW_DIR", ROOT_DIR / "data" / "raw"))
PROCESSED_DIR = Path(os.getenv("AXIS_PROCESSED_DIR", ROOT_DIR / "data" / "processed"))
ARTIFACTS_DIR = Path(os.getenv("AXIS_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
CHROMA_DIR = Path(os.getenv("AXIS_CHROMA_DIR", ROOT_DIR / "chroma"))
EVAL_DIR = Path(os.getenv("AXIS_EVAL_DIR", ROOT_DIR / "eval"))
MODELS_DIR = Path(os.getenv("AXIS_MODELS_DIR", ROOT_DIR / "models"))

SOURCES_CSV = Path(os.getenv("AXIS_SOURCES_CSV", ROOT_DIR / "sources.csv"))
CHUNKS_TXT = ARTIFACTS_DIR / "chunks.txt"
CHUNKS_JSON = ARTIFACTS_DIR / "chunks.json"
INGEST_REPORT = ARTIFACTS_DIR / "ingest_report.md"
INGEST_STATE = ARTIFACTS_DIR / "ingest_state.json"

EMBED_MODEL = os.getenv("AXIS_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("AXIS_EMBED_DIM", "384"))
EMBED_BATCH_SIZE = int(os.getenv("AXIS_EMBED_BATCH_SIZE", "32"))
COLLECTION_NAME = os.getenv("AXIS_COLLECTION_NAME", "axis_mf_faq")

MAX_CHUNK_CHARS = int(os.getenv("AXIS_MAX_CHUNK_CHARS", "900"))
MIN_CHUNK_CHARS = int(os.getenv("AXIS_MIN_CHUNK_CHARS", "120"))
CHUNK_OVERLAP = int(os.getenv("AXIS_CHUNK_OVERLAP", "150"))
TABLE_OVERLAP = int(os.getenv("AXIS_TABLE_OVERLAP", "0"))
MIN_EXTRACTED_CHARS = int(os.getenv("AXIS_MIN_EXTRACTED_CHARS", "2000"))
RECURSIVE_SEPARATORS = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]

# `top_k` and MAX_CONTEXT_CHARS are two independent ceilings and must be raised
# together. The chunk that actually answers "exit load on the Large Cap Regular
# plan" (s02-c016) ranks 11th, so top_k=5 dropped it, and the 6000-char budget was
# already exhausted by the first 8 chunks, so raising top_k alone did nothing and
# the model correctly answered NOT_FOUND. More context does invite unsupported
# numerics, which is what the output verifier now catches.
TOP_K = int(os.getenv("AXIS_TOP_K", "12"))
# Calibrated in eval/eval_report.md: positives bottom out at 0.724, the strongest
# off-topic question reaches 0.348, so the midpoint band is ~0.54. 0.35 sat below
# the band and let the worst off-topic question through as if it were answerable.
SIMILARITY_THRESHOLD = float(os.getenv("AXIS_SIMILARITY_THRESHOLD", "0.53"))
CITATION_SCORE_FLOOR = float(os.getenv("AXIS_CITATION_SCORE_FLOOR", "0.45"))
# Pure-embedding retrieval mis-ranks exact-fact lookups: SID boilerplate that
# repeats "load" out-scored the scheme-page chunk stating "Exit Load NIL". This
# weight blends a lexical term-overlap bonus into the ordering of hits that have
# already cleared SIMILARITY_THRESHOLD, so the calibrated threshold stays valid.
LEXICAL_RERANK_WEIGHT = float(os.getenv("AXIS_LEXICAL_RERANK_WEIGHT", "0.15"))
MAX_CONTEXT_CHARS = int(os.getenv("AXIS_MAX_CONTEXT_CHARS", "12000"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# `llama-3.1-8b-instant`, named in implementation.md, has been retired from the
# Groq catalogue and now 404s. gpt-oss-20b is the smallest fast general model
# this key can reach that handles the 3-sentence facts-only contract.
GROQ_MODEL = os.getenv(
    "AXIS_GROQ_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
)
GROQ_CLASSIFIER_MODEL = os.getenv(
    "AXIS_GROQ_CLASSIFIER_MODEL", os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
)
TEMPERATURE = float(os.getenv("AXIS_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("AXIS_MAX_TOKENS", "220"))
MAX_SENTENCES = int(os.getenv("AXIS_MAX_SENTENCES", "3"))
# `MAX_SENTENCES` alone is not a length bound: the model sometimes returns one
# 700-character comma-spliced sentence, which counts as a single sentence and
# slips past the cap while being unusable in the UI.
MAX_ANSWER_CHARS = int(os.getenv("AXIS_MAX_ANSWER_CHARS", "420"))
# gpt-oss models spend part of `max_tokens` on a separate `reasoning` channel.
# At its default effort a reasoning-heavy question consumes the whole 220-token
# budget and returns an empty answer, so the effort is pinned low.
GROQ_REASONING_EFFORT = os.getenv("AXIS_GROQ_REASONING_EFFORT", "low")

HTTP_TIMEOUT = int(os.getenv("AXIS_HTTP_TIMEOUT", "30"))
HTTP_RETRIES = int(os.getenv("AXIS_HTTP_RETRIES", "3"))
HTTP_BACKOFF_S = float(os.getenv("AXIS_HTTP_BACKOFF_S", "1.0"))
REQUEST_DELAY_S = float(os.getenv("AXIS_REQUEST_DELAY_S", "1.5"))
USER_AGENT = os.getenv(
    "AXIS_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
)

ENABLE_PII_SCREEN = os.getenv("AXIS_ENABLE_PII_SCREEN", "true").lower() == "true"
ENABLE_OUTPUT_VERIFIER = os.getenv("AXIS_ENABLE_OUTPUT_VERIFIER", "true").lower() == "true"
PII_LOG_RAW = os.getenv("AXIS_PII_LOG_RAW", "false").lower() == "true"

# Rolling conversation window, in messages (user + assistant turns combined).
MEMORY_MAX_MESSAGES = int(os.getenv("AXIS_MEMORY_MAX_MESSAGES", "10"))

# The HTTP layer keeps one rolling window per browser session. Both bounds exist
# because that state lives in the worker process, so without them a long-lived
# container would grow until it was OOM-killed.
WEB_MAX_SESSIONS = int(os.getenv("AXIS_WEB_MAX_SESSIONS", "64"))
WEB_SESSION_TTL_S = int(os.getenv("AXIS_WEB_SESSION_TTL_S", "1800"))

assert PII_LOG_RAW is False, "PII_LOG_RAW must stay False; raw PII is never logged (NFR-3)"

SCHEMES = ("large_cap", "flexi_cap", "elss", "midcap", "amc_wide")
PLANS = ("direct", "regular", "n_a")
DOC_TYPES = (
    "scheme_page",
    "sid",
    "kim",
    "factsheet",
    "efactsheet",
    "downloads",
    "homepage",
    "third_party",
)
BLOCK_KINDS = ("heading", "text", "list", "table")
EXTRACT_FLAGS = ("ok", "fetch_failed", "suspected_client_rendered", "extract_failed")


def key_present() -> bool:
    """Report whether a Groq key is configured, without revealing it."""
    return bool(GROQ_API_KEY)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR, CHROMA_DIR, EVAL_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)