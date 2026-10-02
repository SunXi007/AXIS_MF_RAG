"""Persisted ingestion state, so stages can run independently."""

from __future__ import annotations

import json

import config


def read_state() -> dict:
    """Load artifacts/ingest_state.json, or an empty structure when absent."""
    if not config.INGEST_STATE.exists():
        return {"sources": {}, "embedder_fingerprint": None}
    try:
        data = json.loads(config.INGEST_STATE.read_text(encoding="utf-8"))
    except Exception:
        return {"sources": {}, "embedder_fingerprint": None}
    data.setdefault("sources", {})
    data.setdefault("embedder_fingerprint", None)
    return data


def write_state(state: dict) -> None:
    """Persist ingestion state."""
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.INGEST_STATE.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_fetch(result, sources: dict[str, str]) -> None:
    """Store one fetch outcome under its source id."""
    state = read_state()
    state["sources"][result.source_id] = {
        "status": result.status,
        "http_status": result.http_status,
        "byte_size": result.byte_size,
        "content_hash": result.content_hash,
        "fetched_at": result.fetched_at,
        "local_path": result.local_path,
        "url": sources.get(result.source_id, result.url),
        "error": result.error,
    }
    write_state(state)