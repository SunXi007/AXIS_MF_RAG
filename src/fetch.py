"""Stage 2 of ingestion: HTTP retrieval with raw persistence.

A failure here is recorded and never raised, so one dead URL cannot abort the
run (NFR-6). Silent failure is the enemy here: an unfetched source would
otherwise produce a silently smaller corpus.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import httpx

import config
from src.models import FetchResult, Source
from src.sources import load_sources
from src.trace import utc_now

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _target_path(source: Source) -> Path:
    return config.RAW_DIR / source.source_id / source.filename


def _existing(source: Source, force: bool) -> FetchResult | None:
    """Reuse an already-downloaded file unless force is set (idempotency)."""
    if force:
        return None
    path = _target_path(source)
    if not path.exists():
        return None
    payload = path.read_bytes()
    if not payload:
        return None
    return FetchResult(
        source_id=source.source_id,
        url=source.url,
        status="ok",
        http_status=200,
        byte_size=len(payload),
        content_hash=hashlib.sha256(payload).hexdigest(),
        fetched_at=utc_now(),
        local_path=str(path),
    )


def fetch_source(source: Source, *, force: bool = False, client: httpx.Client | None = None) -> FetchResult:
    """Download one source to data/raw/<source_id>/, retrying transient failures."""
    reused = _existing(source, force)
    if reused is not None:
        return reused

    destination = _target_path(source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    owns_client = client is None
    client = client or httpx.Client(
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
    )

    last_error = ""
    last_status: int | None = None
    try:
        for attempt in range(1, config.HTTP_RETRIES + 1):
            try:
                response = client.get(source.url)
                last_status = response.status_code

                if response.status_code in RETRYABLE_STATUS:
                    last_error = f"retryable status {response.status_code}"
                    if attempt < config.HTTP_RETRIES:
                        time.sleep(config.HTTP_BACKOFF_S * attempt)
                        continue
                    break

                if response.status_code >= 400:
                    return FetchResult(
                        source_id=source.source_id,
                        url=source.url,
                        status="fetch_failed",
                        http_status=response.status_code,
                        error=f"HTTP {response.status_code}",
                    )

                payload = response.content
                if not payload:
                    return FetchResult(
                        source_id=source.source_id,
                        url=source.url,
                        status="fetch_failed",
                        http_status=response.status_code,
                        error="empty response body",
                    )

                content_type = response.headers.get("content-type", "")
                if source.is_pdf and "pdf" not in content_type.lower():
                    return FetchResult(
                        source_id=source.source_id,
                        url=source.url,
                        status="fetch_failed",
                        http_status=response.status_code,
                        byte_size=len(payload),
                        error=f"expected PDF, received content-type '{content_type}'",
                    )

                destination.write_bytes(payload)
                return FetchResult(
                    source_id=source.source_id,
                    url=source.url,
                    status="ok",
                    http_status=response.status_code,
                    byte_size=len(payload),
                    content_hash=hashlib.sha256(payload).hexdigest(),
                    fetched_at=utc_now(),
                    local_path=str(destination),
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < config.HTTP_RETRIES:
                    time.sleep(config.HTTP_BACKOFF_S * attempt)
    finally:
        if owns_client:
            client.close()

    return FetchResult(
        source_id=source.source_id,
        url=source.url,
        status="fetch_failed",
        http_status=last_status,
        error=last_error or "exhausted retries",
    )


def fetch_all(
    sources: list[Source] | None = None,
    *,
    force: bool = False,
    delay_s: float | None = None,
    client: httpx.Client | None = None,
) -> list[FetchResult]:
    """Fetch every enabled source, pausing between requests."""
    manifest = sources if sources is not None else load_sources(enabled_only=True)
    pause = config.REQUEST_DELAY_S if delay_s is None else delay_s
    results: list[FetchResult] = []

    owns_client = client is None
    client = client or httpx.Client(
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": config.USER_AGENT, "Accept-Language": "en-IN,en;q=0.9"},
    )
    try:
        for index, source in enumerate(manifest):
            result = fetch_source(source, force=force, client=client)
            results.append(result)
            marker = "ok " if result.status == "ok" else "FAIL"
            detail = result.error or f"{result.byte_size} bytes"
            print(f"  [{marker}] {source.source_id} {source.scheme:<11} {detail}")
            if index < len(manifest) - 1 and pause > 0:
                time.sleep(pause)
    finally:
        if owns_client:
            client.close()

    return results


def main() -> int:
    config.ensure_dirs()
    sources = load_sources(enabled_only=True)
    urls = {s.source_id: s.url for s in sources}
    print(f"fetching {len(sources)} enabled sources")
    results = fetch_all(sources)

    from src.state import record_fetch

    for result in results:
        record_fetch(result, urls)

    ok = [r for r in results if r.status == "ok"]
    failed = [r for r in results if r.status != "ok"]
    print(f"\nfetched ok: {len(ok)}   failed: {len(failed)}")
    for result in failed:
        print(f"  FAILED {result.source_id} {result.url} -> {result.error}")
    print("state written to artifacts/ingest_state.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())