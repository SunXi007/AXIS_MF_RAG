"""Tests for fetch behaviour, using a mocked transport."""

from __future__ import annotations

import hashlib

import httpx
import pytest

import config
from src.fetch import fetch_all, fetch_source
from src.models import Source

HTML_SOURCE = Source(
    source_id="t01",
    scheme="large_cap",
    plan="direct",
    doc_type="scheme_page",
    url="https://example.com/fund",
    enabled=True,
)

PDF_SOURCE = Source(
    source_id="t02",
    scheme="elss",
    plan="n_a",
    doc_type="sid",
    url="https://example.com/docs/SID.pdf",
    enabled=True,
)


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    return tmp_path / "raw"


def client_returning(*responses: httpx.Response) -> httpx.Client:
    queue = list(responses)
    return httpx.Client(transport=httpx.MockTransport(lambda request: queue.pop(0)))


def test_successful_fetch_writes_raw_file(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    body = b"<html><body><h1>Axis Large Cap Fund</h1></body></html>"
    client = client_returning(
        httpx.Response(200, content=body, headers={"content-type": "text/html"})
    )
    result = fetch_source(HTML_SOURCE, client=client)
    assert result.status == "ok"
    assert result.byte_size == len(body)
    assert len(result.content_hash) == 64
    assert (raw_dir / "t01" / "fund.html").exists()


def test_pdf_source_rejects_non_pdf_content_type(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    client = client_returning(
        httpx.Response(200, content=b"<html>404</html>", headers={"content-type": "text/html"})
    )
    result = fetch_source(PDF_SOURCE, client=client)
    assert result.status == "fetch_failed"
    assert "expected PDF" in result.error


def test_pdf_source_accepts_pdf_content_type(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    payload = b"%PDF-1.4\n%mock\n"
    client = client_returning(
        httpx.Response(200, content=payload, headers={"content-type": "application/pdf"})
    )
    result = fetch_source(PDF_SOURCE, client=client)
    assert result.status == "ok"
    assert result.local_path.endswith("SID.pdf")


def test_client_error_is_reported_not_raised(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    client = client_returning(httpx.Response(404, content=b"missing"))
    result = fetch_source(HTML_SOURCE, client=client)
    assert result.status == "fetch_failed"
    assert result.http_status == 404
    assert result.error == "HTTP 404"


def test_retryable_status_is_retried_then_reported(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 2)
    monkeypatch.setattr(config, "HTTP_BACKOFF_S", 0.0)
    client = client_returning(httpx.Response(503), httpx.Response(503))
    result = fetch_source(HTML_SOURCE, client=client)
    assert result.status == "fetch_failed"
    assert "503" in result.error


def test_empty_body_is_reported_as_failure(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    client = client_returning(httpx.Response(200, content=b""))
    result = fetch_source(HTML_SOURCE, client=client)
    assert result.status == "fetch_failed"
    assert "empty" in result.error


def test_second_fetch_reuses_existing_file(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    body = b"<html>content</html>"
    first = client_returning(
        httpx.Response(200, content=body, headers={"content-type": "text/html"})
    )
    assert fetch_source(HTML_SOURCE, client=first).status == "ok"

    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network was used on a cached source")

    second = httpx.Client(transport=httpx.MockTransport(explode))
    reused = fetch_source(HTML_SOURCE, client=second)
    assert reused.status == "ok"
    assert reused.content_hash == hashlib.sha256(body).hexdigest()


def test_force_refetches(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    body = b"<html>v1</html>"
    first = client_returning(
        httpx.Response(200, content=body, headers={"content-type": "text/html"})
    )
    fetch_source(HTML_SOURCE, client=first)

    second = client_returning(
        httpx.Response(200, content=b"<html>v2</html>", headers={"content-type": "text/html"})
    )
    result = fetch_source(HTML_SOURCE, force=True, client=second)
    assert result.byte_size == len(b"<html>v2</html>")


def test_fetch_all_continues_past_a_failure(raw_dir, monkeypatch):
    monkeypatch.setattr(config, "HTTP_RETRIES", 1)
    monkeypatch.setattr(config, "REQUEST_DELAY_S", 0.0)

    responses = {
        "https://example.com/fund": httpx.Response(404, content=b"missing"),
        "https://example.com/docs/SID.pdf": httpx.Response(
            200, content=b"%PDF-1.4\n", headers={"content-type": "application/pdf"}
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return responses[str(request.url)]

    client = httpx.Client(transport=httpx.MockTransport(handler))
    results = fetch_all([HTML_SOURCE, PDF_SOURCE], client=client)
    assert [r.status for r in results] == ["fetch_failed", "ok"]