"""Contract tests for the HTTP layer.

These never load MiniLM or Chroma: `server.get_pipeline` is monkeypatched with a
stub so the API surface can be tested offline and in milliseconds.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import config
import server
from src.models import Hit


def envelope(answer: str = "Exit Load NIL.", *, template_id: str = "FACTUAL") -> object:
    from src.models import AnswerEnvelope

    return AnswerEnvelope(
        answer=answer,
        template_id=template_id,
        citation_url="https://www.axismf.com/x",
        source_title="Axis Large Cap Fund",
        chunk_id="s02-c016",
        last_updated="2026-01-01",
        hits=[
            Hit(
                chunk_id="s02-c016",
                text="Exit Load NIL",
                metadata={"scheme": "large_cap", "source_id": "s02"},
                score=0.68,
            )
        ],
        latency_ms={"retrieval": 4, "generation": 900},
        guardrail_trace={"verdict": "FACTUAL", "matched": []},
    )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose pipeline is stubbed and whose prerequisites pass."""
    monkeypatch.setattr(server, "get_pipeline", lambda: {"embedder": object()})
    monkeypatch.setattr(
        server, "startup_report", lambda: (True, ""), raising=True
    )
    monkeypatch.setattr(config, "CHROMA_DIR", config.ROOT_DIR, raising=False)

    def fake_respond(question: str, **kwargs: object) -> object:
        conversation = kwargs.get("conversation")
        if conversation is not None:
            conversation.add("user", question)
            conversation.add("assistant", "Exit Load NIL.")
        return envelope()

    import src.generate as generate

    monkeypatch.setattr(generate, "respond", fake_respond)
    server.sessions._items.clear()
    return TestClient(server.app)


SESSION = "a" * 32


class TestHealth:
    def test_reports_ready(self, client: TestClient) -> None:
        body = client.get("/api/health").json()
        assert body["ok"] is True
        assert body["model"] == config.GROQ_MODEL
        assert body["memory_window"] == config.MEMORY_MAX_MESSAGES

    def test_reflects_a_missing_key(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server, "startup_report", lambda: (False, "no key"))
        body = client.get("/api/health").json()
        assert body["ok"] is False
        assert body["problem"] == "no key"


class TestStaticShell:
    def test_index_is_served(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>Axis MF Facts Chatbot</title>" in response.text

    def test_css_is_served(self, client: TestClient) -> None:
        assert client.get("/static/styles.css").status_code == 200

    def test_js_is_served(self, client: TestClient) -> None:
        assert client.get("/static/app.js").status_code == 200

    def test_config_endpoint_carries_the_disclaimer(
        self, client: TestClient
    ) -> None:
        body = client.get("/api/config").json()
        assert "advice" in body["disclaimer"].lower()
        assert len(body["suggestions"]) >= 3


class TestChat:
    def test_returns_the_envelope_fields(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": SESSION}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Exit Load NIL."
        assert body["template_id"] == "FACTUAL"
        assert body["citation_url"].startswith("https://")
        assert body["chunk_id"] == "s02-c016"
        assert body["verdict"] == "FACTUAL"

    def test_hit_payload_is_flat_and_bounded(
        self, client: TestClient
    ) -> None:
        body = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": SESSION}
        ).json()
        assert len(body["hits"]) == 1
        hit = body["hits"][0]
        assert hit["chunk_id"] == "s02-c016"
        assert hit["scheme"] == "large_cap"
        assert hit["score"] == 0.68
        assert len(hit["excerpt"]) <= 280

    def test_memory_grows_within_the_window(self, client: TestClient) -> None:
        first = client.post(
            "/api/chat", json={"question": "one", "session_id": SESSION}
        ).json()
        assert first["memory"] == "2/10 messages in memory"

    def test_sessions_are_isolated(self, client: TestClient) -> None:
        client.post("/api/chat", json={"question": "one", "session_id": "a" * 32})
        other = client.post(
            "/api/chat", json={"question": "one", "session_id": "b" * 32}
        ).json()
        assert other["memory"] == "2/10 messages in memory"

    def test_empty_question_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"question": "   ", "session_id": SESSION}
        )
        assert response.status_code == 400

    def test_missing_question_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/chat", json={"session_id": SESSION})
        assert response.status_code == 422

    def test_malformed_session_id_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": "../../etc"}
        )
        assert response.status_code == 422

    def test_non_hex_session_id_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": "z" * 32}
        )
        assert response.status_code == 400

    def test_unavailable_pipeline_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(server, "startup_report", lambda: (False, "no key"))
        response = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": SESSION}
        )
        assert response.status_code == 503

    def test_generation_failure_returns_502(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.generate as generate

        def boom(question: str, **kwargs: object) -> object:
            raise RuntimeError("model exploded")

        monkeypatch.setattr(generate, "respond", boom)
        response = client.post(
            "/api/chat", json={"question": "exit load?", "session_id": SESSION}
        )
        assert response.status_code == 502
        assert "model exploded" in response.json()["detail"]


class TestReset:
    def test_clears_the_window(self, client: TestClient) -> None:
        client.post("/api/chat", json={"question": "one", "session_id": SESSION})
        assert server.sessions.get(SESSION).messages
        assert client.post("/api/reset", json={"session_id": SESSION}).status_code == 200
        assert server.sessions.get(SESSION).messages == []

    def test_rejects_a_bad_session_id(self, client: TestClient) -> None:
        assert client.post("/api/reset", json={"session_id": "nope"}).status_code == 422


class TestSessionStore:
    def test_caps_the_number_of_sessions(self) -> None:
        store = server.SessionStore(max_sessions=3, ttl_s=60)
        for index in range(5):
            store.get(f"{index:032x}")
        assert len(store) == 3

    def test_evicts_the_least_recently_used(self) -> None:
        store = server.SessionStore(max_sessions=2, ttl_s=60)
        first = "a" * 32
        store.get(first)
        store.get("b" * 32)
        store.get(first)
        store.get("c" * 32)
        assert first in store._items

    def test_drops_idle_sessions(self) -> None:
        store = server.SessionStore(max_sessions=10, ttl_s=0)
        store.get("a" * 32)
        assert len(store) == 0

    def test_same_id_returns_the_same_window(self) -> None:
        store = server.SessionStore(max_sessions=4, ttl_s=60)
        assert store.get("a" * 32) is store.get("a" * 32)


class TestPipelineCache:
    def test_embedder_is_loaded_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        class Stub:
            fingerprint = "x"

        def fake_load() -> Stub:
            calls.append(1)
            return Stub()

        import src.embed as embed
        import src.retrieve as retrieve

        monkeypatch.setattr(embed, "load_embedder", fake_load)
        monkeypatch.setattr(retrieve, "open_collection", lambda: object())
        server._pipeline.clear()
        server.get_pipeline()
        server.get_pipeline()
        assert len(calls) == 1
        server._pipeline.clear()