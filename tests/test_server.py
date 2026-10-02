"""Contract tests for the HTTP layer.

These never load MiniLM or Chroma: `server.get_pipeline` is monkeypatched with a
stub so the API surface can be tested offline and in milliseconds.
"""

from __future__ import annotations

import json

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

    def test_reports_pipeline_warm_state(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The embedder loads in the background, so /api/health must say where it got to.

        A container used to report itself healthy while still cold, which hid the
        real cause of a slow first answer.
        """
        monkeypatch.setitem(server._warm, "state", "warming")
        body = client.get("/api/health").json()
        assert body["pipeline"] == "warming"

        monkeypatch.setitem(server._warm, "state", "failed")
        monkeypatch.setitem(server._warm, "error", "no weights")
        body = client.get("/api/health").json()
        assert body["pipeline"] == "failed"
        assert body["pipeline_problem"] == "no weights"


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

    def test_config_never_exposes_the_model_name(self, client: TestClient) -> None:
        """Which LLM answers is an implementation detail, not product copy.

        Guarded because it is easy to reintroduce by adding one key to the
        frontend payload, and the model name is not something a mutual fund
        facts assistant should ask users to judge an answer by.
        """
        body = client.get("/api/config").json()
        assert "model" not in body
        assert config.GROQ_MODEL not in json.dumps(body)

    def test_index_ships_a_history_tab(self, client: TestClient) -> None:
        html = client.get("/").text
        assert 'id="panel-history"' in html
        assert 'id="history-list"' in html
        assert "model-pill" not in html

    def test_config_carries_the_fund_catalogue(self, client: TestClient) -> None:
        """The category browser must only offer schemes we can actually answer.

        A card backed by nothing in sources.csv returns NOT_FOUND, which reads as a
        broken bot rather than an honest gap.
        """
        body = client.get("/api/config").json()
        categories = body["categories"]
        assert categories

        from src.chunk import SCHEME_TITLES

        schemes = [category["scheme"] for category in categories]
        assert schemes == list(config.SCHEMES)
        assert set(schemes) <= set(SCHEME_TITLES)

        for category in categories:
            assert category["title"] == SCHEME_TITLES[category["scheme"]]
            assert category["icon"]
            assert category["category"]
            assert category["tagline"]
            assert category["starters"]
            assert category["documents"]
            for starter in category["starters"]:
                assert starter.endswith("?")

    def test_category_documents_match_the_corpus(self, client: TestClient) -> None:
        """The "Backed by" chips are derived from sources.csv, so they must match.

        A chip promising a factsheet we never ingested is a small lie the user
        only discovers after asking a question that cannot be answered.
        """
        from src.sources import load_sources

        expected: dict[str, set[str]] = {}
        for source in load_sources():
            if source.enabled:
                expected.setdefault(source.scheme, set()).add(source.doc_type)

        for category in client.get("/api/config").json()["categories"]:
            assert set(category["documents"]) == expected[category["scheme"]]

    def test_every_starter_resolves_to_a_real_scheme(self, client: TestClient) -> None:
        """Each starter must name a scheme the retriever can filter on."""
        from src.retrieve import infer_scheme

        body = client.get("/api/config").json()
        for category in body["categories"]:
            for starter in category["starters"]:
                if category["scheme"] == "amc_wide":
                    continue
                assert infer_scheme(starter) == category["scheme"], starter

    def test_index_ships_the_greeting_and_category_browser(
        self, client: TestClient
    ) -> None:
        html = client.get("/").text
        assert "Hello!" in html
        assert "IND Money AI" in html
        assert "/static/indmoney-ai-logo.png" in html
        assert 'id="cat-pills"' in html
        assert 'id="fund-card"' in html
        assert 'id="fund-docs"' in html
        assert 'id="trust-row"' in html
        assert 'id="scope-bar"' in html
        # DESIGN.md forbids fake regulatory markers, so no SEBI number ships.
        assert "SEBI" not in html


class FakeEmbedder:
    """Stand-in for `Embedder` that records whether the lazy model was touched."""

    def __init__(self) -> None:
        self.model_touched = False
        self.encoded: list[str] = []

    @property
    def model(self) -> object:
        self.model_touched = True
        return object()

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encoded.extend(texts)
        return [[0.0] for _ in texts]


class TestWarmUp:
    def test_records_success_and_runs_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(server._warm, "state", "cold")
        monkeypatch.setitem(server._warm, "error", "")
        calls: list[int] = []

        def fake_get() -> dict[str, object]:
            calls.append(1)
            return {"embedder": FakeEmbedder()}

        monkeypatch.setattr(server, "get_pipeline", fake_get)

        server.warm_pipeline()
        server.warm_pipeline()

        assert calls == [1]
        assert server._warm["state"] == "ready"
        assert server._warm["error"] == ""

    def test_forces_the_lazy_encoder_to_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warm-up must touch `.model`, not just build the Embedder.

        `load_embedder()` costs ~0.1s and loads no weights; the first `.model`
        access is ~22s. A warm-up that stops at construction reports `ready`
        while the encoder is still cold, and the first real question pays the
        whole cost on the request path.
        """
        monkeypatch.setitem(server._warm, "state", "cold")
        embedder = FakeEmbedder()
        monkeypatch.setattr(server, "get_pipeline", lambda: {"embedder": embedder})

        server.warm_pipeline()

        assert embedder.model_touched, "warm-up never accessed the lazy model"
        assert embedder.encoded == ["warmup"]

    def test_records_failure_without_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(server._warm, "state", "cold")

        def boom() -> dict[str, object]:
            raise RuntimeError("no weights")

        monkeypatch.setattr(server, "get_pipeline", boom)

        server.warm_pipeline()

        assert server._warm["state"] == "failed"
        assert "no weights" in server._warm["error"]

    def test_encoder_failure_is_reported_not_raised(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken weight load must surface as `failed`, not a dead container."""
        monkeypatch.setitem(server._warm, "state", "cold")

        class Exploding(FakeEmbedder):
            @property
            def model(self) -> object:
                raise RuntimeError("corrupt safetensors")

        monkeypatch.setattr(server, "get_pipeline", lambda: {"embedder": Exploding()})

        server.warm_pipeline()

        assert server._warm["state"] == "failed"
        assert "corrupt safetensors" in server._warm["error"]

    def test_concurrent_callers_load_the_model_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warm-up thread and the first request must not both load MiniLM.

        The lock in get_pipeline is load-bearing, not defensive: two copies of the
        encoder do not fit in the free tier's 512 MB.
        """
        import threading
        import time

        import src.embed as embed_module
        import src.retrieve as retrieve_module

        server._pipeline.clear()
        monkeypatch.setitem(server._warm, "state", "cold")

        loads: list[int] = []
        counter_lock = threading.Lock()

        def fake_load() -> object:
            with counter_lock:
                loads.append(1)
            time.sleep(0.4)
            return "embedder"

        monkeypatch.setattr(embed_module, "load_embedder", fake_load)
        monkeypatch.setattr(retrieve_module, "open_collection", lambda: "collection")

        try:
            threads = [
                threading.Thread(target=server.get_pipeline) for _ in range(4)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            assert len(loads) == 1
            assert "embedder" in server._pipeline
        finally:
            server._pipeline.clear()


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