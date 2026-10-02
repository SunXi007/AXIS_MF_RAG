"""Tests for metadata filter resolution and threshold filtering.

Uses a fake collection and a fake embedder, so nothing here downloads the model
or touches `chroma/`.
"""

from __future__ import annotations

from typing import Any

import pytest

import config
from src.models import Hit
from src.retrieve import infer_metadata_filter, infer_plan, infer_scheme, retrieve


class FakeEmbedder:
    fingerprint = "fake@1:384:True"

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.0, 1.0]
        self.calls: list[list[str]] = []

    def encode(self, texts, *, show_progress: bool = False) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector for _ in texts]


class FakeCollection:
    def __init__(self, ids: list[str], distances: list[float], metadatas: list[dict]) -> None:
        self._ids = ids
        self._distances = distances
        self._metadatas = metadatas
        self.last_where: Any = None
        self.last_n: int | None = None

    def query(self, *, query_embeddings, n_results, where, include):
        self.last_where = where
        self.last_n = n_results
        return {
            "ids": [self._ids],
            "documents": [[f"text of {i}" for i in self._ids]],
            "metadatas": [self._metadatas],
            "distances": [self._distances],
        }


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture(autouse=True)
def _matching_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.retrieve.read_state", lambda: {"fingerprint": FakeEmbedder.fingerprint})


def make(ids: list[str], distances: list[float]) -> FakeCollection:
    return FakeCollection(ids, distances, [{"source_url": "https://x.test"} for _ in ids])


class TestSchemeAliases:
    @pytest.mark.parametrize(
        "question,scheme",
        [
            ("What is the exit load on the Axis Large Cap Fund?", "large_cap"),
            ("largecap direct plan charges", "large_cap"),
            ("Axis Bluechip Fund expense ratio", "large_cap"),
            ("Flexi Cap Fund minimum SIP", "flexi_cap"),
            ("flexicap fund lock-in", "flexi_cap"),
            ("ELSS tax saver fund NAV date", "elss"),
            ("Is 80C deduction available in ELSS?", "elss"),
            ("Midcap fund exit load", "midcap"),
            ("mid cap fund expense ratio", "midcap"),
        ],
    )
    def test_resolves_scheme(self, question: str, scheme: str) -> None:
        assert infer_scheme(question) == scheme

    @pytest.mark.parametrize(
        "question", ["How do I download a statement?", "What is a SEBI-registered adviser?"]
    )
    def test_no_scheme_means_no_filter(self, question: str) -> None:
        assert infer_scheme(question) is None


class TestPlanAliases:
    @pytest.mark.parametrize(
        "question,plan",
        [
            ("exit load on the Direct Growth plan", "direct"),
            ("Direct plan expense ratio", "direct"),
            ("exit load on the Regular plan", "regular"),
            ("Regular Growth charges", "regular"),
        ],
    )
    def test_resolves_plan(self, question: str, plan: str) -> None:
        assert infer_plan(question) == plan

    @pytest.mark.parametrize("question", ["What is the lock-in period?", "Who is the custodian?"])
    def test_no_plan_means_no_filter(self, question: str) -> None:
        assert infer_plan(question) is None


class TestMetadataFilter:
    def test_no_alias_applies_no_filter(self) -> None:
        assert infer_metadata_filter("How do I download a statement?") is None

    def test_scheme_only(self) -> None:
        assert infer_metadata_filter("Flexi Cap Fund minimum SIP") == {"scheme": "flexi_cap"}

    def test_plan_filter_always_includes_na(self) -> None:
        where = infer_metadata_filter("Direct plan exit load")
        assert where == {"$or": [{"plan": "direct"}, {"plan": "n_a"}]}

    def test_scheme_and_plan_are_anded(self) -> None:
        assert infer_metadata_filter("Flexi Cap Direct plan minimum SIP") == {
            "$and": [
                {"scheme": "flexi_cap"},
                {"$or": [{"plan": "direct"}, {"plan": "n_a"}]},
            ]
        }

    def test_statutory_documents_survive_a_direct_filter(self) -> None:
        where = infer_metadata_filter("Direct plan exit load")
        assert {"plan": "n_a"} in where["$or"]


class TestRetrieve:
    def test_score_is_one_minus_distance(self, embedder: FakeEmbedder) -> None:
        col = make(["s01-c001"], [0.25])
        hits = retrieve("q", collection=col, embedder=embedder)
        assert hits[0].score == 0.75
        assert isinstance(hits[0], Hit)
        assert hits[0].chunk_id == "s01-c001"

    def test_hits_below_threshold_are_dropped(self, embedder: FakeEmbedder) -> None:
        col = make(["a", "b", "c"], [0.10, 0.65, 0.90])
        hits = retrieve("q", threshold=0.35, collection=col, embedder=embedder)
        assert [h.chunk_id for h in hits] == ["a", "b"]

    def test_threshold_is_config_driven(self, embedder: FakeEmbedder) -> None:
        col = make(["a"], [0.70])
        assert retrieve("q", collection=col, embedder=embedder) == []
        hits = retrieve("q", threshold=0.29, collection=col, embedder=embedder)
        assert len(hits) == 1

    def test_off_topic_returns_nothing(self, embedder: FakeEmbedder) -> None:
        col = make(["a", "b"], [0.98, 0.99])
        assert retrieve("how do I cook pasta", collection=col, embedder=embedder) == []

    def test_empty_result_is_falsy(self, embedder: FakeEmbedder) -> None:
        assert not retrieve("q", collection=make([], []), embedder=embedder)

    def test_top_k_defaults_to_config(self, embedder: FakeEmbedder) -> None:
        col = make(["a"], [0.1])
        retrieve("q", collection=col, embedder=embedder)
        assert col.last_n == config.TOP_K

    def test_filter_is_passed_to_chroma(self, embedder: FakeEmbedder) -> None:
        col = make(["a"], [0.1])
        retrieve("Flexi Cap Direct exit load", collection=col, embedder=embedder)
        assert col.last_where == {
            "$and": [{"scheme": "flexi_cap"}, {"$or": [{"plan": "direct"}, {"plan": "n_a"}]}]
        }

    def test_question_is_embedded_alone(self, embedder: FakeEmbedder) -> None:
        retrieve("What is the exit load?", collection=make(["a"], [0.1]), embedder=embedder)
        assert embedder.calls == [["What is the exit load?"]]

    def test_hits_carry_metadata_and_url(self, embedder: FakeEmbedder) -> None:
        col = make(["a"], [0.1])
        hits = retrieve("q", collection=col, embedder=embedder)
        assert hits[0].metadata["source_url"] == "https://x.test"
        assert hits[0].source_url == "https://x.test"

    def test_mismatched_fingerprint_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("src.retrieve.read_state", lambda: {"fingerprint": "other@0:768:True"})
        with pytest.raises(RuntimeError, match="different embedder"):
            retrieve("q", collection=make([], []), embedder=FakeEmbedder())