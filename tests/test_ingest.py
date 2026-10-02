"""Tests for the P3 ingest orchestrator and the vector store contract.

A deterministic fake embedder stands in for MiniLM so the idempotency rules can be
exercised in milliseconds, with no model download and no network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import pytest

import config
from src.embed import embedding_text
from src.ingest import (
    chroma_metadata,
    file_hash,
    ingest,
    load_chunks,
    open_collection,
    read_state,
)
from src.models import Chunk, Source

DIRECT = Source("s01", "large_cap", "direct", "scheme_page", "https://example.com/a", True)
REGULAR = Source("s02", "large_cap", "regular", "scheme_page", "https://example.com/b", True)


class FakeEmbedder:
    """Deterministic unit vectors derived from the text, so runs are reproducible."""

    fingerprint = "fake-minilm@9.9.9:384:True"

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0
        self.seen: list[str] = []

    def _vector(self, text: str) -> list[float]:
        seed = text.encode("utf-8")
        stream = b""
        while len(stream) < self.dim:
            stream += hashlib.sha256(seed + len(stream).to_bytes(4, "big")).digest()
        raw = [stream[i] / 255.0 + 0.001 for i in range(self.dim)]
        norm = sum(value * value for value in raw) ** 0.5
        return [value / norm for value in raw]

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        self.calls += 1
        self.seen.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_chunks(
        self, chunks: Sequence[Chunk], *, show_progress: bool = False
    ) -> list[list[float]]:
        return self.encode([embedding_text(c) for c in chunks], show_progress=show_progress)


def make_chunk(chunk_id: str, text: str, source_id: str = "s01", **overrides: object) -> Chunk:
    metadata: dict[str, object] = {
        "source_id": source_id,
        "source_url": f"https://example.com/{source_id}",
        "source_title": "Axis Large Cap Fund",
        "doc_type": "scheme_page",
        "scheme": "large_cap",
        "plan": "direct",
        "page_num": None,
        "section_path": "Axis Large Cap Fund - Direct plan > Key facts",
        "source_date": None,
        "fetched_at": "2026-10-02T09:14:22Z",
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "chunk_index": 0,
        "is_table": False,
        "char_len": len(text),
        "table_part": 1,
        "table_parts": 1,
    }
    metadata.update(overrides)
    return Chunk(chunk_id=chunk_id, text=text, metadata=metadata)


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(config, "INGEST_STATE", tmp_path / "ingest_state.json")
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    for source in (DIRECT, REGULAR):
        raw = config.RAW_DIR / source.source_id / source.filename
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(f"<html>{source.source_id}</html>", encoding="utf-8")
    return tmp_path


def write_chunks_json(path: Path, chunks: Sequence[Chunk]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "chunk_count": len(chunks),
                "chunks": [
                    {"chunk_id": c.chunk_id, "text": c.text, "metadata": c.metadata}
                    for c in chunks
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def sample_chunks() -> list[Chunk]:
    return [
        make_chunk("s01-c000", "Axis Large Cap Fund - Direct plan\nKey facts\nExpense ratio: 0.92"),
        make_chunk("s01-c001", "Axis Large Cap Fund - Direct plan\nExit load\nExit load NIL", "s01", chunk_index=1),
        make_chunk("s02-c000", "Axis Large Cap Fund - Regular plan\nKey facts\nExpense ratio: 1.76", "s02"),
    ]


def run_ingest(workspace: Path, chunks: Sequence[Chunk], **kwargs: object) -> tuple[object, FakeEmbedder]:
    chunks_path = write_chunks_json(workspace / "chunks.json", chunks)
    embedder = FakeEmbedder()
    result = ingest(
        embedder=embedder,
        chunks_path=chunks_path,
        sources=[DIRECT, REGULAR],
        run_upstream=False,
        **kwargs,
    )
    return result, embedder


# ------------------------------------------------------------------ idempotency


def test_first_run_embeds_everything(workspace: Path):
    chunks = sample_chunks()
    result, embedder = run_ingest(workspace, chunks)
    assert result.embedded == 3
    assert result.upserts == 3
    assert result.unchanged == []
    assert sorted(result.reembedded) == ["s01", "s02"]
    assert embedder.calls == 2


def test_second_run_is_a_no_op(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    result, embedder = run_ingest(workspace, chunks)
    assert result.unchanged == ["s01", "s02"]
    assert result.embedded == 0
    assert result.upserts == 0
    assert embedder.calls == 0


def test_second_run_produces_no_duplicates(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    second, _ = run_ingest(workspace, chunks)
    collection = open_collection(config.CHROMA_DIR)
    stored = collection.get(include=[])
    assert len(stored["ids"]) == len(set(stored["ids"]))
    assert collection.count() == len(chunks) == second.stored_total


def test_repeated_runs_keep_the_chunk_id_set_identical(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    collection = open_collection(config.CHROMA_DIR)
    first = set(collection.get(include=[])["ids"])
    run_ingest(workspace, chunks)
    second = set(open_collection(config.CHROMA_DIR).get(include=[])["ids"])
    assert first == second == {c.chunk_id for c in chunks}


def test_changed_chunk_text_re_embeds_only_that_source(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    changed = [
        make_chunk("s01-c000", "Axis Large Cap Fund - Direct plan\nKey facts\nExpense ratio: 0.93"),
        chunks[1],
        chunks[2],
    ]
    result, embedder = run_ingest(workspace, changed)
    assert result.reembedded == ["s01"]
    assert result.unchanged == ["s02"]
    assert result.embedded == 2


def test_force_re_embeds_every_source(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    result, _ = run_ingest(workspace, chunks, force=True)
    assert result.unchanged == []
    assert result.embedded == 3


def test_fingerprint_change_rebuilds_the_whole_index(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)

    class OtherEmbedder(FakeEmbedder):
        fingerprint = "other-model@1.0.0:384:True"

    chunks_path = write_chunks_json(workspace / "chunks.json", chunks)
    result = ingest(
        embedder=OtherEmbedder(),
        chunks_path=chunks_path,
        sources=[DIRECT, REGULAR],
        run_upstream=False,
    )
    assert result.fingerprint_changed is True
    assert result.warnings and "FINGERPRINT" in result.warnings[0]
    assert sorted(result.reembedded) == ["s01", "s02"]
    assert result.unchanged == []


def test_state_records_the_fingerprint_and_per_chunk_hashes(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    state = read_state()
    assert state["fingerprint"] == FakeEmbedder.fingerprint
    assert state["index"]["s01"]["chunk_count"] == 2
    assert state["index"]["s01"]["chunk_hashes"]["s01-c000"] == chunks[0].metadata["content_hash"]
    assert state["index"]["s01"]["file_hash"] == file_hash(config.RAW_DIR / "s01" / DIRECT.filename)


def test_fingerprint_mismatch_is_recorded_loudly(workspace: Path, capsys: pytest.CaptureFixture):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)

    class OtherEmbedder(FakeEmbedder):
        fingerprint = "other-model@1.0.0:384:True"

    chunks_path = write_chunks_json(workspace / "chunks.json", chunks)
    ingest(
        embedder=OtherEmbedder(),
        chunks_path=chunks_path,
        sources=[DIRECT, REGULAR],
        run_upstream=False,
    )
    assert "FINGERPRINT" in capsys.readouterr().err


# --------------------------------------------------------------------- storage


def test_collection_uses_cosine_space(workspace: Path):
    run_ingest(workspace, sample_chunks())
    collection = open_collection(config.CHROMA_DIR)
    assert collection.metadata["hnsw:space"] == "cosine"
    assert collection.name == config.COLLECTION_NAME


def test_stored_vectors_keep_the_configured_dimension(workspace: Path):
    embedder = FakeEmbedder(dim=config.EMBED_DIM)
    chunks_path = write_chunks_json(workspace / "chunks.json", sample_chunks())
    ingest(
        embedder=embedder,
        chunks_path=chunks_path,
        sources=[DIRECT, REGULAR],
        run_upstream=False,
    )
    collection = open_collection(config.CHROMA_DIR)
    stored = collection.get(include=["embeddings"])
    assert len(stored["embeddings"]) == 3
    assert {len(vector) for vector in stored["embeddings"]} == {config.EMBED_DIM}


def test_every_chunk_in_chunks_json_exists_in_the_collection(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    stored = set(open_collection(config.CHROMA_DIR).get(include=[])["ids"])
    assert stored == {c.chunk_id for c in chunks}


def test_stored_metadata_keeps_the_filterable_fields(workspace: Path):
    run_ingest(workspace, sample_chunks())
    collection = open_collection(config.CHROMA_DIR)
    stored = collection.get(include=["metadatas"])
    for metadata in stored["metadatas"]:
        assert metadata["scheme"] == "large_cap"
        assert metadata["plan"] in config.PLANS
        assert metadata["source_id"].startswith("s")
        assert metadata["doc_type"] == "scheme_page"
        assert metadata["section_path"]


def test_stored_document_contains_the_breadcrumb(workspace: Path):
    run_ingest(workspace, sample_chunks())
    stored = open_collection(config.CHROMA_DIR).get(include=["documents"])
    assert "Key facts" in stored["documents"][0]


def test_none_metadata_is_dropped_not_stored(workspace: Path):
    run_ingest(workspace, sample_chunks())
    stored = open_collection(config.CHROMA_DIR).get(include=["metadatas"])
    assert all("page_num" not in metadata for metadata in stored["metadatas"])


def test_chroma_metadata_drops_none_and_encodes_containers():
    chunk = make_chunk("s01-c000", "x", extra={"a": [1, 2]})
    reduced = chroma_metadata(chunk)
    assert "source_date" not in reduced and "page_num" not in reduced
    assert reduced["extra"] == '{"a": [1, 2]}'
    assert reduced["scheme"] == "large_cap"


def test_index_survives_a_restart(workspace: Path):
    run_ingest(workspace, sample_chunks())
    reopened = open_collection(config.CHROMA_DIR)
    assert reopened.count() == 3
    assert set(reopened.get(include=[])["ids"]) == {"s01-c000", "s01-c001", "s02-c000"}


# ----------------------------------------------------------------------- prune


def test_prune_removes_chunks_of_a_disabled_source(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)

    class NoopEmbedder(FakeEmbedder):
        pass

    remaining = [chunks[2]]
    chunks_path = write_chunks_json(workspace / "chunks.json", remaining)
    result = ingest(
        embedder=NoopEmbedder(),
        chunks_path=chunks_path,
        sources=[REGULAR],
        run_upstream=False,
        prune=True,
    )
    assert sorted(result.pruned) == ["s01-c000", "s01-c001"]
    assert set(open_collection(config.CHROMA_DIR).get(include=[])["ids"]) == {"s02-c000"}


def test_prune_is_a_no_op_without_orphans(workspace: Path):
    chunks = sample_chunks()
    run_ingest(workspace, chunks)
    result, _ = run_ingest(workspace, chunks, prune=True)
    assert result.pruned == []
    assert result.stored_total == 3


# ----------------------------------------------------------------- embedding input


def test_embedding_text_does_not_duplicate_the_breadcrumb():
    chunk = make_chunk("s01-c000", "Axis Large Cap Fund - Direct plan\nKey facts\nExpense ratio: 0.92")
    embedded = embedding_text(chunk)
    assert embedded.count("Axis Large Cap Fund - Direct plan") == 1
    assert "Key facts" in embedded


def test_embedding_text_prepends_the_breadcrumb_when_absent():
    chunk = make_chunk("s01-c000", "Expense ratio: 0.92")
    assert embedding_text(chunk).startswith("Axis Large Cap Fund - Direct plan\nKey facts\n")


def test_embed_input_is_not_the_bare_body():
    embedder = FakeEmbedder()
    chunk = make_chunk("s01-c000", "Expense ratio: 0.92")
    embedder.embed_chunks([chunk])
    assert embedder.seen == [embedding_text(chunk)]
    assert embedder.seen[0] != "Expense ratio: 0.92"


# ----------------------------------------------------------------------- inputs


def test_load_chunks_requires_the_artifact(workspace: Path):
    with pytest.raises(FileNotFoundError, match="src.chunk"):
        load_chunks(workspace / "missing.json")


def test_load_chunks_round_trips(workspace: Path):
    chunks = sample_chunks()
    path = write_chunks_json(workspace / "chunks.json", chunks)
    loaded = load_chunks(path)
    assert [c.chunk_id for c in loaded] == [c.chunk_id for c in chunks]
    assert loaded[0].metadata["scheme"] == "large_cap"


def test_file_hash_is_stable_and_empty_when_missing(tmp_path: Path):
    target = tmp_path / "f.txt"
    target.write_text("hello", encoding="utf-8")
    assert file_hash(target) == file_hash(target)
    assert file_hash(tmp_path / "absent.txt") == ""


def test_source_without_chunks_is_reported(workspace: Path):
    chunks_path = write_chunks_json(workspace / "chunks.json", [sample_chunks()[0]])
    result = ingest(
        embedder=FakeEmbedder(),
        chunks_path=chunks_path,
        sources=[DIRECT, REGULAR],
        run_upstream=False,
    )
    assert result.missing_chunks == ["s02"]
    assert result.stored_total == 1


def test_only_fetch_stops_before_embedding(workspace: Path):
    chunks = sample_chunks()
    chunks_path = write_chunks_json(workspace / "chunks.json", chunks)
    embedder = FakeEmbedder()
    result = ingest(
        embedder=embedder,
        chunks_path=chunks_path,
        sources=[],
        run_upstream=False,
        only_fetch=True,
    )
    assert result.upserts == 0
    assert embedder.calls == 0
    assert not (workspace / "chroma").exists() or open_collection(config.CHROMA_DIR).count() == 0


def test_chunking_refuses_to_overwrite_the_artifact_with_nothing(workspace: Path, monkeypatch):
    from src import chunk as chunk_stage

    monkeypatch.setattr(chunk_stage, "load_all_normalized", lambda sources: {})
    with pytest.raises(RuntimeError, match="0 chunks"):
        chunk_stage.chunk_all_sources([DIRECT])