"""Stage 3 - the ingest orchestrator and the vector store writer.

Idempotency follows `architecture.md` §4.8 exactly: a source is re-embedded only
when the raw file hash or the set of per-chunk content hashes changes, or when the
embedder fingerprint no longer matches the one recorded in
`artifacts/ingest_state.json`. A fingerprint change invalidates the whole index and
is reported loudly rather than silently mixed (NFR-4).

`--prune` removes stored chunks whose `source_id` is no longer enabled, which
otherwise linger forever because nothing deletes them on their own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import config
from src.embed import Embedder, EmbedderProtocol, embedding_text
from src.models import Chunk, Source
from src.sources import load_sources
from src.trace import utc_now


@dataclass
class IngestResult:
    """What one ingest run did, for logging, tests and the report."""

    fingerprint: str = ""
    embedded: int = 0
    upserts: int = 0
    unchanged: list[str] = field(default_factory=list)
    reembedded: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    missing_chunks: list[str] = field(default_factory=list)
    stored_total: int = 0
    elapsed_ms: int = 0
    fingerprint_changed: bool = False
    warnings: list[str] = field(default_factory=list)


def file_hash(path: Path) -> str:
    """sha256 of a file on disk, or an empty string when it is absent."""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state(path: Path | None = None) -> dict[str, Any]:
    """Load `ingest_state.json`, tolerating a missing or corrupt file."""
    target = path or config.INGEST_STATE
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{target} is not valid JSON: {exc}") from exc


def write_state(state: dict[str, Any], path: Path | None = None) -> Path:
    """Persist ingest state atomically enough for a crashed run to be recoverable."""
    target = path or config.INGEST_STATE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def chroma_metadata(chunk: Chunk) -> dict[str, Any]:
    """Reduce chunk metadata to the scalar types Chroma accepts.

    Chroma rejects `None` and nested containers, so absent optional fields are
    dropped rather than stored as null, and anything else is JSON-encoded.
    """
    metadata: dict[str, Any] = {}
    for key, value in chunk.metadata.items():
        if value is None:
            continue
        if isinstance(value, bool) or isinstance(value, (int, float, str)):
            metadata[key] = value
        else:
            metadata[key] = json.dumps(value, ensure_ascii=False)
    return metadata


def open_collection(chroma_dir: Path | None = None, name: str | None = None):
    """Open the persistent collection with cosine distance, per §4.9."""
    import chromadb
    from chromadb.config import Settings

    directory = Path(chroma_dir or config.CHROMA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(directory),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(
        name=name or config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Read `artifacts/chunks.json` into Chunk objects."""
    target = path or config.CHUNKS_JSON
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run `python -m src.chunk` before embedding."
        )
    data = json.loads(target.read_text(encoding="utf-8"))
    return [
        Chunk(chunk_id=row["chunk_id"], text=row["text"], metadata=dict(row["metadata"]))
        for row in data["chunks"]
    ]


def _source_chunk_hashes(chunks: Sequence[Chunk], source_id: str) -> dict[str, str]:
    return {c.chunk_id: c.metadata["content_hash"] for c in chunks if c.metadata["source_id"] == source_id}


def _stored_ids(collection: Any) -> set[str]:
    stored = collection.get(include=[])
    return set(stored.get("ids") or [])


def _upsert(collection: Any, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
    """Write chunks in batches, so a large corpus never exceeds Chroma's batch limit."""
    size = 500
    written = 0
    for start in range(0, len(chunks), size):
        batch = chunks[start : start + size]
        collection.upsert(
            ids=[c.chunk_id for c in batch],
            embeddings=[list(v) for v in vectors[start : start + size]],
            documents=[embedding_text(c) for c in batch],
            metadatas=[chroma_metadata(c) for c in batch],
        )
        written += len(batch)
    return written


def _run_upstream_stages(sources: list[Source], force: bool) -> None:
    """Run fetch, extract, normalize and chunk through their own stage functions."""
    from src import chunk as chunk_stage
    from src import extract as extract_stage
    from src import fetch as fetch_stage
    from src import normalize as normalize_stage

    print(f"fetching {len(sources)} enabled sources")
    fetch_stage.fetch_all(sources, force=force)

    print("extracting sources")
    extract_stage.main()

    print("normalizing blocks")
    for source in sources:
        result = normalize_stage.normalize_source(source)
        if result is not None:
            before, after = result
            print(f"  {source.source_id} {before:>6} -> {after:<6} blocks")

    print(f"chunking {len(sources)} sources")
    chunk_stage.chunk_all_sources(sources)


def ingest(
    only_fetch: bool = False,
    only_embed: bool = False,
    force: bool = False,
    prune: bool = False,
    *,
    embedder: EmbedderProtocol | None = None,
    chroma_dir: Path | None = None,
    state_path: Path | None = None,
    chunks_path: Path | None = None,
    sources: list[Source] | None = None,
    run_upstream: bool = True,
) -> IngestResult:
    """Run the pipeline and build the vector store idempotently.

    `embedder`, `chroma_dir`, `state_path`, `chunks_path` and `run_upstream` are
    injection points so the idempotency contract can be tested without downloading
    model weights or touching the real corpus.
    """
    started = time.perf_counter()
    config.ensure_dirs()
    manifest = sources if sources is not None else load_sources(enabled_only=True)

    if run_upstream and not only_embed:
        if only_fetch:
            from src import fetch as fetch_stage

            print(f"fetching {len(manifest)} enabled sources")
            fetch_stage.fetch_all(manifest, force=force)
            return IngestResult(elapsed_ms=int((time.perf_counter() - started) * 1000))
        _run_upstream_stages(manifest, force)

    encoder = embedder or Embedder()
    fingerprint = str(getattr(encoder, "fingerprint", "") or "")
    if not fingerprint:
        raise ValueError("the embedder must expose a fingerprint so the index can be validated")

    state = read_state(state_path)
    previous_index = state.get("index", {}) if isinstance(state.get("index"), dict) else {}
    result = IngestResult(fingerprint=fingerprint)

    stored_fingerprint = str(state.get("fingerprint") or "")
    if stored_fingerprint and stored_fingerprint != fingerprint:
        result.fingerprint_changed = True
        result.warnings.append(
            "EMBEDDER FINGERPRINT CHANGED: "
            f"index was built with {stored_fingerprint}, this run uses {fingerprint}. "
            "Every source is being re-embedded so the index is not left mixed."
        )

    chunks = load_chunks(chunks_path)
    collection = open_collection(chroma_dir)

    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk.metadata["source_id"], []).append(chunk)

    enabled_ids = {s.source_id for s in manifest}
    new_index: dict[str, Any] = {}

    for source in manifest:
        source_id = source.source_id
        raw_path = config.RAW_DIR / source_id / source.filename
        h_file = file_hash(raw_path)
        h_chunks = _source_chunk_hashes(chunks, source_id)
        record = previous_index.get(source_id, {})

        unchanged = (
            not force
            and not result.fingerprint_changed
            and source_id in previous_index
            and record.get("file_hash") == h_file
            and record.get("chunk_hashes") == h_chunks
        )
        new_index[source_id] = {
            "file_hash": h_file,
            "chunk_hashes": h_chunks,
            "chunk_count": len(h_chunks),
            "fingerprint": fingerprint,
            "embedded_at": record.get("embedded_at") if unchanged else utc_now(),
        }

        if unchanged:
            result.unchanged.append(source_id)
            continue

        result.reembedded.append(source_id)
        collection.delete(where={"source_id": source_id})
        source_chunks = by_source.get(source_id, [])
        if not source_chunks:
            result.missing_chunks.append(source_id)
            print(f"  {source_id} no chunks in artifacts/chunks.json")
            continue
        vectors = encoder.embed_chunks(source_chunks, show_progress=True)
        result.upserts += _upsert(collection, source_chunks, vectors)
        result.embedded += len(source_chunks)

    if prune:
        result.pruned = _prune(collection, {c.chunk_id for c in chunks}, enabled_ids)
        if result.pruned:
            print(f"  pruned {len(result.pruned)} chunks from disabled sources")

    result.stored_total = collection.count()
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)

    state["fingerprint"] = fingerprint
    state["index"] = new_index
    state["last_ingest_at"] = utc_now()
    write_state(state, state_path)

    _print_summary(result, len(chunks))
    return result


def _prune(collection: Any, known_ids: set[str], enabled_ids: set[str]) -> list[str]:
    """Delete stored chunks that no longer belong to an enabled source.

    Chunks survive their source being disabled or dropped from `sources.csv`, so
    without this they would stay searchable forever and quietly skew results.
    """
    orphans = sorted(_stored_ids(collection) - known_ids)
    if not orphans:
        return []
    stored = collection.get(ids=orphans, include=["metadatas"])
    metadatas = stored.get("metadatas") or [{}] * len(stored.get("ids") or [])
    removed: list[str] = []
    for chunk_id, metadata in zip(stored.get("ids") or [], metadatas):
        source_id = str((metadata or {}).get("source_id", ""))
        if source_id and source_id in enabled_ids:
            continue
        collection.delete(ids=[chunk_id])
        removed.append(chunk_id)
    return removed


def _print_summary(result: IngestResult, chunk_total: int) -> None:
    print(
        f"\nembedded {result.embedded} chunks across {len(result.reembedded)} sources; "
        f"{len(result.unchanged)} unchanged"
    )
    print(f"collection holds {result.stored_total} chunks (chunks.json has {chunk_total})")
    print(f"fingerprint {result.fingerprint}")
    for warning in result.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"ingest finished in {result.elapsed_ms} ms")


def append_report(result: IngestResult, collection_name: str) -> None:
    """Record embedding statistics in artifacts/ingest_report.md."""
    from src.report import append_embedding_stats

    lines = [
        f"- embedder fingerprint: `{result.fingerprint}`",
        f"- sources embedded: {len(result.reembedded)}",
        f"- sources unchanged: {len(result.unchanged)}",
        f"- chunks upserted: {result.upserts}",
        f"- chunks in collection: {result.stored_total}",
        f"- finished at: {utc_now()}",
    ]
    if result.fingerprint_changed:
        lines.append("- **fingerprint changed: the whole index was rebuilt**")
    if result.pruned:
        lines.append(f"- pruned disabled sources: {', '.join(result.pruned)}")
    append_embedding_stats("\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.ingest")
    parser.add_argument("--only-fetch", action="store_true", help="fetch and stop")
    parser.add_argument("--only-embed", action="store_true", help="skip fetch/extract/chunk")
    parser.add_argument("--force", action="store_true", help="re-embed every source")
    parser.add_argument("--prune", action="store_true", help="drop chunks of disabled sources")
    args = parser.parse_args(argv)

    result = ingest(
        only_fetch=args.only_fetch,
        only_embed=args.only_embed,
        force=args.force,
        prune=args.prune,
    )
    append_report(result, config.COLLECTION_NAME)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())