"""Stage 3 - embedding.

One `Embedder` wraps `all-MiniLM-L6-v2` and is the only component that touches
sentence-transformers. Retrieval (P4) reuses the same instance and asserts that
its fingerprint matches the one recorded in `artifacts/ingest_state.json`, so a
model change can never be silently mixed with an existing index (NFR-4).

The model is loaded lazily on first encode, so importing this module is cheap and
tests can substitute a fake embedder without downloading 90 MB of weights.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import config
from src.models import Chunk


@runtime_checkable
class EmbedderProtocol(Protocol):
    """The surface `ingest` and `retrieve` depend on."""

    fingerprint: str

    def embed_chunks(
        self, chunks: Sequence[Chunk], *, show_progress: bool = False
    ) -> list[list[float]]:
        ...

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        ...


def embedding_text(chunk: Chunk) -> str:
    """Return the exact string that gets embedded.

    `architecture.md` §4.9 requires `section_path + "\n" + text` so the scheme and
    plan context is part of the vector rather than only the metadata (AD-2).
    `chunk.py` already prepends the breadcrumb as the first lines of the text, so
    the breadcrumb is prepended here only when it is genuinely absent, never twice.
    """
    section_path = str(chunk.metadata.get("section_path") or "")
    breadcrumb = "\n".join(part for part in section_path.split(" > ") if part)
    text = chunk.text or ""
    if not breadcrumb:
        return text
    if text.startswith(breadcrumb):
        return text
    return f"{breadcrumb}\n{text}"


def _package_version(name: str) -> str:
    """Read an installed distribution version without importing the package."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except Exception as exc:  # noqa: BLE001 - surfaced as a fingerprint, never swallowed
        raise RuntimeError(f"{name} is not installed; it is required for P3") from exc


class Embedder:
    """Lazily-loaded MiniLM encoder producing L2-normalized 384-dim vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        dim: int | None = None,
        normalize: bool = True,
        batch_size: int | None = None,
    ) -> None:
        self.model_name = model_name or config.EMBED_MODEL
        self.dim = dim or config.EMBED_DIM
        self.normalize = normalize
        self.batch_size = batch_size or config.EMBED_BATCH_SIZE
        self.version = _package_version("sentence-transformers")
        self._model: Any | None = None
        self.fingerprint = f"{self.model_name}@{self.version}:{self.dim}:{self.normalize}"

    @property
    def model(self) -> Any:
        """Load the sentence-transformers model on first access."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str], *, show_progress: bool = False) -> list[list[float]]:
        """Encode texts into unit-length vectors, in input order."""
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return [[float(value) for value in row] for row in vectors]

    def embed_chunks(
        self, chunks: Sequence[Chunk], *, show_progress: bool = False
    ) -> list[list[float]]:
        """Encode chunks using the AD-2 embedding input."""
        return self.encode(
            [embedding_text(chunk) for chunk in chunks], show_progress=show_progress
        )


def load_embedder() -> Embedder:
    """Build the production embedder."""
    return Embedder()