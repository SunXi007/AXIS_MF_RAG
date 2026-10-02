"""Environment, dependency and corpus health check.

Prints the key presence as a boolean only; the key value is never emitted.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys

import config
from src.sources import counts_by_scheme, excluded_sources, load_sources

OPTIONAL_PACKAGES = (
    "sentence_transformers",
    "chromadb",
    "groq",
    "streamlit",
    "tiktoken",
    "tqdm",
)

REQUIRED_PACKAGES = {
    "httpx": "httpx",
    "pymupdf": "pymupdf",
    "trafilatura": "trafilatura",
    "bs4": "beautifulsoup4",
    "lxml": "lxml",
    "dotenv": "python-dotenv",
    "pytest": "pytest",
}


def _installed(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _importable(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def _collection_exists() -> bool:
    if not config.CHROMA_DIR.exists():
        return False
    try:
        import chromadb
    except ImportError:
        return False
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return config.COLLECTION_NAME in [
            c.name for c in client.list_collections()
        ]
    except Exception:
        return False


def main() -> int:
    print("=" * 62)
    print("healthcheck - Mutual Fund FAQ Assistant")
    print("=" * 62)

    print("\n[python]")
    print(f"  version        {sys.version.split()[0]}")

    print("\n[required packages]")
    for module, distribution in REQUIRED_PACKAGES.items():
        version = _installed(distribution)
        importable = "importable" if _importable(module) else "IMPORT FAILED"
        print(f"  {module:<22}{version:<18}{importable}")

    print("\n[deferred packages]")
    for module in OPTIONAL_PACKAGES:
        installed = _importable(module)
        print(f"  {module:<22}{'installed' if installed else 'not installed (expected before its phase)'}")

    print("\n[config]")
    print(f"  PII_LOG_RAW            {config.PII_LOG_RAW}")
    print(f"  EMBED_MODEL            {config.EMBED_MODEL}")
    print(f"  EMBED_DIM              {config.EMBED_DIM}")
    print(f"  COLLECTION_NAME        {config.COLLECTION_NAME}")
    print(f"  MAX_CHUNK_CHARS        {config.MAX_CHUNK_CHARS}")
    print(f"  MIN_CHUNK_CHARS        {config.MIN_CHUNK_CHARS}")
    print(f"  CHUNK_OVERLAP          {config.CHUNK_OVERLAP}")
    print(f"  TABLE_OVERLAP          {config.TABLE_OVERLAP}")
    print(f"  MIN_EXTRACTED_CHARS    {config.MIN_EXTRACTED_CHARS}")
    print(f"  TOP_K                  {config.TOP_K}")
    print(f"  SIMILARITY_THRESHOLD   {config.SIMILARITY_THRESHOLD}")
    print(f"  CITATION_SCORE_FLOOR   {config.CITATION_SCORE_FLOOR}")
    print(f"  MAX_CONTEXT_CHARS      {config.MAX_CONTEXT_CHARS}")
    print(f"  TEMPERATURE            {config.TEMPERATURE}")
    print(f"  MAX_TOKENS             {config.MAX_TOKENS}")
    print(f"  GROQ_MODEL             {config.GROQ_MODEL}")
    print(f"  groq_key_present       {config.key_present()}")

    print("\n[corpus]")
    try:
        all_sources = load_sources()
        enabled = [s for s in all_sources if s.enabled]
        disabled = excluded_sources()
        print(f"  rows in manifest       {len(all_sources)}")
        print(f"  enabled                {len(enabled)}")
        print(f"  excluded               {len(disabled)}")
        for source in disabled:
            print(f"    {source.source_id} excluded: {source.notes.split(' - ')[0]}")
        print()
        for scheme, count in sorted(counts_by_scheme(enabled).items()):
            print(f"    {scheme:<14}{count}")
    except Exception as exc:
        print(f"  FAILED to load sources.csv: {exc}")
        return 1

    print("\n[paths]")
    for label, path in (
        ("RAW_DIR", config.RAW_DIR),
        ("PROCESSED_DIR", config.PROCESSED_DIR),
        ("ARTIFACTS_DIR", config.ARTIFACTS_DIR),
        ("CHROMA_DIR", config.CHROMA_DIR),
    ):
        state = "exists" if path.exists() else "missing"
        print(f"  {label:<20}{state:<10}{path}")

    print("\n[vector store]")
    print(f"  collection present     {_collection_exists()}")

    print("\n" + "=" * 62)
    if config.PII_LOG_RAW:
        print("RED: PII_LOG_RAW is enabled")
        return 1
    print("GREEN: scaffold is valid")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())