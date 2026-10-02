"""Tests for config invariants."""

from __future__ import annotations

import config


def test_pii_log_raw_is_false():
    assert config.PII_LOG_RAW is False


def test_groq_key_is_never_exposed_as_value():
    assert isinstance(config.key_present(), bool)


def test_chunk_params_match_prd():
    assert config.MAX_CHUNK_CHARS == 900
    assert config.MIN_CHUNK_CHARS == 120
    assert config.CHUNK_OVERLAP == 150
    assert config.TABLE_OVERLAP == 0


def test_embedding_params_match_prd():
    assert config.EMBED_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.EMBED_DIM == 384
    assert config.COLLECTION_NAME == "axis_mf_faq"


def test_retrieval_params_match_prd():
    # top_k and MAX_CONTEXT_CHARS are ceilings on the same pipeline; both were
    # raised from 5/6000 so the rank-11 answer chunk can reach the prompt.
    assert config.TOP_K == 12
    assert config.CITATION_SCORE_FLOOR == 0.45
    assert config.MAX_CONTEXT_CHARS == 12000


def test_generation_params_match_prd():
    assert config.TEMPERATURE == 0.0
    assert config.MAX_TOKENS == 220


def test_config_imports_no_project_modules():
    import config as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "from src" not in source
    assert "import src" not in source


def test_ensure_dirs_creates_paths(tmp_path, monkeypatch):
    created = []
    for name in ("RAW_DIR", "PROCESSED_DIR", "ARTIFACTS_DIR", "CHROMA_DIR", "EVAL_DIR", "MODELS_DIR"):
        target = tmp_path / name
        setattr(config, name, target)
        created.append(target)
    config.ensure_dirs()
    for target in created:
        assert target.exists()