"""Tests for the secret and storage helpers."""

from pathlib import Path

import pytest

from airllm_lab.shared import secrets, storage


def test_get_hf_token_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-blank HF_TOKEN is returned trimmed."""
    monkeypatch.setattr(secrets, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("HF_TOKEN", "  hf_abc  ")
    assert secrets.get_hf_token() == "hf_abc"


def test_get_hf_token_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank/missing HF_TOKEN yields None."""
    monkeypatch.setattr(secrets, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert secrets.get_hf_token() is None


def test_save_json_writes_and_creates_dirs(tmp_path: Path) -> None:
    """save_json creates parent dirs and writes readable JSON."""
    import json

    target = tmp_path / "nested" / "out.json"
    result = storage.save_json(target, {"a": 1, "b": "x"})
    assert result == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": "x"}


def test_configure_hf_cache_sets_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_hf_cache creates the dir and points HF env vars at it."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    cache = tmp_path / "hf"
    result = storage.configure_hf_cache(cache)
    assert result == cache
    assert cache.is_dir()
    import os

    assert os.environ["HF_HOME"] == str(cache)
    assert os.environ["HF_HUB_CACHE"] == str(cache / "hub")
