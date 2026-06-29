"""Tests for the curl-based model downloader (network mocked)."""

from pathlib import Path

import pytest

from airllm_lab.services import model_download
from airllm_lab.services.model_download import ModelDownloader


def test_download_fetches_each_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """download lists repo files and curls each (skipping .gitattributes)."""
    monkeypatch.setattr(
        model_download,
        "list_repo_files",
        lambda model_id, token=None: [".gitattributes", "config.json", "model.safetensors"],
    )
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool = False) -> None:
        calls.append(cmd)

    monkeypatch.setattr(model_download.subprocess, "run", _fake_run)

    dest = ModelDownloader(hf_token="tok").download("org/Model-7B", tmp_path)

    assert dest == tmp_path / "Model-7B"
    assert dest.is_dir()
    # .gitattributes skipped -> two fetches; token passed as a header.
    assert len(calls) == 2
    fetched = [c[-1] for c in calls]
    assert "org/Model-7B/resolve/main/config.json" in fetched[0]
    assert any("Authorization: Bearer tok" in part for part in calls[0])


def test_download_no_token_no_auth_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a token, no Authorization header is added."""
    monkeypatch.setattr(
        model_download, "list_repo_files", lambda model_id, token=None: ["config.json"]
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(
        model_download.subprocess, "run", lambda cmd, check=False: calls.append(cmd)
    )

    ModelDownloader().download("org/Model", tmp_path)

    assert not any("Authorization" in part for part in calls[0])
