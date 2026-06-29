"""Tests for the configuration loader."""

import json
from pathlib import Path

import pytest

from airllm_lab.shared.config import Config, ConfigError, load_config
from airllm_lab.shared.version import __version__


def _write_setup(tmp_path: Path, version: str) -> Path:
    """Create a minimal config dir with a setup.json of the given version."""
    (tmp_path / "setup.json").write_text(json.dumps({"version": version}), encoding="utf-8")
    return tmp_path


def test_load_config_ok(tmp_path: Path) -> None:
    """A matching-version config loads successfully."""
    cfg = load_config(_write_setup(tmp_path, __version__))
    assert cfg.version == __version__


def test_load_config_missing_file(tmp_path: Path) -> None:
    """A missing setup.json raises ConfigError."""
    with pytest.raises(ConfigError, match="Missing config file"):
        load_config(tmp_path)


def test_load_config_version_mismatch(tmp_path: Path) -> None:
    """A mismatched version raises ConfigError."""
    with pytest.raises(ConfigError, match="!="):
        load_config(_write_setup(tmp_path, "9.9.9"))


def test_config_get_default() -> None:
    """Config.get returns the default for absent keys."""
    cfg = Config(raw={"version": "1.0.0"}, config_dir=Path("."))
    assert cfg.get("nope", 42) == 42
