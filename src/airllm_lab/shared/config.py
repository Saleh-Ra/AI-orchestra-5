"""Configuration loading and version validation.

Loads versioned JSON config from ``config/`` and validates that the config
version matches the code version, so configuration drift is caught early
at startup rather than mid-experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from airllm_lab.shared.version import __version__


class ConfigError(RuntimeError):
    """Raised when configuration is missing or inconsistent."""


@dataclass(frozen=True)
class Config:
    """In-memory view of the project configuration."""

    raw: dict[str, Any]
    config_dir: Path

    @property
    def version(self) -> str:
        """Version string declared in ``setup.json``."""
        return str(self.raw.get("version", ""))

    def get(self, key: str, default: Any = None) -> Any:
        """Return a top-level config value, or ``default`` if absent."""
        return self.raw.get(key, default)


def load_config(config_dir: Path | str = "config") -> Config:
    """Load ``setup.json`` and validate its version against the code version.

    Raises:
        ConfigError: if the file is missing or the versions disagree.
    """
    config_path = Path(config_dir) / "setup.json"
    if not config_path.is_file():
        raise ConfigError(f"Missing config file: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = Config(raw=raw, config_dir=Path(config_dir))
    if cfg.version != __version__:
        raise ConfigError(f"Config version {cfg.version!r} != code version {__version__!r}")
    return cfg
