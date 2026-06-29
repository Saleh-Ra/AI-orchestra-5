"""Filesystem/path helpers: force heavy artifacts onto the configured drive.

The host ``C:`` drive is nearly full, so model weights and the Hugging Face
cache must live on ``D:``. These helpers set the relevant env vars at runtime.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def save_json(path: str | Path, data: Any) -> Path:
    """Write ``data`` as pretty JSON to ``path``, creating parent dirs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def configure_hf_cache(hf_cache_dir: str | Path) -> Path:
    """Point the Hugging Face cache at ``hf_cache_dir`` (kept off ``C:``).

    Creates the directory and sets ``HF_HOME`` / ``HF_HUB_CACHE`` so downloads
    land there. Returns the resolved cache path.
    """
    path = Path(hf_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(path)
    os.environ["HF_HUB_CACHE"] = str(path / "hub")
    # The Xet download backend stalls on large files on this Windows host, so
    # force the standard, reliable HTTP downloader instead.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    return path
