"""Hardware probe: collect CPU / RAM / GPU / disk facts for the report.

Uses ``psutil`` for CPU/RAM/disk and ``nvidia-smi`` (via subprocess) for GPU,
so it has no heavy ML dependency and stays easy to unit-test.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

import psutil


@dataclass(frozen=True)
class HardwareSpec:
    """Snapshot of the host machine relevant to LLM inference."""

    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpu_name: str
    vram_gb: float
    free_disk_gb: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


def parse_gpu_csv(text: str) -> tuple[str, float]:
    """Parse the first ``nvidia-smi`` CSV row into ``(name, vram_gb)``.

    Expects rows like ``NVIDIA GeForce GTX 1650, 4096`` (MiB). Returns
    ``("none", 0.0)`` for empty input.
    """
    first = text.splitlines()[0] if text else ""
    if not first.strip():
        return ("none", 0.0)
    name, _, mem = first.partition(",")
    mib = float(mem.strip() or 0)
    return (name.strip(), round(mib / 1024, 2))


def query_gpu() -> tuple[str, float]:
    """Return ``(gpu_name, vram_gb)`` via nvidia-smi, or ``("none", 0.0)``."""
    if shutil.which("nvidia-smi") is None:
        return ("none", 0.0)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return ("none", 0.0)
    return parse_gpu_csv(result.stdout.strip())


class HardwareProbe:
    """Collects a :class:`HardwareSpec` from the running machine."""

    def __init__(self, root: str = ".") -> None:
        """Store the path used to measure free disk space."""
        self._root = root

    def probe(self) -> HardwareSpec:
        """Gather CPU, RAM, GPU, and free-disk facts."""
        gpu_name, vram_gb = query_gpu()
        return HardwareSpec(
            cpu_cores=psutil.cpu_count(logical=False) or 0,
            cpu_threads=psutil.cpu_count(logical=True) or 0,
            ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            free_disk_gb=round(shutil.disk_usage(self._root).free / 1024**3, 1),
        )
