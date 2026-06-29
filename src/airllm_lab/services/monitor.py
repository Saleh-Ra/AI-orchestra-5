"""Peak memory sampling during a generation run.

VRAM peak comes from torch's allocator counters (cheap, exact). Host RAM is
sampled on a background thread polling this process's RSS, so we capture the
peak working set across a run. Both are reported in GB. Heavy/threaded I/O, so
this module is exercised by real runs rather than unit tests.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType

import psutil


def reset_vram_peak() -> None:
    """Reset torch's CUDA peak-memory counter (no-op without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def vram_peak_gb() -> float:
    """Return torch's CUDA peak allocated memory in GB (0.0 without CUDA)."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    except Exception:
        pass
    return 0.0


class PeakRamSampler:
    """Context manager sampling this process's RSS; exposes the peak in GB."""

    def __init__(self, interval_s: float = 0.2) -> None:
        """Store the polling interval and prepare sampling state."""
        self._interval = interval_s
        self._proc = psutil.Process()
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        """Background loop recording the maximum observed RSS."""
        while not self._stop.is_set():
            self._peak_bytes = max(self._peak_bytes, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def __enter__(self) -> PeakRamSampler:
        """Start the background sampler thread."""
        self._peak_bytes = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop and join the sampler thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    @property
    def peak_gb(self) -> float:
        """Peak resident set size observed, in GB."""
        return round(self._peak_bytes / 1e9, 3)
