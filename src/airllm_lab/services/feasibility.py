"""Pre-flight feasibility checks for naive (baseline) in-memory execution.

The baseline loads the whole model into RAM before moving it to the GPU. On
modest hardware this thrashes/crashes (observed: the 7B FP16 model crashed the
process at ~61% load). These pure helpers let the runner predict that and fail
fast with a clear message instead of hanging the machine.
"""

from __future__ import annotations

from pathlib import Path


def weights_size_bytes(model_dir: str | Path) -> int:
    """Sum of ``*.safetensors`` sizes in a local model dir (0 if not local)."""
    path = Path(model_dir)
    if not path.is_dir():
        return 0
    return sum(f.stat().st_size for f in path.glob("*.safetensors"))


def baseline_fits(weights_bytes: int, available_ram_bytes: int, safety: float = 1.2) -> bool:
    """Whether weights (+ overhead) plausibly fit in available RAM.

    Returns ``True`` when ``weights_bytes`` is unknown (0), so remote model ids
    are still attempted rather than pre-emptively blocked.
    """
    if weights_bytes <= 0:
        return True
    return weights_bytes * safety <= available_ram_bytes
