"""Roofline plot (original extension): is the workload compute- or memory-bound?

The roofline model bounds attainable performance by ``min(peak_compute,
bandwidth × operational_intensity)``. LLM decode has an intensity of ~1 FLOP/byte
(it reads ~2 bytes per weight to do ~2 FLOPs), so it lives on the *slanted*
bandwidth roof — it is memory-bound. We draw two roofs: GPU memory (for a model
that fits) and disk (for AirLLM's layer streaming), then place each measured run
at its achieved FLOP/s to show which roof it is pinned under.

Matplotlib I/O only (no unit tests; exercised by the ``roofline`` command).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from airllm_lab.services.charts import _collect_runs  # noqa: E402

_GFLOP = 1e9


def _params_from_label(label: str) -> float | None:
    """Parse a parameter count from a model label like ``Qwen2.5-7B``."""
    match = re.search(r"([\d.]+)\s*([BM])", label, re.IGNORECASE)
    if not match:
        return None
    scale = 1e9 if match.group(2).upper() == "B" else 1e6
    return float(match.group(1)) * scale


def _achieved_gflops(params: float, throughput_tok_s: float) -> float:
    """Achieved GFLOP/s ≈ 2 × params × tokens-per-second (forward-pass FLOPs)."""
    return 2.0 * params * throughput_tok_s / _GFLOP


def render_roofline(results_dir: Path, out_dir: Path, peaks: dict[str, float]) -> Path | None:
    """Draw the roofline with GPU + disk roofs and the measured run points."""
    peak = peaks["gpu_peak_gflops_fp16"]
    bw_gpu = peaks["gpu_mem_bandwidth_gb_s"]
    bw_disk = peaks["disk_read_gb_s"]
    intensity = peaks["decode_intensity_flops_per_byte"]

    points = [p for p in _collect_runs(results_dir) if p["throughput"] > 0]
    if not points:
        return None

    oi = [10**e for e in range(-2, 3)]  # operational intensity sweep (FLOP/byte)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(oi, [min(peak, bw_gpu * x) for x in oi], color="#2563eb", label=f"GPU mem roof ({bw_gpu:g} GB/s)")
    ax.plot(oi, [min(peak, bw_disk * x) for x in oi], color="#dc2626", label=f"disk roof ({bw_disk:g} GB/s)")
    ax.axhline(peak, ls=":", color="gray", label=f"compute roof ({peak:g} GFLOP/s)")

    for point in points:
        params = _params_from_label(point["label"])
        if not params:
            continue
        achieved = _achieved_gflops(params, point["throughput"])
        ax.scatter([intensity], [achieved], s=70, zorder=3)
        ax.annotate(point["label"].replace("\n", " "), (intensity, achieved),
                    textcoords="offset points", xytext=(8, 0), fontsize=8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operational intensity (FLOP / byte)")
    ax.set_ylabel("performance (GFLOP/s)")
    ax.set_title("Roofline: where each run is bottlenecked")
    ax.legend(loc="lower right", fontsize=8)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "roofline.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
