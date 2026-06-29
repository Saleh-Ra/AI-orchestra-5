"""Render benchmark + cost figures from stored JSON results into ``assets/``.

Matplotlib I/O only (no unit tests; exercised by the ``charts`` command). Each
function reads the artifacts the SDK already wrote and emits one PNG. Missing
inputs are skipped so the command works with whatever results exist so far.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from airllm_lab.services import cost_model  # noqa: E402

_API, _ONPREM, _CLOUD = "#2563eb", "#dc2626", "#16a34a"


def _load(path: Path) -> dict[str, Any] | None:
    """Read a JSON artifact, or ``None`` if it does not exist."""
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Write ``fig`` to ``out_dir/name`` and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _short(model_id: str) -> str:
    """Compact model label for axis ticks."""
    return Path(model_id).name.replace("-Instruct", "")


def _collect_runs(results_dir: Path) -> list[dict[str, Any]]:
    """Gather comparable run points from benchmark + single-run artifacts."""
    points: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("benchmark_*.json")):
        data = _load(path)
        if not data or data.get("n_ok", 0) == 0:
            continue
        agg = data["aggregates"]
        points.append({
            "label": f"{_short(data['model_id'])}\n{data['mode']} {data['quant']}",
            "throughput": agg["throughput_tok_s"]["mean"],
            "ttft": agg["ttft_s"]["mean"],
            "peak_ram": agg.get("peak_ram_gb", {}).get("mean", 0.0),
            "peak_vram": agg.get("peak_vram_gb", {}).get("mean", 0.0),
        })
    for pattern in ("airllm_*.json", "baseline_*.json"):
        for path in sorted(results_dir.glob(pattern)):
            data = _load(path)
            if not data or not data.get("ok"):
                continue
            points.append({
                "label": f"{_short(data['model_id'])}\n{data['mode']} {data['quant']}",
                "throughput": data.get("throughput_tok_s", 0.0),
                "ttft": data.get("ttft_s", 0.0),
                "peak_ram": data.get("peak_ram_gb", 0.0),
                "peak_vram": data.get("peak_vram_gb", 0.0),
            })
    return points


def _bar_chart(points: list[dict[str, Any]], key: str, title: str, ylabel: str, out: Path) -> Path | None:
    """Log-scale bar chart of one metric across runs."""
    usable = [p for p in points if p[key] > 0]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [p["label"] for p in usable]
    values = [p[key] for p in usable]
    bars = ax.bar(labels, values, color=["#16a34a", "#dc2626", "#2563eb", "#9333ea"][: len(usable)])
    ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for rect, value in zip(bars, values, strict=False):
        ax.annotate(f"{value:g}", (rect.get_x() + rect.get_width() / 2, value),
                    ha="center", va="bottom", fontsize=9)
    return _save(fig, out, f"{key}.png")


def _memory_chart(points: list[dict[str, Any]], out: Path) -> Path | None:
    """Grouped bars of peak RAM vs VRAM per run."""
    usable = [p for p in points if p["peak_ram"] > 0 or p["peak_vram"] > 0]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    idx = range(len(usable))
    ax.bar([i - 0.2 for i in idx], [p["peak_ram"] for p in usable], 0.4, label="peak RAM", color="#2563eb")
    ax.bar([i + 0.2 for i in idx], [p["peak_vram"] for p in usable], 0.4, label="peak VRAM", color="#f59e0b")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([p["label"] for p in usable])
    ax.set_ylabel("GB")
    ax.set_title("Peak memory footprint")
    ax.legend()
    return _save(fig, out, "memory.png")


def _breakeven_chart(report: dict[str, Any], out: Path) -> Path:
    """Cumulative monthly cost vs request volume, with the crossover marked."""
    asm = report["assumptions"]
    onprem = cost_model.OnPremSpec(**asm["onprem"])
    fixed = cost_model.onprem_fixed_monthly(onprem)
    energy = cost_model.energy_cost_per_request(asm["runtime_s_per_request"], onprem)
    api_req, cloud_req = report["api_per_request"], report["cloud_per_request"]
    be = report["breakeven_requests_per_month"]
    max_x = (be * 2) if be else asm["usage"]["requests_per_month"] * 2
    xs = [max_x * i / 50 for i in range(51)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, [api_req * x for x in xs], color=_API, label="Hosted API")
    ax.plot(xs, [fixed + energy * x for x in xs], color=_ONPREM, label="OnPrem (CAPEX+energy)")
    ax.plot(xs, [cloud_req * x for x in xs], color=_CLOUD, label="Cloud GPU")
    if be:
        ax.axvline(be, ls="--", color="gray")
        ax.annotate(f"break-even ≈ {be:,.0f}/mo", (be, fixed), fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("requests / month")
    ax.set_ylabel("monthly cost (USD)")
    ax.set_title("Break-even: OnPrem vs API vs Cloud")
    ax.legend()
    return _save(fig, out, "cost_breakeven.png")


def render_all(results_dir: Path, out_dir: Path) -> list[Path]:
    """Render every figure for which inputs exist; return written paths."""
    points = _collect_runs(results_dir)
    created = [
        _bar_chart(points, "throughput", "Throughput (tokens/sec, log scale)", "tok/s", out_dir),
        _bar_chart(points, "ttft", "Time to first token (log scale)", "seconds", out_dir),
        _memory_chart(points, out_dir),
    ]
    report = _load(results_dir / "cost_analysis.json")
    if report:
        created.append(_breakeven_chart(report, out_dir))
    return [p for p in created if p is not None]
