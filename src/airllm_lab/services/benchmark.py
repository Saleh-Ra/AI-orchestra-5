"""BenchmarkHarness: run one config N times and aggregate the metrics.

Pure orchestration: it takes any object with a ``run(cfg) -> RunResult`` method,
so it is unit-testable with a fake runner (no torch). Warmup runs are executed
but discarded; aggregates (mean/median/std/min/max) are computed over the
successful runs only.
"""

from __future__ import annotations

from typing import Protocol

from airllm_lab.services import metrics
from airllm_lab.services.models import BenchmarkSummary, RunConfig, RunResult

# Numeric metrics we aggregate across repeats.
_AGGREGATED = (
    "ttft_s",
    "tpot_s",
    "throughput_tok_s",
    "runtime_s",
    "peak_ram_gb",
    "peak_vram_gb",
    "energy_wh",
)


class Runner(Protocol):
    """Anything that can execute a :class:`RunConfig` into a :class:`RunResult`."""

    def run(self, cfg: RunConfig) -> RunResult:
        """Execute the run and return its measured result."""
        ...


class BenchmarkHarness:
    """Repeats a run and aggregates its metrics into a :class:`BenchmarkSummary`."""

    def __init__(self, runner: Runner) -> None:
        """Store the runner whose ``run`` will be repeated."""
        self._runner = runner

    def run(self, cfg: RunConfig, repeats: int = 1, warmup: int = 0) -> BenchmarkSummary:
        """Execute ``warmup`` discarded runs, then ``repeats`` measured runs."""
        for _ in range(max(0, warmup)):
            self._runner.run(cfg)
        results = [self._runner.run(cfg) for _ in range(max(1, repeats))]
        ok = [r for r in results if r.ok]
        aggregates = {
            metric: metrics.summarize([getattr(r, metric) for r in ok]) for metric in _AGGREGATED
        }
        return BenchmarkSummary(
            model_id=cfg.model_id,
            mode=cfg.mode,
            quant=cfg.quant,
            repeats=len(results),
            n_ok=len(ok),
            runs=[r.to_dict() for r in results],
            aggregates=aggregates,
        )
