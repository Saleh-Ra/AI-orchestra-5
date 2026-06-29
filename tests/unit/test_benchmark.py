"""Tests for the BenchmarkHarness (pure orchestration with a fake runner)."""

from airllm_lab.services.benchmark import BenchmarkHarness
from airllm_lab.services.models import RunConfig, RunResult


def _result(ok: bool = True, runtime: float = 1.0) -> RunResult:
    """Build a RunResult with a given runtime/ok for aggregation tests."""
    return RunResult(
        model_id="m",
        mode="airllm",
        quant="fp16",
        device="cpu",
        n_input_tokens=1,
        n_output_tokens=2,
        ttft_s=0.5,
        tpot_s=0.1,
        throughput_tok_s=2.0,
        runtime_s=runtime,
        output_text="x",
        ok=ok,
    )


class _SequenceRunner:
    """Runner that yields predefined results and counts its calls."""

    def __init__(self, results: list[RunResult]) -> None:
        self._results = list(results)
        self.calls = 0

    def run(self, cfg: RunConfig) -> RunResult:
        self.calls += 1
        return self._results.pop(0)


def _cfg() -> RunConfig:
    return RunConfig(model_id="m", prompt="hi", mode="airllm")


def test_harness_repeats_and_aggregates() -> None:
    """It runs ``repeats`` times and aggregates runtime across them."""
    runner = _SequenceRunner([_result(runtime=2.0), _result(runtime=4.0)])
    summary = BenchmarkHarness(runner).run(_cfg(), repeats=2)
    assert runner.calls == 2
    assert summary.repeats == 2
    assert summary.n_ok == 2
    assert summary.aggregates["runtime_s"]["mean"] == 3.0
    assert len(summary.runs) == 2


def test_harness_discards_warmup() -> None:
    """Warmup runs execute but are not counted in the summary."""
    runner = _SequenceRunner([_result(), _result(), _result()])
    summary = BenchmarkHarness(runner).run(_cfg(), repeats=2, warmup=1)
    assert runner.calls == 3
    assert summary.repeats == 2


def test_harness_aggregates_only_successful_runs() -> None:
    """Failed runs are excluded from the metric aggregates."""
    runner = _SequenceRunner([_result(ok=True, runtime=2.0), _result(ok=False, runtime=99.0)])
    summary = BenchmarkHarness(runner).run(_cfg(), repeats=2)
    assert summary.n_ok == 1
    assert summary.aggregates["runtime_s"]["max"] == 2.0
