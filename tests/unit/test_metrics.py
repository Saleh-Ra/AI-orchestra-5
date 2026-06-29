"""Tests for the pure latency/throughput metrics."""

import pytest

from airllm_lab.services import metrics


def test_ttft_is_difference() -> None:
    """TTFT is the gap between start and the first token."""
    assert metrics.time_to_first_token(10.0, 10.5) == pytest.approx(0.5)


def test_ttft_never_negative() -> None:
    """Out-of-order timestamps clamp TTFT to zero."""
    assert metrics.time_to_first_token(10.0, 9.0) == 0.0


def test_tpot_averages_decode_phase() -> None:
    """TPOT divides decode time by (n_tokens - 1)."""
    assert metrics.time_per_output_token(1.0, 5.0, 5) == pytest.approx(1.0)


def test_tpot_zero_for_single_token() -> None:
    """A single token has no inter-token interval."""
    assert metrics.time_per_output_token(1.0, 5.0, 1) == 0.0


def test_throughput_tokens_per_second() -> None:
    """Throughput is tokens divided by runtime."""
    assert metrics.throughput(100, 4.0) == pytest.approx(25.0)


def test_throughput_zero_runtime() -> None:
    """Zero runtime yields zero throughput (no divide-by-zero)."""
    assert metrics.throughput(100, 0.0) == 0.0


def test_energy_wh_power_times_time() -> None:
    """Energy is watts × hours: 3600 s at 65 W = 65 Wh."""
    assert metrics.energy_wh(3600.0, 65.0) == pytest.approx(65.0)


def test_energy_wh_zero_for_nonpositive() -> None:
    """Non-positive runtime or watts yields zero energy."""
    assert metrics.energy_wh(0.0, 65.0) == 0.0
    assert metrics.energy_wh(10.0, 0.0) == 0.0


def test_summarize_empty_is_zeros() -> None:
    """Summarizing an empty list returns zeroed stats."""
    assert metrics.summarize([]) == {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


def test_summarize_odd_count() -> None:
    """Median of an odd-length list is the middle value."""
    out = metrics.summarize([3.0, 1.0, 2.0])
    assert out["mean"] == pytest.approx(2.0)
    assert out["median"] == pytest.approx(2.0)
    assert out["min"] == 1.0
    assert out["max"] == 3.0


def test_summarize_even_count_averages_middle() -> None:
    """Median of an even-length list averages the two middle values."""
    out = metrics.summarize([1.0, 2.0, 3.0, 4.0])
    assert out["median"] == pytest.approx(2.5)
    assert out["std"] == pytest.approx(1.1180, abs=1e-3)
