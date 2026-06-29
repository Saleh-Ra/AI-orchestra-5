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
