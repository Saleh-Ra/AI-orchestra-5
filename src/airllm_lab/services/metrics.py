"""Latency / throughput metrics derived from per-token timestamps.

These are pure functions (no I/O) so they are trivially unit-testable and
reused by every runner. Timestamps are seconds from ``time.perf_counter()``.
"""

from __future__ import annotations


def time_to_first_token(start: float, first_token_at: float) -> float:
    """TTFT: seconds from generation start to the first output token.

    Reflects the compute-bound *prefill* phase (KV-cache build).
    """
    return max(0.0, first_token_at - start)


def time_per_output_token(first_token_at: float, end: float, n_output_tokens: int) -> float:
    """TPOT: mean seconds per output token after the first one.

    Reflects the memory-bandwidth-bound *decode* phase. Returns ``0.0`` when
    there are not at least two tokens to measure between.
    """
    if n_output_tokens <= 1:
        return 0.0
    return max(0.0, (end - first_token_at) / (n_output_tokens - 1))


def throughput(n_output_tokens: int, runtime_s: float) -> float:
    """Overall output tokens per second across the whole run."""
    if runtime_s <= 0:
        return 0.0
    return n_output_tokens / runtime_s
