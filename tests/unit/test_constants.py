"""Tests for constants and enumerations."""

from airllm_lab.constants import BYTES_PER_MB, SECONDS_PER_HOUR, QuantLevel, RunMode


def test_run_mode_values() -> None:
    """RunMode exposes the two supported execution modes."""
    assert RunMode.BASELINE.value == "baseline"
    assert RunMode.AIRLLM.value == "airllm"


def test_quant_levels_are_complete() -> None:
    """QuantLevel covers the planned bit-widths."""
    assert {level.value for level in QuantLevel} == {"fp16", "q8", "q4", "q2"}


def test_unit_conversions() -> None:
    """Unit constants have their expected integer values."""
    assert BYTES_PER_MB == 1_048_576
    assert SECONDS_PER_HOUR == 3_600
