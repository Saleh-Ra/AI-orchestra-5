"""Tests for the baseline feasibility helpers."""

from pathlib import Path

from airllm_lab.services.feasibility import baseline_fits, weights_size_bytes


def test_weights_size_sums_safetensors(tmp_path: Path) -> None:
    """It sums only the *.safetensors files in the directory."""
    (tmp_path / "a.safetensors").write_bytes(b"x" * 100)
    (tmp_path / "b.safetensors").write_bytes(b"y" * 50)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert weights_size_bytes(tmp_path) == 150


def test_weights_size_zero_for_missing_dir() -> None:
    """A non-existent / remote id path yields 0 (unknown)."""
    assert weights_size_bytes("org/remote-model-id") == 0


def test_baseline_fits_true_when_small() -> None:
    """Weights well under available RAM fit."""
    assert baseline_fits(weights_bytes=1_000, available_ram_bytes=10_000) is True


def test_baseline_fits_false_when_too_big() -> None:
    """Weights exceeding RAM (with safety margin) do not fit."""
    assert baseline_fits(weights_bytes=9_000, available_ram_bytes=10_000) is False


def test_baseline_fits_true_when_unknown() -> None:
    """Unknown size (0) is treated as fitting so remote ids are attempted."""
    assert baseline_fits(weights_bytes=0, available_ram_bytes=1) is True
