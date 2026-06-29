"""Tests for the hardware probe."""

import subprocess

import pytest

from airllm_lab.services import hardware
from airllm_lab.services.hardware import HardwareProbe, HardwareSpec, parse_gpu_csv


def test_parse_gpu_csv_valid() -> None:
    """A typical nvidia-smi row parses to name + VRAM in GiB."""
    name, vram = parse_gpu_csv("NVIDIA GeForce GTX 1650, 4096")
    assert name == "NVIDIA GeForce GTX 1650"
    assert vram == pytest.approx(4.0, abs=0.01)


def test_parse_gpu_csv_empty() -> None:
    """Empty input yields the 'none' sentinel."""
    assert parse_gpu_csv("") == ("none", 0.0)


def test_query_gpu_no_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nvidia-smi is absent, query_gpu reports no GPU."""
    monkeypatch.setattr(hardware.shutil, "which", lambda _: None)
    assert hardware.query_gpu() == ("none", 0.0)


def test_query_gpu_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess failure degrades gracefully to no GPU."""
    monkeypatch.setattr(hardware.shutil, "which", lambda _: "nvidia-smi")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise subprocess.SubprocessError("boom")

    monkeypatch.setattr(hardware.subprocess, "run", _boom)
    assert hardware.query_gpu() == ("none", 0.0)


def test_probe_returns_sane_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """probe() returns a HardwareSpec with plausible values."""
    monkeypatch.setattr(hardware, "query_gpu", lambda: ("FakeGPU", 8.0))
    spec = HardwareProbe().probe()
    assert isinstance(spec, HardwareSpec)
    assert spec.cpu_threads >= spec.cpu_cores >= 1
    assert spec.ram_gb > 0
    assert spec.gpu_name == "FakeGPU"
    assert "gpu_name" in spec.to_dict()
