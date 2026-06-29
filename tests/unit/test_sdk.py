"""Tests for the LabSDK facade."""

from pathlib import Path
from typing import Any

import pytest

from airllm_lab.sdk import LabSDK
from airllm_lab.services.hardware import HardwareSpec
from airllm_lab.services.models import RunConfig, RunResult
from airllm_lab.shared.config import Config
from airllm_lab.shared.version import __version__


def _sdk(raw: dict[str, Any] | None = None) -> LabSDK:
    """Build an SDK with an in-memory config (no disk access)."""
    raw = raw or {"version": __version__}
    return LabSDK(config=Config(raw=raw, config_dir=Path(".")))


def _fake_result() -> RunResult:
    """A minimal RunResult stand-in for mocked runner calls."""
    return RunResult(
        model_id="m",
        mode="baseline",
        quant="fp16",
        device="cpu",
        n_input_tokens=1,
        n_output_tokens=1,
        ttft_s=0.0,
        tpot_s=0.0,
        throughput_tok_s=0.0,
        runtime_s=0.0,
        output_text="ok",
    )


def test_sdk_version() -> None:
    """SDK exposes the code version."""
    assert _sdk().version == __version__


def test_sdk_config_passthrough() -> None:
    """SDK returns the config it was constructed with."""
    assert _sdk().config.version == __version__


def test_sdk_probe_hardware() -> None:
    """SDK delegates to the hardware probe and returns a spec."""
    spec = _sdk().probe_hardware()
    assert isinstance(spec, HardwareSpec)
    assert spec.ram_gb > 0


def test_run_baseline_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_baseline configures the cache + token and calls the runner."""
    import airllm_lab.services.baseline_runner as br
    import airllm_lab.shared.secrets as secrets
    import airllm_lab.shared.storage as storage

    seen: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, hf_token: str | None = None) -> None:
            seen["token"] = hf_token

        def run(self, cfg: RunConfig) -> RunResult:
            seen["cfg"] = cfg
            return _fake_result()

    monkeypatch.setattr(storage, "configure_hf_cache", lambda p: Path(p))
    monkeypatch.setattr(secrets, "get_hf_token", lambda: "tok")
    monkeypatch.setattr(br, "BaselineRunner", _FakeRunner)

    cfg = RunConfig(model_id="m", prompt="hi")
    result = _sdk({"version": __version__, "storage": {"hf_cache": "x"}}).run_baseline(cfg)
    assert result.output_text == "ok"
    assert seen == {"token": "tok", "cfg": cfg}


def test_run_smoke_builds_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_smoke builds a capped RunConfig from config and delegates."""
    captured: dict[str, Any] = {}

    def _fake_run_baseline(self: LabSDK, cfg: RunConfig) -> RunResult:
        captured["cfg"] = cfg
        return _fake_result()

    monkeypatch.setattr(LabSDK, "run_baseline", _fake_run_baseline)
    raw = {
        "version": __version__,
        "models": {"smoke": "tiny/model"},
        "experiment": {"prompt": "P", "max_new_tokens": 999},
    }
    _sdk(raw).run_smoke()
    assert captured["cfg"].model_id == "tiny/model"
    assert captured["cfg"].prompt == "P"
    assert captured["cfg"].max_new_tokens == 64


def test_run_baseline_model_uses_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_baseline_model builds a config (cap 128) and persists the result."""
    import json

    captured: dict[str, Any] = {}

    def _fake_run_baseline(self: LabSDK, cfg: RunConfig) -> RunResult:
        captured["cfg"] = cfg
        return _fake_result()

    monkeypatch.setattr(LabSDK, "run_baseline", _fake_run_baseline)
    raw = {
        "version": __version__,
        "experiment": {"prompt": "Q", "max_new_tokens": 200},
        "storage": {"results_dir": str(tmp_path)},
    }
    _sdk(raw).run_baseline_model("org/7b")
    assert captured["cfg"].model_id == "org/7b"
    assert captured["cfg"].max_new_tokens == 128
    saved = json.loads((tmp_path / "baseline_7b.json").read_text(encoding="utf-8"))
    assert saved["output_text"] == "ok"


def test_run_airllm_delegates_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_airllm configures cache, runs the AirLLM runner, and persists JSON."""
    import json

    import airllm_lab.services.airllm_runner as ar
    import airllm_lab.shared.secrets as secrets
    import airllm_lab.shared.storage as storage

    seen: dict[str, Any] = {}

    class _FakeRunner:
        def __init__(self, shards_dir: str, hf_token: str | None = None) -> None:
            seen["shards_dir"] = shards_dir
            seen["token"] = hf_token

        def run(self, cfg: RunConfig) -> RunResult:
            seen["cfg"] = cfg
            return _fake_result()

    monkeypatch.setattr(storage, "configure_hf_cache", lambda p: Path(p))
    monkeypatch.setattr(secrets, "get_hf_token", lambda: "tok")
    monkeypatch.setattr(ar, "AirLLMRunner", _FakeRunner)

    raw = {
        "version": __version__,
        "experiment": {"prompt": "P", "max_new_tokens": 200},
        "storage": {"layer_shards_saving_path": "D:/shards", "results_dir": str(tmp_path)},
    }
    result = _sdk(raw).run_airllm("D:/models/Qwen2.5-7B-Instruct", quant="4bit")
    assert result.output_text == "ok"
    assert seen["shards_dir"] == "D:/shards"
    assert seen["cfg"].mode == "airllm"
    assert seen["cfg"].quant == "4bit"
    assert seen["cfg"].max_new_tokens == 20
    saved = json.loads((tmp_path / "airllm_Qwen2.5-7B-Instruct_4bit.json").read_text("utf-8"))
    assert saved["output_text"] == "ok"


def test_run_airllm_captures_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An AirLLM runner exception becomes a failed RunResult, not a raise."""
    import airllm_lab.services.airllm_runner as ar
    import airllm_lab.shared.secrets as secrets
    import airllm_lab.shared.storage as storage

    class _BoomRunner:
        def __init__(self, shards_dir: str, hf_token: str | None = None) -> None:
            pass

        def run(self, cfg: RunConfig) -> RunResult:
            raise RuntimeError("airllm boom")

    monkeypatch.setattr(storage, "configure_hf_cache", lambda p: Path(p))
    monkeypatch.setattr(secrets, "get_hf_token", lambda: None)
    monkeypatch.setattr(ar, "AirLLMRunner", _BoomRunner)

    raw = {"version": __version__, "storage": {"results_dir": str(tmp_path)}}
    result = _sdk(raw).run_airllm("org/7b")
    assert result.ok is False
    assert "airllm boom" in result.error


def test_run_baseline_captures_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A runner exception becomes a failed RunResult, not a raise."""
    import airllm_lab.services.baseline_runner as br
    import airllm_lab.shared.secrets as secrets
    import airllm_lab.shared.storage as storage

    class _BoomRunner:
        def __init__(self, hf_token: str | None = None) -> None:
            pass

        def run(self, cfg: RunConfig) -> RunResult:
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(storage, "configure_hf_cache", lambda p: Path(p))
    monkeypatch.setattr(secrets, "get_hf_token", lambda: None)
    monkeypatch.setattr(br, "BaselineRunner", _BoomRunner)

    result = _sdk().run_baseline(RunConfig(model_id="big", prompt="hi"))
    assert result.ok is False
    assert "CUDA out of memory" in result.error


def test_hardware_report_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """hardware_report writes results/hardware.json and returns the spec."""
    import json

    raw = {"version": __version__, "storage": {"results_dir": str(tmp_path)}}
    spec = _sdk(raw).hardware_report()
    saved = json.loads((tmp_path / "hardware.json").read_text(encoding="utf-8"))
    assert isinstance(spec, HardwareSpec)
    assert saved["ram_gb"] == spec.ram_gb


def test_download_model_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """download_model configures the cache and calls the downloader."""
    import airllm_lab.services.model_download as md
    import airllm_lab.shared.secrets as secrets

    seen: dict[str, Any] = {}

    class _FakeDownloader:
        def __init__(self, hf_token: str | None = None) -> None:
            seen["token"] = hf_token

        def download(self, model_id: str, dest_root: str) -> Path:
            seen["model_id"] = model_id
            seen["dest_root"] = dest_root
            return Path(dest_root) / "model"

    monkeypatch.setattr(secrets, "get_hf_token", lambda: "tok")
    monkeypatch.setattr(md, "ModelDownloader", _FakeDownloader)

    raw = {"version": __version__, "storage": {"models_dir": "D:/models"}}
    path = _sdk(raw).download_model("org/7b")
    assert path == Path("D:/models") / "model"
    assert seen == {"token": "tok", "model_id": "org/7b", "dest_root": "D:/models"}
