"""LabSDK: the single entry point for all lab capabilities.

CLI, notebooks, and any other consumer must go through this facade rather
than importing services directly (per the SDK-architecture rule).
"""

from __future__ import annotations

from pathlib import Path

from airllm_lab.services.hardware import HardwareProbe, HardwareSpec
from airllm_lab.services.models import RunConfig, RunResult
from airllm_lab.shared.config import Config, load_config
from airllm_lab.shared.storage import save_json
from airllm_lab.shared.version import __version__


class LabSDK:
    """Facade exposing every lab capability to external consumers."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize with a Config (loaded from disk when not provided)."""
        self._config = config or load_config()
        self._probe = HardwareProbe()

    @property
    def version(self) -> str:
        """Return the code version."""
        return __version__

    @property
    def config(self) -> Config:
        """Return the active configuration."""
        return self._config

    def probe_hardware(self) -> HardwareSpec:
        """Collect and return the machine's hardware spec."""
        return self._probe.probe()

    def hardware_report(self) -> HardwareSpec:
        """Probe hardware and persist it to ``<results_dir>/hardware.json``."""
        spec = self._probe.probe()
        results_dir = self._config.get("storage", {}).get("results_dir", "results")
        save_json(Path(results_dir) / "hardware.json", spec.to_dict())
        return spec

    def download_model(self, model_id: str) -> Path:
        """Download a model's files to the configured ``D:`` models dir.

        Returns the local directory; pass it to ``run_baseline_model`` so the
        model loads from disk (avoiding the hub downloader hang on large files).
        """
        from airllm_lab.services.model_download import ModelDownloader
        from airllm_lab.shared.secrets import get_hf_token

        models_dir = self._config.get("storage", {}).get("models_dir", "models")
        return ModelDownloader(hf_token=get_hf_token()).download(model_id, models_dir)

    def run_baseline(self, cfg: RunConfig) -> RunResult:
        """Run an in-memory Transformers baseline generation for ``cfg``.

        Heavy torch imports are deferred to call time. Any failure (e.g. CUDA
        OOM on a model too big for the GPU) is captured as a failed RunResult
        rather than raised, since that failure is itself a finding.
        """
        from airllm_lab.shared.secrets import get_hf_token
        from airllm_lab.shared.storage import configure_hf_cache

        # Configure the HF cache BEFORE importing transformers, which reads the
        # cache location at import time (otherwise weights land on C:).
        configure_hf_cache(self._config.get("storage", {}).get("hf_cache", "hf_cache"))
        from airllm_lab.services.baseline_runner import BaselineRunner

        try:
            return BaselineRunner(hf_token=get_hf_token()).run(cfg)
        except Exception as exc:
            return RunResult.failed(cfg, error=f"{type(exc).__name__}: {exc}")

    def _run_config_for(
        self, model_id: str, max_cap: int, mode: str = "baseline", quant: str = "fp16"
    ) -> RunConfig:
        """Build a RunConfig from config for ``model_id`` (tokens capped)."""
        experiment = self._config.get("experiment", {})
        return RunConfig(
            model_id=model_id,
            prompt=experiment.get("prompt", "Hello, who are you?"),
            max_new_tokens=min(int(experiment.get("max_new_tokens", 64)), max_cap),
            mode=mode,
            quant=quant,
        )

    def run_smoke(self) -> RunResult:
        """Run the tiny smoke-test model end-to-end to prove the pipeline."""
        model_id = self._config.get("models", {}).get("smoke", "Qwen/Qwen2.5-0.5B-Instruct")
        return self.run_baseline(self._run_config_for(model_id, max_cap=64))

    def run_baseline_model(self, model_id: str) -> RunResult:
        """Run the baseline on an explicit ``model_id`` and persist the result.

        The result (success or a captured failure) is written to
        ``<results_dir>/baseline_<name>.json`` so it can be cited in the report.
        """
        result = self.run_baseline(self._run_config_for(model_id, max_cap=128))
        results_dir = self._config.get("storage", {}).get("results_dir", "results")
        name = Path(model_id).name or "model"
        save_json(Path(results_dir) / f"baseline_{name}.json", result.to_dict())
        return result

    def run_airllm(self, model_id: str, quant: str = "fp16", max_cap: int = 20) -> RunResult:
        """Run layered AirLLM inference on ``model_id`` and persist the result.

        Tokens are capped low (AirLLM streams every layer from disk per token, so
        it is slow). Any failure is captured as a failed RunResult, and the result
        is written to ``<results_dir>/airllm_<name>_<quant>.json`` for the report.
        """
        from airllm_lab.shared.secrets import get_hf_token
        from airllm_lab.shared.storage import configure_hf_cache

        storage = self._config.get("storage", {})
        configure_hf_cache(storage.get("hf_cache", "hf_cache"))
        from airllm_lab.services.airllm_runner import AirLLMRunner

        cfg = self._run_config_for(model_id, max_cap=max_cap, mode="airllm", quant=quant)
        shards_dir = storage.get("layer_shards_saving_path", "airllm_shards")
        try:
            result = AirLLMRunner(shards_dir=shards_dir, hf_token=get_hf_token()).run(cfg)
        except Exception as exc:
            result = RunResult.failed(cfg, error=f"{type(exc).__name__}: {exc}")
        name = Path(model_id).name or "model"
        save_json(Path(storage.get("results_dir", "results")) / f"airllm_{name}_{quant}.json",
                  result.to_dict())
        return result
