"""LabSDK: the single entry point for all lab capabilities.

CLI, notebooks, and any other consumer must go through this facade rather
than importing services directly (per the SDK-architecture rule). The
capabilities are split across mixins to keep each file within the line limit:
:class:`RunsMixin` (model execution) and :class:`AnalysisMixin` (cost + charts).
"""

from __future__ import annotations

from pathlib import Path

from airllm_lab.sdk.analysis_mixin import AnalysisMixin
from airllm_lab.sdk.runs_mixin import RunsMixin
from airllm_lab.services.hardware import HardwareProbe, HardwareSpec
from airllm_lab.shared.config import Config, load_config
from airllm_lab.shared.storage import save_json
from airllm_lab.shared.version import __version__


class LabSDK(RunsMixin, AnalysisMixin):
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
