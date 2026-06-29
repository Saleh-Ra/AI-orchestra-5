"""AnalysisMixin: the cost + visualization capabilities of :class:`LabSDK`.

Split out of ``sdk.py`` to respect the 150-line file limit. Assumes the
composing class provides ``self._config`` and ``self._power_watts()``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from airllm_lab.services import cost_model
from airllm_lab.shared.config import Config
from airllm_lab.shared.storage import save_json


class AnalysisMixin:
    """Cost-model and figure-rendering methods for the SDK facade.

    ``_config`` and ``_power_watts`` are supplied by the composing ``LabSDK``
    (the latter via :class:`RunsMixin`); declared here only for type clarity.
    """

    _config: Config
    _power_watts: Callable[[], float]

    def run_cost_analysis(self, runtime_s: float | None = None) -> cost_model.CostReport:
        """Compute the OnPrem-vs-API-vs-Cloud cost report from config + a runtime.

        ``runtime_s`` is the measured local seconds per request (defaults to the
        config value). The report and all assumptions are saved to
        ``<results_dir>/cost_analysis.json``.
        """
        cost = self._config.get("cost", {})
        usage, api = cost.get("usage", {}), cost.get("api_pricing_per_mtok", {})
        op, cloud = cost.get("onprem", {}), cost.get("cloud_gpu", {})
        profile = cost_model.UsageProfile(
            requests_per_month=int(usage.get("requests_per_month", 1000)),
            avg_input_tokens=int(usage.get("avg_input_tokens", 500)),
            avg_output_tokens=int(usage.get("avg_output_tokens", 300)),
            cached_fraction=float(usage.get("cached_fraction", 0.0)),
        )
        pricing = cost_model.ApiPricing(
            input_per_mtok=float(api.get("input", 0.0)),
            output_per_mtok=float(api.get("output", 0.0)),
            cached_input_per_mtok=float(api.get("cached_input", 0.0)),
        )
        onprem = cost_model.OnPremSpec(
            capex_usd=float(op.get("capex_usd", 0.0)),
            lifetime_months=int(op.get("lifetime_months", 36)),
            electricity_per_kwh=float(op.get("electricity_per_kwh", 0.0)),
            avg_power_watts=float(op.get("avg_power_watts", self._power_watts())),
            maintenance_per_month=float(op.get("maintenance_per_month", 0.0)),
        )
        cloud_spec = cost_model.CloudGpuSpec(usd_per_hour=float(cloud.get("usd_per_hour", 0.0)))
        runtime = runtime_s if runtime_s is not None else float(usage.get("runtime_s_per_request", 30))
        report = cost_model.build_report(profile, pricing, onprem, cloud_spec, runtime)
        results_dir = self._config.get("storage", {}).get("results_dir", "results")
        save_json(Path(results_dir) / "cost_analysis.json", report.to_dict())
        return report

    def generate_charts(self) -> list[str]:
        """Render benchmark + cost figures from stored results to the assets dir."""
        from airllm_lab.services import charts

        storage = self._config.get("storage", {})
        results_dir = Path(storage.get("results_dir", "results"))
        out_dir = Path(storage.get("assets_dir", "assets"))
        return [str(p) for p in charts.render_all(results_dir, out_dir)]

    def generate_roofline(self) -> str | None:
        """Render the roofline figure (the original extension) from results."""
        from airllm_lab.services import roofline

        storage = self._config.get("storage", {})
        results_dir = Path(storage.get("results_dir", "results"))
        out_dir = Path(storage.get("assets_dir", "assets"))
        peaks = self._config.get("roofline", {})
        path = roofline.render_roofline(results_dir, out_dir, peaks)
        return str(path) if path is not None else None
