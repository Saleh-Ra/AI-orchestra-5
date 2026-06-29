"""Cost model: OnPrem vs hosted API vs Cloud-GPU, with a break-even solver.

Pure, config-driven economics (no I/O): prices/assumptions are passed in so the
analysis is transparent and unit-testable. All monetary values are USD; token
prices are per one million tokens (``*_per_mtok``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_MILLION = 1_000_000
_SECONDS_PER_HOUR = 3600
_WATTS_PER_KW = 1000


@dataclass(frozen=True)
class UsageProfile:
    """Workload shape: monthly volume and average tokens per request."""

    requests_per_month: int
    avg_input_tokens: int
    avg_output_tokens: int
    cached_fraction: float = 0.0


@dataclass(frozen=True)
class ApiPricing:
    """Hosted-API token prices (USD per 1M tokens)."""

    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0


@dataclass(frozen=True)
class OnPremSpec:
    """Local hardware economics: up-front cost amortized + electricity."""

    capex_usd: float
    lifetime_months: int
    electricity_per_kwh: float
    avg_power_watts: float
    maintenance_per_month: float = 0.0


@dataclass(frozen=True)
class CloudGpuSpec:
    """Rented-GPU economics: pay-as-you-go hourly rate."""

    usd_per_hour: float


@dataclass
class CostReport:
    """Per-request and monthly costs for each option, plus the break-even."""

    assumptions: dict[str, Any]
    api_per_request: float
    onprem_per_request: float
    cloud_per_request: float
    api_monthly: float
    onprem_monthly: float
    cloud_monthly: float
    breakeven_requests_per_month: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


def api_cost_per_request(profile: UsageProfile, pricing: ApiPricing) -> float:
    """API cost for one request, applying the cached-prefix discount."""
    cached = profile.avg_input_tokens * profile.cached_fraction
    full = profile.avg_input_tokens - cached
    cost = (
        full * pricing.input_per_mtok
        + cached * pricing.cached_input_per_mtok
        + profile.avg_output_tokens * pricing.output_per_mtok
    ) / _MILLION
    return round(cost, 6)


def energy_cost_per_request(runtime_s: float, onprem: OnPremSpec) -> float:
    """Electricity cost (USD) of one local request from its runtime."""
    kwh = runtime_s * onprem.avg_power_watts / _SECONDS_PER_HOUR / _WATTS_PER_KW
    return round(kwh * onprem.electricity_per_kwh, 6)


def onprem_fixed_monthly(onprem: OnPremSpec) -> float:
    """Fixed monthly OnPrem cost: amortized CAPEX + maintenance."""
    return round(onprem.capex_usd / onprem.lifetime_months + onprem.maintenance_per_month, 4)


def onprem_cost_per_request(runtime_s: float, onprem: OnPremSpec, requests_per_month: int) -> float:
    """Effective OnPrem cost/request: amortized fixed share + energy."""
    fixed_share = onprem_fixed_monthly(onprem) / max(1, requests_per_month)
    return round(fixed_share + energy_cost_per_request(runtime_s, onprem), 6)


def cloud_cost_per_request(runtime_s: float, cloud: CloudGpuSpec) -> float:
    """Rented-GPU cost for one request (runtime × hourly rate)."""
    return round(runtime_s / _SECONDS_PER_HOUR * cloud.usd_per_hour, 6)


def breakeven_requests(
    pricing: ApiPricing, onprem: OnPremSpec, profile: UsageProfile, runtime_s: float
) -> float | None:
    """Monthly request volume where OnPrem total equals API total.

    Returns ``None`` when local *energy alone* already costs more per request
    than the API (the lines never cross — API always wins).
    """
    margin = api_cost_per_request(profile, pricing) - energy_cost_per_request(runtime_s, onprem)
    if margin <= 0:
        return None
    return round(onprem_fixed_monthly(onprem) / margin, 1)


def build_report(
    profile: UsageProfile,
    pricing: ApiPricing,
    onprem: OnPremSpec,
    cloud: CloudGpuSpec,
    runtime_s: float,
) -> CostReport:
    """Assemble a full :class:`CostReport` from the inputs."""
    requests = profile.requests_per_month
    api_req = api_cost_per_request(profile, pricing)
    cloud_req = cloud_cost_per_request(runtime_s, cloud)
    energy_req = energy_cost_per_request(runtime_s, onprem)
    return CostReport(
        assumptions={
            "usage": asdict(profile),
            "api_pricing": asdict(pricing),
            "onprem": asdict(onprem),
            "cloud_gpu": asdict(cloud),
            "runtime_s_per_request": runtime_s,
        },
        api_per_request=api_req,
        onprem_per_request=onprem_cost_per_request(runtime_s, onprem, requests),
        cloud_per_request=cloud_req,
        api_monthly=round(api_req * requests, 4),
        onprem_monthly=round(onprem_fixed_monthly(onprem) + energy_req * requests, 4),
        cloud_monthly=round(cloud_req * requests, 4),
        breakeven_requests_per_month=breakeven_requests(pricing, onprem, profile, runtime_s),
    )
