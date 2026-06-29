"""Tests for the pure cost model (hand-computed expected values)."""

import pytest

from airllm_lab.services import cost_model as cm


def _profile(**kw: object) -> cm.UsageProfile:
    base = {
        "requests_per_month": 100,
        "avg_input_tokens": 1000,
        "avg_output_tokens": 1000,
        "cached_fraction": 0.0,
    }
    base.update(kw)
    return cm.UsageProfile(**base)  # type: ignore[arg-type]


def test_api_cost_no_caching() -> None:
    """1000 in @ $1/Mtok + 1000 out @ $2/Mtok = $0.003/request."""
    pricing = cm.ApiPricing(input_per_mtok=1.0, output_per_mtok=2.0)
    assert cm.api_cost_per_request(_profile(), pricing) == pytest.approx(0.003)


def test_api_cost_with_caching_discount() -> None:
    """Half the input billed at the cheaper cached rate lowers the cost."""
    pricing = cm.ApiPricing(input_per_mtok=1.0, output_per_mtok=2.0, cached_input_per_mtok=0.5)
    cost = cm.api_cost_per_request(_profile(cached_fraction=0.5), pricing)
    assert cost == pytest.approx(0.00275)


def test_energy_cost_per_request() -> None:
    """3600 s at 1000 W = 1 kWh; at $0.15/kWh = $0.15."""
    onprem = cm.OnPremSpec(
        capex_usd=0, lifetime_months=36, electricity_per_kwh=0.15, avg_power_watts=1000
    )
    assert cm.energy_cost_per_request(3600, onprem) == pytest.approx(0.15)


def test_onprem_fixed_monthly_amortizes_capex() -> None:
    """$3600 over 36 months + $0 maintenance = $100/month."""
    onprem = cm.OnPremSpec(
        capex_usd=3600, lifetime_months=36, electricity_per_kwh=0.0, avg_power_watts=0
    )
    assert cm.onprem_fixed_monthly(onprem) == pytest.approx(100.0)


def test_cloud_cost_per_request() -> None:
    """1 hour of runtime at $0.50/hour = $0.50."""
    assert cm.cloud_cost_per_request(3600, cm.CloudGpuSpec(usd_per_hour=0.5)) == pytest.approx(0.5)


def test_breakeven_positive_when_api_dearer() -> None:
    """Break-even = fixed_monthly / (api_per_req - energy_per_req)."""
    pricing = cm.ApiPricing(input_per_mtok=1000.0, output_per_mtok=1000.0)  # $2/request
    onprem = cm.OnPremSpec(
        capex_usd=3600, lifetime_months=36, electricity_per_kwh=0.0, avg_power_watts=0
    )  # fixed $100/mo, $0 energy
    assert cm.breakeven_requests(pricing, onprem, _profile(), runtime_s=10) == pytest.approx(50.0)


def test_breakeven_none_when_api_always_cheaper() -> None:
    """If local energy already beats API price, the lines never cross."""
    pricing = cm.ApiPricing(input_per_mtok=0.0, output_per_mtok=0.0)
    onprem = cm.OnPremSpec(
        capex_usd=3600, lifetime_months=36, electricity_per_kwh=0.15, avg_power_watts=1000
    )
    assert cm.breakeven_requests(pricing, onprem, _profile(), runtime_s=3600) is None


def test_build_report_serializes_assumptions() -> None:
    """build_report returns a serializable report carrying its assumptions."""
    report = cm.build_report(
        _profile(),
        cm.ApiPricing(input_per_mtok=1.0, output_per_mtok=2.0),
        cm.OnPremSpec(
            capex_usd=3600, lifetime_months=36, electricity_per_kwh=0.15, avg_power_watts=65
        ),
        cm.CloudGpuSpec(usd_per_hour=0.5),
        runtime_s=30,
    )
    data = report.to_dict()
    assert data["api_per_request"] == pytest.approx(0.003)
    assert "usage" in data["assumptions"]
    assert data["assumptions"]["runtime_s_per_request"] == 30
