# PRD — Cost Model (OnPrem vs API vs Cloud-GPU)

**Version:** 1.00 · **Mechanism owner module:** `services/cost_model.py`

## 1. Background (theory)
Local deployment trades a large up-front hardware cost (**CAPEX**) plus electricity and
maintenance (**OPEX**) against the per-token pricing of a hosted **API**. As usage volume
grows, amortized local cost per request falls, so there is a **break-even** volume beyond
which OnPrem is cheaper. **Prompt/Context caching** (providers built on PagedAttention)
discounts repeated fixed-prefix tokens, shifting the break-even for repetitive workloads.

## 2. Requirements
Compute and compare three cost lines vs monthly usage volume:
1. **API:** `(input_tokens + output_tokens) × price_per_token`, with an optional
   **cached-prefix discount** on a configurable fraction of input tokens.
2. **OnPrem:** `CAPEX amortized over lifetime + electricity (kWh × tariff) + maintenance`,
   divided by request volume → effective cost/request.
3. **Cloud-GPU (optional):** `hourly_rate × runtime_hours_per_request`.
- Find the **break-even** request volume(s); produce a cumulative-cost-vs-volume curve.
- **All assumptions** (prices, tokens/request, hardware price + lifetime, tariff, power)
  come from `config/setup.json` so the analysis is transparent and reproducible.

## 3. Inputs / Outputs / Setup
- **Input:** `UsageProfile(requests_per_month, avg_input_tokens, avg_output_tokens,
  cached_fraction)` + measured `runtime`/`energy` from the benchmark.
- **Output:** `CostReport(api_cost, onprem_cost, cloudgpu_cost, breakeven_requests,
  assumptions)` + data for the break-even chart.
- **Setup:** price tables, hardware price, lifetime months, electricity tariff, avg power.

## 4. Performance expectations
- A clear crossover point on the curve; recommendation text per scenario (incl. privacy
  / data-security considerations, not only cost).

## 5. Constraints / limitations
- Prices change over time → values are dated assumptions in config.
- OnPrem ignores amortized human time; state this explicitly.

## 6. Alternatives considered
- Pure per-request comparison (no amortization) — rejected as misleading for CAPEX.

## 7. Success criteria & tests
- [ ] API/OnPrem/Cloud cost functions unit-tested against hand-computed values.
- [ ] Break-even solver returns the correct crossover on known inputs.
- [ ] Caching discount correctly reduces API cost.
- [ ] `CostReport` serializes with all assumptions recorded.
