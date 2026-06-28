# AirLLM Lab — Running a Massive LLM Locally

Quantization & performance benchmarking of large-language-model inference on
modest, consumer-grade hardware, using **AirLLM** (layered inference) and
**quantization** to run a model that does not fit in VRAM.

> **Status:** scaffolding (Phase 1). This README will grow into the full deep-dive
> technical report: hardware documentation, baseline failure analysis, AirLLM +
> quantization benchmarks, an OnPrem-vs-API economic analysis, and the link from
> measurements back to execution concepts (Prefill/Decode, compute- vs
> memory-bound, virtual memory / paging).

## Planning documents

| Document | Purpose |
|----------|---------|
| [`docs/PRD.md`](docs/PRD.md) | Product requirements, goals, KPIs, acceptance criteria |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture (C4), interfaces, ADRs |
| [`docs/TODO.md`](docs/TODO.md) | Standards & rules + the ordered working map |
| [`docs/PRD_airllm.md`](docs/PRD_airllm.md) | AirLLM layered-inference mechanism |
| [`docs/PRD_quantization.md`](docs/PRD_quantization.md) | Quantization mechanism |
| [`docs/PRD_benchmark.md`](docs/PRD_benchmark.md) | Benchmarking & metrics |
| [`docs/PRD_cost_model.md`](docs/PRD_cost_model.md) | OnPrem vs API cost model |

## Quick start (dev)

```bash
uv sync                  # create the env and install dependencies
uv run ruff check        # lint
uv run pytest            # tests + coverage (must stay >= 85%)
uv run airllm-lab        # version banner (CLI expands in later phases)
```

## License

MIT (see `LICENSE`, to be added).
