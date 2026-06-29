# AirLLM Lab — Running a Massive LLM on Modest Hardware

**Can you run a 7-billion-parameter language model on a laptop with only 4 GB of
GPU memory?** Normally, no. This project shows *why* it fails, then uses
**AirLLM (layered inference)** and **quantization** to make it work anyway — and
measures the cost in speed, memory, and money.

This README is the project's **technical report**: it grows as the work
progresses, linking every claim back to a measurement saved under
[`results/`](results/).

---

## 1. The problem in one picture

A model's weights must fit in memory to run it the normal way. Our target model
does not:

| | Size |
|---|---|
| `Qwen2.5-7B-Instruct` weights (FP16) | **~15 GB** |
| This machine's RAM | 15.8 GB (≈ 8 GB free in practice) |
| This machine's **GPU memory (VRAM)** | **4 GB** |

The weights are bigger than the GPU by almost **4×**, and bigger than the *free*
RAM too. So the naive "load it and run it" approach is doomed — and we prove it
below before fixing it.

## 2. The hardware (the constraint we design around)

Measured automatically and saved to [`results/hardware.json`](results/hardware.json):

| Component | Spec |
|-----------|------|
| CPU | Intel i7-9750H — 6 cores / 12 threads |
| RAM | 15.8 GB |
| GPU | NVIDIA GTX 1650 — **4 GB VRAM** |
| Model storage | `D:` HDD (spinning disk, ~753 GB free) |

Two facts shape everything: **tiny VRAM** (can't hold the model) and a **slow
HDD** (AirLLM streams layers from here, so disk speed becomes the bottleneck).

## 3. The experiment, in stages

```
  Stage 1  Baseline    →  load 7B the normal way        →  FAILS (proof of the wall)
  Stage 2  AirLLM      →  stream the model layer-by-layer →  RUNS on 4 GB VRAM
  Stage 3  Quantization→  shrink 16-bit → 8/4-bit         →  smaller + faster
  Stage 4  Benchmark   →  measure speed, memory, energy   →  compare them all
  Stage 5  Economics   →  local cost vs. paying an API    →  when is each worth it?
```

### Stage 1 — Baseline: watch it fail ✅ done

We deliberately tried to run the 7B model the normal way (load all weights into
RAM, then move to the GPU). The result, captured in
[`results/baseline_Qwen2.5-7B-Instruct.json`](results/baseline_Qwen2.5-7B-Instruct.json):

> **It never finished loading.** The weights overflowed RAM, Windows began
> paging to the HDD, and after **~28 minutes** of swap-thrashing the process
> **crashed** (`0xC0000005`, access violation) at only **61 %** of the
> weight-load — it never even reached the GPU.

This is the finding that justifies the whole project: **direct execution is
infeasible here.** To avoid freezing the machine on every run, the code now does
a **pre-flight memory check** that reports this cleanly in seconds instead of
crashing:

```json
{
  "mode": "baseline",
  "ok": false,
  "error": "baseline infeasible: ~15.2 GB FP16 weights exceed ~8.3 GB available RAM (and 4 GB VRAM); the naive load thrashes swap and crashes the process"
}
```

### Stage 2 — AirLLM: the rescue ✅ done

**AirLLM** loads the model **one layer at a time**: load layer → compute → free
it → load the next. Peak memory becomes the size of a *single layer*, not the
whole model — so a 7B model fits in 4 GB of VRAM. The price is speed: every token
requires reading all 31 layers from the (slow) HDD. This is the classic
**virtual-memory / paging** trade-off applied to neural-network weights.

**Result — the same 7B that crashed the baseline now generates**, on the 4 GB
GPU ([`results/airllm_Qwen2.5-7B-Instruct_fp16.json`](results/airllm_Qwen2.5-7B-Instruct_fp16.json)):

| | Baseline (naive load) | **AirLLM (layered)** |
|---|---|---|
| Outcome | crashed at 61 % of load | **generated `"Virtual memory is a"`** |
| Reached the GPU? | no | yes (`cuda:0`) |
| Time to first token | — | **~930 s** (~15.5 min) |
| Throughput | — | **~0.0011 tok/s** (~15 min / token) |

So AirLLM turns "impossible" into "possible but slow": feasibility is bought with
heavy per-token disk I/O. The next stages attack that slowness with quantization.

> **Engineering note.** AirLLM 2.11 is unmaintained against current `transformers`,
> so the stack is pinned to the compatible `transformers 4.40` era (see
> `pyproject.toml`). Generation is driven by a small greedy-decode loop over the
> model's `forward` (in `services/airllm_runner.py`) rather than `transformers`'
> `generate`, whose newer KV-cache API AirLLM does not support.

### Stage 3 — Quantization 🔜

Store weights in fewer bits (FP16 → INT8 → INT4). Less data means less memory and
less disk traffic, so it runs smaller and faster — until precision drops too far
and answer quality degrades (the "accuracy red line" we'll measure).

### Stage 4 — Benchmarking 🔜

For each configuration we record:
- **TTFT** (time to first token) — the compute-bound *prefill* phase.
- **TPOT** (time per output token) — the memory-bandwidth-bound *decode* phase.
- **Throughput** (tokens/sec), **peak RAM/VRAM**, total runtime, energy estimate.

### Stage 5 — Economic analysis 🔜

Compare the real cost of running locally (hardware + electricity + time) against
calling a hosted API per token, and find the break-even point.

## 4. How the code is organized

A single **SDK facade** (`LabSDK`) is the one entry point; the CLI and notebooks
call it, never the internals (keeps business logic in one place).

```
src/airllm_lab/
  sdk/            LabSDK facade — the only public entry point
  services/       hardware probe, model download, baseline runner,
                  metrics, feasibility check  (AirLLM + benchmark land here)
  shared/         config, secrets (.env), storage/paths, version
  main.py         thin CLI
config/           versioned JSON config (no hardcoded values)
results/          saved measurements (hardware, baseline, benchmarks)
docs/             PRD, PLAN, per-mechanism PRDs, working map
tests/            unit + integration tests (100 % coverage today)
```

## 5. Try it yourself

```bash
uv sync                                     # set up env + install deps
uv run airllm-lab hardware                  # probe + save this machine's spec
uv run airllm-lab smoke                     # run a tiny 0.5B model end-to-end
uv run airllm-lab download Qwen/Qwen2.5-7B-Instruct   # fetch the big model to D:
uv run airllm-lab baseline <path-to-model>  # the (failing) naive baseline
```

Secrets (e.g. a Hugging Face token) go in a local, git-ignored `.env`
(see `.env.example`). Large model files are kept off the small `C:` drive.

## 6. Planning & design documents

| Document | Purpose |
|----------|---------|
| [`docs/PRD.md`](docs/PRD.md) | Product requirements, goals, KPIs, acceptance criteria |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture (C4), interfaces, ADRs |
| [`docs/TODO.md`](docs/TODO.md) | Standards & rules + the ordered working map |
| [`docs/PRD_airllm.md`](docs/PRD_airllm.md) | AirLLM layered-inference mechanism |
| [`docs/PRD_quantization.md`](docs/PRD_quantization.md) | Quantization mechanism |
| [`docs/PRD_benchmark.md`](docs/PRD_benchmark.md) | Benchmarking & metrics |
| [`docs/PRD_cost_model.md`](docs/PRD_cost_model.md) | OnPrem vs API cost model |

## 7. Progress

- [x] **Phase 0–1** — planning docs + project scaffold (SDK, CLI, config, tests)
- [x] **Phase 2** — pipeline smoke test (0.5B model runs end-to-end on GPU)
- [x] **Phase 3** — hardware report, 7B download, **baseline failure documented**
- [~] **Phase 4** — **AirLLM runner done (7B runs!)**; quantization + benchmark harness next
- [ ] **Phase 5** — results notebook, economic analysis, final report

**Quality gates (current):** ruff clean · 45 tests passing · 100 % coverage.

## License

MIT (see `LICENSE`, to be added).
