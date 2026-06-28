# Product Requirements Document (PRD)

**Project:** Running a Massive LLM Locally — AirLLM, Quantization & Performance Benchmarking
**Course / Assignment:** EX05 (L08) — Dr. Yoram Segal
**Version:** 1.00
**Status:** DRAFT (awaiting approval)

---

## 1. Overview & Context

This project is an **open, research-style engineering experiment**. The goal is to
take a Large Language Model that is **deliberately too big** for our hardware, prove
that it **fails or runs unbearably slowly** when executed directly (the *baseline*),
then use **AirLLM layered inference + quantization** to make it run anyway — and to
**measure, compare, and deeply analyze** the cost/benefit, both **technically** and
**economically**.

The deliverable is a GitHub repository whose `README.md` doubles as a **deep-dive
technical report** (graphs, tables, and screenshots embedded inline). The emphasis is
**not** on the model's answer quality, but on understanding the *execution mechanics*
(Prefill/Decode, compute- vs memory-bound, VRAM, virtual memory / paging), drawing
data-backed conclusions, and producing a rigorous engineering + economic analysis.

> A well-analyzed **negative result** (e.g. "AirLLM did not improve wall-clock time on
> an HDD, and here is exactly why") is explicitly acceptable and valued.

### 1.1 The user / problem
- **Primary user:** an engineer/researcher with modest, consumer-grade hardware who
  wants to run an LLM that does not fit in their GPU VRAM (or comfortably in RAM).
- **Problem:** large models exceed VRAM/RAM; naive local execution either OOMs,
  thrashes swap, or is far too slow. Is local deployment ever worth it vs. a paid API?

### 1.2 Target audience for the report
External technical reader (peer engineer / reviewer) who can reproduce the experiment
from the README alone.

---

## 2. Goals, KPIs & Acceptance Criteria

### 2.1 Measurable goals
- **G1** — Document exact hardware and justify a model choice that is "too big but not
  hopeless" for this machine.
- **G2** — Establish a reproducible **baseline** (direct execution) and capture its
  bottleneck with evidence.
- **G3** — Integrate **AirLLM + quantization** and run the *same* task; show how
  resource allocation changes.
- **G4** — Measure a standard metric suite across configurations and present it in
  tables + graphs.
- **G5** — Deliver an **OnPrem vs API** economic analysis with a break-even graph and
  explicit assumptions.
- **G6** — Tie every result back to execution concepts; propose ≥ 1 original extension.

### 2.2 KPIs (the standard metric suite)
| KPI | Definition | What it reveals |
|-----|------------|-----------------|
| **TTFT** | Time To First Token | Prefill load (compute, KV-cache build) |
| **TPOT / ITL** | Time Per Output Token | Decode load (memory bandwidth) |
| **Throughput** | tokens / second | Overall generation rate |
| **Peak RAM** | max resident RAM | host memory pressure |
| **Peak VRAM** | max GPU memory | GPU fit / offload behavior |
| **Total runtime** | wall-clock per run | end-to-end cost |
| **Est. energy** | Wh per run (estimated) | electricity cost input |
| **Output quality** | qualitative per quant level | accuracy "red line" |

### 2.3 Acceptance criteria
- [ ] Hardware documented; model choice justified (params, format, size vs VRAM/RAM).
- [ ] Baseline run documented with evidence of the bottleneck (logs/screenshots/metrics).
- [ ] At least **two** quantization levels compared (e.g. FP16/Q8/Q4) under AirLLM.
- [ ] Full metric suite captured per configuration; raw numbers stored in `results/`.
- [ ] Comparative tables + graphs generated programmatically and embedded in README.
- [ ] Break-even graph (OnPrem vs API, optional Cloud GPU) with documented assumptions.
- [ ] Analysis explicitly answers the 6 research questions (Section 4).
- [ ] Repo meets core engineering rules (see `docs/TODO.md` → Standards & Rules).

---

## 3. Functional Requirements

- **FR1 — Hardware probe:** a routine that collects CPU/RAM/GPU/VRAM/disk and writes a
  machine-readable spec to `results/` for the report.
- **FR2 — Model acquisition:** documented (manual) HF login/license acceptance + a
  download routine that stores weights on `D:` (never `C:`), token via env var.
- **FR3 — Baseline runner:** run the chosen model directly (e.g. HF Transformers and/or
  Ollama) on a fixed prompt + fixed `max_new_tokens`, capturing metrics or the failure.
- **FR4 — AirLLM runner:** run the same task via AirLLM with an explicit
  `layer_shards_saving_path` on `D:`, using `AutoModel`, at configurable quant levels.
- **FR5 — Benchmark harness:** measure TTFT, TPOT, throughput, peak RAM/VRAM, runtime,
  energy estimate; repeat N times; persist raw + aggregated results.
- **FR6 — Cost model:** compute API cost (input+output tokens × price, incl. optional
  prompt-caching) and OnPrem cost (CAPEX amortization + electricity/OPEX); find
  break-even; optional Cloud-GPU option.
- **FR7 — API client (via Gatekeeper):** optional real API calls for token-count/latency
  reference, routed through the `ApiGatekeeper` (rate-limited, queued, logged).
- **FR8 — Visualization:** generate all charts to `assets/` from stored results.
- **FR9 — SDK entry point:** every capability above is invokable through a single SDK
  facade; a thin CLI delegates to it.
- **FR10 — Reproducibility:** one documented command sequence reruns the experiment.

---

## 4. Research Questions (must be answered in the report)

1. What is the bottleneck blocking direct execution — memory (VRAM/RAM) or compute?
   How was it identified?
2. How does AirLLM change resource allocation, and how does it relate to virtual
   memory / OS paging?
3. Impact of quantization on memory, speed, and quality — where is the accuracy "red line"?
4. How do Prefill vs Decode show up, split across TTFT (compute) vs TPOT (memory)?
5. What latency/throughput price is paid to run a big model on modest hardware?
6. When is OnPrem cheaper than an external API, and when not?

---

## 5. Non-Functional Requirements

- **NFR1 — Code quality:** files ≤ 150 lines, ruff-clean, docstrings, DRY, OOP, SDK layer.
- **NFR2 — Tests:** TDD, ≥ 85% coverage, edge + error paths, external deps mocked.
- **NFR3 — Tooling:** `uv` only; `pyproject.toml` + `uv.lock`; Python 3.12.
- **NFR4 — Security:** no secrets in code; `.env` + `.env.example`; tokens via env vars.
- **NFR5 — Config-driven:** no hardcoded URLs/limits/timeouts; versioned config files.
- **NFR6 — Reproducibility & versioning:** start at `1.00`; clean Git history; Prompt Book.
- **NFR7 — Storage safety:** all large artifacts on `D:`; guard against filling `C:`.

---

## 6. Assumptions, Dependencies, Constraints

### 6.1 Assumptions
- The user creates the HF account, accepts model licenses, and provides tokens/API keys
  (these manual, web-based steps are out of scope for automation).
- Cost analysis uses documented, stated price assumptions (subject to change over time).

### 6.2 Dependencies (external)
- Hugging Face (model hub + token), AirLLM, PyTorch (CUDA 13.x build), Transformers,
  optionally Ollama; Matplotlib/Seaborn for charts; pytest + coverage + ruff; uv.

### 6.3 Constraints
- **Hardware:** i7-9750H, 15.8 GB RAM, GTX 1650 **4 GB VRAM**, model cache on **HDD**.
- **Disk:** `C:` ~9.8 GB free → unusable for models; everything heavy on `D:`.
- **Time:** keep it a focused experiment (target ~2–3 h active work), not a final project.

### 6.4 Out of scope
- Account creation, license acceptance, key provisioning (manual user steps).
- Training a model from scratch; production deployment; full GUI (optional nice-to-have).

---

## 7. User Stories

- **US1:** As a researcher, I run one command to capture my hardware spec for the report.
- **US2:** As a researcher, I run the baseline and get clear evidence of why it fails/stalls.
- **US3:** As a researcher, I run the model via AirLLM at several quant levels and get a
  metrics table.
- **US4:** As a researcher, I generate all comparison charts from stored results.
- **US5:** As a decision-maker, I see a break-even graph telling me when OnPrem beats API.

---

## 8. Milestones (high-level; detailed tasks live in TODO.md)

| # | Milestone | Definition of Done |
|---|-----------|--------------------|
| M0 | Planning docs | PRD, PLAN, TODO, per-mechanism PRDs approved |
| M1 | Env & scaffold | uv project on `D:`, Python 3.12, repo skeleton, ruff/pytest wired |
| M2 | Pipeline smoke test | tiny model + aggressive quant runs end-to-end |
| M3 | Baseline | direct run measured/failed + documented |
| M4 | AirLLM + quant | same task across quant levels, metrics persisted |
| M5 | Analysis & viz | charts + notebook + economic break-even |
| M6 | Report (README) | full deep-dive README with embedded visuals; checklist passed |

---

## 9. Decisions (locked 2026-06-28; override anytime)

1. **Model choice — Qwen2.5 family** (recommended by assignment; `AutoModel`-friendly,
   strong AirLLM support, multiple sizes for the size-sweep extension):
   - **Smoke test (tiny):** `Qwen/Qwen2.5-0.5B-Instruct` — prove the pipeline.
   - **Primary target:** `Qwen/Qwen2.5-7B-Instruct` (~15 GB FP16) — exceeds 4 GB VRAM and
     stresses 16 GB RAM → painful/failing baseline, clear AirLLM rescue.
   - **Stretch (dramatic baseline failure):** `Qwen/Qwen2.5-14B-Instruct` (~29 GB FP16) —
     exceeds RAM outright; used if HDD layer-streaming throughput proves tolerable.
   - *Exact final target confirmed at M3 after measuring D: (HDD) read speed.*
2. **Baseline tool — HF Transformers (primary)** for full control of dtype/device_map and
   precise TTFT/TPOT timing. **Ollama (optional contrast)** via the GGUF/llama.cpp path if
   time permits.
3. **API cost reference — modeled/published prices (primary)** for OpenAI + Anthropic
   (no live key needed for the core analysis). **Real API calls optional**, only through
   the `ApiGatekeeper`, if we want an empirical token-count/latency datapoint.
4. **Cloud-GPU — included** as the optional third break-even line (pure calculation:
   hourly rate of a rented T4/L4/A10 × measured runtime).
5. **Original extension — Roofline model plot (primary):** arithmetic intensity vs
   achievable throughput, marking Prefill vs Decode and the compute-/memory-bound
   crossover. **Secondary (time permitting): model-size sweep** (0.5B / 7B / 14B) showing
   how the bottleneck shifts with scale.
