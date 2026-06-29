# TODO

> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done
> This file tracks tasks **and** holds the project-wide Standards & Rules we must
> follow throughout (from `software_submission_guidelines-V3.pdf`). The rules are
> not tasks — they are constraints we keep in mind on every task.

---

## Project context (our machine)

- **CPU:** Intel i7-9750H — 6 cores / 12 threads @ 2.6 GHz
- **RAM:** 15.8 GB
- **GPU:** NVIDIA GTX 1650, **4 GB VRAM** (+ Intel UHD 630) · CUDA 13.1
- **Storage:** `C:` SSD 237 GB (**~9.8 GB free** — tight!), `D:` HDD 932 GB (**~765 GB free**)
- **Python:** use **3.12** (3.14 is too new; 3.9 too old) · **uv** not yet installed
- **Hard constraints:**
  - Keep models, HF cache, AirLLM layer shards, and the venv **on `D:`** (C: is full).
  - `D:` is an **HDD** → AirLLM per-layer disk I/O will be slow (expected, and a finding).
  - Never hardcode the Hugging Face token or any API key.

---

## Standards & Rules (keep in mind on every task)

### A. Mandatory docs & workflow (do docs before code)
- [ ] `docs/PRD.md` — product requirements (overview, goals, KPIs, acceptance criteria,
      functional/non-functional reqs, assumptions, constraints, milestones).
- [ ] `docs/PLAN.md` — architecture (C4 diagrams, UML for complex flows, ADRs with
      rationale/trade-offs, API/interface docs, data schemas).
- [ ] `docs/TODO.md` — this file (tasks with priority + status, phases, milestones,
      definition-of-done per task).
- [ ] `docs/PRD_<mechanism>.md` — one dedicated PRD per core mechanism:
      AirLLM, quantization, benchmarking/metrics, cost model.
- [ ] Approve all docs **before** development starts; update TODO as we progress;
      save results + visualizations + update README at the end.
- [ ] Root `README.md` = full user manual (install, usage, examples + **screenshots**,
      config guide, contribution guidelines, license & credits). The deep-dive report
      lives here too, with all graphs/tables/screenshots embedded inline.

### B. Code structure & quality
- [ ] **Every code file ≤ 150 lines** (blank/comment lines excluded). Split, never compress.
- [ ] **SDK layer is the single entry point** for all business logic; CLI/notebooks/GUI
      only delegate to the SDK (no business logic in them).
- [ ] OOP, **DRY** (extract shared logic to module/base class/mixin), single responsibility.
- [ ] Docstrings on every module/class/function; comments explain **why**, not what.
- [ ] Descriptive names, consistent style across the project.
- [ ] Package layout: `src/<pkg>/{sdk,services,shared}` with
      `shared/{gatekeeper.py,config.py,version.py}`, plus `constants.py`.
- [ ] Building blocks defined by Input / Output / Setup data.

### C. API Gatekeeper & rate limiting (for the API cost-comparison part)
- [ ] All external API calls go through a central `ApiGatekeeper`.
- [ ] Enforce rate limits **before** each call; overflow → **FIFO queue** (never drop);
      retry on transient failures; log every call.
- [ ] Rate limits read from `config/rate_limits.json` (versioned), never hardcoded.

### D. Testing (TDD)
- [ ] TDD: RED → GREEN → REFACTOR; tests written before/with code.
- [ ] `tests/` mirrors `src/`; `conftest.py` for shared fixtures; mock external deps.
- [ ] Every module has a test file; every public function has ≥ 1 test (happy + error path).
- [ ] **≥ 85% coverage** (suite fails below; `fail_under = 85` in `pyproject.toml`).
- [ ] Test files also ≤ 150 lines; no tests depend on live external services.
- [ ] Document expected results; produce pass/fail reports; keep run logs.

### E. Linting, config, security
- [ ] **Ruff** clean: `ruff check` with no errors
      (line-length 100, target py310, select `E,F,W,I,N,UP,B,C4,SIM`, ignore `E501`).
- [ ] No hardcoded config (URLs, rate limits, timeouts, secrets). Allowed in code:
      physical/math constants, default params, `constants.py`, Enum values.
- [ ] Config hierarchy: `config/{setup.json,rate_limits.json,logging_config.json}`,
      `.env` (gitignored), `.env.example` (committed), `pyproject.toml`, `constants.py`.
- [ ] Secrets only via `os.environ.get(...)`. `.gitignore` includes
      `.env, *.pem, *.key, credentials.json`. `.env.example` present with dummy values.

### F. Tooling, versioning, Git
- [ ] **uv is mandatory** — `uv sync` / `uv add` / `uv run python` / `uv run pytest` /
      `uv lock`. No `pip`, `python -m`, `venv`, `virtualenv`. No `requirements.txt`.
- [ ] `pyproject.toml` = single source of truth; `uv.lock` committed.
- [ ] Versioning starts at `1.00` (`version.py`, JSON `version`, `rate_limits.version`);
      validate config version at startup.
- [ ] Clean Git history (meaningful commits, feature branches, tags for versions).
- [ ] Maintain a **Prompt Book** (log of significant AI prompts: context, goal, outputs).

### G. Research, visualization, costs
- [ ] Systematic parameter/sensitivity analysis (e.g. OAT across quantization levels).
- [ ] **Jupyter analysis notebook** in `notebooks/` (LaTeX for equations, academic refs).
- [ ] High-quality charts (bar/line/scatter/heatmap/box/waterfall): clear labels,
      consistent accessible colors, captions, legends, high resolution.
- [ ] Token cost tables + break-even analysis (OnPrem vs API, optional Cloud GPU),
      all assumptions explicit.

### H. Nice-to-have (raise grade; not strictly required for a research experiment)
- [ ] Plugins/extension points (hooks, middleware), ISO/IEC 25010 self-assessment,
      parallel processing where it helps (multiprocessing CPU-bound / threading I/O-bound),
      UI/UX with Nielsen heuristics if a GUI is added.

---

## Working Map — ordered steps

This is the exact order we execute. Each step has a **DoD** (definition of done).
`[U]` = needs your (user) action; `[A]` = I (agent) do it; `[U+A]` = we do together.

### Phase 0 — Planning docs  (Milestone M0) — ✅ DONE
1. [x] `[A]` Draft `docs/PRD.md`. **DoD:** PRD covers goals/KPIs/FRs/constraints.
2. [x] `[U]` Review & approve `docs/PRD.md`. (approved: "we seem to be set")
3. [x] `[A]` Draft `docs/PLAN.md` — architecture: C4, SDK + Gatekeeper interfaces,
   data schemas, ADRs.
4. [x] `[A]` Draft per-mechanism PRDs: `PRD_airllm.md`, `PRD_quantization.md`,
   `PRD_benchmark.md`, `PRD_cost_model.md`.
5. [x] `[U+A]` Decide the 5 open questions → recorded in PRD §9.
6. [x] `[U]` Approve all docs → green light to build.

### Phase 1 — Environment & scaffold  (Milestone M1) — ✅ DONE
7. [x] `[A]` Install `uv` (0.11.25); create project with **Python 3.12**, venv on `D:`.
8. [x] `[A]` Create repo skeleton (`src/airllm_lab/{shared,...}`, `tests/`,
   `config/`, `data/`, `results/`, `assets/`, `notebooks/`).
9. [x] `[A]` Write `pyproject.toml` (deps, ruff, coverage `fail_under=85`), `uv.lock`,
   `.gitignore`, `.env.example`, `version.py` (1.0.0), `constants.py`.
10. [x] `[A]` Configure storage paths (HF cache + AirLLM `layer_shards_saving_path` → `D:`)
    in `config/setup.json`. *(exported as env vars at runtime in Phase 2/3.)*
11. [ ] `[U]` Free up `C:` if feasible; confirm `D:` has room for the model.
    **DoD:** ≥ model-size free on D:. *(still pending — your action before M3 download.)*
12. [x] `[A]` `git init` + first commit; ruff clean + pytest pass (5 tests, 100% cov).

### Phase 2 — Pipeline smoke test  (Milestone M2) — ✅ DONE
13. [x] `[A]` Build the **SDK facade** + thin CLI (entry point only). **DoD:** `--help` works, tests pass.
    *(done: `LabSDK`, CLI `version`/`hardware`/`smoke`, config loader, hardware probe.)*
14. [~] `[U]` Create HF account, accept model license, put `HF_TOKEN` in `.env`.
    **DoD:** token present (never committed). *(token pasted but needs **save** to disk;*
    *not blocking — Qwen2.5 models are open, so the smoke ran token-less.)*
15. [x] `[A]` Run a tiny model through the full path to prove the pipeline.
    **DoD:** end-to-end generation succeeds on the small model.
    *(done: Qwen2.5-0.5B-Instruct generated on **CUDA**, weights on **D:**, FP16. ML stack*
    *installed: torch 2.6.0+cu124, transformers 5.12.1, accelerate, airllm 2.11.0. Quant*
    *(Q8/Q4) deferred to Phase 4; smoke used FP16 to prove load→generate→measure.)*

### Phase 3 — Baseline  (Milestone M3) — ✅ DONE
16. [x] `[A]` Implement hardware-probe service; write spec to `results/`. **DoD:** spec JSON saved.
    *(done: `results/hardware.json`.)*
17. [x] `[U+A]` Download the chosen "too big" model to `D:`. **DoD:** weights on disk.
    *(done: Qwen2.5-7B-Instruct, 14.2 GB on D:, via curl in ~56 min.)*
18. [x] `[A]` Run **baseline direct execution** (Transformers ± Ollama); capture metrics
    or the failure (OOM/swap-thrash/too-slow) with logs + screenshots. **DoD:** baseline documented.
    *(FINDING: naive FP16 load thrashed swap and **crashed the process** — exit*
    *`0xC0000005` ACCESS_VIOLATION at 207/339 (~61%) of weight-load after ~28 min,*
    *never reaching the GPU. Added a pre-flight feasibility guard so it now fails*
    *fast & clean: `results/baseline_Qwen2.5-7B-Instruct.json` (ok=false, ~15.2 GB*
    *weights > ~8.3 GB free RAM + 4 GB VRAM). This is the motivation for AirLLM.)*

### Phase 4 — AirLLM + quantization  (Milestone M4)
19. [x] `[A]` Implement the **AirLLM runner** (AutoModel, shards on D:, quant configurable).
    **DoD:** big model generates via AirLLM.
    *(DONE: `services/airllm_runner.py` + `LabSDK.run_airllm` + CLI `airllm`. The 7B*
    *that crashed at baseline GENERATED via AirLLM on the 4 GB GPU: `"Virtual memory*
    *is a"`, TTFT≈930 s, ≈0.0011 tok/s — feasible but disk-bound. Result:*
    *`results/airllm_Qwen2.5-7B-Instruct_fp16.json`. Stack pinned to transformers*
    *4.40 (AirLLM 2.11 incompatible with 5.x); greedy-decode loop over `forward`.)*
20. [x] `[A]` Implement the **benchmark harness** (TTFT, TPOT, throughput, peak RAM/VRAM,
    runtime, energy est.), N repeats, persist raw+aggregated. **DoD:** results in `results/`.
    *(DONE: `services/benchmark.py` (harness, repeats+warmup, mean/median/std/min/max),*
    *`services/monitor.py` (peak RAM sampler + CUDA VRAM peak), `metrics.energy_wh`/*
    *`summarize`, RunResult extended with peak_ram_gb/peak_vram_gb/energy_wh. SDK*
    *`run_benchmark` + CLI `benchmark`. Verified: 0.5B baseline ×2 → peak RAM 2.21 GB,*
    *VRAM 1.25 GB, 0.42 Wh. 56 tests, 100% cov.)*
21. [ ] `[A]` Run the same task across **≥ 2 quant levels** (e.g. FP16/Q8/Q4) + record
    qualitative output quality. **DoD:** full metric matrix captured.

### Phase 5 — Analysis & visualization  (Milestone M5)
22. [x] `[A]` Implement the **cost model** (API tokens×price incl. prompt-caching; OnPrem
    CAPEX+OPEX; optional Cloud-GPU) → break-even. **DoD:** break-even point computed.
    *(DONE: `services/cost_model.py` (pure, dataclasses + functions), config `cost`*
    *section (dated assumptions), `LabSDK.run_cost_analysis` + CLI `cost`. Result:*
    *`results/cost_analysis.json`. Break-even ≈ 239k req/mo at 30 s/request; with the*
    *measured AirLLM runtime, local energy alone beats no API → never breaks even.*
    *8 unit tests vs hand-computed values.)*
23. [x] `[A]` Generate all **charts** to `assets/` from stored results. **DoD:** figures render.
    *(DONE: `services/charts.py` (matplotlib, auto-discovers benchmark/airllm/baseline*
    *artifacts), `LabSDK.generate_charts` + CLI `charts`. Renders throughput.png,*
    *ttft.png, memory.png, cost_breakeven.png.)*
24. [x] `[A]` Build the **analysis notebook** (`notebooks/`) tying results to execution
    concepts (Prefill/Decode, compute/memory-bound, paging). **DoD:** notebook runs top-to-bottom.
    *(DONE: `notebooks/analysis.ipynb` — loads artifacts, rebuilds cost+charts via the*
    *SDK, explains prefill/decode, paging, memory, economics. Executed end-to-end with*
    *nbconvert --execute (exit 0).)*
25. [x] `[A]` Implement the **original extension** chosen in step 5. **DoD:** extension produces a result.
    *(DONE: **Roofline model** — `services/roofline.py` + `LabSDK.generate_roofline` +*
    *CLI `roofline` + config `roofline` peaks. Plots GPU-mem/disk/compute roofs and*
    *places each run at its achieved GFLOP/s → both memory-bound; 7B AirLLM pinned to*
    *the disk roof. Result: `assets/roofline.png`. 69 tests, ruff clean, 100% cov.)*

### Phase 6 — Report & submission  (Milestone M6)
26. [ ] `[A]` Write the deep-dive **`README.md`** (install, usage, findings, economic
    analysis, concept analysis, embedded graphs/tables/screenshots, reproduce steps).
    **DoD:** README answers all 6 research questions.
27. [ ] `[A]` Final pass: ruff clean, coverage ≥ 85%, Prompt Book updated, Git tidy.
    **DoD:** final checklist (guidelines §17) all green.
28. [ ] `[U]` Final review & submit. **DoD:** you approve the repo for submission.
