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

### Phase 0 — Planning docs  (Milestone M0)
1. [x] `[A]` Draft `docs/PRD.md`. **DoD:** PRD covers goals/KPIs/FRs/constraints. *(done, in review)*
2. [ ] `[U]` Review & approve `docs/PRD.md`. **DoD:** you confirm or request edits.
3. [ ] `[A]` Draft `docs/PLAN.md` — architecture: C4 (context/container/component),
   SDK + Gatekeeper interfaces, data schemas, ADRs. **DoD:** plan reviewed by you.
4. [ ] `[A]` Draft per-mechanism PRDs: `PRD_airllm.md`, `PRD_quantization.md`,
   `PRD_benchmark.md`, `PRD_cost_model.md`. **DoD:** each defines I/O, success criteria.
5. [ ] `[U+A]` Decide the 5 open questions (model, baseline tool, API provider,
   cloud-GPU, extension). **DoD:** decisions recorded in PRD §9.
6. [ ] `[U]` Approve all docs → green light to build. **DoD:** explicit go-ahead.

### Phase 1 — Environment & scaffold  (Milestone M1)
7. [ ] `[A]` Install `uv`; create project with **Python 3.12**, venv on **`D:`**.
   **DoD:** `uv run python --version` shows 3.12 from a D: venv.
8. [ ] `[A]` Create repo skeleton (`src/<pkg>/{sdk,services,shared}`, `tests/`,
   `config/`, `data/`, `results/`, `assets/`, `notebooks/`). **DoD:** tree matches PLAN.
9. [ ] `[A]` Write `pyproject.toml` (deps, ruff, coverage `fail_under=85`), `uv.lock`,
   `.gitignore`, `.env.example`, `version.py` (1.00), `constants.py`. **DoD:** `uv sync` ok.
10. [ ] `[A]` Configure storage env (HF cache + AirLLM `layer_shards_saving_path` → `D:`).
    **DoD:** config file points all caches to D:.
11. [ ] `[U]` Free up `C:` if feasible; confirm `D:` has room for the model. **DoD:** ≥ model-size free on D:.
12. [ ] `[A]` `git init` + first commit; wire ruff + pytest. **DoD:** `uv run ruff check` &
    `uv run pytest` run clean on the empty skeleton.

### Phase 2 — Pipeline smoke test  (Milestone M2)
13. [ ] `[A]` Build the **SDK facade** + thin CLI (entry point only). **DoD:** `--help` works, tests pass.
14. [ ] `[U]` Create HF account, accept model license, put `HF_TOKEN` in `.env`.
    **DoD:** token present (never committed).
15. [ ] `[A]` Run a **tiny model at aggressive quant (e.g. Q2)** through the full path
    to prove the pipeline. **DoD:** end-to-end generation succeeds on the small model.

### Phase 3 — Baseline  (Milestone M3)
16. [ ] `[A]` Implement hardware-probe service; write spec to `results/`. **DoD:** spec JSON saved.
17. [ ] `[U+A]` Download the chosen "too big" model to `D:`. **DoD:** weights on disk.
18. [ ] `[A]` Run **baseline direct execution** (Transformers ± Ollama); capture metrics
    or the failure (OOM/swap-thrash/too-slow) with logs + screenshots. **DoD:** baseline documented.

### Phase 4 — AirLLM + quantization  (Milestone M4)
19. [ ] `[A]` Implement the **AirLLM runner** (AutoModel, shards on D:, quant configurable).
    **DoD:** big model generates via AirLLM.
20. [ ] `[A]` Implement the **benchmark harness** (TTFT, TPOT, throughput, peak RAM/VRAM,
    runtime, energy est.), N repeats, persist raw+aggregated. **DoD:** results in `results/`.
21. [ ] `[A]` Run the same task across **≥ 2 quant levels** (e.g. FP16/Q8/Q4) + record
    qualitative output quality. **DoD:** full metric matrix captured.

### Phase 5 — Analysis & visualization  (Milestone M5)
22. [ ] `[A]` Implement the **cost model** (API tokens×price incl. prompt-caching; OnPrem
    CAPEX+OPEX; optional Cloud-GPU) → break-even. **DoD:** break-even point computed.
23. [ ] `[A]` Generate all **charts** to `assets/` from stored results. **DoD:** figures render.
24. [ ] `[A]` Build the **analysis notebook** (`notebooks/`) tying results to execution
    concepts (Prefill/Decode, compute/memory-bound, paging). **DoD:** notebook runs top-to-bottom.
25. [ ] `[A]` Implement the **original extension** chosen in step 5. **DoD:** extension produces a result.

### Phase 6 — Report & submission  (Milestone M6)
26. [ ] `[A]` Write the deep-dive **`README.md`** (install, usage, findings, economic
    analysis, concept analysis, embedded graphs/tables/screenshots, reproduce steps).
    **DoD:** README answers all 6 research questions.
27. [ ] `[A]` Final pass: ruff clean, coverage ≥ 85%, Prompt Book updated, Git tidy.
    **DoD:** final checklist (guidelines §17) all green.
28. [ ] `[U]` Final review & submit. **DoD:** you approve the repo for submission.
