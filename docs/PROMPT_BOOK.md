# Prompt Book

A log of the **significant prompts** used to build this project with an AI coding
assistant, as required by the submission guidelines (transparency of AI use). Each
entry records the *context*, the *goal/prompt*, and the *outcome*. Routine
follow-ups ("continue", "go ahead") are omitted; this captures the decisions that
shaped the codebase.

> Convention: the human set direction and made the external/account decisions
> (Hugging Face token, model choice, Git/GitHub); the assistant did the coding,
> debugging, measurement, and documentation.

---

## Phase 0 — Framing & planning

- **Context:** empty repo + the assignment PDF and a separate "good software
  practices" guidelines PDF.
- **Prompt:** "Read the assignment carefully; create PLAN and TODO; turn the
  guidelines (≤150-line files, TDD, linting, docs-first, SDK facade, config-driven,
  `uv`) into a working map we follow in order."
- **Outcome:** `docs/PRD.md`, `docs/PLAN.md`, `docs/TODO.md` (rules + ordered
  step list), and per-mechanism PRDs (`PRD_airllm`, `PRD_quantization`,
  `PRD_benchmark`, `PRD_cost_model`). Six research questions fixed in the PRD.

## Phase 1 — Scaffold

- **Prompt:** "Stand up the project skeleton: `uv` + `pyproject.toml`, the `LabSDK`
  facade as the single entry point, a thin CLI, versioned JSON config, `.env`
  handling, and tests — keep every file ≤150 lines."
- **Outcome:** `src/airllm_lab/{sdk,services,shared}`, `config/setup.json`, CLI,
  and a passing test suite with 100% coverage as the baseline standard.

## Phase 2–3 — Hardware, download, and the deliberate failure

- **Context:** modest laptop (4 GB VRAM, ~16 GB RAM, model storage on a `D:` HDD).
- **Prompt:** "Probe and record the hardware. Download the 7B model to `D:`. Then
  run it the naive way so we can *see* it fail."
- **Outcome:** `results/hardware.json`; resumable downloader; the baseline crash
  (`0xC0000005` after swap-thrashing) captured, then a **pre-flight feasibility
  check** added so it fails fast and cleanly instead of freezing the machine.

## Phase 4 — AirLLM + quantization

- **Prompt:** "Make the 7B actually run via AirLLM (layered, disk-streamed
  inference); measure TTFT/TPOT/throughput/peak memory/energy with a reusable
  benchmark harness."
- **Key debugging prompts & decisions:**
  - AirLLM 2.11 was incompatible with `transformers 5.x` (KV-cache / RoPE refactor).
    **Decision (human):** *"Pin transformers to the AirLLM-compatible ~4.40 era and
    use the real AirLLM library."* → stack pinned; generation driven by a manual
    greedy-decode loop over `forward`.
  - Fixed cascade: `optimum.bettertransformer` shim, `sentencepiece`, per-model/
    per-quant shard directories to avoid mixing layers.
- **Prompt:** "Run the same task across ≥2 quant levels and record quality."
- **Outcome:** `bitsandbytes` added; FP16/INT8/INT4 matrix captured — INT4 ≈ 21×
  FP16 but crosses the accuracy red line; INT8 is the sweet spot.

## Phase 5 — Analysis & visualization

- **Prompt:** "Implement the cost model (API vs OnPrem vs Cloud-GPU + break-even),
  generate all charts from stored results, build a reproducible analysis notebook
  tying numbers to systems concepts, and implement an original extension."
- **Outcome:** `services/cost_model.py` (pure, unit-tested), `services/charts.py`,
  `notebooks/analysis.ipynb` (runs top-to-bottom), and the **Roofline** extension
  (`services/roofline.py`) proving the workload is bandwidth/disk-bound. The SDK
  was split into mixins to keep every file ≤150 lines.

## Phase 6 — Report & submission

- **Prompt:** "Make it submission-ready."
- **Outcome:** README finalized as a technical report (explicitly answering the six
  research questions, with embedded figures), `LICENSE` added, this Prompt Book
  written, and a final quality pass (ruff clean, 100% coverage, file-size limit).
