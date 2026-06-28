# PRD — Quantization

**Version:** 1.00 · **Mechanism owner module:** `services/airllm_runner.py` (+ config)

## 1. Background (theory)
Quantization reduces the numeric precision of weights (and sometimes activations) from
FP16 to 8-bit (Q8) or 4-bit (Q4/Q2). This shrinks the memory footprint roughly
proportionally (FP16→Q4 ≈ 4× smaller) and reduces the bytes moved per token, which can
**raise decode throughput** on memory-bandwidth-bound systems. The trade-off is a
potential **loss of output quality/accuracy**; below some bit-width the model degrades
sharply — the "accuracy red line".

## 2. Requirements
- Support at least **two** levels for comparison; target set: **FP16, Q8, Q4** (+ Q2 for
  the aggressive smoke test).
- Quant level is **config-driven** (enum `QuantLevel`), never hardcoded at a call site.
- Same prompt, seed, and `max_new_tokens` across levels for fair comparison.

## 3. Inputs / Outputs / Setup
- **Input:** `quant: QuantLevel`, plus the shared `RunConfig`.
- **Output:** per-level `RunResult` + a short **qualitative quality note** per level.
- **Setup:** quant backend params (e.g. bits, group size) in `config/setup.json`.

## 4. Performance expectations
- Memory: FP16 > Q8 > Q4 > Q2 (monotonic decrease).
- Throughput: typically increases as bits drop (less data moved), hardware permitting.
- Quality: stable down to Q8/Q4, then degrades — locate the red line.

## 5. Constraints / limitations
- Not all quant kernels support CPU/old GPUs equally; document what actually ran.
- Quality assessment is qualitative (small fixed prompt set), not a full benchmark.

## 6. Alternatives considered
- GPTQ / AWQ / bitsandbytes / GGUF: choose whichever AirLLM + our hardware supports
  cleanly; record the choice and why.

## 7. Success criteria & tests
- [ ] ≥ 2 quant levels run on the primary model with metrics captured.
- [ ] Memory decreases monotonically with bit-width.
- [ ] Quality red line identified and discussed.
- [ ] Unit test: `QuantLevel` parsing/validation and config wiring.
