# PRD — Benchmarking & Metrics

**Version:** 1.00 · **Mechanism owner modules:** `services/metrics.py`, `services/benchmark.py`

## 1. Background (theory)
Generation has two phases. **Prefill** processes the whole prompt in parallel and builds
the KV-cache — compute-heavy, reflected by **TTFT** (Time To First Token). **Decode**
emits tokens one at a time, each needing a full pass over weights + KV-cache — memory-
bandwidth-heavy, reflected by **TPOT** (Time Per Output Token, a.k.a. ITL). Splitting
TTFT vs TPOT exposes whether a phase is **compute-bound** or **memory-bound**.

## 2. Requirements
Measure per run and persist raw + aggregated:
- **TTFT** (s) — request → first token.
- **TPOT / ITL** (s/token) — mean inter-token time after the first.
- **Throughput** (tokens/s) — output tokens / generation time.
- **Peak RAM** and **Peak VRAM** (MB) — sampled during generation.
- **Total runtime** (s) and **estimated energy** (Wh).
- **Token counts** (input/output) and the **output text**.
- **Qualitative quality** note.

## 3. Inputs / Outputs / Setup
- **Input:** `RunConfig`, `repeats: int`.
- **Output:** `BenchmarkReport` (per-run `RunResult` + aggregates: mean/median/std).
- **Setup:** memory-sampling interval, warmup count, fixed seed, energy model (W × time).

## 4. Method
- Use **streaming** generation to timestamp each token (TTFT = t[0]−t_start;
  TPOT = mean(diff(t[1:]))).
- Sample RAM via `psutil`, VRAM via `torch.cuda.max_memory_allocated` / `nvidia-smi`,
  on a background poller; record the peak.
- Energy ≈ estimated average package+GPU power × runtime (assumption documented).
- Repeat N times; discard warmup; report mean ± std.

## 5. Constraints / limitations
- Energy is an **estimate** (no wall meter); state the power assumptions.
- Background sampling adds tiny overhead; keep interval modest.

## 6. Success criteria & tests
- [ ] TTFT, TPOT, throughput computed correctly from synthetic timestamp sequences.
- [ ] Aggregation (mean/median/std) unit-tested.
- [ ] Report serializes to JSON in `results/`.
- [ ] Memory sampler returns a peak ≥ any sampled value (unit-tested with a fake source).
