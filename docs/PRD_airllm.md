# PRD — AirLLM Layered Inference

**Version:** 1.00 · **Mechanism owner module:** `services/airllm_runner.py`

## 1. Background (theory)
AirLLM enables running a model far larger than available VRAM/RAM by performing
**layer-by-layer inference**. Each transformer layer's weights are streamed from disk
(memory-mapped), used for the forward pass, then released before the next layer loads.
Peak memory is therefore ~one layer instead of the whole model. This is the practical
analogue of **virtual memory / demand paging**: the "working set" in fast memory is tiny,
while the full model lives on slower storage and is paged in on demand. The cost is heavy
**Disk I/O per token** (every generated token re-reads all layers).

## 2. Requirements
- Load a Qwen2.5 model via **`AutoModel`**-style API to avoid class-mismatch.
- Set an explicit **`layer_shards_saving_path` on `D:`** (never `C:`).
- Support a configurable **compression/quant level** (FP16 / 8-bit / 4-bit).
- Configurable `max_new_tokens`; deterministic decoding for fair comparison.
- Stream tokens so per-token timing is observable.

## 3. Inputs / Outputs / Setup
- **Input:** `RunConfig(model_id, quant, prompt, max_new_tokens)`.
- **Output:** `RunResult` (TTFT, TPOT, throughput, peak RAM/VRAM, runtime, energy, text).
- **Setup:** shard path, device, dtype, HF token (env), seed.

## 4. Performance expectations
- Peak RAM/VRAM **much lower** than baseline (≈ one layer + activations).
- Wall-clock **slower** per token than an in-memory run would be — dominated by disk reads
  (especially on our HDD). This trade-off is the core finding.

## 5. Constraints / limitations
- HDD on `D:` → high per-token latency; mitigate by capping tokens and starting at 7B.
- First run also performs **sharding** (one-time heavy write) of the model to layers.

## 6. Alternatives considered
- `accelerate` disk-offload / `device_map="auto"`: similar idea, less explicit per-layer
  control; AirLLM chosen per assignment focus.
- llama.cpp mmap (GGUF): used as the **quantization/Ollama** contrast, not the AirLLM path.

## 7. Success criteria & tests
- [ ] 0.5B model generates via AirLLM in the smoke test.
- [ ] 7B model that fails/stalls at baseline **completes** via AirLLM.
- [ ] Peak memory recorded and clearly below baseline.
- [ ] Unit tests mock the AirLLM model object and assert the runner's measure/flow logic
      (no real download in tests).
