"""AirLLMRunner: layered (disk-streamed) inference with per-token timing.

AirLLM loads one transformer layer at a time from disk, runs it, then frees it,
so a model far larger than VRAM fits (peak memory ~ one layer). We drive the
model's ``forward`` in a manual greedy loop instead of ``transformers.generate``:
AirLLM 2.11's generate integration relies on the legacy KV-cache and is brittle,
while the explicit loop makes TTFT/TPOT timing obvious. Disk-bound by design, so
this is slow per token — that trade-off is the core finding versus the baseline.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import torch

from airllm_lab.services import metrics
from airllm_lab.services.models import RunConfig, RunResult

# Map our quant labels to AirLLM's ``compression`` argument (4/8-bit need
# bitsandbytes; ``fp16`` is the uncompressed layered path).
_QUANT_TO_COMPRESSION: dict[str, str | None] = {"fp16": None, "8bit": "8bit", "4bit": "4bit"}


def _ensure_bettertransformer_importable() -> None:
    """Shim ``optimum.bettertransformer`` if a newer optimum removed it.

    AirLLM imports it at module load; its ``transform`` is wrapped in a
    ``try/except ValueError`` that falls back to sdpa attention, so a stub that
    raises ``ValueError`` keeps AirLLM importable and working.
    """
    try:
        import optimum.bettertransformer  # noqa: F401
    except Exception:
        shim = types.ModuleType("optimum.bettertransformer")

        class BetterTransformer:
            @staticmethod
            def transform(*args: object, **kwargs: object) -> object:
                raise ValueError("BetterTransformer shim: fall back to sdpa")

        shim.BetterTransformer = BetterTransformer  # type: ignore[attr-defined]
        sys.modules["optimum.bettertransformer"] = shim


class AirLLMRunner:
    """Runs layered AirLLM inference for a model that does not fit in VRAM."""

    def __init__(
        self, shards_dir: str, hf_token: str | None = None, max_seq_len: int = 512
    ) -> None:
        """Store the layer-shard output dir (on D:), HF token, and max seq len."""
        self._shards_dir = shards_dir
        self._token = hf_token
        self._max_seq_len = max_seq_len

    def _load(self, cfg: RunConfig) -> tuple[object, str]:
        """Load the AirLLM model for ``cfg`` (splits layers to disk on first run)."""
        _ensure_bettertransformer_importable()
        from airllm import AutoModel

        # Isolate shards per model+quant: AirLLM writes to ``<path>/splitted_model``,
        # so a shared path would mix layers from different models. Key by name.
        per_model = Path(self._shards_dir) / f"{Path(cfg.model_id).name}-{cfg.quant}"
        per_model.mkdir(parents=True, exist_ok=True)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(
            cfg.model_id,
            device=device,
            compression=_QUANT_TO_COMPRESSION.get(cfg.quant),
            layer_shards_saving_path=str(per_model),
            max_seq_len=self._max_seq_len,
            hf_token=self._token,
        )
        return model, device

    def run(self, cfg: RunConfig) -> RunResult:
        """Greedy-decode up to ``cfg.max_new_tokens`` with per-token timing."""
        model, device = self._load(cfg)
        tok = model.tokenizer
        messages = [{"role": "user", "content": cfg.prompt}]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(device)
        n_in = int(ids.shape[-1])
        eos = tok.eos_token_id

        start = time.perf_counter()
        first_at, generated = start, []
        for index in range(cfg.max_new_tokens):
            out = model(input_ids=ids, use_cache=False, return_dict=True)
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if index == 0:
                first_at = time.perf_counter()
            token_id = int(nxt.item())
            generated.append(token_id)
            ids = torch.cat([ids, nxt], dim=1)
            if eos is not None and token_id == eos:
                break
        end = time.perf_counter()

        n_out = len(generated)
        return RunResult(
            model_id=cfg.model_id,
            mode="airllm",
            quant=cfg.quant,
            device=device,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            ttft_s=round(metrics.time_to_first_token(start, first_at), 4),
            tpot_s=round(metrics.time_per_output_token(first_at, end, n_out), 4),
            throughput_tok_s=round(metrics.throughput(n_out, end - start), 4),
            runtime_s=round(end - start, 3),
            output_text=tok.decode(generated, skip_special_tokens=True),
        )
