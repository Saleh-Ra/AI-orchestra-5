"""BaselineRunner: direct Transformers load + streaming generation with timing.

This is the in-memory baseline (no AirLLM). It streams tokens so we can split
first-token latency (TTFT, prefill) from per-token latency (TPOT, decode).
Torch/transformers I/O is kept thin here; the measurable logic lives in
``metrics.py`` (pure, unit-tested). Proven end-to-end via the smoke run.
"""

from __future__ import annotations

import threading
import time

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from airllm_lab.services import metrics
from airllm_lab.services.feasibility import baseline_fits, weights_size_bytes
from airllm_lab.services.models import RunConfig, RunResult


class BaselineRunner:
    """Loads a model via Transformers and generates with per-token timing."""

    def __init__(self, hf_token: str | None = None) -> None:
        """Store the optional Hugging Face token used for downloads."""
        self._token = hf_token

    def run(self, cfg: RunConfig) -> RunResult:
        """Load the model, stream a generation, and return measured metrics.

        A pre-flight check fails fast when the weights cannot fit in RAM, so a
        too-big model does not thrash/crash the host during the naive load.
        """
        available = psutil.virtual_memory().available
        weights = weights_size_bytes(cfg.model_id)
        if not baseline_fits(weights, available):
            return RunResult.failed(
                cfg,
                error=(
                    f"baseline infeasible: ~{weights / 1e9:.1f} GB FP16 weights exceed "
                    f"~{available / 1e9:.1f} GB available RAM (and 4 GB VRAM); the naive "
                    "load thrashes swap and crashes the process"
                ),
                device="skipped",
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, token=self._token)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            token=self._token,
        ).to(device)

        messages = [{"role": "user", "content": cfg.prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=180
        )
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        start = time.perf_counter()
        thread.start()
        first_at, pieces = start, []
        for index, piece in enumerate(streamer):
            if index == 0:
                first_at = time.perf_counter()
            pieces.append(piece)
        end = time.perf_counter()
        thread.join()

        text = "".join(pieces)
        n_out = len(tokenizer(text, add_special_tokens=False).input_ids)
        return RunResult(
            model_id=cfg.model_id,
            mode=cfg.mode,
            quant=cfg.quant,
            device=device,
            n_input_tokens=int(input_ids.shape[-1]),
            n_output_tokens=n_out,
            ttft_s=round(metrics.time_to_first_token(start, first_at), 4),
            tpot_s=round(metrics.time_per_output_token(first_at, end, n_out), 4),
            throughput_tok_s=round(metrics.throughput(n_out, end - start), 2),
            runtime_s=round(end - start, 3),
            output_text=text,
        )
