"""Domain data models (dataclasses) shared across services.

These are plain, serializable records describing what to run (``RunConfig``)
and what happened (``RunResult``); they are persisted as JSON in ``results/``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Inputs that fully specify a single generation run."""

    model_id: str
    prompt: str
    max_new_tokens: int = 64
    mode: str = "baseline"
    quant: str = "fp16"


@dataclass
class RunResult:
    """Measured outcome of a single generation run."""

    model_id: str
    mode: str
    quant: str
    device: str
    n_input_tokens: int
    n_output_tokens: int
    ttft_s: float
    tpot_s: float
    throughput_tok_s: float
    runtime_s: float
    output_text: str
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def failed(cls, cfg: RunConfig, error: str, device: str = "n/a") -> RunResult:
        """Build a zeroed result marking a run that errored (e.g. OOM)."""
        return cls(
            model_id=cfg.model_id,
            mode=cfg.mode,
            quant=cfg.quant,
            device=device,
            n_input_tokens=0,
            n_output_tokens=0,
            ttft_s=0.0,
            tpot_s=0.0,
            throughput_tok_s=0.0,
            runtime_s=0.0,
            output_text="",
            ok=False,
            error=error,
        )
