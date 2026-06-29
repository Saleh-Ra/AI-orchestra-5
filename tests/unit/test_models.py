"""Tests for the domain dataclasses."""

from airllm_lab.services.models import RunConfig, RunResult


def test_run_config_defaults() -> None:
    """RunConfig fills sensible defaults for optional fields."""
    cfg = RunConfig(model_id="m", prompt="hi")
    assert cfg.max_new_tokens == 64
    assert cfg.mode == "baseline"
    assert cfg.quant == "fp16"


def test_run_result_to_dict_roundtrip() -> None:
    """RunResult serializes every field to a plain dict."""
    result = RunResult(
        model_id="m",
        mode="baseline",
        quant="fp16",
        device="cpu",
        n_input_tokens=3,
        n_output_tokens=5,
        ttft_s=0.1,
        tpot_s=0.2,
        throughput_tok_s=10.0,
        runtime_s=0.5,
        output_text="hello",
    )
    data = result.to_dict()
    assert data["output_text"] == "hello"
    assert data["ok"] is True
    assert set(data) >= {"ttft_s", "tpot_s", "throughput_tok_s", "device"}


def test_run_result_failed_factory() -> None:
    """RunResult.failed records the error and marks the run not-ok."""
    cfg = RunConfig(model_id="big/model", prompt="hi")
    result = RunResult.failed(cfg, error="CUDA out of memory")
    assert result.ok is False
    assert result.error == "CUDA out of memory"
    assert result.model_id == "big/model"
    assert result.n_output_tokens == 0
