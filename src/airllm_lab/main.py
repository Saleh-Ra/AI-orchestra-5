"""Thin CLI entry point.

Contains no business logic; it only wires user commands to the SDK facade.
Excluded from coverage by design (see pyproject ``coverage`` config).
"""

from __future__ import annotations

import argparse
import json

from airllm_lab.sdk.sdk import LabSDK


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="airllm-lab", description="AirLLM Lab CLI")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("version", help="print the version")
    sub.add_parser("hardware", help="probe + save the hardware spec to results/")
    sub.add_parser("smoke", help="run the tiny smoke-test model end-to-end")
    dl = sub.add_parser("download", help="download a model to the D: cache")
    dl.add_argument("model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    bl = sub.add_parser("baseline", help="run a baseline generation on a model")
    bl.add_argument("model", help="HF model id, e.g. Qwen/Qwen2.5-7B-Instruct")
    air = sub.add_parser("airllm", help="run layered AirLLM inference on a model")
    air.add_argument("model", help="local model dir or HF id (must have a sharded index)")
    air.add_argument("--quant", default="fp16", choices=["fp16", "8bit", "4bit"])
    air.add_argument("--max-tokens", type=int, default=20, help="cap on generated tokens")
    bench = sub.add_parser("benchmark", help="benchmark a model over N repeats")
    bench.add_argument("model", help="local model dir or HF id")
    bench.add_argument("--mode", default="airllm", choices=["airllm", "baseline"])
    bench.add_argument("--quant", default="fp16", choices=["fp16", "8bit", "4bit"])
    bench.add_argument("--repeats", type=int, default=1)
    bench.add_argument("--warmup", type=int, default=0)
    bench.add_argument("--max-tokens", type=int, default=20, help="cap on generated tokens")
    cost = sub.add_parser("cost", help="compute the OnPrem vs API vs Cloud cost report")
    cost.add_argument("--runtime", type=float, default=None, help="measured local s per request")
    sub.add_parser("charts", help="render figures from stored results into assets/")
    sub.add_parser("roofline", help="render the roofline figure (original extension)")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch to the SDK."""
    args = build_parser().parse_args(argv)
    sdk = LabSDK()
    if args.command == "hardware":
        print(json.dumps(sdk.hardware_report().to_dict(), indent=2))
    elif args.command == "smoke":
        print(json.dumps(sdk.run_smoke().to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "download":
        print(sdk.download_model(args.model))
    elif args.command == "baseline":
        print(json.dumps(sdk.run_baseline_model(args.model).to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "airllm":
        result = sdk.run_airllm(args.model, quant=args.quant, max_cap=args.max_tokens)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "benchmark":
        summary = sdk.run_benchmark(
            args.model, mode=args.mode, quant=args.quant,
            repeats=args.repeats, warmup=args.warmup, max_cap=args.max_tokens,
        )
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
    elif args.command == "cost":
        print(json.dumps(sdk.run_cost_analysis(runtime_s=args.runtime).to_dict(),
                         indent=2, ensure_ascii=False))
    elif args.command == "charts":
        print("\n".join(sdk.generate_charts()))
    elif args.command == "roofline":
        print(sdk.generate_roofline() or "no run data for roofline")
    else:
        print(f"airllm-lab v{sdk.version}")


if __name__ == "__main__":
    main()
