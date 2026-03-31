#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Track B Qwen support-query grounding SFT with swift.")
    parser.add_argument("--model", required=True, help="Local Qwen model path")
    parser.add_argument("--train-jsonl", required=True, help="Track B train.jsonl path")
    parser.add_argument("--eval-jsonl", required=True, help="Track B eval.jsonl path")
    parser.add_argument("--output-dir", default=None, help="Output directory for swift checkpoints")
    parser.add_argument("--train-type", default="lora", choices=["lora", "full"], help="Training type for swift")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--truncation-strategy", default="right")
    parser.add_argument("--load-best-model-at-end", default="true")
    parser.add_argument("--metric-for-best-model", default="eval_loss")
    parser.add_argument("--greater-is-better", default="false")
    parser.add_argument("--save-only-model", default="false")
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--early-stop-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=448)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-impl", default="flash_attn")
    parser.add_argument("--system", default=None, help="Optional override system prompt")
    parser.add_argument("--dry-run", action="store_true", help="Print the swift command without running it")
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    now = datetime.now()
    return (
        Path(__file__).resolve().parents[2] / 'results' / 'track_b_qwen_sft'
        / now.strftime("%Y%m%d")
        / now.strftime("%H%M%S")
    )


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [
        "swift",
        "sft",
        "--model",
        str(Path(args.model).resolve()),
        "--model_name",
        str(Path(args.model).resolve()),
        "--dataset",
        str(Path(args.train_jsonl).resolve()),
        "--val_dataset",
        str(Path(args.eval_jsonl).resolve()),
        "--output_dir",
        str(output_dir),
        "--use_hf",
        "true",
        "--train_type",
        args.train_type,
        "--tuner_backend",
        "peft",
        "--max_steps",
        str(args.max_steps),
        "--max_length",
        str(args.max_length),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--lora_rank",
        str(args.lora_rank),
        "--lora_alpha",
        str(args.lora_alpha),
        "--target_modules",
        args.target_modules,
        "--save_strategy",
        "steps",
        "--eval_strategy",
        "steps",
        "--save_steps",
        str(args.save_steps),
        "--eval_steps",
        str(args.eval_steps),
        "--logging_steps",
        str(args.logging_steps),
        "--save_total_limit",
        str(args.save_total_limit),
        "--warmup_ratio",
        str(args.warmup_ratio),
        "--dataloader_num_workers",
        str(args.dataloader_num_workers),
        "--seed",
        str(args.seed),
        "--torch_dtype",
        args.torch_dtype,
        "--truncation_strategy",
        args.truncation_strategy,
        "--max_pixels",
        str(args.max_pixels),
        "--attn_impl",
        args.attn_impl,
        "--load_best_model_at_end",
        args.load_best_model_at_end,
        "--metric_for_best_model",
        args.metric_for_best_model,
        "--greater_is_better",
        args.greater_is_better,
        "--save_only_model",
        args.save_only_model,
        "--report_to",
        args.report_to,
        "--run_name",
        args.run_name or Path(args.model).name,
        "--early_stop_interval",
        str(args.early_stop_interval),
        "--split_dataset_ratio",
        "0",
    ]
    if args.system is not None:
        cmd.extend(["--system", args.system])
    return cmd


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args, output_dir)

    print("Track B swift SFT command:")
    print(shlex.join(cmd))
    print(f"Output dir: {output_dir}")

    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
