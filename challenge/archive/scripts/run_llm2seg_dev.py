#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM2Seg baseline on the 158-image dev metadata split.")
    parser.add_argument("--query-dir", required=True, help="Path to query_images")
    parser.add_argument("--support-dir", required=True, help="Path to support_images")
    parser.add_argument("--support-json", required=True, help="Path to support_set.json")
    parser.add_argument("--json-path", required=True, help="Path to dev_set_images_info.json")
    parser.add_argument("--output-dir", required=True, help="Root directory for inference outputs")
    parser.add_argument("--config-path", required=True, help="Grounding DINO config path")
    parser.add_argument("--checkpoint-path", required=True, help="Trained Grounding DINO checkpoint")
    parser.add_argument("--sam-checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--llm-model", required=True, help="Local Qwen model directory")
    parser.add_argument("--date-tag", default=None, help="Optional date tag override")
    parser.add_argument("--run-id", default=None, help="Optional run id override")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for quick tests")
    parser.add_argument("--device", default="cuda", help="Inference device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    llm2seg_dir = Path(__import__('os').environ.get('LLM2SEG_DIR', repo_root / 'third_party' / 'LLM2Seg'))
    main_py = llm2seg_dir / 'main.py'
    cmd = [
        sys.executable,
        str(main_py),
        "--query_dir",
        args.query_dir,
        "--support_dir",
        args.support_dir,
        "--support_json",
        args.support_json,
        "--json_path",
        args.json_path,
        "--output_dir",
        args.output_dir,
        "--config_path",
        args.config_path,
        "--checkpoint_path",
        args.checkpoint_path,
        "--sam_checkpoint",
        args.sam_checkpoint,
        "--method",
        "qwen",
        "--llm-model",
        args.llm_model,
        "--device",
        args.device,
    ]
    if args.date_tag:
        cmd.extend(["--date-tag", args.date_tag])
    if args.run_id:
        cmd.extend(["--run-id", args.run_id])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
