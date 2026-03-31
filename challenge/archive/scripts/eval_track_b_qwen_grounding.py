#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM2SEG_DIR = Path(os.environ.get('LLM2SEG_DIR', REPO_ROOT / 'third_party' / 'LLM2Seg'))
if str(LLM2SEG_DIR) not in sys.path:
    sys.path.insert(0, str(LLM2SEG_DIR))

from call_vlm import SwiftVLMCaller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Track B Qwen grounding checkpoint on eval.jsonl.")
    parser.add_argument("--model", required=True, help="Local Qwen model path")
    parser.add_argument("--eval-jsonl", required=True, help="Track B eval.jsonl path")
    parser.add_argument("--output-dir", default=None, help="Directory for eval outputs")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA checkpoint path")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=448)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    now = datetime.now()
    return (
        REPO_ROOT / 'results' / 'track_b_qwen_eval'
        / now.strftime("%Y%m%d")
        / now.strftime("%H%M%S")
    )


BOX_PATTERN = re.compile(r"<box>\s*\((\d+)\s*,\s*(\d+)\)\s*,\s*\((\d+)\s*,\s*(\d+)\)\s*</box>", re.IGNORECASE)
NONE_PATTERN = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)


def parse_grounding_output(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if NONE_PATTERN.search(text):
        return {"exists": False, "bbox_xyxy_norm": None}
    match = BOX_PATTERN.search(text)
    if not match:
        return None
    x1, y1, x2, y2 = [int(v) for v in match.groups()]
    bbox_xyxy_norm = [round(v / 1000.0, 6) for v in [x1, y1, x2, y2]]
    return {"exists": True, "bbox_xyxy_norm": bbox_xyxy_norm}


def xyxy_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def is_valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def main() -> None:
    args = parse_args()
    out_dir = resolve_output_dir(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with Path(args.eval_jsonl).open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if args.limit is not None:
        rows = rows[: args.limit]

    caller = SwiftVLMCaller(
        model_path=str(Path(args.model).resolve()),
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
        lora_path=args.lora_path,
    )

    records = []
    parse_ok = 0
    exists_acc = 0
    positive_rows = 0
    positive_iou_sum = 0.0
    positive_hit50 = 0

    for idx, row in enumerate(rows, start=1):
        response = caller.generate_images(
            row["images"],
            instruction=f"{row['messages'][0]['content']}\n\n{row['messages'][1]['content']}",
        )
        parsed = parse_grounding_output(response)
        target = row["target"]
        target_exists = bool(target["exists"])
        pred_exists = None
        pred_box = None
        iou = None

        if parsed is not None:
            parse_ok += 1
            pred_exists = bool(parsed.get("exists"))
            pred_box = parsed.get("bbox_xyxy_norm")
            if pred_exists == target_exists:
                exists_acc += 1
            if target_exists:
                positive_rows += 1
                if pred_exists and is_valid_box(pred_box):
                    iou = xyxy_iou(pred_box, target["bbox_xyxy_norm"])
                else:
                    iou = 0.0
                positive_iou_sum += iou
                if iou >= 0.5:
                    positive_hit50 += 1
        elif target_exists:
            positive_rows += 1
            positive_iou_sum += 0.0

        records.append(
            {
                "index": idx,
                "example_type": row["example_type"],
                "category_name": row["category_name"],
                "support_image_id": row["support_image_id"],
                "query_image_id": row["query_image_id"],
                "target": target,
                "response_raw": response,
                "response_parsed": parsed,
                "pred_exists": pred_exists,
                "pred_box": pred_box,
                "iou": iou,
            }
        )

    total = len(rows)
    summary = {
        "model": str(Path(args.model).resolve()),
        "lora_path": str(Path(args.lora_path).resolve()) if args.lora_path else None,
        "eval_jsonl": str(Path(args.eval_jsonl).resolve()),
        "num_examples": total,
        "parse_success_rate": parse_ok / total if total else 0.0,
        "exists_accuracy": exists_acc / total if total else 0.0,
        "positive_rows": positive_rows,
        "positive_mean_iou": positive_iou_sum / positive_rows if positive_rows else 0.0,
        "positive_hit50": positive_hit50 / positive_rows if positive_rows else 0.0,
    }

    (out_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
