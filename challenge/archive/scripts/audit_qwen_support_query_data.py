#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Track B support-query grounding JSONL files."
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to train.jsonl")
    parser.add_argument("--eval-jsonl", required=True, help="Path to eval.jsonl")
    parser.add_argument("--output-json", default=None, help="Optional output path for the audit summary")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def bbox_area_ratio(example: dict[str, Any]) -> float | None:
    target = example["target"]
    bbox = target.get("bbox_xyxy_norm")
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(row["example_type"] for row in rows)
    positive_ratios = []
    per_category = defaultdict(lambda: Counter())
    large_examples = []
    tiny_examples = []

    for row in rows:
        per_category[row["category_name"]][row["example_type"]] += 1
        ratio = bbox_area_ratio(row)
        if ratio is None:
            continue
        positive_ratios.append(ratio)
        if ratio >= 0.6:
            large_examples.append(
                {
                    "category_name": row["category_name"],
                    "query_file_name": row["query_file_name"],
                    "area_ratio": round(ratio, 4),
                }
            )
        if ratio <= 0.02:
            tiny_examples.append(
                {
                    "category_name": row["category_name"],
                    "query_file_name": row["query_file_name"],
                    "area_ratio": round(ratio, 4),
                }
            )

    positive_ratios_sorted = sorted(positive_ratios)
    def percentile(p: float) -> float:
        if not positive_ratios_sorted:
            return 0.0
        idx = min(len(positive_ratios_sorted) - 1, int(round((len(positive_ratios_sorted) - 1) * p)))
        return positive_ratios_sorted[idx]

    return {
        "num_examples": len(rows),
        "example_type_counts": dict(type_counts),
        "positive_bbox_area_ratio": {
            "count": len(positive_ratios),
            "mean": round(sum(positive_ratios) / len(positive_ratios), 4) if positive_ratios else 0.0,
            "p10": round(percentile(0.10), 4),
            "p50": round(percentile(0.50), 4),
            "p90": round(percentile(0.90), 4),
        },
        "categories": {
            category_name: dict(counts)
            for category_name, counts in sorted(per_category.items())
        },
        "large_bbox_examples": large_examples[:20],
        "tiny_bbox_examples": tiny_examples[:20],
        "num_large_bbox_examples": len(large_examples),
        "num_tiny_bbox_examples": len(tiny_examples),
    }


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(Path(args.train_jsonl))
    eval_rows = load_jsonl(Path(args.eval_jsonl))

    summary = {
        "train": summarize_split(train_rows),
        "eval": summarize_split(eval_rows),
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
