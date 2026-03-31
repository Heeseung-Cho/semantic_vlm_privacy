#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are given a support image and a query image. "
    "Find the object in the query image that belongs to the same category as the object shown in the support image. "
    "Return only native grounding text. "
    "If present, output <ref>category name</ref><box>(x1,y1),(x2,y2)</box> with coordinates normalized to [0,1000]. "
    "If the support-category object is absent in the query image, return <ref>category name</ref><box>none</box>."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare support-query grounding examples from VizWiz base annotations for Track B."
    )
    parser.add_argument("--base-annotations", required=True, help="Path to base_annotations.json")
    parser.add_argument("--image-dir", required=True, help="Directory containing the base images")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--examples-per-category", type=int, default=20, help="Max positive examples per category")
    parser.add_argument("--min-area-ratio", type=float, default=0.0, help="Minimum normalized bbox area ratio for positive examples")
    parser.add_argument("--max-area-ratio", type=float, default=1.0, help="Maximum normalized bbox area ratio for positive examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-config", default=None, help="Optional pseudo_novel_3split.json for train/eval partitioning")
    parser.add_argument("--split-name", default=None, help="Optional split name when using split-config")
    return parser.parse_args()


def normalize_category_name(name: str) -> str:
    return name.replace("_", " ").strip()


def bbox_xywh_to_xyxy_norm(bbox: list[float], width: float, height: float) -> list[float]:
    x, y, w, h = bbox
    x1 = x / width
    y1 = y / height
    x2 = (x + w) / width
    y2 = (y + h) / height
    return [round(v, 6) for v in [x1, y1, x2, y2]]


def bbox_xywh_to_xyxy_1000(bbox: list[float], width: float, height: float) -> list[int]:
    x1, y1, x2, y2 = bbox_xywh_to_xyxy_norm(bbox, width, height)
    return [int(round(v * 1000)) for v in [x1, y1, x2, y2]]


def make_grounding_text(category_name: str, exists: bool, bbox_1000: list[int] | None) -> str:
    if not exists or bbox_1000 is None:
        return f"<ref>{category_name}</ref><box>none</box>"
    x1, y1, x2, y2 = bbox_1000
    return f"<ref>{category_name}</ref><box>({x1},{y1}),({x2},{y2})</box>"


def bbox_area_ratio_xywh(bbox: list[float], width: float, height: float) -> float:
    _, _, w, h = bbox
    return max(0.0, w / width) * max(0.0, h / height)


def make_example(
    support_image: dict[str, Any],
    query_image: dict[str, Any],
    ann: dict[str, Any],
    category_name: str,
    image_dir: Path,
    example_type: str,
) -> dict[str, Any]:
    if example_type == "positive":
        bbox_xyxy_norm = bbox_xywh_to_xyxy_norm(
            ann["bbox"],
            width=query_image["width"],
            height=query_image["height"],
        )
        bbox_xyxy_1000 = bbox_xywh_to_xyxy_1000(
            ann["bbox"],
            width=query_image["width"],
            height=query_image["height"],
        )
        target = {
            "exists": True,
            "bbox_xyxy_norm": bbox_xyxy_norm,
            "bbox_xyxy_1000": bbox_xyxy_1000,
        }
    else:
        target = {
            "exists": False,
            "bbox_xyxy_norm": None,
            "bbox_xyxy_1000": None,
        }

    assistant_content = make_grounding_text(
        category_name=category_name,
        exists=target["exists"],
        bbox_1000=target["bbox_xyxy_1000"],
    )

    return {
        "example_type": example_type,
        "category_name": category_name,
        "support_image_id": support_image["id"],
        "support_file_name": support_image["file_name"],
        "query_image_id": query_image["id"],
        "query_file_name": query_image["file_name"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Support image: <image>\n"
                    "Query image: <image>\n"
                    "Find the object in the query image that matches the category shown in the support image.\n"
                    "Return only native grounding text."
                ),
            },
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [
            str((image_dir / support_image["file_name"]).resolve()),
            str((image_dir / query_image["file_name"]).resolve()),
        ],
        "target": target,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    base = json.loads(Path(args.base_annotations).read_text())
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_by_id = {img["id"]: img for img in base["images"]}
    categories_by_id = {cat["id"]: normalize_category_name(cat["name"]) for cat in base["categories"]}

    pseudo_novel_ids: set[int] = set()
    if args.split_config and args.split_name:
        split_cfg = json.loads(Path(args.split_config).read_text())
        pseudo_novel_ids = set(split_cfg["folds"][args.split_name]["pseudo_novel_ids"])

    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in base["annotations"]:
        anns_by_cat[ann["category_id"]].append(ann)
        anns_by_img[ann["image_id"]].append(ann)

    train_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for category_id, anns in sorted(anns_by_cat.items()):
        if len(anns) < 2:
            continue
        category_name = categories_by_id[category_id]
        category_anns = sorted(anns, key=lambda ann: (-float(ann.get("area", 0.0)), ann["image_id"], ann["id"]))
        support_ann = category_anns[0]
        support_image = images_by_id[support_ann["image_id"]]
        remaining = [ann for ann in category_anns[1:] if ann["image_id"] != support_image["id"]]
        if not remaining:
            continue

        filtered_remaining = []
        for ann in remaining:
            query_image = images_by_id[ann["image_id"]]
            area_ratio = bbox_area_ratio_xywh(
                ann["bbox"],
                width=query_image["width"],
                height=query_image["height"],
            )
            if area_ratio < args.min_area_ratio:
                continue
            if area_ratio > args.max_area_ratio:
                continue
            filtered_remaining.append(ann)
        positive_candidates = filtered_remaining[: args.examples_per_category]
        if not positive_candidates:
            continue
        negative_pool = [
            img for img in base["images"]
            if img["id"] != support_image["id"] and all(a["category_id"] != category_id for a in anns_by_img[img["id"]])
        ]
        rng.shuffle(negative_pool)
        negative_candidates = negative_pool[: min(len(positive_candidates), len(negative_pool))]

        target_bucket = eval_examples if category_id in pseudo_novel_ids else train_examples
        for ann in positive_candidates:
            query_image = images_by_id[ann["image_id"]]
            target_bucket.append(
                make_example(
                    support_image=support_image,
                    query_image=query_image,
                    ann=ann,
                    category_name=category_name,
                    image_dir=image_dir,
                    example_type="positive",
                )
            )
        for query_image in negative_candidates:
            target_bucket.append(
                make_example(
                    support_image=support_image,
                    query_image=query_image,
                    ann=support_ann,
                    category_name=category_name,
                    image_dir=image_dir,
                    example_type="negative",
                )
            )

        meta_rows.append(
            {
                "category_id": category_id,
                "category_name": category_name,
                "support_image_id": support_image["id"],
                "support_file_name": support_image["file_name"],
                "positive_examples": len(positive_candidates),
                "negative_examples": len(negative_candidates),
                "bucket": "eval" if category_id in pseudo_novel_ids else "train",
            }
        )

    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"
    meta_path = out_dir / "meta.json"

    with train_path.open("w", encoding="utf-8") as handle:
        for row in train_examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as handle:
        for row in eval_examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "base_annotations": str(Path(args.base_annotations).resolve()),
        "image_dir": str(image_dir.resolve()),
        "examples_per_category": args.examples_per_category,
        "min_area_ratio": args.min_area_ratio,
        "max_area_ratio": args.max_area_ratio,
        "seed": args.seed,
        "split_config": str(Path(args.split_config).resolve()) if args.split_config else None,
        "split_name": args.split_name,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "categories": meta_rows,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
