#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare image-level K-fold COCO splits for VizWiz base detector optimization."
    )
    parser.add_argument("--base-annotations", required=True, help="Path to base_annotations.json")
    parser.add_argument("--output-dir", required=True, help="Directory to write fold subdirectories")
    parser.add_argument("--num-folds", type=int, default=3, help="Number of image-level folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie-breaking")
    return parser.parse_args()


def filter_coco(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    keep_image_ids: set[int],
) -> dict[str, Any]:
    image_ids = set(keep_image_ids)
    kept_images = [img for img in images if img["id"] in image_ids]
    kept_annotations = []
    next_ann_id = 1
    for ann in annotations:
        if ann["image_id"] not in image_ids:
            continue
        normalized = dict(ann)
        normalized["id"] = next_ann_id
        next_ann_id += 1
        normalized.setdefault("iscrowd", 0)
        normalized.setdefault("ignore", 0)
        kept_annotations.append(normalized)
    used_category_ids = {ann["category_id"] for ann in kept_annotations}
    kept_categories = [cat for cat in categories if cat["id"] in used_category_ids]
    return {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": kept_categories,
    }


def score_fold_assignment(
    fold_index: int,
    image_categories: set[int],
    fold_sizes: list[int],
    fold_cat_counts: list[Counter],
    target_fold_size: float,
) -> tuple[float, int, int]:
    current_size = fold_sizes[fold_index]
    size_penalty = abs((current_size + 1) - target_fold_size)
    rarity_penalty = sum(fold_cat_counts[fold_index][cat_id] for cat_id in image_categories)
    overlap_penalty = sum(1 for cat_id in image_categories if fold_cat_counts[fold_index][cat_id] > 0)
    return (rarity_penalty, overlap_penalty, int(size_penalty * 1000))


def build_balanced_folds(base: dict[str, Any], num_folds: int, seed: int) -> list[list[int]]:
    annotations = base["annotations"]
    images = base["images"]

    image_to_categories: dict[int, set[int]] = defaultdict(set)
    image_to_ann_count: Counter[int] = Counter()
    category_to_images: dict[int, set[int]] = defaultdict(set)
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        image_to_categories[image_id].add(category_id)
        image_to_ann_count[image_id] += 1
        category_to_images[category_id].add(image_id)

    rng = random.Random(seed)
    image_ids = [img["id"] for img in images]
    image_ids.sort(
        key=lambda image_id: (
            -len(image_to_categories[image_id]),
            -sum(len(category_to_images[cat_id]) for cat_id in image_to_categories[image_id]),
            -image_to_ann_count[image_id],
            rng.random(),
        )
    )

    fold_image_ids: list[list[int]] = [[] for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]
    fold_cat_counts = [Counter() for _ in range(num_folds)]
    target_fold_size = len(image_ids) / num_folds

    for image_id in image_ids:
        categories = image_to_categories[image_id]
        candidate_order = list(range(num_folds))
        rng.shuffle(candidate_order)
        best_fold = min(
            candidate_order,
            key=lambda fold_idx: score_fold_assignment(
                fold_idx,
                categories,
                fold_sizes,
                fold_cat_counts,
                target_fold_size,
            ),
        )
        fold_image_ids[best_fold].append(image_id)
        fold_sizes[best_fold] += 1
        for category_id in categories:
            fold_cat_counts[best_fold][category_id] += 1

    return fold_image_ids


def summarize_fold(
    fold_name: str,
    train_coco: dict[str, Any],
    val_coco: dict[str, Any],
) -> dict[str, Any]:
    train_cat_image_counts: dict[int, set[int]] = defaultdict(set)
    val_cat_image_counts: dict[int, set[int]] = defaultdict(set)
    for ann in train_coco["annotations"]:
        train_cat_image_counts[ann["category_id"]].add(ann["image_id"])
    for ann in val_coco["annotations"]:
        val_cat_image_counts[ann["category_id"]].add(ann["image_id"])

    train_min = min((len(v) for v in train_cat_image_counts.values()), default=0)
    val_min = min((len(v) for v in val_cat_image_counts.values()), default=0)
    train_mean = (
        sum(len(v) for v in train_cat_image_counts.values()) / len(train_cat_image_counts)
        if train_cat_image_counts
        else 0.0
    )
    val_mean = (
        sum(len(v) for v in val_cat_image_counts.values()) / len(val_cat_image_counts)
        if val_cat_image_counts
        else 0.0
    )
    return {
        "fold_name": fold_name,
        "train_images": len(train_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "val_images": len(val_coco["images"]),
        "val_annotations": len(val_coco["annotations"]),
        "train_categories": len(train_coco["categories"]),
        "val_categories": len(val_coco["categories"]),
        "train_min_images_per_category": train_min,
        "val_min_images_per_category": val_min,
        "train_mean_images_per_category": round(train_mean, 2),
        "val_mean_images_per_category": round(val_mean, 2),
    }


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.base_annotations).read_text())

    fold_image_ids = build_balanced_folds(base, num_folds=args.num_folds, seed=args.seed)
    all_image_ids = {img["id"] for img in base["images"]}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "base_annotations": str(Path(args.base_annotations).resolve()),
        "num_folds": args.num_folds,
        "seed": args.seed,
        "folds": [],
    }

    for fold_idx, val_ids_list in enumerate(fold_image_ids, start=1):
        fold_name = f"fold{fold_idx}"
        val_ids = set(val_ids_list)
        train_ids = all_image_ids - val_ids
        train_coco = filter_coco(base["images"], base["annotations"], base["categories"], train_ids)
        val_coco = filter_coco(base["images"], base["annotations"], base["categories"], val_ids)

        fold_dir = out_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "train.json").write_text(json.dumps(train_coco, ensure_ascii=False, indent=2))
        (fold_dir / "val.json").write_text(json.dumps(val_coco, ensure_ascii=False, indent=2))

        meta = summarize_fold(fold_name, train_coco, val_coco)
        meta["val_image_ids"] = sorted(val_ids)
        (fold_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        summary["folds"].append(meta)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
