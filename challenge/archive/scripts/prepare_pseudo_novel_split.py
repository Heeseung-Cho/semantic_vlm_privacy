#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/support/query-eval COCO files for a pseudo-novel split.")
    parser.add_argument("--base-annotations", required=True, help="Path to base_annotations.json")
    parser.add_argument("--split-config", required=True, help="Path to pseudo_novel_3split.json")
    parser.add_argument("--split-name", required=True, help="Split name, e.g. split1")
    parser.add_argument("--output-dir", required=True, help="Output directory for the prepared split")
    parser.add_argument(
        "--support-policy",
        choices=["largest"],
        default="largest",
        help="How to choose the single support instance for each pseudo-novel category.",
    )
    return parser.parse_args()


def filter_coco(
    images: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    keep_image_ids: set[int],
    keep_category_ids: set[int] | None = None,
) -> dict[str, Any]:
    image_ids = set(keep_image_ids)
    kept_images = [img for img in images if img["id"] in image_ids]
    kept_annotations = []
    for ann in annotations:
        if ann["image_id"] not in image_ids:
            continue
        if keep_category_ids is not None and ann["category_id"] not in keep_category_ids:
            continue
        normalized = dict(ann)
        normalized.setdefault("iscrowd", 0)
        normalized.setdefault("ignore", 0)
        kept_annotations.append(normalized)
    if keep_category_ids is None:
        used_category_ids = {ann["category_id"] for ann in kept_annotations}
    else:
        used_category_ids = set(keep_category_ids)
    kept_categories = [cat for cat in categories if cat["id"] in used_category_ids]
    return {
        "images": kept_images,
        "annotations": kept_annotations,
        "categories": kept_categories,
    }


def choose_support_annotations(
    annotations: list[dict[str, Any]],
    image_by_id: dict[int, dict[str, Any]],
    pseudo_ids: list[int],
) -> list[dict[str, Any]]:
    anns_by_cat = defaultdict(list)
    for ann in annotations:
        if ann["category_id"] in pseudo_ids:
            anns_by_cat[ann["category_id"]].append(ann)

    selected = []
    used_image_ids: set[int] = set()
    for cat_id in pseudo_ids:
        candidates = anns_by_cat[cat_id]
        if not candidates:
            raise ValueError(f"No annotation found for pseudo-novel category {cat_id}")
        ranked = sorted(
            candidates,
            key=lambda ann: (
                ann["image_id"] in used_image_ids,
                -float(ann.get("area", 0.0)),
                ann["image_id"],
            ),
        )
        chosen = ranked[0]
        selected.append(chosen)
        used_image_ids.add(chosen["image_id"])
    return selected


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.base_annotations).read_text())
    split_cfg = json.loads(Path(args.split_config).read_text())
    split = split_cfg["folds"][args.split_name]

    pseudo_ids = list(split["pseudo_novel_ids"])
    pseudo_set = set(pseudo_ids)

    images = base["images"]
    categories = base["categories"]
    annotations = base["annotations"]
    image_by_id = {img["id"]: img for img in images}

    img_to_cats = defaultdict(set)
    anns_by_img = defaultdict(list)
    for ann in annotations:
        img_to_cats[ann["image_id"]].add(ann["category_id"])
        anns_by_img[ann["image_id"]].append(ann)

    pseudo_image_ids = {img_id for img_id, cats in img_to_cats.items() if cats & pseudo_set}
    train_image_ids = set(img_to_cats.keys()) - pseudo_image_ids

    support_annotations = choose_support_annotations(annotations, image_by_id, pseudo_ids)
    support_annotations = [
        {**ann, "iscrowd": ann.get("iscrowd", 0), "ignore": ann.get("ignore", 0)}
        for ann in support_annotations
    ]
    support_image_ids = {ann["image_id"] for ann in support_annotations}
    query_eval_image_ids = pseudo_image_ids - support_image_ids

    train_coco = filter_coco(images, annotations, categories, train_image_ids)
    support_categories = [cat for cat in categories if cat["id"] in pseudo_set]
    support_coco = {
        "images": [img for img in images if img["id"] in support_image_ids],
        "annotations": support_annotations,
        "categories": support_categories,
    }

    support_ann_ids = {ann["id"] for ann in support_annotations}
    query_annotations = []
    for ann in annotations:
        if ann["image_id"] not in query_eval_image_ids:
            continue
        if ann["category_id"] not in pseudo_set:
            continue
        if ann["id"] in support_ann_ids:
            continue
        query_annotations.append({**ann, "iscrowd": ann.get("iscrowd", 0), "ignore": ann.get("ignore", 0)})
    query_coco = {
        "images": [img for img in images if img["id"] in query_eval_image_ids],
        "annotations": query_annotations,
        "categories": [cat for cat in categories if cat["id"] in pseudo_set],
    }

    meta = {
        "split_name": args.split_name,
        "pseudo_novel_ids": pseudo_ids,
        "private_like_ids": split.get("private_like_ids", []),
        "train_images": len(train_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "support_images": len(support_coco["images"]),
        "support_annotations": len(support_coco["annotations"]),
        "query_eval_images": len(query_coco["images"]),
        "query_eval_annotations": len(query_coco["annotations"]),
        "contaminated_image_policy": "exclude any image containing pseudo-novel category from train",
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_base.json").write_text(json.dumps(train_coco, ensure_ascii=False, indent=2))
    (out_dir / "support_1shot.json").write_text(json.dumps(support_coco, ensure_ascii=False, indent=2))
    (out_dir / "query_eval.json").write_text(json.dumps(query_coco, ensure_ascii=False, indent=2))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
