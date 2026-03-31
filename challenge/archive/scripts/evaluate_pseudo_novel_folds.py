#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate purge-aware pseudo-novel fold proposals.")
    parser.add_argument("--base-annotations", required=True, help="Path to base_annotations.json")
    parser.add_argument("--fold-config", required=True, help="Path to fold proposal JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = json.loads(Path(args.base_annotations).read_text())
    fold_cfg = json.loads(Path(args.fold_config).read_text())

    cat_id_to_name = {c["id"]: c["name"] for c in base["categories"]}
    img_to_cats = defaultdict(set)
    cat_to_imgs = defaultdict(set)
    cat_to_instances = Counter()
    for ann in base["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        img_to_cats[img_id].add(cat_id)
        cat_to_imgs[cat_id].add(img_id)
        cat_to_instances[cat_id] += 1

    total_images = len(base["images"])
    total_instances = len(base["annotations"])
    print(f"total_images={total_images}")
    print(f"total_instances={total_instances}")
    print()

    for fold_name, fold in fold_cfg["folds"].items():
        pseudo_ids = fold["pseudo_novel_ids"]
        pseudo_set = set(pseudo_ids)
        pseudo_img_ids = {img_id for img_id, cats in img_to_cats.items() if cats & pseudo_set}
        train_img_ids = set(img_to_cats.keys()) - pseudo_img_ids
        pseudo_instances = sum(cat_to_instances[cat_id] for cat_id in pseudo_ids)
        private_like_ids = set(fold.get("private_like_ids", []))

        pseudo_multi = sum(1 for img_id in pseudo_img_ids if len(img_to_cats[img_id]) > 1)
        train_multi = sum(1 for img_id in train_img_ids if len(img_to_cats[img_id]) > 1)
        train_instances = sum(
            1 for ann in base["annotations"] if ann["image_id"] in train_img_ids
        )

        print(f"[{fold_name}]")
        print(f"pseudo_novel_categories={len(pseudo_ids)}")
        print(f"private_like_pseudo={len(private_like_ids)}")
        print(f"pseudo_novel_images={len(pseudo_img_ids)}")
        print(f"pseudo_novel_instances={pseudo_instances}")
        print(f"pseudo_multi_ratio={pseudo_multi / len(pseudo_img_ids):.3f}" if pseudo_img_ids else "pseudo_multi_ratio=0.000")
        print(f"train_images_after_purge={len(train_img_ids)}")
        print(f"train_instances_after_purge={train_instances}")
        print(f"train_multi_ratio={train_multi / len(train_img_ids):.3f}" if train_img_ids else "train_multi_ratio=0.000")
        print("private_like_names=" + ", ".join(cat_id_to_name[c] for c in sorted(private_like_ids)))
        print("pseudo_names=" + ", ".join(cat_id_to_name[c] for c in pseudo_ids))
        print()


if __name__ == "__main__":
    main()
