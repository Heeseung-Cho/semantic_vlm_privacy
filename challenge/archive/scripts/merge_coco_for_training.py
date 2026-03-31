#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge base-train COCO and augmented-support COCO into one training dataset.")
    parser.add_argument("--base-ann", required=True, help="Path to train_base.json")
    parser.add_argument("--base-image-dir", required=True, help="Directory for base train images")
    parser.add_argument("--support-ann", required=True, help="Path to augmented support annotations json")
    parser.add_argument("--support-image-dir", required=True, help="Directory for augmented support images")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged dataset")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    base = load_json(args.base_ann)
    support = load_json(args.support_ann)
    base_image_dir = Path(args.base_image_dir).resolve()
    support_image_dir = Path(args.support_image_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    img_out_dir = out_dir / "images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    categories = {cat["id"]: cat for cat in base["categories"]}
    for cat in support["categories"]:
        categories.setdefault(cat["id"], cat)

    merged_images = []
    merged_annotations = []
    next_image_id = 1
    next_ann_id = 1
    image_id_map: dict[tuple[str, int], int] = {}

    def add_dataset(payload: dict[str, Any], source: str, image_dir: Path) -> None:
        nonlocal next_image_id, next_ann_id
        for img in payload["images"]:
            new_file_name = f"{source}__{img['file_name']}"
            src_path = image_dir / img["file_name"]
            dst_path = img_out_dir / new_file_name
            safe_link(src_path, dst_path)
            new_img = dict(img)
            new_img["id"] = next_image_id
            new_img["file_name"] = new_file_name
            merged_images.append(new_img)
            image_id_map[(source, img["id"])] = next_image_id
            next_image_id += 1

        for ann in payload["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[(source, ann["image_id"])]
            new_ann.setdefault("iscrowd", 0)
            new_ann.setdefault("ignore", 0)
            merged_annotations.append(new_ann)
            next_ann_id += 1

    add_dataset(base, "base", base_image_dir)
    add_dataset(support, "support", support_image_dir)

    merged = {
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": [categories[k] for k in sorted(categories)],
    }
    ann_out = out_dir / "annotations.json"
    ann_out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    print(f"merged_images={len(merged_images)}")
    print(f"merged_annotations={len(merged_annotations)}")
    print(f"categories={len(merged['categories'])}")
    print(f"annotations_path={ann_out}")


if __name__ == "__main__":
    main()
