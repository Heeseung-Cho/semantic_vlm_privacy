#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LLM2Seg-style augmented support set.")
    parser.add_argument("--support-json", required=True, help="Path to support_set.json")
    parser.add_argument("--support-dir", required=True, help="Directory containing support images")
    parser.add_argument("--output-dir", required=True, help="Output directory for augmented images/json")
    parser.add_argument("--augmentations-per-image", type=int, default=32, help="Number of synthetic variants per support image")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG save quality")
    return parser.parse_args()


def polygon_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in segmentation:
        pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def mask_to_polygons(mask: np.ndarray) -> tuple[list[list[float]], list[int], float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    x1 = y1 = None
    x2 = y2 = None
    area = 0.0
    for contour in contours:
        if contour.shape[0] < 3:
            continue
        contour = contour.reshape(-1, 2)
        segmentation.append(contour.astype(float).flatten().tolist())
        bx, by, bw, bh = cv2.boundingRect(contour.astype(np.int32))
        area += float(cv2.contourArea(contour.astype(np.float32)))
        x1 = bx if x1 is None else min(x1, bx)
        y1 = by if y1 is None else min(y1, by)
        x2 = bx + bw if x2 is None else max(x2, bx + bw)
        y2 = by + bh if y2 is None else max(y2, by + bh)
    if not segmentation or x1 is None:
        return [], [0, 0, 0, 0], 0.0
    return segmentation, [int(x1), int(y1), int(x2 - x1), int(y2 - y1)], area


def apply_affine(image: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    warped_image = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped_image, warped_mask


def random_translate_rotate(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    angle = rng.uniform(-20.0, 20.0)
    scale = rng.uniform(0.9, 1.08)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += rng.uniform(-0.08, 0.08) * width
    matrix[1, 2] += rng.uniform(-0.08, 0.08) * height
    return apply_affine(image, mask, matrix)


def random_crop_resize(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return image, mask
    height, width = image.shape[:2]
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    margin_x = rng.uniform(0.15, 0.55) * box_w
    margin_y = rng.uniform(0.15, 0.55) * box_h
    crop_x1 = max(0, int(math.floor(x1 - margin_x)))
    crop_y1 = max(0, int(math.floor(y1 - margin_y)))
    crop_x2 = min(width, int(math.ceil(x2 + margin_x)))
    crop_y2 = min(height, int(math.ceil(y2 + margin_y)))
    if crop_x2 - crop_x1 < 8 or crop_y2 - crop_y1 < 8:
        return image, mask
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_mask = mask[crop_y1:crop_y2, crop_x1:crop_x2]
    resized_img = cv2.resize(crop_img, (width, height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized_img, resized_mask


def random_brightness(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    scale = rng.uniform(0.75, 1.3)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def build_variant(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    out_img, out_mask = image.copy(), mask.copy()
    out_img, out_mask = random_translate_rotate(out_img, out_mask, rng)
    if rng.random() < 0.85:
        out_img, out_mask = random_crop_resize(out_img, out_mask, rng)
    if rng.random() < 0.5:
        out_img = cv2.flip(out_img, 1)
        out_mask = cv2.flip(out_mask, 1)
    out_img = random_brightness(out_img, rng)
    out_mask = (out_mask > 0).astype(np.uint8)
    return out_img, out_mask


def save_image(image: np.ndarray, path: Path, jpeg_quality: int) -> None:
    pil_image = Image.fromarray(image)
    pil_image.save(path, quality=jpeg_quality)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    support_json_path = Path(args.support_json)
    support_dir = Path(args.support_dir)
    output_dir = Path(args.output_dir)
    image_out_dir = output_dir / "augmented_images_v2"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    support_data = json.loads(support_json_path.read_text())
    image_by_id = {item["id"]: item for item in support_data["images"]}
    anns_by_image = {ann["image_id"]: ann for ann in support_data["annotations"]}

    output_images = []
    output_annotations = []
    next_image_id = 1
    next_ann_id = 1

    for existing_path in image_out_dir.glob("*.jpg"):
        existing_path.unlink()
    annotations_existing = image_out_dir / "augmented_annotations.json"
    if annotations_existing.exists():
        annotations_existing.unlink()

    def add_sample(file_name: str, category_id: int, image: np.ndarray, mask: np.ndarray, source_id: int) -> None:
        nonlocal next_image_id, next_ann_id
        segmentation, bbox, area = mask_to_polygons(mask)
        if not segmentation or area <= 4.0 or bbox[2] <= 1 or bbox[3] <= 1:
            return
        image_path = image_out_dir / file_name
        save_image(image, image_path, args.jpeg_quality)
        height, width = image.shape[:2]
        output_images.append(
            {
                "id": next_image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "source_image_id": source_id,
            }
        )
        output_annotations.append(
            {
                "id": next_ann_id,
                "image_id": next_image_id,
                "category_id": category_id,
                "bbox": bbox,
                "area": float(area),
                "segmentation": segmentation,
                "iscrowd": 0,
            }
        )
        next_image_id += 1
        next_ann_id += 1

    for image_id, image_info in image_by_id.items():
        ann = anns_by_image.get(image_id)
        if ann is None:
            continue
        image_path = support_dir / image_info["file_name"]
        image = np.array(ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"))
        mask = polygon_to_mask(ann["segmentation"], image.shape[0], image.shape[1])

        stem = Path(image_info["file_name"]).stem
        add_sample(f"{stem}_orig.jpg", ann["category_id"], image, mask, image_id)
        for aug_idx in range(args.augmentations_per_image):
            aug_img, aug_mask = build_variant(image, mask, rng)
            add_sample(f"{stem}_aug_{aug_idx:03d}.jpg", ann["category_id"], aug_img, aug_mask, image_id)

    output_payload = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": support_data["categories"],
    }
    annotations_path = image_out_dir / "augmented_annotations.json"
    annotations_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2))

    print(f"Saved {len(output_images)} augmented images to {image_out_dir}")
    print(f"Saved annotations to {annotations_path}")


if __name__ == "__main__":
    main()
