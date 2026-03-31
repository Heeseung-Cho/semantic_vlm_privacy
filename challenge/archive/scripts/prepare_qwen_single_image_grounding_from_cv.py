#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    'You are a visual grounding model. '
    'Given one image and one target category, return only native grounding text. '
    'Use coordinates normalized to [0,1000]. '
    'Return <ref>category name</ref><box>(x1,y1),(x2,y2)</box> for the target object.'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare single-image Qwen grounding SFT data from a VizWiz CV fold.')
    parser.add_argument('--train-ann', required=True)
    parser.add_argument('--eval-ann', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--examples-per-category', type=int, default=30)
    parser.add_argument('--min-area-ratio', type=float, default=0.0)
    parser.add_argument('--max-area-ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def normalize_category_name(name: str) -> str:
    return name.replace('_', ' ').strip()


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


def make_grounding_text(category_name: str, bbox_1000: list[int]) -> str:
    x1, y1, x2, y2 = bbox_1000
    return f'<ref>{category_name}</ref><box>({x1},{y1}),({x2},{y2})</box>'


def bbox_area_ratio_xywh(bbox: list[float], width: float, height: float) -> float:
    _, _, w, h = bbox
    return max(0.0, w / width) * max(0.0, h / height)


def make_example(image: dict[str, Any], ann: dict[str, Any] | None, category_name: str, image_dir: Path) -> dict[str, Any]:
    exists = ann is not None
    bbox_1000 = None
    bbox_norm = None
    if ann is not None:
        bbox_1000 = bbox_xywh_to_xyxy_1000(ann['bbox'], width=image['width'], height=image['height'])
        bbox_norm = bbox_xywh_to_xyxy_norm(ann['bbox'], width=image['width'], height=image['height'])

    assistant_content = make_grounding_text(category_name, exists, bbox_1000)
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': SYSTEM_PROMPT},
                    {'type': 'image', 'image': str((image_dir / image['file_name']).resolve())},
                    {'type': 'text', 'text': (
                        f'Target category: {category_name}. '                        'Find the target object in the image. '                        'Return only native grounding text with 0-1000 coordinates.'
                    )},
                ],
            },
            {'role': 'assistant', 'content': assistant_content},
        ],
        'image_id': image['id'],
        'file_name': image['file_name'],
        'category_name': category_name,
        'target': {
            'exists': True,
            'bbox_xyxy_norm': bbox_norm,
            'bbox_xyxy_1000': bbox_1000,
        },
    }


def build_split(data: dict[str, Any], image_dir: Path, examples_per_category: int, min_area_ratio: float, max_area_ratio: float) -> list[dict[str, Any]]:
    images_by_id = {img['id']: img for img in data['images']}
    categories_by_id = {cat['id']: normalize_category_name(cat['name']) for cat in data['categories']}
    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data['annotations']:
        anns_by_cat[ann['category_id']].append(ann)

    examples: list[dict[str, Any]] = []
    for category_id, anns in sorted(anns_by_cat.items()):
        category_name = categories_by_id[category_id]
        filtered = []
        for ann in sorted(anns, key=lambda a: (-float(a.get('area', 0.0)), a['image_id'], a['id'])):
            image = images_by_id[ann['image_id']]
            area_ratio = bbox_area_ratio_xywh(ann['bbox'], width=image['width'], height=image['height'])
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            filtered.append(ann)
        positives = filtered[:examples_per_category]
        for ann in positives:
            image = images_by_id[ann['image_id']]
            examples.append(make_example(image, ann, category_name, image_dir))

    return examples


def main() -> None:
    args = parse_args()
    train_data = json.loads(Path(args.train_ann).read_text())
    eval_data = json.loads(Path(args.eval_ann).read_text())
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_examples = build_split(train_data, image_dir, args.examples_per_category, args.min_area_ratio, args.max_area_ratio)
    eval_examples = build_split(eval_data, image_dir, args.examples_per_category, args.min_area_ratio, args.max_area_ratio)

    with (out_dir / 'train.jsonl').open('w', encoding='utf-8') as f:
        for row in train_examples:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (out_dir / 'eval.jsonl').open('w', encoding='utf-8') as f:
        for row in eval_examples:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    meta = {
        'train_ann': str(Path(args.train_ann).resolve()),
        'eval_ann': str(Path(args.eval_ann).resolve()),
        'image_dir': str(image_dir.resolve()),
        'examples_per_category': args.examples_per_category,
        'min_area_ratio': args.min_area_ratio,
        'max_area_ratio': args.max_area_ratio,
        'seed': args.seed,
        'train_examples': len(train_examples),
        'eval_examples': len(eval_examples),
        'format': 'single-image grounding with interleaved user content',
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
