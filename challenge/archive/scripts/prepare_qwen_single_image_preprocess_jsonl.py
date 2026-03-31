#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROMPT_TEMPLATE = (
    'Locate every instance that belongs to the following category: {category}. '
    'Return only native grounding text in this format: '
    '<ref>{category}</ref><box>(x1,y1),(x2,y2)</box>. '
    'If there are multiple instances, concatenate multiple <box> entries after the same <ref>. '
    'Coordinates must be normalized to a 0-1000 scale. '
    'Do not output explanations.\n<image>'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare Swift-style multi-bbox single-image Qwen grounding JSONL from a VizWiz CV fold.'
    )
    parser.add_argument('--train-ann', required=True)
    parser.add_argument('--eval-ann', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--examples-per-category', type=int, default=30)
    parser.add_argument('--min-area-ratio', type=float, default=0.0)
    parser.add_argument('--max-area-ratio', type=float, default=1.0)
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


def bbox_area_ratio_xywh(bbox: list[float], width: float, height: float) -> float:
    _, _, w, h = bbox
    return max(0.0, w / width) * max(0.0, h / height)


def make_answer(category_name: str, bboxes_1000: list[list[int]]) -> str:
    boxes_text = ''.join(f'<box>({x1},{y1}),({x2},{y2})</box>' for x1, y1, x2, y2 in bboxes_1000)
    return f'<ref>{category_name}</ref>{boxes_text}'


def make_conversations(category_name: str, answer: str) -> list[dict[str, str]]:
    return [
        {
            'from': 'human',
            'value': PROMPT_TEMPLATE.format(category=category_name),
        },
        {
            'from': 'gpt',
            'value': answer,
        },
    ]


def make_record(image: dict[str, Any], anns: list[dict[str, Any]], category_name: str, image_dir: Path) -> dict[str, Any]:
    anns_sorted = sorted(anns, key=lambda ann: (-float(ann.get('area', 0.0)), ann['id']))
    bboxes_norm = [bbox_xywh_to_xyxy_norm(ann['bbox'], width=image['width'], height=image['height']) for ann in anns_sorted]
    bboxes_1000 = [bbox_xywh_to_xyxy_1000(ann['bbox'], width=image['width'], height=image['height']) for ann in anns_sorted]
    answer = make_answer(category_name, bboxes_1000)
    return {
        'image': [str((image_dir / image['file_name']).resolve())],
        'conversations': make_conversations(category_name, answer),
        'info': {
            'image_id': image['id'],
            'file_name': image['file_name'],
            'category_name': category_name,
            'num_instances': len(anns_sorted),
            'bbox_xyxy_norm_list': bboxes_norm,
            'bbox_xyxy_1000_list': bboxes_1000,
        },
    }


def build_split(
    data: dict[str, Any],
    image_dir: Path,
    examples_per_category: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[dict[str, Any]]:
    images_by_id = {img['id']: img for img in data['images']}
    categories_by_id = {cat['id']: normalize_category_name(cat['name']) for cat in data['categories']}
    anns_by_cat_img: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann in data['annotations']:
        image = images_by_id[ann['image_id']]
        area_ratio = bbox_area_ratio_xywh(ann['bbox'], width=image['width'], height=image['height'])
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        anns_by_cat_img[(ann['category_id'], ann['image_id'])].append(ann)

    grouped_keys_by_cat: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for key in anns_by_cat_img:
        grouped_keys_by_cat[key[0]].append(key)

    records: list[dict[str, Any]] = []
    for category_id, keys in sorted(grouped_keys_by_cat.items()):
        category_name = categories_by_id[category_id]
        sorted_keys = sorted(
            keys,
            key=lambda key: (-sum(float(ann.get('area', 0.0)) for ann in anns_by_cat_img[key]), key[1]),
        )
        for key in sorted_keys[:examples_per_category]:
            _, image_id = key
            image = images_by_id[image_id]
            records.append(make_record(image, anns_by_cat_img[key], category_name, image_dir))
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    train_data = json.loads(Path(args.train_ann).read_text())
    eval_data = json.loads(Path(args.eval_ann).read_text())
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = build_split(train_data, image_dir, args.examples_per_category, args.min_area_ratio, args.max_area_ratio)
    eval_rows = build_split(eval_data, image_dir, args.examples_per_category, args.min_area_ratio, args.max_area_ratio)
    write_jsonl(out_dir / 'train.jsonl', train_rows)
    write_jsonl(out_dir / 'eval.jsonl', eval_rows)

    meta = {
        'train_ann': str(Path(args.train_ann).resolve()),
        'eval_ann': str(Path(args.eval_ann).resolve()),
        'image_dir': str(image_dir.resolve()),
        'examples_per_category': args.examples_per_category,
        'min_area_ratio': args.min_area_ratio,
        'max_area_ratio': args.max_area_ratio,
        'train_examples': len(train_rows),
        'eval_examples': len(eval_rows),
        'format': 'swift-style image+conversations with single-image single-category multi-bbox grounding',
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
