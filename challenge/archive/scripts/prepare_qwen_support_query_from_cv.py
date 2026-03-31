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
    parser = argparse.ArgumentParser(description='Prepare Qwen support-query grounding data from a VizWiz base CV fold.')
    parser.add_argument('--train-ann', required=True, help='Fold train.json path')
    parser.add_argument('--eval-ann', required=True, help='Fold val.json path')
    parser.add_argument('--image-dir', required=True, help='Directory containing images for both splits')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--examples-per-category', type=int, default=20, help='Max positive examples per category per split')
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


def make_grounding_text(category_name: str, exists: bool, bbox_1000: list[int] | None) -> str:
    if not exists or bbox_1000 is None:
        return f'<ref>{category_name}</ref><box>none</box>'
    x1, y1, x2, y2 = bbox_1000
    return f'<ref>{category_name}</ref><box>({x1},{y1}),({x2},{y2})</box>'


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
    if example_type == 'positive':
        bbox_xyxy_norm = bbox_xywh_to_xyxy_norm(ann['bbox'], width=query_image['width'], height=query_image['height'])
        bbox_xyxy_1000 = bbox_xywh_to_xyxy_1000(ann['bbox'], width=query_image['width'], height=query_image['height'])
        target = {'exists': True, 'bbox_xyxy_norm': bbox_xyxy_norm, 'bbox_xyxy_1000': bbox_xyxy_1000}
    else:
        target = {'exists': False, 'bbox_xyxy_norm': None, 'bbox_xyxy_1000': None}

    assistant_content = make_grounding_text(category_name=category_name, exists=target['exists'], bbox_1000=target['bbox_xyxy_1000'])
    return {
        'example_type': example_type,
        'category_name': category_name,
        'support_image_id': support_image['id'],
        'support_file_name': support_image['file_name'],
        'query_image_id': query_image['id'],
        'query_file_name': query_image['file_name'],
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': (
                    'Support image: <image>\n'
                    'Query image: <image>\n'
                    'Find the object in the query image that matches the category shown in the support image.\n'
                    'Return only native grounding text.'
                ),
            },
            {'role': 'assistant', 'content': assistant_content},
        ],
        'images': [
            str((image_dir / support_image['file_name']).resolve()),
            str((image_dir / query_image['file_name']).resolve()),
        ],
        'target': target,
    }


def select_supports(train_data: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images_by_id = {img['id']: img for img in train_data['images']}
    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in train_data['annotations']:
        anns_by_cat[ann['category_id']].append(ann)

    supports: dict[int, dict[str, Any]] = {}
    for category_id, anns in anns_by_cat.items():
        category_anns = sorted(anns, key=lambda ann: (-float(ann.get('area', 0.0)), ann['image_id'], ann['id']))
        if not category_anns:
            continue
        support_ann = category_anns[0]
        support_image = images_by_id[support_ann['image_id']]
        supports[category_id] = {'ann': support_ann, 'image': support_image}
    return supports


def build_bucket(
    source_data: dict[str, Any],
    supports: dict[int, dict[str, Any]],
    categories_by_id: dict[int, str],
    image_dir: Path,
    examples_per_category: int,
    min_area_ratio: float,
    max_area_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    images_by_id = {img['id']: img for img in source_data['images']}
    anns_by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    anns_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in source_data['annotations']:
        anns_by_cat[ann['category_id']].append(ann)
        anns_by_img[ann['image_id']].append(ann)

    examples: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []

    for category_id, support in sorted(supports.items()):
        if category_id not in categories_by_id:
            continue
        support_image = support['image']
        support_ann = support['ann']
        category_name = categories_by_id[category_id]

        remaining = [ann for ann in anns_by_cat.get(category_id, []) if ann['image_id'] != support_image['id']]
        filtered_remaining = []
        for ann in sorted(remaining, key=lambda a: (-float(a.get('area', 0.0)), a['image_id'], a['id'])):
            query_image = images_by_id[ann['image_id']]
            area_ratio = bbox_area_ratio_xywh(ann['bbox'], width=query_image['width'], height=query_image['height'])
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            filtered_remaining.append(ann)

        positive_candidates = filtered_remaining[:examples_per_category]
        negative_pool = [
            img for img in source_data['images']
            if img['id'] != support_image['id'] and all(a['category_id'] != category_id for a in anns_by_img[img['id']])
        ]
        rng.shuffle(negative_pool)
        negative_candidates = negative_pool[: min(len(positive_candidates), len(negative_pool))]

        for ann in positive_candidates:
            query_image = images_by_id[ann['image_id']]
            examples.append(make_example(support_image, query_image, ann, category_name, image_dir, 'positive'))
        for query_image in negative_candidates:
            examples.append(make_example(support_image, query_image, support_ann, category_name, image_dir, 'negative'))

        meta_rows.append({
            'category_id': category_id,
            'category_name': category_name,
            'support_image_id': support_image['id'],
            'support_file_name': support_image['file_name'],
            'positive_examples': len(positive_candidates),
            'negative_examples': len(negative_candidates),
        })

    return examples, meta_rows


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    train_data = json.loads(Path(args.train_ann).read_text())
    eval_data = json.loads(Path(args.eval_ann).read_text())
    image_dir = Path(args.image_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    categories_by_id = {cat['id']: normalize_category_name(cat['name']) for cat in train_data['categories']}
    supports = select_supports(train_data)
    train_examples, train_meta = build_bucket(
        source_data=train_data,
        supports=supports,
        categories_by_id=categories_by_id,
        image_dir=image_dir,
        examples_per_category=args.examples_per_category,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        rng=rng,
    )
    eval_examples, eval_meta = build_bucket(
        source_data=eval_data,
        supports=supports,
        categories_by_id=categories_by_id,
        image_dir=image_dir,
        examples_per_category=args.examples_per_category,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        rng=rng,
    )

    (out_dir / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in train_examples), encoding='utf-8')
    (out_dir / 'eval.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in eval_examples), encoding='utf-8')
    meta = {
        'train_ann': str(Path(args.train_ann).resolve()),
        'eval_ann': str(Path(args.eval_ann).resolve()),
        'image_dir': str(image_dir.resolve()),
        'examples_per_category': args.examples_per_category,
        'min_area_ratio': args.min_area_ratio,
        'max_area_ratio': args.max_area_ratio,
        'seed': args.seed,
        'num_support_categories': len(supports),
        'train_examples': len(train_examples),
        'eval_examples': len(eval_examples),
        'train_categories': train_meta,
        'eval_categories': eval_meta,
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
