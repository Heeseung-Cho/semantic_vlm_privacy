#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from challenge.protocols.qwen_gdino_sam import GroundingDinoLocalizer, QwenController, QwenGdinoSamProtocol, SamSegmenter


def load_categories_dict_from_support(dataset: dict[str, Any]) -> dict[str, int]:
    return {category['name']: int(category['id']) for category in dataset['categories']}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Qwen-GDINO-SAM on BIV support images with GT bbox.')
    parser.add_argument('--support-dir', required=True)
    parser.add_argument('--support-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--controller-mode', choices=['query_only', 'support_query'], default='query_only')
    parser.add_argument('--llm-max-new-tokens', type=int, default=256)
    parser.add_argument('--llm-decoding-mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-max-pixels', type=int, default=448)
    parser.add_argument('--box-threshold', type=float, default=0.35)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--date-tag', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--flat-output', action='store_true')
    return parser.parse_args()


def resolve_output_dir(base_dir: str, flat_output: bool, date_tag: str | None, run_id: str | None) -> Path:
    if flat_output:
        return Path(base_dir).resolve()
    now = datetime.now()
    return Path(base_dir).resolve() / (date_tag or now.strftime('%Y%m%d')) / (run_id or now.strftime('%H%M%S'))


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def compute_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def save_overlay(image_path: Path, ann: dict[str, Any], record: dict[str, Any], output_dir: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    draw = ImageDraw.Draw(image)

    gx1, gy1, gx2, gy2 = xywh_to_xyxy(ann['bbox'])
    draw.rectangle([gx1, gy1, gx2, gy2], outline='lime', width=4)
    draw.text((gx1, max(0, gy1 - 18)), 'GT bbox', fill='lime')

    for polygon in ann.get('segmentation', []):
        gt_points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        if len(gt_points) >= 2:
            draw.line(gt_points + [gt_points[0]], fill='yellow', width=3)

    for result in record.get('results', []):
        x, y, w, h = result['bbox']
        draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
        draw.text((x, max(0, y - 18)), f"PRED {result['label_text']} {result['score']:.2f}", fill='red')
        for polygon in result.get('segmentation', []):
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill='cyan', width=2)

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    image.save(vis_dir / f'{image_path.stem}_gt_pred_overlay.jpg')


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.flat_output, args.date_tag, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.support_json).read_text())
    images = payload['images']
    annotations = payload['annotations']
    categories = payload['categories']
    if args.limit is not None:
        images = images[:args.limit]

    image_id_to_ann = {ann['image_id']: ann for ann in annotations}
    category_id_to_name = {cat['id']: cat['name'].replace('_', ' ') for cat in categories}
    categories_dict = {cat['name'].replace('_', ' '): cat['id'] for cat in categories}

    controller = QwenController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        allowed_categories=list(categories_dict.keys()),
    )
    localizer = GroundingDinoLocalizer(args.config_path, args.checkpoint_path, device=args.device)
    segmenter = SamSegmenter(args.sam_checkpoint, device=args.device)
    protocol = QwenGdinoSamProtocol(controller, localizer, segmenter)

    records = []
    ious = []
    hit50 = 0

    for image_info in images:
        image_id = image_info['id']
        ann = image_id_to_ann[image_id]
        image_path = Path(args.support_dir) / image_info['file_name']
        gt_bbox_xyxy = xywh_to_xyxy(ann['bbox'])
        gt_label = category_id_to_name[ann['category_id']]

        if args.controller_mode == 'query_only':
            record = protocol.run_query_only(
                query_image_path=str(image_path.resolve()),
                image_id=image_id,
                categories_dict=categories_dict,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        else:
            record = protocol.run_support_query(
                support_image_paths=[str(image_path.resolve())],
                query_image_path=str(image_path.resolve()),
                image_id=image_id,
                categories_dict=categories_dict,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )

        pred_iou = 0.0
        pred_label = None
        pred_score = None
        pred_bbox = None
        if record['results']:
            best_result = max(record['results'], key=lambda r: compute_iou_xyxy(gt_bbox_xyxy, xywh_to_xyxy(r['bbox'])))
            pred_bbox = best_result['bbox']
            pred_label = best_result['label_text']
            pred_score = best_result['score']
            pred_iou = compute_iou_xyxy(gt_bbox_xyxy, xywh_to_xyxy(best_result['bbox']))
            if pred_iou >= 0.5:
                hit50 += 1
        ious.append(pred_iou)

        enriched = {
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'controller_mode': record['controller_mode'],
            'controller_instruction': record['controller_instruction'],
            'controller_raw_text': record['controller_raw_text'],
            'cue_text': record['cue_text'],
            'gt_label': gt_label,
            'gt_bbox': ann['bbox'],
            'gt_segmentation': ann.get('segmentation', []),
            'pred_best_label': pred_label,
            'pred_best_score': pred_score,
            'pred_best_bbox': pred_bbox,
            'best_iou': pred_iou,
            'num_results': len(record['results']),
            'results': record['results'],
        }
        records.append(enriched)
        save_overlay(image_path, ann, record, output_dir)
        print(f"Processed support image_id={image_id} gt={gt_label} cue={record['cue_text']} iou={pred_iou:.3f}")

    summary = {
        'controller_mode': args.controller_mode,
        'num_images': len(records),
        'mean_iou': sum(ious) / len(ious) if ious else 0.0,
        'hit50': hit50 / len(records) if records else 0.0,
        'num_nonempty_results': sum(1 for rec in records if rec['num_results'] > 0),
    }

    (output_dir / 'support_eval_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    (output_dir / 'support_eval_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
