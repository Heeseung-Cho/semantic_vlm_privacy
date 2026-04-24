#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.overlay_utils import draw_gt_annotation, draw_xywh_box, load_display_image, xywh_to_xyxy
from baseline.qwen_gdino_sam import GroundingDinoLocalizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate detector-only G-DINO on BIV support images.')
    parser.add_argument('--support-dir', required=True)
    parser.add_argument('--support-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--device', default='cuda')
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
    return 0.0 if union <= 0 else inter_area / union


def save_overlay(image_path: Path, ann: dict[str, Any], results: list[dict[str, Any]], output_dir: Path) -> None:
    image = load_display_image(image_path)
    draw = ImageDraw.Draw(image)
    draw_gt_annotation(draw, ann, label=None, bbox_color='lime', polygon_color='yellow')
    for result in results:
        draw_xywh_box(draw, result['bbox'], f"PRED {result['label_text']} {result['score']:.2f}", color='red', width=4)
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
    category_id_to_name = {cat['id']: cat['name'] for cat in categories}
    categories_dict = {cat['name']: int(cat['id']) for cat in categories}

    localizer = GroundingDinoLocalizer(args.config_path, args.checkpoint_path, device=args.device)

    records = []
    ious = []
    hits = 0
    num_nonempty = 0
    for image in images:
        ann = image_id_to_ann[image['id']]
        gt_label = category_id_to_name[ann['category_id']]
        image_path = Path(args.support_dir) / image['file_name']
        detections = localizer.detect(
            image_path=str(image_path),
            cue_text=gt_label,
            categories_dict=categories_dict,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        gt_xyxy = xywh_to_xyxy(ann['bbox'])
        best_iou = 0.0
        best_label = None
        results = []
        for det in detections:
            pred_xyxy = [float(v) for v in det.xyxy]
            iou = compute_iou_xyxy(gt_xyxy, pred_xyxy)
            x1, y1, x2, y2 = pred_xyxy
            results.append({
                'score': float(det.score),
                'label_text': det.label_text,
                'category_id': int(det.category_id),
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'iou_to_gt': iou,
            })
            if iou > best_iou:
                best_iou = iou
                best_label = det.label_text
        if results:
            num_nonempty += 1
        if best_iou >= 0.5:
            hits += 1
        ious.append(best_iou)
        record = {
            'image_id': image['id'],
            'file_name': image['file_name'],
            'gt_label': gt_label,
            'gt_bbox': ann['bbox'],
            'best_iou': best_iou,
            'pred_best_label': best_label,
            'results': results,
        }
        records.append(record)
        save_overlay(image_path, ann, results, output_dir)
        print(f"Processed support image_id={image['id']} gt={gt_label} best_iou={best_iou:.3f} detections={len(results)}")

    summary = {
        'checkpoint_path': str(Path(args.checkpoint_path).resolve()),
        'num_images': len(images),
        'mean_iou': sum(ious) / len(ious) if ious else 0.0,
        'hit50': hits / len(images) if images else 0.0,
        'num_nonempty_results': num_nonempty,
    }
    (output_dir / 'support_eval_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'support_eval_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
