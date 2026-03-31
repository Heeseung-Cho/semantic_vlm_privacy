#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export GT-only overlays for BIV support images.')
    parser.add_argument('--support-dir', required=True)
    parser.add_argument('--support-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--draw-bbox', action='store_true', help='Draw GT bbox')
    parser.add_argument('--draw-polygon', action='store_true', help='Draw GT segmentation polygon')
    return parser.parse_args()


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.support_json).read_text())
    images = payload['images']
    annotations = payload['annotations']
    categories = payload['categories']
    if args.limit is not None:
        images = images[:args.limit]

    image_id_to_ann = {ann['image_id']: ann for ann in annotations}
    category_id_to_name = {cat['id']: cat['name'].replace('_', ' ') for cat in categories}

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for image_info in images:
        image_id = image_info['id']
        ann = image_id_to_ann[image_id]
        image_path = Path(args.support_dir) / image_info['file_name']

        image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
        draw = ImageDraw.Draw(image)
        label = category_id_to_name[ann['category_id']]

        if args.draw_bbox:
            x1, y1, x2, y2 = xywh_to_xyxy(ann['bbox'])
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
            draw.text((x1, max(0, y1 - 18)), f'GT bbox: {label}', fill='lime')

        if args.draw_polygon:
            for polygon in ann.get('segmentation', []):
                pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                if len(pts) >= 2:
                    draw.line(pts + [pts[0]], fill='yellow', width=3)

        out_path = output_dir / f"{Path(image_info['file_name']).stem}_gt_overlay.jpg"
        image.save(out_path)
        manifest.append({
            'image_id': image_id,
            'file_name': image_info['file_name'],
            'category_name': label,
            'bbox': ann['bbox'],
            'num_polygons': len(ann.get('segmentation', [])),
            'output_path': str(out_path),
        })
        print(f"Saved GT overlay for image_id={image_id} label={label} -> {out_path}")

    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
