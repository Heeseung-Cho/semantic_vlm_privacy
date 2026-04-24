#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import ImageDraw

from common.overlay_utils import draw_gt_annotation, load_display_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export GT-only overlays for BIV support images.')
    parser.add_argument('--support-dir', required=True)
    parser.add_argument('--support-json', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--draw-bbox', action='store_true', help='Draw GT bbox')
    parser.add_argument('--draw-polygon', action='store_true', help='Draw GT segmentation polygon')
    return parser.parse_args()


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

        image = load_display_image(image_path)
        draw = ImageDraw.Draw(image)
        label = category_id_to_name[ann['category_id']]

        if args.draw_bbox or args.draw_polygon:
            draw_gt_annotation(
                draw,
                ann,
                label=label if args.draw_bbox else None,
                bbox_color='lime',
                polygon_color='yellow',
            )

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
