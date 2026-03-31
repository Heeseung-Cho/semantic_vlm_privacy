#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from challenge.protocols.qwen_gdino_sam import (
    GroundingDinoLocalizer,
    QwenController,
    QwenGdinoSamProtocol,
    SamSegmenter,
    load_categories_dict,
    load_support_image_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the modular Qwen-GDINO-SAM protocol.')
    parser.add_argument('--query-dir', required=True)
    parser.add_argument('--json-path', required=True, help='Query/dev metadata json with categories/images')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--controller-mode', choices=['query_only', 'support_query'], default='query_only')
    parser.add_argument('--support-dir', default=None)
    parser.add_argument('--support-json', default=None)
    parser.add_argument('--llm-max-new-tokens', type=int, default=256)
    parser.add_argument('--llm-decoding-mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-max-pixels', type=int, default=448)
    parser.add_argument('--box-threshold', type=float, default=0.35)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image-id', type=int, default=None)
    parser.add_argument('--date-tag', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--flat-output', action='store_true')
    parser.add_argument('--save-vis', action='store_true', help='Save bbox and mask overlays per query image')
    return parser.parse_args()


def resolve_output_dir(base_dir: str, flat_output: bool, date_tag: str | None, run_id: str | None) -> Path:
    if flat_output:
        return Path(base_dir).resolve()
    now = datetime.now()
    return Path(base_dir).resolve() / (date_tag or now.strftime('%Y%m%d')) / (run_id or now.strftime('%H%M%S'))


def save_visualization(record: dict[str, Any], output_dir: Path) -> None:
    image_path = Path(record['query_image_path'])
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for result in record.get('results', []):
        x, y, w, h = result['bbox']
        draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
        draw.text((x, max(0, y - 18)), f"{result['label_text']} {result['score']:.2f}", fill='red')

        for polygon in result.get('segmentation', []):
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill='cyan', width=2)

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    image.save(vis_dir / f'{image_path.stem}_overlay.jpg')


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.flat_output, args.date_tag, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    categories_dict = load_categories_dict(args.json_path)
    support_image_paths: list[str] = []
    if args.controller_mode == 'support_query':
        if not args.support_dir or not args.support_json:
            raise ValueError('support_query mode requires --support-dir and --support-json')
        support_image_paths = load_support_image_paths(args.support_json, args.support_dir)

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

    outputs = []
    for image_info in images:
        query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
        if args.controller_mode == 'query_only':
            record = protocol.run_query_only(
                query_image_path=query_image_path,
                image_id=image_info['id'],
                categories_dict=categories_dict,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        else:
            record = protocol.run_support_query(
                support_image_paths=support_image_paths,
                query_image_path=query_image_path,
                image_id=image_info['id'],
                categories_dict=categories_dict,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
        outputs.append(record)
        print(f"Processed image_id={image_info['id']} cue={record['cue_text']}")
        if args.save_vis:
            save_visualization(record, output_dir)

    (output_dir / 'protocol_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f'Saved protocol outputs to: {output_dir / "protocol_results.json"}')


if __name__ == '__main__':
    main()
