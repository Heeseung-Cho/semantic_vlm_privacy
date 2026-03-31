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

from challenge.semantic.semantic_gdino_sam import (
    GroundingDinoLocalizer,
    ProposalReranker,
    SamSegmenter,
    SemanticController,
    SemanticGdinoSamPipeline,
    load_support_image_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the no-train semantic controller -> G-DINO -> VLM rerank -> SAM pipeline.')
    parser.add_argument('--query-dir', required=True)
    parser.add_argument('--json-path', required=True)
    parser.add_argument('--support-dir', default=None)
    parser.add_argument('--support-json', default=None)
    parser.add_argument('--controller-mode', choices=['query_only', 'support_query'], default='support_query')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm-max-new-tokens', type=int, default=256)
    parser.add_argument('--llm-decoding-mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-max-pixels', type=int, default=448)
    parser.add_argument('--box-threshold', type=float, default=0.25)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--proposal-nms-iou', type=float, default=0.6)
    parser.add_argument('--max-candidates', type=int, default=5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image-id', type=int, default=None)
    parser.add_argument('--date-tag', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--flat-output', action='store_true')
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--disable-sam', action='store_true')
    parser.add_argument('--null-policy', choices=['strict', 'skip', 'ignore'], default='strict')
    return parser.parse_args()


def resolve_output_dir(base_dir: str, flat_output: bool, date_tag: str | None, run_id: str | None) -> Path:
    if flat_output:
        return Path(base_dir).resolve()
    now = datetime.now()
    return Path(base_dir).resolve() / (date_tag or now.strftime('%Y%m%d')) / (run_id or now.strftime('%H%M%S'))


def save_visualization(record: dict[str, Any], output_dir: Path, gt_ann: dict[str, Any] | None = None, gt_label: str | None = None) -> None:
    image_path = Path(record['query_image_path'])
    image = ImageOps.exif_transpose(Image.open(image_path)).convert('RGB')
    draw = ImageDraw.Draw(image)

    if gt_ann is not None:
        gx, gy, gw, gh = gt_ann['bbox']
        draw.rectangle([gx, gy, gx + gw, gy + gh], outline='lime', width=4)
        label_text = f"GT {gt_label}" if gt_label else 'GT'
        draw.text((gx, max(0, gy - 18)), label_text, fill='lime')
        for polygon in gt_ann.get('segmentation', []):
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill='yellow', width=3)

    for idx, candidate in enumerate(record.get('proposal_candidates', []), start=1):
        x1, y1, x2, y2 = candidate['bbox_xyxy']
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=2)
        draw.text((x1, max(0, y1 - 16)), f"P{idx}:{candidate['label_text']} {candidate['score']:.2f}", fill='yellow')

    for result in record.get('results', []):
        x, y, w, h = result['bbox']
        draw.rectangle([x, y, x + w, y + h], outline='red', width=4)
        draw.text((x, max(0, y - 18)), f"SEL {result['label_text']} {result['score']:.2f}", fill='red')
        for polygon in result.get('segmentation', []):
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            if len(points) >= 2:
                draw.line(points + [points[0]], fill='cyan', width=2)

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    image.save(vis_dir / f'{image_path.stem}_semantic_overlay.jpg')


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir, args.flat_output, args.date_tag, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    annotations_by_image = {ann['image_id']: ann for ann in dataset.get('annotations', [])}
    categories_by_id = {cat['id']: cat['name'] for cat in dataset.get('categories', [])}
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    support_image_paths = []
    if args.controller_mode == 'support_query':
        if not args.support_dir or not args.support_json:
            raise ValueError('support_query mode requires --support-dir and --support-json')
        support_image_paths = load_support_image_paths(args.support_json, args.support_dir)
    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )
    localizer = GroundingDinoLocalizer(args.config_path, args.checkpoint_path, device=args.device)
    reranker = ProposalReranker(
        model_path=args.llm_model,
        max_new_tokens=128,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )
    segmenter = SamSegmenter(args.sam_checkpoint, device=args.device)
    pipeline = SemanticGdinoSamPipeline(controller, localizer, reranker, segmenter)

    outputs = []
    for image_info in images:
        query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
        record = pipeline.run(
            support_image_paths=support_image_paths if args.controller_mode == 'support_query' else [],
            query_image_path=query_image_path,
            image_id=image_info['id'],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            proposal_nms_iou=args.proposal_nms_iou,
            max_candidates=args.max_candidates,
            use_sam=not args.disable_sam,
            null_policy=args.null_policy,
        )
        outputs.append(record)
        print(
            f"Processed image_id={image_info['id']} family={record['semantic_family']} "
            f"null={record['null_likely']} proposals={len(record['proposal_candidates'])} selected={len(record['results'])}"
        )
        if args.save_vis:
            gt_ann = annotations_by_image.get(image_info['id'])
            gt_label = categories_by_id.get(gt_ann['category_id']) if gt_ann else None
            save_visualization(record, output_dir, gt_ann=gt_ann, gt_label=gt_label)

    (output_dir / 'semantic_pipeline_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    print(f'Saved semantic pipeline outputs to: {output_dir / "semantic_pipeline_results.json"}')


if __name__ == '__main__':
    main()
