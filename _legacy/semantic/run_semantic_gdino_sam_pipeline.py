#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import ImageDraw
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.overlay_utils import (
    draw_gt_annotation,
    draw_segmentation_polygons,
    draw_xywh_box,
    draw_xyxy_box,
    load_display_image,
)
from common.vlm import SwiftVLMCaller, release_torch_runtime
from semantic.family_config import set_active_family_config
from semantic.semantic_gdino_sam import (
    GroundingDinoLocalizer,
    ProposalCalibrator,
    SamSegmenter,
    SemanticController,
    SemanticGdinoSamPipeline,
    load_support_image_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the no-train semantic controller -> G-DINO -> VLM category calibration -> optional SAM pipeline.')
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
    parser.add_argument('--family-config', '--family_config', dest='family_config', default=None)
    parser.add_argument('--box-threshold', type=float, default=0.25)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--proposal-nms-iou', type=float, default=0.6)
    parser.add_argument('--max-candidates', type=int, default=5)
    parser.add_argument('--final-score-threshold', type=float, default=0.40)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image-id', type=int, default=None)
    parser.add_argument('--date-tag', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--flat-output', action='store_true')
    parser.add_argument('--save-vis', action='store_true')
    parser.add_argument('--disable-sam', action='store_true')
    parser.add_argument('--null-policy', choices=['strict', 'skip', 'ignore'], default='strict')
    parser.add_argument('--use-hybrid-category-scores', action='store_true')
    parser.add_argument('--hybrid-score-threshold', type=float, default=0.35)
    parser.add_argument('--hybrid-margin', type=float, default=0.05)
    parser.add_argument('--hybrid-iou-threshold', type=float, default=0.50)
    parser.add_argument('--hybrid-exact-confidence-threshold', type=int, default=95)
    parser.add_argument('--calibration-mode', choices=['legacy', 'reference_match'], default='legacy')
    parser.add_argument('--reference-source', choices=['crop', 'full_image'], default='crop')
    parser.add_argument('--classification-top-k', type=int, default=None)
    parser.add_argument('--disable-document-text', action='store_true')
    parser.add_argument('--runtime-stats-jsonl', default=None)
    parser.add_argument('--cuda-cleanup-interval', type=int, default=1)
    return parser.parse_args()


def resolve_output_dir(base_dir: str, flat_output: bool, date_tag: str | None, run_id: str | None) -> Path:
    if flat_output:
        return Path(base_dir).resolve()
    now = datetime.now()
    return Path(base_dir).resolve() / (date_tag or now.strftime('%Y%m%d')) / (run_id or now.strftime('%H%M%S'))


def save_visualization(record: dict[str, Any], output_dir: Path, gt_ann: dict[str, Any] | None = None, gt_label: str | None = None) -> None:
    image_path = Path(record['query_image_path'])
    image = load_display_image(image_path)
    draw = ImageDraw.Draw(image)

    if gt_ann is not None:
        draw_gt_annotation(draw, gt_ann, label=gt_label, bbox_color='lime', polygon_color='yellow')

    for idx, candidate in enumerate(record.get('proposal_candidates', []), start=1):
        draw_xyxy_box(
            draw,
            candidate['bbox_xyxy'],
            f"P{idx}:{candidate['label_text']} {candidate['score']:.2f}",
            color='yellow',
            width=2,
        )

    for result in record.get('results', []):
        score_text = result.get('final_score', result['score'])
        draw_xywh_box(draw, result['bbox'], f"SEL {result['label_text']} {score_text:.2f}", color='red', width=4)
        draw_segmentation_polygons(draw, result.get('segmentation', []), color='cyan', width=2)

    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    image.save(vis_dir / f'{image_path.stem}_semantic_overlay.jpg')


def build_submission_records(outputs: list[dict[str, Any]], category_name_to_id: dict[str, int]) -> list[dict[str, Any]]:
    submission: list[dict[str, Any]] = []
    for record in outputs:
        image_id = record['image_id']
        for result in record.get('results', []):
            category_name = result.get('matched_category')
            if not category_name:
                continue
            category_id = category_name_to_id.get(category_name)
            if category_id is None:
                continue
            segmentation = result.get('segmentation', [])
            bbox = result.get('bbox', [])
            area = result.get('area')
            if area is None and len(bbox) == 4:
                area = float(bbox[2]) * float(bbox[3])
            submission.append({
                'image_id': image_id,
                'score': float(result.get('score', result.get('final_score', 0.0))),
                'category_id': category_id,
                'area': float(area or 0.0),
                'bbox': bbox,
                'segmentation': segmentation,
            })
    return submission


def main() -> None:
    args = parse_args()
    set_active_family_config(args.family_config)
    output_dir = resolve_output_dir(args.output_dir, args.flat_output, args.date_tag, args.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    annotations_by_image = {ann['image_id']: ann for ann in dataset.get('annotations', [])}
    categories_by_id = {cat['id']: cat['name'] for cat in dataset.get('categories', [])}
    category_name_to_id = {cat['name'].replace('_', ' '): cat['id'] for cat in dataset.get('categories', [])}
    category_names = list(category_name_to_id.keys())
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    support_image_paths = []
    if args.controller_mode == 'support_query':
        if not args.support_dir or not args.support_json:
            raise ValueError('support_query mode requires --support-dir and --support-json')
        support_image_paths = load_support_image_paths(args.support_json, args.support_dir)

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=max(args.llm_max_new_tokens, 128),
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )
    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        client=shared_vlm,
    )
    localizer = GroundingDinoLocalizer(args.config_path, args.checkpoint_path, device=args.device)
    calibrator = ProposalCalibrator(
        model_path=args.llm_model,
        max_new_tokens=128,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        calibration_mode=args.calibration_mode,
        support_json_path=args.support_json,
        support_dir=args.support_dir,
        reference_source=args.reference_source,
        client=shared_vlm,
    )
    segmenter = None if args.disable_sam else SamSegmenter(args.sam_checkpoint, device=args.device)
    pipeline = SemanticGdinoSamPipeline(controller, localizer, calibrator, segmenter)

    outputs = []
    runtime_stats_path = Path(args.runtime_stats_jsonl).resolve() if args.runtime_stats_jsonl else None
    runtime_stats_fh = runtime_stats_path.open('w') if runtime_stats_path else None
    progress = tqdm(images, desc='challenge_repo dev', unit='image')
    try:
        for index, image_info in enumerate(progress, start=1):
            query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
            image_start = time.perf_counter()
            record = pipeline.run(
                support_image_paths=support_image_paths if args.controller_mode == 'support_query' else [],
                query_image_path=query_image_path,
                image_id=image_info['id'],
                category_names=category_names,
                category_name_to_id=category_name_to_id,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                proposal_nms_iou=args.proposal_nms_iou,
                max_candidates=args.max_candidates,
                final_score_threshold=args.final_score_threshold,
                use_sam=not args.disable_sam,
                null_policy=args.null_policy,
                use_hybrid_category_scores=args.use_hybrid_category_scores,
                hybrid_score_threshold=args.hybrid_score_threshold,
                hybrid_margin=args.hybrid_margin,
                hybrid_iou_threshold=args.hybrid_iou_threshold,
                hybrid_exact_confidence_threshold=args.hybrid_exact_confidence_threshold,
                use_document_text=not args.disable_document_text,
                classification_top_k=args.classification_top_k,
            )
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed_sec = time.perf_counter() - image_start
            outputs.append(record)
            runtime_stats: dict[str, Any] = {
                'image_index': index,
                'image_id': image_info['id'],
                'elapsed_sec': round(elapsed_sec, 3),
                'semantic_family': record['semantic_family'],
                'null_likely': record['null_likely'],
                'proposal_count': len(record['proposal_candidates']),
                'selected_count': len(record['results']),
            }
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                runtime_stats['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1)
                torch.cuda.reset_peak_memory_stats(args.device)
            top_hybrid = record.get('category_hybrid_signals', [])
            hybrid_text = ''
            if top_hybrid:
                hybrid_text = f" hybrid_top={top_hybrid[0]['category']}:{top_hybrid[0]['score']:.2f}"
            progress.set_postfix({
                'image_id': image_info['id'],
                'family': record['semantic_family'] or '-',
                'selected': len(record['results']),
                'sec': f'{elapsed_sec:.1f}',
            })
            message = (
                f"Processed image_id={image_info['id']} family={record['semantic_family']} "
                f"null={record['null_likely']} proposals={len(record['proposal_candidates'])} "
                f"selected={len(record['results'])} elapsed_sec={elapsed_sec:.1f}{hybrid_text}"
            )
            if 'gpu_memory_reserved_mb' in runtime_stats:
                message += (
                    f" gpu_alloc_mb={runtime_stats['gpu_memory_allocated_mb']:.1f}"
                    f" gpu_reserved_mb={runtime_stats['gpu_memory_reserved_mb']:.1f}"
                    f" gpu_peak_mb={runtime_stats['gpu_max_memory_allocated_mb']:.1f}"
                )
            tqdm.write(message)
            if runtime_stats_fh is not None:
                runtime_stats_fh.write(json.dumps(runtime_stats) + '\n')
                runtime_stats_fh.flush()
            if args.save_vis:
                gt_ann = annotations_by_image.get(image_info['id'])
                gt_label = categories_by_id.get(gt_ann['category_id']) if gt_ann else None
                save_visualization(record, output_dir, gt_ann=gt_ann, gt_label=gt_label)
            if args.cuda_cleanup_interval > 0 and index % args.cuda_cleanup_interval == 0:
                release_torch_runtime()
        progress.close()
    finally:
        if runtime_stats_fh is not None:
            runtime_stats_fh.close()

    submission_records = build_submission_records(outputs, category_name_to_id)
    (output_dir / 'semantic_pipeline_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'query_submission.json').write_text(json.dumps(submission_records, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    tqdm.write(f"Saved semantic pipeline outputs to: {output_dir / 'semantic_pipeline_results.json'}")
    tqdm.write(f"Saved submission-format outputs to: {output_dir / 'query_submission.json'}")


if __name__ == '__main__':
    main()
