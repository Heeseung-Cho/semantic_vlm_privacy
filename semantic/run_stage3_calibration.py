#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller
from semantic.semantic_gdino_sam import (
    DetectionCandidate,
    ProposalCalibrator,
    SamSegmenter,
    SemanticCue,
    build_category_shortlist,
    finalize_candidate_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 3 calibration/finalization only.')
    parser.add_argument('--json-path', required=True)
    parser.add_argument('--stage1-path', required=True)
    parser.add_argument('--stage2-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm-decoding-mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-max-pixels', type=int, default=448)
    parser.add_argument('--calibration-mode', choices=['legacy', 'reference_match'], default='legacy')
    parser.add_argument('--reference-source', choices=['crop', 'full_image'], default='crop')
    parser.add_argument('--support-dir', default=None)
    parser.add_argument('--support-json', default=None)
    parser.add_argument('--disable-sam', action='store_true')
    parser.add_argument('--final-score-threshold', type=float, default=0.30)
    parser.add_argument('--classification-top-k', type=int, default=None)
    return parser.parse_args()


def build_submission_records(outputs: list[dict[str, object]], category_name_to_id: dict[str, int]) -> list[dict[str, object]]:
    submission: list[dict[str, object]] = []
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


def _normalize_category_name(text: str) -> str:
    return ' '.join((text or '').strip().replace('_', ' ').lower().split())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    category_name_to_id = {cat['name'].replace('_', ' '): cat['id'] for cat in dataset.get('categories', [])}
    category_names = list(category_name_to_id.keys())

    stage1_payload = json.loads(Path(args.stage1_path).read_text())
    stage2_payload = json.loads(Path(args.stage2_path).read_text())
    stage1_by_image = {int(record['image_id']): record for record in stage1_payload['records']}
    stage2_records = stage2_payload['records']

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=128,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
    )
    if args.calibration_mode == 'reference_match' and (not args.support_json or not args.support_dir):
        raise ValueError('reference_match mode requires --support-json and --support-dir')
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

    outputs: list[dict[str, object]] = []
    progress = tqdm(stage2_records, desc='stage3 calibration', unit='image')
    for detection_record in progress:
        image_id = int(detection_record['image_id'])
        semantic_record = stage1_by_image[image_id]
        semantic = SemanticCue(
            family=str(semantic_record['semantic_family']),
            proposal_prompts=list(semantic_record['proposal_prompts']),
            null_likely=bool(semantic_record['null_likely']),
        )
        category_shortlist = build_category_shortlist(semantic, category_names)
        kept_candidates: list[dict[str, object]] = []
        calibration_logs: list[dict[str, object]] = []

        raw_candidates = list(detection_record.get('proposal_candidates', []))
        candidates_for_scoring = (
            raw_candidates[:args.classification_top_k]
            if args.classification_top_k
            else raw_candidates
        )
        for candidate_record in candidates_for_scoring:
            candidate = DetectionCandidate(
                score=float(candidate_record['score']),
                label_text='candidate',
                category_id=-1,
                xyxy=[float(value) for value in candidate_record['bbox_xyxy']],
            )
            decision = calibrator.score_candidate(
                support_image_paths=detection_record.get('support_image_paths', []),
                query_image_path=str(detection_record['query_image_path']),
                candidate=candidate,
                semantic=semantic,
                allowed_categories=category_shortlist,
            )
            matched_category = None
            normalized_allowed = {
                _normalize_category_name(name): name for name in category_shortlist
            }
            normalized_decision = _normalize_category_name(decision.category)
            if normalized_decision in normalized_allowed:
                matched_category = normalized_allowed[normalized_decision]
            accepted = bool(
                matched_category
                and candidate.score >= args.final_score_threshold
                and (decision.decision is not False)
                and (decision.object_valid is not False)
                and (decision.family_match is not False)
                and (decision.exact_match is not False)
            )
            calibration_logs.append({
                'candidate_score': candidate.score,
                'candidate_bbox_xyxy': list(candidate.xyxy),
                'calibration_decision': decision.decision,
                'object_valid': decision.object_valid,
                'family_match': decision.family_match,
                'exact_match': decision.exact_match,
                'calibration_score': decision.score,
                'calibration_label': decision.label,
                'calibration_category': decision.category,
                'calibration_reason': decision.reason,
                'matched_category': matched_category,
                'accepted': accepted,
                'final_score': candidate.score,
            })
            if not accepted:
                continue
            kept_candidates.append({
                'detection': DetectionCandidate(
                    score=candidate.score,
                    label_text=matched_category,
                    category_id=category_name_to_id.get(matched_category, -1),
                    xyxy=list(candidate.xyxy),
                ),
                'detector_score': candidate.score,
                'calibration_category': decision.category,
                'matched_category': matched_category,
                'final_score': candidate.score,
            })

        kept_candidates.sort(key=lambda item: item['final_score'], reverse=True)
        finalized = finalize_candidate_results(
            segmenter=segmenter,
            query_image_path=str(detection_record['query_image_path']),
            candidates=[item['detection'] for item in kept_candidates],
            image_id=image_id,
            use_sam=not args.disable_sam,
        )
        for result, item in zip(finalized, kept_candidates):
            result['detector_score'] = item['detector_score']
            result['calibration_category'] = item['calibration_category']
            result['matched_category'] = item['matched_category']
            result['final_score'] = item['final_score']

        outputs.append({
            'image_id': image_id,
            'query_image_path': detection_record['query_image_path'],
            'support_image_paths': detection_record.get('support_image_paths', []),
            'controller_mode': detection_record['controller_mode'],
            'semantic_family': semantic.family,
            'proposal_prompts': semantic.proposal_prompts,
            'null_likely': semantic.null_likely,
            'null_policy': detection_record['null_policy'],
            'proposal_candidates': raw_candidates,
            'category_shortlist': category_shortlist,
            'calibration_logs': calibration_logs,
            'rerank_logs': calibration_logs,
            'results': finalized,
        })
        progress.set_postfix({
            'image_id': image_id,
            'selected': len(finalized),
        })
        tqdm.write(
            f"Stage3 image_id={image_id} family={semantic.family} "
            f"candidates={len(raw_candidates)} selected={len(finalized)}"
        )
    progress.close()

    submission_records = build_submission_records(outputs, category_name_to_id)
    (output_dir / 'semantic_pipeline_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'query_submission.json').write_text(json.dumps(submission_records, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    tqdm.write(f"Saved Stage 3 outputs to: {output_dir / 'semantic_pipeline_results.json'}")
    tqdm.write(f"Saved Stage 3 submission to: {output_dir / 'query_submission.json'}")


if __name__ == '__main__':
    main()
