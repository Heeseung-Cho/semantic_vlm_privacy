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

from semantic.semantic_gdino_sam import GroundingDinoLocalizer, detect_free_text, should_run_detection


INVALID_STAGE2_PROMPTS = {'null', 'none', 'empty', 'n/a'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 2 detection only.')
    parser.add_argument('--stage1_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--box_threshold', type=float, default=0.25)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--proposal_nms_iou', type=float, default=0.6)
    parser.add_argument('--max_candidates', type=int, default=5)
    return parser.parse_args()


def _nms_rank(candidates: list[dict[str, object]], proposal_nms_iou: float, max_candidates: int) -> list[dict[str, object]]:
    import torch
    from torchvision import ops

    if not candidates:
        return []
    boxes = torch.tensor([candidate['bbox_xyxy'] for candidate in candidates], dtype=torch.float32)
    scores = torch.tensor([candidate['score'] for candidate in candidates], dtype=torch.float32)
    keep = ops.nms(boxes, scores, iou_threshold=proposal_nms_iou)
    ranked = [candidates[idx] for idx in keep.tolist()]
    ranked.sort(key=lambda item: float(item['score']), reverse=True)
    return ranked[:max_candidates]


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage1_payload = json.loads(Path(args.stage1_path).read_text())
    records = stage1_payload['records']

    localizer = GroundingDinoLocalizer(args.config_path, args.checkpoint_path, device=args.device)
    outputs: list[dict[str, object]] = []
    progress = tqdm(records, desc='stage2 detection', unit='image')
    for record in progress:
        run_detection = should_run_detection(bool(record['null_likely']), str(record['null_policy']))
        skip_reason = None
        if bool(record['null_likely']):
            run_detection = False
            skip_reason = 'stage1_null'
        proposal_candidates: list[dict[str, object]] = []
        prompt_list = []
        if run_detection:
            prompt_list = [
                str(prompt).strip()
                for prompt in record.get('proposal_prompts', [])
                if str(prompt).strip() and str(prompt).strip().lower() not in INVALID_STAGE2_PROMPTS
            ]
            if not prompt_list:
                skip_reason = 'empty_prompt_list'
        if run_detection and prompt_list:
            collected: list[dict[str, object]] = []
            for prompt in prompt_list:
                detections = detect_free_text(
                    localizer=localizer,
                    image_path=str(record['query_image_path']),
                    cue_text=str(prompt),
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                )
                for detection in detections:
                    collected.append({
                        'score': float(detection.score),
                        'label_text': detection.label_text,
                        'source_prompt': str(prompt),
                        'bbox_xyxy': list(detection.xyxy),
                    })
            proposal_candidates = _nms_rank(
                candidates=collected,
                proposal_nms_iou=args.proposal_nms_iou,
                max_candidates=args.max_candidates,
            )

        outputs.append({
            'image_id': record['image_id'],
            'query_image_path': record['query_image_path'],
            'support_image_paths': record['support_image_paths'],
            'controller_mode': record['controller_mode'],
            'null_policy': record['null_policy'],
            'semantic_family': record['semantic_family'],
            'route_type': record.get('route_type', ''),
            'semantic_categories': record.get('semantic_categories', []),
            'proposal_prompts': prompt_list,
            'null_likely': record['null_likely'],
            'stage2_skip_reason': skip_reason,
            'proposal_candidates': proposal_candidates,
        })
        progress.set_postfix({
            'image_id': record['image_id'],
            'selected': len(proposal_candidates),
        })
        tqdm.write(
            f"Stage2 image_id={record['image_id']} null={record['null_likely']} "
            f"candidates={len(proposal_candidates)}"
        )
    progress.close()

    payload = {
        'config': vars(args),
        'records': outputs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tqdm.write(f'Saved Stage 2 outputs to: {output_path}')


if __name__ == '__main__':
    main()
