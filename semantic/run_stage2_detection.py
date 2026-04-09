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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 2 detection only.')
    parser.add_argument('--stage1-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--config-path', required=True)
    parser.add_argument('--checkpoint-path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--box-threshold', type=float, default=0.25)
    parser.add_argument('--text-threshold', type=float, default=0.25)
    parser.add_argument('--proposal-nms-iou', type=float, default=0.6)
    parser.add_argument('--max-candidates', type=int, default=5)
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
        proposal_candidates: list[dict[str, object]] = []
        if run_detection and record['proposal_prompts']:
            collected: list[dict[str, object]] = []
            for prompt in record['proposal_prompts']:
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
            'proposal_prompts': record['proposal_prompts'],
            'null_likely': record['null_likely'],
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
