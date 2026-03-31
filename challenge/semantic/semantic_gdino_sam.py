from __future__ import annotations

import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mmcv
import torch
from PIL import Image
from torchvision import ops

REPO_ROOT = Path(__file__).resolve().parents[2]
LLM2SEG_DIR = Path(os.environ.get('LLM2SEG_DIR', REPO_ROOT / 'third_party' / 'LLM2Seg'))
if str(LLM2SEG_DIR) not in sys.path:
    sys.path.insert(0, str(LLM2SEG_DIR))

from call_vlm import SwiftVLMCaller
from utils import preprocess_caption
from challenge.baseline.qwen_gdino_sam import (
    DetectionCandidate,
    GroundingDinoLocalizer,
    SamSegmenter,
    dedupe_preserve_order,
    load_support_image_paths,
)

QUERY_ONLY_SEMANTIC_PROMPT = """
You are given one query image.
Analyze the image and identify the most likely private-object family and short detector-friendly prompts.

Return only the following XML-like fields:
<family>short coarse family such as financial document, receipt-like paper, newspaper page, payment card, medical form, plastic test device, pill bottle, tattoo sleeve</family>
<summary>one short sentence describing the most likely target object appearance</summary>
<cue>comma-separated detector prompts from coarse to specific</cue>
<null>yes or no</null>

Rules:
- Prefer coarse family-level wording and short content descriptions that a detector can use.
- Keep the cue list short, usually 2 to 4 items.
- If no likely private target appears in the image, output <null>yes</null> and keep <cue>empty</cue>.
- Do not output any extra text.
""".strip()

SEMANTIC_CONTROLLER_PROMPT = """
You are given support image(s) followed by one query image.
Use the support images as the target reference and analyze the query image.

Return only the following XML-like fields:
<family>short coarse family such as financial document, receipt-like paper, newspaper page, payment card, medical form, plastic test device, pill bottle, tattoo sleeve</family>
<summary>one short sentence describing the target object appearance in the query image</summary>
<cue>comma-separated detector prompts from coarse to specific</cue>
<null>yes or no</null>

Rules:
- The support images define the target object type.
- Prefer coarse family-level wording and short content descriptions that a detector can use.
- Keep the cue list short, usually 2 to 4 items.
- If the query image likely does not contain the target, output <null>yes</null> and keep <cue>empty</cue>.
- Do not output any extra text.
""".strip()

RERANK_PROMPT = """
You are given support image(s) followed by one candidate crop from the query image.
The support images show the target private object type.
Decide whether the final candidate crop matches the same object type.

Return only the following XML-like fields:
<decision>yes or no</decision>
<score>integer 0 to 100</score>
<label>short object label</label>
<reason>short phrase</reason>

Rules:
- Be strict.
- Use a high score only if the crop clearly matches the support images.
- If the crop is background or the wrong object, return <decision>no</decision>.
- Do not output any extra text.
""".strip()

NEGATIVE_VALUES = {'yes', 'true', '1'}
LOW_SIGNAL_PROMPTS = {
    'blurry', 'white', 'black', 'small', 'large', 'background', 'table', 'wooden table',
    'wooden surface', 'surface', 'floor', 'room', 'photo', 'image', 'object'
}
PRIORITY_FAMILY_TERMS = (
    'document', 'paper', 'receipt', 'newspaper', 'card', 'bottle', 'prescription',
    'record', 'statement', 'report', 'transcript', 'test', 'box', 'sleeve', 'letter'
)


@dataclass
class SemanticCue:
    raw_text: str
    family: str
    summary: str
    proposal_prompts: list[str]
    null_likely: bool


@dataclass
class RerankDecision:
    raw_text: str
    decision: bool
    score: int
    label: str
    reason: str


class SemanticController:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        decoding_mode: str = 'deterministic',
        seed: int | None = None,
        max_pixels: int = 448,
        instruction: str | None = None,
        query_only_instruction: str | None = None,
    ) -> None:
        self.client = SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.instruction = instruction or SEMANTIC_CONTROLLER_PROMPT
        self.query_only_instruction = query_only_instruction or QUERY_ONLY_SEMANTIC_PROMPT

    def infer_query_only(self, query_image_path: str) -> SemanticCue:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text)

    def infer(self, support_image_paths: Sequence[str], query_image_path: str) -> SemanticCue:
        image_paths = [*map(str, support_image_paths), str(query_image_path)]
        raw_text = self.client.generate_images(image_paths, instruction=self.instruction)
        return _parse_semantic_cue(raw_text)


class ProposalReranker:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        decoding_mode: str = 'deterministic',
        seed: int | None = None,
        max_pixels: int = 448,
        instruction: str | None = None,
    ) -> None:
        self.client = SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.instruction = instruction or RERANK_PROMPT

    def score_candidate(
        self,
        support_image_paths: Sequence[str],
        query_image_path: str,
        candidate: DetectionCandidate,
    ) -> RerankDecision:
        crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
        try:
            raw_text = self.client.generate_images([*map(str, support_image_paths), crop_path], instruction=self.instruction)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        decision = _extract_tag(raw_text, 'decision').strip().lower() == 'yes'
        score_text = _extract_tag(raw_text, 'score').strip()
        try:
            score = int(score_text)
        except ValueError:
            score = 0
        score = max(0, min(100, score))
        return RerankDecision(
            raw_text=raw_text,
            decision=decision,
            score=score,
            label=_extract_tag(raw_text, 'label').strip(),
            reason=_extract_tag(raw_text, 'reason').strip(),
        )


class SemanticGdinoSamPipeline:
    def __init__(
        self,
        controller: SemanticController,
        localizer: GroundingDinoLocalizer,
        reranker: ProposalReranker,
        segmenter: SamSegmenter,
    ) -> None:
        self.controller = controller
        self.localizer = localizer
        self.reranker = reranker
        self.segmenter = segmenter

    def run(
        self,
        support_image_paths: Sequence[str] | None,
        query_image_path: str,
        image_id: int,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        proposal_nms_iou: float = 0.6,
        max_candidates: int = 5,
        use_sam: bool = True,
        null_policy: str = 'strict',
    ) -> dict[str, Any]:
        support_image_paths = list(support_image_paths or [])
        semantic = self.controller.infer(support_image_paths, query_image_path) if support_image_paths else self.controller.infer_query_only(query_image_path)
        proposal_candidates: list[dict[str, Any]] = []
        selected_result: list[dict[str, Any]] = []
        rerank_logs: list[dict[str, Any]] = []

        run_detection = _should_run_detection(semantic.null_likely, null_policy)
        if run_detection and semantic.proposal_prompts:
            candidates = self._collect_candidates(
                query_image_path=query_image_path,
                prompts=semantic.proposal_prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                proposal_nms_iou=proposal_nms_iou,
                max_candidates=max_candidates,
            )
            for candidate in candidates:
                proposal_candidates.append({
                    'score': candidate.score,
                    'label_text': candidate.label_text,
                    'bbox_xyxy': candidate.xyxy,
                })

            if candidates:
                if not support_image_paths:
                    best_candidate = candidates[0]
                    selected_result = _finalize_candidate_result(
                        self.segmenter,
                        query_image_path,
                        best_candidate,
                        image_id=image_id,
                        use_sam=use_sam,
                    )
                    if selected_result:
                        selected_result[0]['rerank_score'] = None
                        selected_result[0]['rerank_label'] = ''
                        selected_result[0]['rerank_reason'] = 'query_only_top_detection'
                        selected_result[0]['final_score'] = best_candidate.score
                else:
                    best_candidate = None
                    best_decision = None
                    best_final_score = -1.0
                    for candidate in candidates:
                        decision = self.reranker.score_candidate(support_image_paths, query_image_path, candidate)
                        final_score = candidate.score * (decision.score / 100.0)
                        rerank_logs.append({
                            'candidate_label_text': candidate.label_text,
                            'candidate_score': candidate.score,
                            'candidate_bbox_xyxy': candidate.xyxy,
                            'decision': decision.decision,
                            'semantic_score': decision.score,
                            'label': decision.label,
                            'reason': decision.reason,
                            'raw_text': decision.raw_text,
                            'final_score': final_score,
                        })
                        if decision.decision and final_score > best_final_score:
                            best_candidate = candidate
                            best_decision = decision
                            best_final_score = final_score
                    if best_candidate is not None and best_decision is not None:
                        selected_result = _finalize_candidate_result(
                            self.segmenter,
                            query_image_path,
                            best_candidate,
                            image_id=image_id,
                            use_sam=use_sam,
                        )
                        if selected_result:
                            selected_result[0]['rerank_score'] = best_decision.score
                            selected_result[0]['rerank_label'] = best_decision.label
                            selected_result[0]['rerank_reason'] = best_decision.reason
                            selected_result[0]['final_score'] = best_final_score

        return {
            'image_id': image_id,
            'query_image_path': str(query_image_path),
            'support_image_paths': [str(p) for p in support_image_paths],
            'controller_mode': 'support_query' if support_image_paths else 'query_only',
            'semantic_raw_text': semantic.raw_text,
            'semantic_family': semantic.family,
            'semantic_summary': semantic.summary,
            'proposal_prompts': semantic.proposal_prompts,
            'null_likely': semantic.null_likely,
            'null_policy': null_policy,
            'proposal_candidates': proposal_candidates,
            'rerank_logs': rerank_logs,
            'results': selected_result,
        }

    def _collect_candidates(
        self,
        query_image_path: str,
        prompts: Sequence[str],
        box_threshold: float,
        text_threshold: float,
        proposal_nms_iou: float,
        max_candidates: int,
    ) -> list[DetectionCandidate]:
        collected: list[DetectionCandidate] = []
        for prompt in prompts:
            collected.extend(
                detect_free_text(
                    self.localizer,
                    query_image_path,
                    prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            )
        if not collected:
            return []

        boxes = torch.tensor([candidate.xyxy for candidate in collected], dtype=torch.float32)
        scores = torch.tensor([candidate.score for candidate in collected], dtype=torch.float32)
        keep = ops.nms(boxes, scores, iou_threshold=proposal_nms_iou)
        ranked = [collected[idx] for idx in keep.tolist()]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:max_candidates]


def detect_free_text(
    localizer: GroundingDinoLocalizer,
    image_path: str,
    cue_text: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    max_dets: int = 50,
) -> list[DetectionCandidate]:
    if not cue_text:
        return []
    image = mmcv.imread(image_path, channel_order='rgb')
    prompt = preprocess_caption(cue_text)
    label_texts = [label.strip() for label in cue_text.split(',') if label.strip()]
    if not label_texts:
        return []
    try:
        result = localizer.model(inputs=image, texts=[prompt])
    except RuntimeError as exc:
        if 'selected index k out of range' in str(exc):
            return []
        raise
    if isinstance(result, list):
        result = result[0]
    predictions = result.get('predictions', []) if isinstance(result, dict) else []
    if not predictions or not isinstance(predictions[0], dict):
        return []
    first_pred = predictions[0]
    boxes = first_pred.get('bboxes', [])
    scores = first_pred.get('scores', [])
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    if len(boxes) == 0 or len(scores) == 0:
        return []

    top_indices = torch.argsort(scores, descending=True)[:max_dets]
    detections: list[DetectionCandidate] = []
    for idx in top_indices.tolist():
        score = float(scores[idx].item())
        if score < box_threshold or score < text_threshold:
            continue
        label_text = label_texts[idx] if idx < len(label_texts) else label_texts[0]
        detections.append(
            DetectionCandidate(
                score=score,
                label_text=label_text,
                category_id=-1,
                xyxy=[float(v) for v in boxes[idx].tolist()],
            )
        )
    return detections


def _parse_semantic_cue(raw_text: str) -> SemanticCue:
    family = _extract_tag(raw_text, 'family')
    summary = _extract_tag(raw_text, 'summary')
    cue = _extract_tag(raw_text, 'cue')
    null_text = _extract_tag(raw_text, 'null').lower()
    proposal_prompts = _normalize_prompt_list(cue, family, summary)
    return SemanticCue(
        raw_text=raw_text,
        family=family,
        summary=summary,
        proposal_prompts=proposal_prompts,
        null_likely=null_text in NEGATIVE_VALUES,
    )


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return match.group(1).strip()


def _normalize_prompt_list(cue: str, family: str, summary: str) -> list[str]:
    items = []
    for value in (cue, family, summary):
        if not value:
            continue
        if value == cue:
            items.extend(part.strip() for part in cue.split(',') if part.strip())
        else:
            items.append(value.strip())

    deduped = []
    for item in dedupe_preserve_order(items):
        normalized = item.strip()
        lowered = normalized.lower()
        if lowered in {'empty', 'none', 'no', 'n/a'}:
            continue
        if lowered in LOW_SIGNAL_PROMPTS:
            continue
        if len(lowered.split()) == 1 and lowered not in PRIORITY_FAMILY_TERMS and lowered in {'white', 'black', 'blurry', 'small', 'large'}:
            continue
        deduped.append(normalized)

    ranked = sorted(deduped, key=_prompt_priority)
    return ranked[:4]


def _prompt_priority(prompt: str) -> tuple[int, int]:
    lowered = prompt.lower()
    if any(term in lowered for term in PRIORITY_FAMILY_TERMS):
        return (0, len(prompt))
    if len(lowered.split()) <= 2:
        return (1, len(prompt))
    return (2, len(prompt))


def _should_run_detection(null_likely: bool, null_policy: str) -> bool:
    if null_policy == 'ignore':
        return True
    if null_policy == 'skip':
        return not null_likely
    if null_policy == 'strict':
        return not null_likely
    raise ValueError(f'Unknown null_policy: {null_policy}')


def _save_candidate_crop(query_image_path: str, xyxy: Sequence[float]) -> str:
    image = Image.open(query_image_path).convert('RGB')
    width, height = image.size
    x1, y1, x2, y2 = xyxy
    left_f, right_f = sorted((float(x1), float(x2)))
    top_f, bottom_f = sorted((float(y1), float(y2)))
    left = max(0, min(width - 1, int(round(left_f))))
    top = max(0, min(height - 1, int(round(top_f))))
    right = max(left + 1, min(width, int(round(right_f))))
    bottom = max(top + 1, min(height, int(round(bottom_f))))
    crop = image.crop((left, top, right, bottom))
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        crop.save(tmp.name)
        return tmp.name


def _finalize_candidate_result(
    segmenter: SamSegmenter,
    query_image_path: str,
    candidate: DetectionCandidate,
    image_id: int,
    use_sam: bool,
) -> list[dict[str, Any]]:
    if use_sam:
        return segmenter.segment(query_image_path, [candidate], image_id=image_id)
    x1, y1, x2, y2 = candidate.xyxy
    return [{
        'image_id': image_id,
        'score': candidate.score,
        'category_id': candidate.category_id,
        'bbox': [x1, y1, x2 - x1, y2 - y1],
        'area': float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
        'segmentation': [],
        'label_text': candidate.label_text,
    }]


__all__ = [
    'GroundingDinoLocalizer',
    'SamSegmenter',
    'SemanticController',
    'ProposalReranker',
    'SemanticGdinoSamPipeline',
    'load_support_image_paths',
]
