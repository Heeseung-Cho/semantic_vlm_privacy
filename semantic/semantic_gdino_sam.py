from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mmcv
import torch
from PIL import Image
from torchvision import ops

from baseline.qwen_gdino_sam import (
    DetectionCandidate,
    GroundingDinoLocalizer,
    SamSegmenter,
    dedupe_preserve_order,
    load_support_image_paths,
)
from common.text_utils import preprocess_caption
from common.vlm import SwiftVLMCaller, release_torch_runtime

PROMPTS_DIR = Path(__file__).resolve().parents[1] / 'prompts' / 'active'


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text().strip()


QUERY_ONLY_SEMANTIC_PROMPT = _load_prompt('semantic_query_only.txt')
SEMANTIC_CONTROLLER_PROMPT = _load_prompt('semantic_support_query.txt')
CALIBRATION_PROMPT = _load_prompt('semantic_rerank.txt')
REFERENCE_MATCH_PROMPT = _load_prompt('semantic_reference_match.txt')

NEGATIVE_VALUES = {'yes', 'true', '1'}
LOW_SIGNAL_PROMPTS = {
    'blurry', 'white', 'black', 'small', 'large', 'background', 'table', 'wooden table',
    'wooden surface', 'surface', 'floor', 'room', 'photo', 'image', 'object',
}
PRIORITY_FAMILY_TERMS = (
    'document', 'paper', 'receipt', 'newspaper', 'card', 'bottle', 'prescription',
    'record', 'statement', 'report', 'transcript', 'test', 'box', 'sleeve', 'letter',
)
DESCRIPTIVE_SPLIT_PATTERNS = (
    r'\bsuggesting\b',
    r'\bshowing\b',
    r'\bwith\b',
    r'\bsitting\b',
    r'\blying\b',
    r'\bplaced\b',
    r'\bon top of\b',
    r'\bin the\b',
    r'\bon the\b',
    r'\bpartially\b',
    r'\bnext to\b',
)


@dataclass
class SemanticCue:
    family: str
    proposal_prompts: list[str]
    null_likely: bool


@dataclass
class CalibrationDecision:
    category: str
    decision: bool | None = None
    object_valid: bool | None = None
    family_match: bool | None = None
    exact_match: bool | None = None
    score: int | None = None
    label: str = ''
    reason: str = ''


@dataclass
class SupportReferenceCrop:
    image_id: int
    category_name: str
    crop_path: str


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
        client: SwiftVLMCaller | None = None,
    ) -> None:
        self.client = client or SwiftVLMCaller(
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


class ProposalCalibrator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        decoding_mode: str = 'deterministic',
        seed: int | None = None,
        max_pixels: int = 448,
        instruction: str | None = None,
        calibration_mode: str = 'legacy',
        support_json_path: str | None = None,
        support_dir: str | None = None,
        reference_instruction: str | None = None,
        reference_source: str = 'crop',
        client: SwiftVLMCaller | None = None,
    ) -> None:
        self.client = client or SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.mode = calibration_mode
        self.instruction = instruction or CALIBRATION_PROMPT
        self.reference_instruction = reference_instruction or REFERENCE_MATCH_PROMPT
        self.reference_source = reference_source
        self.support_references: list[SupportReferenceCrop] = []
        if self.mode == 'reference_match' and support_json_path and support_dir:
            self.support_references = _build_support_reference_crops(
                support_json_path,
                support_dir,
                reference_source=self.reference_source,
            )

    def score_candidate(
        self,
        support_image_paths: Sequence[str],
        query_image_path: str,
        candidate: DetectionCandidate,
        semantic: SemanticCue,
        allowed_categories: Sequence[str] | None = None,
    ) -> CalibrationDecision:
        if self.mode == 'reference_match' and self.support_references:
            return self._score_candidate_by_reference(
                query_image_path=query_image_path,
                candidate=candidate,
                allowed_categories=allowed_categories or [],
            )

        crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
        try:
            instruction = _build_calibration_instruction(
                base_instruction=self.instruction,
                semantic=semantic,
                allowed_categories=allowed_categories or [],
                has_support_images=bool(support_image_paths),
            )
            image_paths = [*map(str, support_image_paths), str(query_image_path), crop_path]
            raw_text = self.client.generate_images(image_paths, instruction=instruction)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        return _parse_calibration_decision(raw_text)

    def _score_candidate_by_reference(
        self,
        query_image_path: str,
        candidate: DetectionCandidate,
        allowed_categories: Sequence[str],
    ) -> CalibrationDecision:
        crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
        try:
            filtered_references = [
                entry for entry in self.support_references
                if not allowed_categories or entry.category_name in allowed_categories
            ]
            if not filtered_references:
                filtered_references = self.support_references
            instruction = _build_reference_match_instruction(
                base_instruction=self.reference_instruction,
                support_references=filtered_references,
                allowed_categories=allowed_categories,
            )
            image_paths = [entry.crop_path for entry in filtered_references]
            image_paths.append(crop_path)
            raw_text = self.client.generate_images(image_paths, instruction=instruction)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        return _parse_calibration_decision(raw_text)


class SemanticGdinoSamPipeline:
    def __init__(
        self,
        controller: SemanticController,
        localizer: GroundingDinoLocalizer,
        calibrator: ProposalCalibrator,
        segmenter: SamSegmenter | None,
    ) -> None:
        self.controller = controller
        self.localizer = localizer
        self.calibrator = calibrator
        self.segmenter = segmenter

    def run(
        self,
        support_image_paths: Sequence[str] | None,
        query_image_path: str,
        image_id: int,
        category_names: Sequence[str] | None = None,
        category_name_to_id: dict[str, int] | None = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        proposal_nms_iou: float = 0.6,
        max_candidates: int = 5,
        final_score_threshold: float = 0.30,
        use_sam: bool = True,
        null_policy: str = 'strict',
        classification_top_k: int | None = None,
    ) -> dict[str, Any]:
        support_image_paths = list(support_image_paths or [])
        category_names = list(category_names or [])
        category_name_to_id = dict(category_name_to_id or {})
        semantic = (
            self.controller.infer(support_image_paths, query_image_path)
            if support_image_paths
            else self.controller.infer_query_only(query_image_path)
        )
        category_shortlist = _build_category_shortlist(semantic, category_names)
        proposal_candidates: list[dict[str, Any]] = []
        calibration_logs: list[dict[str, Any]] = []
        selected_result: list[dict[str, Any]] = []

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

            kept_candidates: list[dict[str, Any]] = []
            candidates_for_scoring = candidates[:classification_top_k] if classification_top_k else candidates
            for candidate in candidates_for_scoring:
                decision = self.calibrator.score_candidate(
                    support_image_paths=support_image_paths,
                    query_image_path=query_image_path,
                    candidate=candidate,
                    semantic=semantic,
                    allowed_categories=category_shortlist,
                )
                matched_category = _match_allowed_category(decision.category, category_shortlist)
                accepted = _should_accept_calibration(
                    decision=decision,
                    matched_category=matched_category,
                    detector_score=candidate.score,
                    final_score_threshold=final_score_threshold,
                )
                final_score = candidate.score
                calibration_logs.append({
                    'candidate_score': candidate.score,
                    'candidate_bbox_xyxy': candidate.xyxy,
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
                    'final_score': final_score,
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
                    'detector_label_text': candidate.label_text,
                    'calibration_category': decision.category,
                    'matched_category': matched_category,
                    'final_score': final_score,
                })

            kept_candidates.sort(key=lambda item: item['final_score'], reverse=True)
            finalized = _finalize_candidate_results(
                segmenter=self.segmenter,
                query_image_path=query_image_path,
                candidates=[item['detection'] for item in kept_candidates],
                image_id=image_id,
                use_sam=use_sam,
            )
            for result, item in zip(finalized, kept_candidates):
                result['detector_score'] = item['detector_score']
                result['detector_label_text'] = item['detector_label_text']
                result['calibration_category'] = item['calibration_category']
                result['matched_category'] = item['matched_category']
                result['final_score'] = item['final_score']
            selected_result = finalized

        return {
            'image_id': image_id,
            'query_image_path': str(query_image_path),
            'support_image_paths': [str(path) for path in support_image_paths],
            'controller_mode': 'support_query' if support_image_paths else 'query_only',
            'semantic_family': semantic.family,
            'proposal_prompts': semantic.proposal_prompts,
            'null_likely': semantic.null_likely,
            'null_policy': null_policy,
            'proposal_candidates': proposal_candidates,
            'category_shortlist': category_shortlist,
            'calibration_logs': calibration_logs,
            'rerank_logs': calibration_logs,
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
        with torch.inference_mode():
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
                xyxy=[float(value) for value in boxes[idx].tolist()],
            )
        )
    return detections


def _parse_semantic_cue(raw_text: str) -> SemanticCue:
    family = _extract_tag(raw_text, 'family')
    cue = _extract_tag(raw_text, 'cue')
    null_text = _extract_tag(raw_text, 'null').lower()
    return SemanticCue(
        family=family,
        proposal_prompts=_normalize_prompt_list(cue, family),
        null_likely=null_text in NEGATIVE_VALUES,
    )


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return match.group(1).strip()


def _normalize_prompt_list(cue: str, family: str) -> list[str]:
    items: list[str] = []
    for value in (cue, family):
        if not value:
            continue
        if value == cue:
            items.extend(part.strip() for part in cue.split(',') if part.strip())
        else:
            items.append(value.strip())

    deduped: list[str] = []
    for item in dedupe_preserve_order(items):
        normalized = item.strip()
        lowered = normalized.lower()
        if lowered in {'empty', 'none', 'no', 'n/a', 'null', 'unknown'}:
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
    if null_policy in {'skip', 'strict'}:
        return not null_likely
    raise ValueError(f'Unknown null_policy: {null_policy}')


def _parse_yes_no_tag(raw_text: str, tag: str) -> bool | None:
    value = _extract_tag(raw_text, tag).strip().lower()
    if not value:
        return None
    if value in {'yes', 'true', '1'}:
        return True
    if value in {'no', 'false', '0'}:
        return False
    return None


def _parse_int_tag(raw_text: str, tag: str) -> int | None:
    value = _extract_tag(raw_text, tag).strip()
    if not value:
        return None
    match = re.search(r'-?\d+', value)
    if not match:
        return None
    return int(match.group(0))


def _parse_calibration_decision(raw_text: str) -> CalibrationDecision:
    return CalibrationDecision(
        category=_extract_tag(raw_text, 'category').strip(),
        decision=_parse_yes_no_tag(raw_text, 'decision'),
        object_valid=_parse_yes_no_tag(raw_text, 'object_valid'),
        family_match=_parse_yes_no_tag(raw_text, 'family_match'),
        exact_match=_parse_yes_no_tag(raw_text, 'exact_match'),
        score=_parse_int_tag(raw_text, 'score'),
        label=_extract_tag(raw_text, 'label').strip(),
        reason=_extract_tag(raw_text, 'reason').strip(),
    )


def _should_accept_calibration(
    decision: CalibrationDecision,
    matched_category: str | None,
    detector_score: float,
    final_score_threshold: float,
) -> bool:
    if not matched_category:
        return False
    if detector_score < final_score_threshold:
        return False
    if decision.decision is not None and not decision.decision:
        return False
    if decision.object_valid is not None and not decision.object_valid:
        return False
    if decision.family_match is not None and not decision.family_match:
        return False
    if decision.exact_match is not None and not decision.exact_match:
        return False
    return True


def _save_candidate_crop(query_image_path: str, xyxy: Sequence[float]) -> str:
    with Image.open(query_image_path) as src_image:
        image = src_image.convert('RGB')
        width, height = image.size
        x1, y1, x2, y2 = xyxy
        left_f, right_f = sorted((float(x1), float(x2)))
        top_f, bottom_f = sorted((float(y1), float(y2)))
        left = max(0, min(width - 1, int(round(left_f))))
        top = max(0, min(height - 1, int(round(top_f))))
        right = max(left + 1, min(width, int(round(right_f))))
        bottom = max(top + 1, min(height, int(round(bottom_f))))

        crop_w = right - left
        crop_h = bottom - top
        min_side = 28
        max_aspect_ratio = 50.0

        if crop_w < min_side:
            pad = min_side - crop_w
            left = max(0, left - pad // 2)
            right = min(width, right + (pad - pad // 2))
        if crop_h < min_side:
            pad = min_side - crop_h
            top = max(0, top - pad // 2)
            bottom = min(height, bottom + (pad - pad // 2))

        crop_w = right - left
        crop_h = bottom - top
        if crop_w / max(crop_h, 1) > max_aspect_ratio:
            target_h = min(height, max(min_side, int(round(crop_w / max_aspect_ratio))))
            extra = max(0, target_h - crop_h)
            top = max(0, top - extra // 2)
            bottom = min(height, bottom + (extra - extra // 2))
        elif crop_h / max(crop_w, 1) > max_aspect_ratio:
            target_w = min(width, max(min_side, int(round(crop_h / max_aspect_ratio))))
            extra = max(0, target_w - crop_w)
            left = max(0, left - extra // 2)
            right = min(width, right + (extra - extra // 2))

        if right <= left:
            right = min(width, left + 1)
        if bottom <= top:
            bottom = min(height, top + 1)

        crop = image.crop((left, top, right, bottom))
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                crop.save(tmp.name)
                return tmp.name
        finally:
            crop.close()
            image.close()
            release_torch_runtime()


def _build_support_reference_crops(
    support_json_path: str | Path,
    support_dir: str | Path,
    reference_source: str = 'crop',
) -> list[SupportReferenceCrop]:
    payload = json.loads(Path(support_json_path).read_text())
    support_dir = Path(support_dir)
    categories_by_id = {int(cat['id']): cat['name'].replace('_', ' ') for cat in payload.get('categories', [])}
    images_by_id = {int(image_info['id']): image_info for image_info in payload.get('images', [])}
    references: list[SupportReferenceCrop] = []
    for ann in payload.get('annotations', []):
        image_id = int(ann['image_id'])
        image_info = images_by_id.get(image_id)
        if image_info is None:
            continue
        category_name = categories_by_id.get(int(ann['category_id']))
        bbox = ann.get('bbox', [])
        if not category_name or len(bbox) != 4:
            continue
        image_path = support_dir / image_info['file_name']
        x, y, w, h = bbox
        if reference_source == 'full_image':
            reference_path = str(image_path)
        else:
            reference_path = _save_candidate_crop(str(image_path), [x, y, x + w, y + h])
        references.append(SupportReferenceCrop(image_id=image_id, category_name=category_name, crop_path=reference_path))
    references.sort(key=lambda item: item.image_id)
    return references


def _build_reference_match_instruction(
    base_instruction: str,
    support_references: Sequence[SupportReferenceCrop],
    allowed_categories: Sequence[str],
) -> str:
    support_lines = [f'{idx}. {entry.category_name}' for idx, entry in enumerate(support_references, start=1)]
    allowed_text = ', '.join(allowed_categories) if allowed_categories else 'all support categories'
    return '\n'.join([
        base_instruction,
        '',
        'The images are ordered as: support reference images first, then one candidate crop last.',
        'Support reference labels in order:',
        *support_lines,
        f'Allowed categories: {allowed_text}',
    ])


def _finalize_candidate_results(
    segmenter: SamSegmenter | None,
    query_image_path: str,
    candidates: Sequence[DetectionCandidate],
    image_id: int,
    use_sam: bool,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if use_sam:
        if segmenter is None:
            raise ValueError('SAM segmentation requested but segmenter is not initialized')
        return segmenter.segment(query_image_path, list(candidates), image_id=image_id)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate.xyxy
        results.append({
            'image_id': image_id,
            'score': candidate.score,
            'category_id': candidate.category_id,
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'area': float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
            'segmentation': [],
            'label_text': candidate.label_text,
        })
    return results


def finalize_candidate_results(
    segmenter: SamSegmenter | None,
    query_image_path: str,
    candidates: Sequence[DetectionCandidate],
    image_id: int,
    use_sam: bool,
) -> list[dict[str, Any]]:
    return _finalize_candidate_results(segmenter, query_image_path, candidates, image_id, use_sam)


def _context_contains_any(context: str, tokens: Sequence[str]) -> bool:
    return any(token in context for token in tokens)


def _infer_health_indicator_subfamily(context: str) -> str:
    if _context_contains_any(context, ['pill bottle', 'medicine bottle', 'prescription bottle', 'medication bottle', 'cylindrical bottle', 'container', 'bottle']):
        return 'bottle'
    if _context_contains_any(context, ['pregnancy test box', 'boxed pregnancy test', 'clearblue box', 'test kit box', 'retail box', 'box']):
        return 'box'
    if _context_contains_any(context, ['pregnancy test', 'plastic test device', 'urine test', 'medical test', 'test stick', 'test device']):
        return 'test'
    if _context_contains_any(context, ['condom', 'plastic bag', 'wrapped package', 'foil packet', 'small packet', 'wrapped item', 'packet']):
        return 'wrapped item'
    return 'generic'


def _build_category_shortlist(semantic: SemanticCue, allowed_categories: Sequence[str]) -> list[str]:
    allowed_categories = list(allowed_categories)
    if not allowed_categories:
        return []

    family = _normalize_phrase(semantic.family)
    prompt_text = _normalize_phrase(' '.join(semantic.proposal_prompts))
    context = ' '.join(part for part in [family, prompt_text] if part)

    def pick(*names: str) -> list[str]:
        seen = set()
        picked: list[str] = []
        normalized_allowed = {
            _normalize_phrase(category).replace(' ', '_'): category for category in allowed_categories
        }
        for name in names:
            resolved_name = normalized_allowed.get(_normalize_phrase(name).replace(' ', '_'))
            if resolved_name is None or resolved_name in seen:
                continue
            picked.append(resolved_name)
            seen.add(resolved_name)
        return picked

    shortlist: list[str] = []
    if family == 'printed financial card':
        shortlist = pick('credit or debit card')
    elif family == 'visual identity card':
        shortlist = pick('business card')
    elif family == 'print media':
        shortlist = pick('local newspaper')
    elif family == 'medical document':
        if any(token in context for token in ['prescription', 'rx', 'doctor']):
            shortlist = pick('doctors prescription', 'medical record document')
        else:
            shortlist = pick('medical record document', 'doctors prescription')
    elif family == 'educational record':
        shortlist = pick('transcript')
    elif family == 'addressed correspondence':
        shortlist = pick('letters with address', 'bills or receipt')
    elif family == 'statement or report document':
        if _context_contains_any(context, ['mortgage', 'investment', 'annual report', 'financial report']):
            shortlist = pick('mortgage or investment report', 'bank statement')
        elif _context_contains_any(context, ['bank statement', 'account statement', 'account details', 'account', 'balance', 'routing', 'deposit']):
            shortlist = pick('bank statement', 'mortgage or investment report')
        else:
            shortlist = pick('mortgage or investment report', 'bank statement')
    elif family == 'health indicator':
        health_subfamily = _infer_health_indicator_subfamily(context)
        if health_subfamily == 'bottle':
            shortlist = pick('empty pill bottle')
        elif health_subfamily == 'box':
            if _context_contains_any(context, ['condom box', 'boxed condom', 'retail condom package', 'contraceptive box']):
                shortlist = pick('condom box', 'condom with plastic bag')
            else:
                shortlist = pick('pregnancy test box', 'condom box', 'pregnancy test')
        elif health_subfamily == 'test':
            shortlist = pick('pregnancy test', 'pregnancy test box')
        elif health_subfamily == 'wrapped item':
            shortlist = pick('condom with plastic bag', 'condom box')
        else:
            shortlist = pick(
                'pregnancy test',
                'pregnancy test box',
                'condom with plastic bag',
                'condom box',
                'empty pill bottle',
            )
    elif family == 'tattoo sleeve':
        shortlist = pick('tattoo sleeve')

    if not shortlist:
        if 'business card' in context:
            shortlist = pick('business card')
        elif any(token in context for token in ['visa', 'credit card', 'debit card', 'payment card', 'plastic card']):
            shortlist = pick('credit or debit card')
        elif any(token in context for token in ['pregnancy test box', 'boxed pregnancy test', 'boxed test', 'box test', 'test kit box']):
            shortlist = pick('pregnancy test box', 'pregnancy test')
        elif any(token in context for token in ['pregnancy test', 'plastic test device', 'urine test', 'medical test', 'test stick']):
            shortlist = pick('pregnancy test', 'pregnancy test box')
        elif any(token in context for token in ['tattoo sleeve', 'patterned sleeve', 'sleeve']) or family == 'clothing':
            shortlist = pick('tattoo sleeve')
        elif any(token in context for token in ['newspaper', 'qr code']):
            shortlist = pick('local newspaper')
        elif any(token in context for token in ['prescription', 'doctor', 'rx']):
            shortlist = pick('doctors prescription', 'medical record document')
        elif any(token in context for token in ['medical form', 'health form', 'patient form', 'medical record']):
            shortlist = pick('medical record document', 'doctors prescription')
        elif any(token in context for token in ['mortgage', 'investment report', 'financial document']):
            shortlist = pick('mortgage or investment report', 'bank statement', 'transcript')
        elif any(token in context for token in ['bank statement', 'account statement']):
            shortlist = pick('bank statement', 'mortgage or investment report')
        elif 'address' in context or 'letter' in context or 'envelope' in context:
            shortlist = pick('letters with address', 'bills or receipt')
        elif 'transcript' in context:
            shortlist = pick('transcript')
        elif any(token in context for token in ['receipt', 'bills', 'transaction details', 'merchant receipt', 'itemized receipt']):
            shortlist = pick('bills or receipt', 'letters with address')
        elif any(token in context for token in ['pill bottle', 'medicine bottle', 'prescription bottle', 'medical container', 'blue bottle', 'white bottle', 'cylindrical container']):
            shortlist = pick('empty pill bottle')
        elif 'condom box' in context:
            shortlist = pick('condom box')
        elif any(token in context for token in ['condom', 'plastic bag', 'foil packet', 'small packet']):
            shortlist = pick('condom with plastic bag', 'condom box')

    if not shortlist:
        shortlist = allowed_categories
    return shortlist


def build_category_shortlist(semantic: SemanticCue, allowed_categories: Sequence[str]) -> list[str]:
    return _build_category_shortlist(semantic, allowed_categories)


def _build_calibration_instruction(
    base_instruction: str,
    semantic: SemanticCue,
    allowed_categories: Sequence[str],
    has_support_images: bool,
) -> str:
    semantic_family = semantic.family or 'unknown'
    proposal_prompts = ', '.join(semantic.proposal_prompts) if semantic.proposal_prompts else 'none'
    allowed_text = ', '.join(allowed_categories) if allowed_categories else 'none'
    mode_text = (
        'The images are ordered as: support reference image(s), full query image, candidate crop.'
        if has_support_images
        else 'The images are ordered as: full query image, candidate crop.'
    )
    return '\n'.join([
        base_instruction,
        '',
        mode_text,
        f'Semantic family prior: {semantic_family}',
        f'Detector prompt set: {proposal_prompts}',
        f'Allowed categories: {allowed_text}',
    ])


def _match_allowed_category(text: str, allowed_categories: Sequence[str]) -> str | None:
    normalized_item = _normalize_phrase(text)
    if not normalized_item:
        return None
    normalized_allowed = {_normalize_phrase(category): category for category in allowed_categories}
    return normalized_allowed.get(normalized_item)


def _normalize_phrase(text: str) -> str:
    phrase = (text or '').strip().strip('[](){}')
    phrase = phrase.replace('_', ' ')
    phrase = re.sub(r'^\d+\.\s*', '', phrase)
    phrase = re.sub(r'\s+', ' ', phrase)
    lowered = phrase.lower()
    for pattern in DESCRIPTIVE_SPLIT_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            phrase = phrase[:match.start()].strip(' ,.;:-')
            break
    phrase = re.sub(r'[.;:]+$', '', phrase.strip())
    return phrase.lower()


def should_run_detection(null_likely: bool, null_policy: str) -> bool:
    return _should_run_detection(null_likely, null_policy)


__all__ = [
    'CalibrationDecision',
    'DetectionCandidate',
    'GroundingDinoLocalizer',
    'SamSegmenter',
    'SemanticCue',
    'SemanticController',
    'ProposalCalibrator',
    'SemanticGdinoSamPipeline',
    'build_category_shortlist',
    'detect_free_text',
    'finalize_candidate_results',
    'load_support_image_paths',
    'should_run_detection',
]
