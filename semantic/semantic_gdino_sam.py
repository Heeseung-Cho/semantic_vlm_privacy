from __future__ import annotations

import re
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mmcv
import torch
from PIL import Image
from torchvision import ops

from common.vlm import SwiftVLMCaller
from common.text_utils import preprocess_caption
from baseline.qwen_gdino_sam import (
    DetectionCandidate,
    GroundingDinoLocalizer,
    SamSegmenter,
    dedupe_preserve_order,
    load_support_image_paths,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / 'prompts' / 'active'


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text().strip()


QUERY_ONLY_SEMANTIC_PROMPT = _load_prompt('semantic_query_only.txt')
SEMANTIC_CONTROLLER_PROMPT = _load_prompt('semantic_support_query.txt')
CALIBRATION_PROMPT = _load_prompt('semantic_rerank.txt')
DOCUMENT_TEXT_PROMPT = _load_prompt('semantic_document_text.txt')
TRANSACTIONAL_TEXT_PROMPT = _load_prompt('semantic_transactional_text.txt')
REFERENCE_MATCH_PROMPT = _load_prompt('semantic_reference_match.txt')

NEGATIVE_VALUES = {'yes', 'true', '1'}
LOW_SIGNAL_PROMPTS = {
    'blurry', 'white', 'black', 'small', 'large', 'background', 'table', 'wooden table',
    'wooden surface', 'surface', 'floor', 'room', 'photo', 'image', 'object'
}
PRIORITY_FAMILY_TERMS = (
    'document', 'paper', 'receipt', 'newspaper', 'card', 'bottle', 'prescription',
    'record', 'statement', 'report', 'transcript', 'test', 'box', 'sleeve', 'letter'
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
    raw_text: str
    family: str
    summary: str
    proposal_prompts: list[str]
    null_likely: bool
    text_hint_raw: str = ''
    text_hint_summary: str = ''
    text_hint_tokens: list[str] = field(default_factory=list)


@dataclass
class CalibrationDecision:
    raw_text: str
    decision: bool
    object_valid: bool
    family_match: bool
    exact_match: bool
    score: int
    label: str
    category: str
    reason: str


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
        document_text_instruction: str | None = None,
        transactional_text_instruction: str | None = None,
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
        self.document_text_instruction = document_text_instruction or DOCUMENT_TEXT_PROMPT
        self.transactional_text_instruction = transactional_text_instruction or TRANSACTIONAL_TEXT_PROMPT

    def infer_query_only(self, query_image_path: str) -> SemanticCue:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text)

    def infer(self, support_image_paths: Sequence[str], query_image_path: str) -> SemanticCue:
        image_paths = [*map(str, support_image_paths), str(query_image_path)]
        raw_text = self.client.generate_images(image_paths, instruction=self.instruction)
        return _parse_semantic_cue(raw_text)

    def enrich_with_document_text(self, query_image_path: str, semantic: SemanticCue) -> SemanticCue:
        if not _should_extract_document_text(semantic):
            return semantic
        instruction = self.transactional_text_instruction if _should_extract_transactional_text(semantic) else self.document_text_instruction
        raw_text = self.client.generate(query_image_path, instruction=instruction)
        text_summary = _extract_tag(raw_text, 'document_hint')
        text_tokens = _normalize_text_token_list(_extract_tag(raw_text, 'text'))
        return SemanticCue(
            raw_text=semantic.raw_text,
            family=semantic.family,
            summary=semantic.summary,
            proposal_prompts=list(semantic.proposal_prompts),
            null_likely=semantic.null_likely,
            text_hint_raw=raw_text,
            text_hint_summary=text_summary,
            text_hint_tokens=text_tokens,
        )


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
    ) -> None:
        self.client = SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        self.mode = calibration_mode
        self.instruction = instruction or CALIBRATION_PROMPT
        self.reference_instruction = reference_instruction or REFERENCE_MATCH_PROMPT
        self.support_references: list[SupportReferenceCrop] = []
        if self.mode == 'reference_match' and support_json_path and support_dir:
            self.support_references = _build_support_reference_crops(support_json_path, support_dir)

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
                candidate=candidate,
                allowed_categories=allowed_categories or [],
                has_support_images=bool(support_image_paths),
            )
            image_paths = [*map(str, support_image_paths), str(query_image_path), crop_path]
            raw_text = self.client.generate_images(image_paths, instruction=instruction)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        decision = _extract_bool_tag(raw_text, 'decision')
        object_valid = _extract_bool_tag(raw_text, 'object_valid', default=decision)
        family_match = _extract_bool_tag(raw_text, 'family_match', default=decision)
        exact_match = _extract_bool_tag(raw_text, 'exact_match', default=decision)
        score_text = _extract_tag(raw_text, 'score').strip()
        try:
            score = int(score_text)
        except ValueError:
            score = 0
        score = max(0, min(100, score))
        return CalibrationDecision(
            raw_text=raw_text,
            decision=decision,
            object_valid=object_valid,
            family_match=family_match,
            exact_match=exact_match,
            score=score,
            label=_extract_tag(raw_text, 'label').strip(),
            category=_extract_tag(raw_text, 'category').strip(),
            reason=_extract_tag(raw_text, 'reason').strip(),
        )

    def _score_candidate_by_reference(
        self,
        query_image_path: str,
        candidate: DetectionCandidate,
        allowed_categories: Sequence[str],
    ) -> CalibrationDecision:
        crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
        try:
            instruction = _build_reference_match_instruction(
                base_instruction=self.reference_instruction,
                support_references=self.support_references,
                allowed_categories=allowed_categories,
            )
            image_paths = [entry.crop_path for entry in self.support_references]
            image_paths.append(crop_path)
            raw_text = self.client.generate_images(image_paths, instruction=instruction)
        finally:
            Path(crop_path).unlink(missing_ok=True)
        category = _extract_tag(raw_text, 'category').strip()
        score_text = _extract_tag(raw_text, 'score').strip()
        try:
            score = int(score_text)
        except ValueError:
            score = 0
        score = max(0, min(100, score))
        return CalibrationDecision(
            raw_text=raw_text,
            decision=bool(category),
            object_valid=bool(category),
            family_match=True,
            exact_match=bool(category),
            score=score,
            label=category,
            category=category,
            reason=_extract_tag(raw_text, 'reason').strip(),
        )


class SemanticGdinoSamPipeline:
    def __init__(
        self,
        controller: SemanticController,
        localizer: GroundingDinoLocalizer,
        calibrator: ProposalCalibrator,
        segmenter: SamSegmenter,
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
        use_hybrid_category_scores: bool = False,
        hybrid_score_threshold: float = 0.35,
        hybrid_margin: float = 0.05,
        hybrid_iou_threshold: float = 0.50,
        hybrid_exact_confidence_threshold: int = 95,
        use_document_text: bool = True,
        classification_top_k: int | None = None,
    ) -> dict[str, Any]:
        support_image_paths = list(support_image_paths or [])
        category_names = list(category_names or [])
        category_name_to_id = dict(category_name_to_id or {})
        semantic = self.controller.infer(support_image_paths, query_image_path) if support_image_paths else self.controller.infer_query_only(query_image_path)
        if use_document_text:
            semantic = self.controller.enrich_with_document_text(query_image_path, semantic)
        if self.calibrator.mode == 'reference_match':
            use_hybrid_category_scores = False
        category_shortlist = _build_category_shortlist(semantic, category_names)
        proposal_candidates: list[dict[str, Any]] = []
        selected_result: list[dict[str, Any]] = []
        calibration_logs: list[dict[str, Any]] = []
        category_hybrid_signals: list[dict[str, Any]] = []

        run_detection = _should_run_detection(semantic.null_likely, null_policy)
        if run_detection and use_hybrid_category_scores and category_shortlist:
            category_hybrid_signals = self._collect_category_signals(
                query_image_path=query_image_path,
                categories=category_shortlist,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
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
                if self.calibrator.mode == 'reference_match':
                    hybrid_resolution = {
                        'applied': False,
                        'resolved_category': matched_category,
                        'reason': 'reference_match',
                        'top_category': None,
                        'top_score': 0.0,
                        'second_category': None,
                        'second_score': 0.0,
                        'margin': 0.0,
                        'overlap': 0.0,
                        'text_hint_category': None,
                        'score_factor': 1.0,
                    }
                else:
                    hybrid_resolution = _resolve_hybrid_category(
                        category_signals=category_hybrid_signals,
                        candidate_xyxy=candidate.xyxy,
                        matched_category=matched_category,
                        calibration_category=decision.category,
                        calibration_label=decision.label,
                        exact_match=decision.exact_match,
                        calibration_score=decision.score,
                        exact_confidence_threshold=hybrid_exact_confidence_threshold,
                        score_threshold=hybrid_score_threshold,
                        margin_threshold=hybrid_margin,
                        iou_threshold=hybrid_iou_threshold,
                        semantic_family=semantic.family or '',
                        text_hint=semantic.text_hint_summary,
                        allowed_categories=category_shortlist,
                    )
                    if hybrid_resolution['resolved_category'] is not None:
                        matched_category = hybrid_resolution['resolved_category']
                final_score = candidate.score * (decision.score / 100.0) * hybrid_resolution['score_factor']
                semantic_family = semantic.family or ''
                exact_threshold = _family_exact_score_threshold(semantic_family, final_score_threshold)
                soft_threshold = _family_soft_score_threshold(semantic_family, final_score_threshold)

                if self.calibrator.mode == 'reference_match':
                    confident_exact = matched_category is not None and decision.exact_match and decision.decision
                else:
                    confident_exact = _has_confident_exact_consensus(
                        matched_category=matched_category,
                        calibration_category=decision.category,
                        calibration_label=decision.label,
                        exact_match=decision.exact_match,
                        calibration_score=decision.score,
                        exact_confidence_threshold=hybrid_exact_confidence_threshold,
                    )
                exact_keep = (
                    decision.object_valid
                    and decision.family_match
                    and decision.decision
                    and confident_exact
                    and matched_category is not None
                    and final_score >= exact_threshold
                )
                soft_family_keep = (
                    decision.object_valid
                    and decision.family_match
                    and final_score >= soft_threshold
                    and not _requires_exact_category(semantic_family)
                )
                accepted = exact_keep or soft_family_keep
                acceptance_mode = 'rejected'
                if exact_keep:
                    acceptance_mode = 'exact'
                elif soft_family_keep:
                    acceptance_mode = 'family_soft'
                category_id = category_name_to_id.get(matched_category, -1) if matched_category else -1
                final_label = matched_category or decision.label.strip() or candidate.label_text
                calibration_logs.append({
                    'candidate_label_text': candidate.label_text,
                    'candidate_score': candidate.score,
                    'candidate_bbox_xyxy': candidate.xyxy,
                    'decision': decision.decision,
                    'object_valid': decision.object_valid,
                    'family_match': decision.family_match,
                    'exact_match': decision.exact_match,
                    'calibration_score': decision.score,
                    'calibration_label': decision.label,
                    'calibration_category': decision.category,
                    'matched_category': matched_category,
                    'hybrid_category_applied': hybrid_resolution['applied'],
                    'hybrid_category': hybrid_resolution['resolved_category'],
                    'hybrid_reason': hybrid_resolution['reason'],
                    'hybrid_top_category': hybrid_resolution['top_category'],
                    'hybrid_top_score': hybrid_resolution['top_score'],
                    'hybrid_second_category': hybrid_resolution['second_category'],
                    'hybrid_second_score': hybrid_resolution['second_score'],
                    'hybrid_margin': hybrid_resolution['margin'],
                    'hybrid_overlap': hybrid_resolution['overlap'],
                    'text_hint_category': hybrid_resolution['text_hint_category'],
                    'hybrid_score_factor': hybrid_resolution['score_factor'],
                    'evidence_table': {
                        'calibration_category': matched_category,
                        'hybrid_top_category': hybrid_resolution['top_category'],
                        'hybrid_second_category': hybrid_resolution['second_category'],
                        'hybrid_margin': hybrid_resolution['margin'],
                        'hybrid_overlap': hybrid_resolution['overlap'],
                        'text_hint_category': hybrid_resolution['text_hint_category'],
                        'semantic_family': semantic.family or '',
                    },
                    'reason': decision.reason,
                    'raw_text': decision.raw_text,
                    'accepted': accepted,
                    'acceptance_mode': acceptance_mode,
                    'soft_family_keep': soft_family_keep,
                    'final_score': final_score,
                })
                if not accepted:
                    continue
                kept_candidates.append({
                    'detection': DetectionCandidate(
                        score=candidate.score,
                        label_text=final_label,
                        category_id=category_id,
                        xyxy=list(candidate.xyxy),
                    ),
                    'detector_label_text': candidate.label_text,
                    'detector_score': candidate.score,
                    'calibration_score': decision.score,
                    'calibration_label': decision.label,
                    'calibration_category': decision.category,
                    'matched_category': matched_category,
                    'hybrid_category_applied': hybrid_resolution['applied'],
                    'hybrid_category': hybrid_resolution['resolved_category'],
                    'hybrid_reason': hybrid_resolution['reason'],
                    'hybrid_top_category': hybrid_resolution['top_category'],
                    'hybrid_top_score': hybrid_resolution['top_score'],
                    'hybrid_score_factor': hybrid_resolution['score_factor'],
                    'calibration_reason': decision.reason,
                    'acceptance_mode': acceptance_mode,
                    'object_valid': decision.object_valid,
                    'family_match': decision.family_match,
                    'exact_match': decision.exact_match,
                    'final_score': final_score,
                })

            kept_candidates.sort(key=lambda item: item['final_score'], reverse=True)
            finalized = _finalize_candidate_results(
                self.segmenter,
                query_image_path,
                [item['detection'] for item in kept_candidates],
                image_id=image_id,
                use_sam=use_sam,
            )
            for result, item in zip(finalized, kept_candidates):
                result['detector_score'] = item['detector_score']
                result['detector_label_text'] = item['detector_label_text']
                result['calibration_score'] = item['calibration_score']
                result['calibration_label'] = item['calibration_label']
                result['calibration_category'] = item['calibration_category']
                result['matched_category'] = item['matched_category']
                result['hybrid_category_applied'] = item['hybrid_category_applied']
                result['hybrid_category'] = item['hybrid_category']
                result['hybrid_reason'] = item['hybrid_reason']
                result['hybrid_top_category'] = item['hybrid_top_category']
                result['hybrid_top_score'] = item['hybrid_top_score']
                result['hybrid_score_factor'] = item['hybrid_score_factor']
                result['calibration_reason'] = item['calibration_reason']
                result['acceptance_mode'] = item['acceptance_mode']
                result['object_valid'] = item['object_valid']
                result['family_match'] = item['family_match']
                result['exact_match'] = item['exact_match']
                result['final_score'] = item['final_score']
            selected_result = finalized

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
            'document_text_hint': semantic.text_hint_summary,
            'document_text_tokens': semantic.text_hint_tokens,
            'document_text_raw_text': semantic.text_hint_raw,
            'null_policy': null_policy,
            'proposal_candidates': proposal_candidates,
            'category_shortlist': category_shortlist,
            'category_hybrid_signals': category_hybrid_signals,
            'calibration_logs': calibration_logs,
            'rerank_logs': calibration_logs,
            'results': selected_result,
        }

    def _collect_category_signals(
        self,
        query_image_path: str,
        categories: Sequence[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for category in categories:
            detections = detect_free_text(
                self.localizer,
                query_image_path,
                category,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                max_dets=1,
            )
            best = detections[0] if detections else None
            records.append({
                'category': category,
                'score': float(best.score) if best else 0.0,
                'bbox_xyxy': list(best.xyxy) if best else [],
                'label_text': best.label_text if best else category,
            })
        records.sort(key=lambda item: item['score'], reverse=True)
        return records

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


def _extract_bool_tag(text: str, tag: str, default: bool = False) -> bool:
    value = _extract_tag(text, tag).strip().lower()
    if not value:
        return default
    return value in NEGATIVE_VALUES


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
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        crop.save(tmp.name)
        return tmp.name


def _build_support_reference_crops(support_json_path: str | Path, support_dir: str | Path) -> list[SupportReferenceCrop]:
    payload = json.loads(Path(support_json_path).read_text())
    support_dir = Path(support_dir)
    categories_by_id = {int(cat['id']): cat['name'].replace('_', ' ') for cat in payload.get('categories', [])}
    images_by_id = {int(image_info['id']): image_info for image_info in payload.get('images', [])}
    crop_dir = Path(tempfile.mkdtemp(prefix='support_reference_crops_'))
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
        crop_path = _save_candidate_crop(str(image_path), [x, y, x + w, y + h])
        references.append(SupportReferenceCrop(image_id=image_id, category_name=category_name, crop_path=str(crop_path)))
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
        'The images are ordered as: support reference crops first, then one candidate crop last.',
        'Support reference labels in order:',
        *support_lines,
        f'Allowed categories: {allowed_text}',
    ])


def _finalize_candidate_results(
    segmenter: SamSegmenter,
    query_image_path: str,
    candidates: Sequence[DetectionCandidate],
    image_id: int,
    use_sam: bool,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    if use_sam:
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


def _xyxy_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    if not box1 or not box2:
        return 0.0
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def _map_text_hint_to_category(text_hint: str, allowed_categories: Sequence[str]) -> str | None:
    normalized_hint = _normalize_phrase(text_hint)
    hint_to_candidates = {
        'receipt': ['bills or receipt', 'bills_or_receipt'],
        'bank statement': ['bank statement', 'bank_statement'],
        'addressed letter': ['letters with address', 'letters_with_address'],
        'transcript': ['transcript'],
        'prescription': ['doctors prescription', 'doctors_prescription', 'medical record document', 'medical_record_document'],
    }
    for candidate_name in hint_to_candidates.get(normalized_hint, []):
        matched = _match_allowed_category(candidate_name, allowed_categories)
        if matched is not None:
            return matched
    return None


def _has_confident_exact_consensus(
    *,
    matched_category: str | None,
    calibration_category: str,
    calibration_label: str,
    exact_match: bool,
    calibration_score: int,
    exact_confidence_threshold: int,
) -> bool:
    if matched_category is None:
        return False
    if exact_match and calibration_score >= exact_confidence_threshold:
        return True
    normalized_match = _normalize_phrase(matched_category)
    normalized_category = _normalize_phrase(calibration_category)
    normalized_label = _normalize_phrase(calibration_label)
    if calibration_score < exact_confidence_threshold:
        return False
    if normalized_category == normalized_match:
        return True
    return normalized_label == normalized_match


def _resolve_hybrid_category(
    category_signals: Sequence[dict[str, Any]],
    candidate_xyxy: Sequence[float],
    matched_category: str | None,
    calibration_category: str,
    calibration_label: str,
    exact_match: bool,
    calibration_score: int,
    exact_confidence_threshold: int,
    score_threshold: float,
    margin_threshold: float,
    iou_threshold: float,
    semantic_family: str,
    text_hint: str,
    allowed_categories: Sequence[str],
) -> dict[str, Any]:
    text_hint_category = _map_text_hint_to_category(text_hint, allowed_categories)
    if not category_signals:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'no_category_signals',
            'top_category': None,
            'top_score': 0.0,
            'second_category': None,
            'second_score': 0.0,
            'margin': 0.0,
            'overlap': 0.0,
            'text_hint_category': text_hint_category,
            'score_factor': 1.0,
        }

    ranked = sorted(category_signals, key=lambda item: float(item.get('score', 0.0)), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    top_category = top.get('category')
    top_score = float(top.get('score', 0.0))
    second_category = second.get('category') if second else None
    second_score = float(second.get('score', 0.0)) if second else 0.0
    margin = top_score - second_score
    overlap = _xyxy_iou(candidate_xyxy, top.get('bbox_xyxy', []))

    confident_exact = _has_confident_exact_consensus(
        matched_category=matched_category,
        calibration_category=calibration_category,
        calibration_label=calibration_label,
        exact_match=exact_match,
        calibration_score=calibration_score,
        exact_confidence_threshold=exact_confidence_threshold,
    )
    score_factor = 0.75 + 0.25 * top_score
    base = {
        'top_category': top_category,
        'top_score': top_score,
        'second_category': second_category,
        'second_score': second_score,
        'margin': margin,
        'overlap': overlap,
        'text_hint_category': text_hint_category,
        'score_factor': 1.0,
    }

    if text_hint_category is not None and text_hint_category == matched_category == top_category:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'three_way_consensus_retained',
            **base,
        }

    if text_hint_category is not None and text_hint_category == matched_category and not confident_exact:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'text_hint_supported_retained',
            **base,
        }

    if (
        text_hint_category is not None
        and text_hint_category != matched_category
        and _normalize_phrase(text_hint) in {'addressed letter', 'bank statement', 'transcript', 'prescription'}
        and not confident_exact
    ):
        return {
            'applied': True,
            'resolved_category': text_hint_category,
            'reason': 'specific_text_hint_override',
            **base,
        }

    if (
        _normalize_phrase(semantic_family) == 'statement or report document'
        and matched_category is not None
        and top_category is not None
        and matched_category != top_category
        and top_score >= score_threshold
        and margin >= margin_threshold
        and overlap >= iou_threshold
    ):
        return {
            'applied': True,
            'resolved_category': top_category,
            'reason': 'statement_report_hybrid_margin_override',
            **base,
            'score_factor': score_factor,
        }

    if confident_exact:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'confident_exact_retained',
            **base,
        }

    if top_score < score_threshold:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'top_score_below_threshold',
            **base,
        }
    if margin < margin_threshold:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'top2_margin_too_small',
            **base,
        }
    if overlap < iou_threshold:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'top_category_box_mismatch',
            **base,
        }

    if matched_category == top_category:
        return {
            'applied': False,
            'resolved_category': matched_category,
            'reason': 'same_category_score_boost',
            **base,
            'score_factor': score_factor,
        }

    return {
        'applied': True,
        'resolved_category': top_category,
        'reason': 'low_confidence_hybrid_fallback',
        **base,
        'score_factor': score_factor,
    }



def _should_extract_document_text(semantic: SemanticCue) -> bool:
    context = ' '.join(
        part for part in [semantic.family, semantic.summary, ' '.join(semantic.proposal_prompts)] if part
    ).lower()
    return any(token in context for token in [
        'document', 'paper', 'receipt', 'statement', 'report', 'transcript', 'letter', 'address', 'newspaper', 'prescription', 'medical form'
    ])


def _should_extract_transactional_text(semantic: SemanticCue) -> bool:
    context = ' '.join(
        part for part in [semantic.family, semantic.summary, ' '.join(semantic.proposal_prompts)] if part
    ).lower()
    return any(token in context for token in [
        'transactional paper', 'receipt', 'bill', 'statement', 'report', 'letter', 'addressed correspondence', 'educational record', 'medical document', 'transcript', 'prescription'
    ])


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


def _normalize_text_token_list(raw_text: str) -> list[str]:
    tokens = []
    for item in raw_text.split(','):
        normalized = _normalize_phrase(item)
        if not normalized or normalized in {'none', 'unknown', 'unreadable', 'n a'}:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens[:8]


def _build_category_shortlist(semantic: SemanticCue, allowed_categories: Sequence[str]) -> list[str]:
    allowed_categories = list(allowed_categories)
    if not allowed_categories:
        return []

    family = _normalize_phrase(semantic.family)
    summary = _normalize_phrase(semantic.summary)
    prompt_text = _normalize_phrase(' '.join(semantic.proposal_prompts))
    text_hint = _normalize_phrase(semantic.text_hint_summary)
    text_tokens = ' '.join(_normalize_phrase(token) for token in semantic.text_hint_tokens)
    context = ' '.join(part for part in [family, summary, prompt_text, text_hint, text_tokens] if part)

    def pick(*names: str) -> list[str]:
        seen = set()
        picked = []
        normalized_allowed = {
            _normalize_phrase(category).replace(' ', '_'): category for category in allowed_categories
        }
        for name in names:
            resolved_name = normalized_allowed.get(_normalize_phrase(name).replace(' ', '_'))
            if resolved_name is None:
                continue
            if resolved_name not in seen:
                picked.append(resolved_name)
                seen.add(resolved_name)
        return picked

    shortlist: list[str] = []

    # 1) Family-first mapping with family-bounded shortlist.
    if family == 'printed financial card':
        shortlist = pick('credit or debit card', 'business card')
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
            shortlist = pick('bank statement', 'mortgage or investment report', 'transcript')
    elif family == 'transactional paper':
        if text_hint == 'addressed letter':
            shortlist = pick('letters with address', 'bills or receipt')
        elif text_hint == 'bank statement':
            shortlist = pick('bank statement', 'bills or receipt')
        elif text_hint == 'transcript':
            shortlist = pick('transcript')
        elif text_hint == 'prescription':
            shortlist = pick('doctors prescription', 'medical record document')
        elif text_hint == 'receipt':
            shortlist = pick('bills or receipt', 'bank statement')
        elif _context_contains_any(context, ['address', 'mail', 'letter', 'contact information', 'dear', 'street', 'avenue', 'zip', 'recipient']):
            shortlist = pick('letters with address', 'bills or receipt')
        elif _context_contains_any(context, ['account', 'balance', 'routing', 'statement', 'deposit', 'withdrawal', 'checking', 'savings']):
            shortlist = pick('bank statement', 'bills or receipt')
        elif _context_contains_any(context, ['grade', 'course', 'school', 'student', 'transcript', 'gpa', 'semester', 'credits', 'university']):
            shortlist = pick('transcript')
        elif _context_contains_any(context, ['prescription', 'rx', 'doctor', 'dosage', 'refill', 'pharmacy', 'medication', 'prescriber']):
            shortlist = pick('doctors prescription', 'medical record document')
        elif _context_contains_any(context, ['subtotal', 'total', 'amount due', 'cash', 'card', 'change due', 'item', 'receipt']):
            shortlist = pick('bills or receipt', 'bank statement')
        else:
            shortlist = pick('bills or receipt', 'bank statement', 'letters with address', 'doctors prescription', 'transcript')
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

    # 2) Backward-compatible keyword fallback only when family-first mapping did not fire.
    if not shortlist:
        if 'business card' in context:
            shortlist = pick('business card')
        elif any(token in context for token in ['visa', 'credit card', 'debit card', 'payment card', 'plastic card']):
            shortlist = pick('credit or debit card', 'business card')
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
            shortlist = pick('bank statement', 'bills or receipt')
        elif 'address' in context or 'letter' in context:
            shortlist = pick('letters with address', 'bills or receipt')
        elif 'transcript' in context:
            shortlist = pick('transcript')
        elif any(token in context for token in ['receipt', 'bills', 'transaction details']):
            shortlist = pick('bills or receipt', 'bank statement', 'doctors prescription', 'letters with address', 'transcript')
        elif any(token in context for token in ['pill bottle', 'medicine bottle', 'prescription bottle', 'medical container', 'blue bottle', 'white bottle', 'cylindrical container']):
            shortlist = pick('empty pill bottle')
        elif 'condom box' in context:
            shortlist = pick('condom box')
        elif any(token in context for token in ['condom', 'plastic bag', 'foil packet', 'small packet']):
            shortlist = pick('condom with plastic bag', 'condom box')

    if not shortlist:
        shortlist = allowed_categories
    return shortlist

def _requires_exact_category(semantic_family: str) -> bool:
    family = _normalize_phrase(semantic_family)
    return family in {
        'printed financial card',
        'visual identity card',
        'health indicator',
        'tattoo sleeve',
    }


def _family_exact_score_threshold(semantic_family: str, base_threshold: float) -> float:
    family = _normalize_phrase(semantic_family)
    floor_by_family = {
        'health indicator': 0.35,
        'printed financial card': 0.40,
        'statement or report document': 0.35,
        'medical document': 0.35,
    }
    return max(base_threshold, floor_by_family.get(family, base_threshold))


def _family_soft_score_threshold(semantic_family: str, base_threshold: float) -> float:
    family = _normalize_phrase(semantic_family)
    floor_by_family = {
        'transactional paper': 0.30,
        'statement or report document': 0.42,
        'medical document': 0.42,
        'print media': 0.30,
        'addressed correspondence': 0.32,
        'educational record': 0.35,
    }
    return max(base_threshold, floor_by_family.get(family, base_threshold))


def _build_calibration_instruction(
    base_instruction: str,
    semantic: SemanticCue,
    candidate: DetectionCandidate,
    allowed_categories: Sequence[str],
    has_support_images: bool,
) -> str:
    semantic_family = semantic.family or 'unknown'
    semantic_summary = semantic.summary or 'unknown'
    proposal_prompts = ', '.join(semantic.proposal_prompts) if semantic.proposal_prompts else 'none'
    document_text_hint = semantic.text_hint_summary or 'none'
    document_text_tokens = ', '.join(semantic.text_hint_tokens) if semantic.text_hint_tokens else 'none'
    allowed_text = ', '.join(allowed_categories) if allowed_categories else 'none'
    mode_text = (
        'The images are ordered as: support reference image(s), full query image, candidate crop.'
        if has_support_images
        else 'The images are ordered as: full query image, candidate crop.'
    )
    return "\n".join([
        base_instruction,
        "",
        mode_text,
        f"Semantic family prior: {semantic_family}",
        f"Semantic summary prior: {semantic_summary}",
        f"Detector proposal text: {candidate.label_text}",
        f"Detector prompt set: {proposal_prompts}",
        f"Document text hint: {document_text_hint}",
        f"Visible text tokens: {document_text_tokens}",
        f"Allowed categories: {allowed_text}",
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


__all__ = [
    'GroundingDinoLocalizer',
    'SamSegmenter',
    'SemanticController',
    'ProposalCalibrator',
    'SemanticGdinoSamPipeline',
    'load_support_image_paths',
]
