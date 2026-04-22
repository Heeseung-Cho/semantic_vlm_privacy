from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mmcv
import torch
from PIL import Image
from torchvision import ops

from common.text_utils import preprocess_caption
from common.vlm import SwiftVLMCaller, release_torch_runtime
from baseline.qwen_gdino_sam import (
    DetectionCandidate,
    GroundingDinoLocalizer,
    SamSegmenter,
    dedupe_preserve_order,
    load_support_image_paths,
)
from semantic.family_config import (
    canonicalize_family_name,
    get_category_family,
    get_family_categories,
    render_prompt_with_family_config,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / 'prompts' / 'active'


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text().strip()


QUERY_ONLY_SEMANTIC_PROMPT_TEMPLATE = _load_prompt('stage1/semantic_query_only.txt')
DOCUMENT_TEXT_PROMPT = _load_prompt('stage1/semantic_document_text.txt')
TRANSACTIONAL_TEXT_PROMPT = _load_prompt('stage1/semantic_transactional_text.txt')
REFERENCE_MATCH_PROMPT = _load_prompt('stage3/semantic_reference_match.txt')

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
    raw_text: str = ''
    family: str = ''
    route_type: str = ''
    route_confidence: str = ''
    categories: list[str] = field(default_factory=list)
    summary: str = ''
    proposal_prompts: list[str] = field(default_factory=list)
    null_likely: bool = False
    text_hint_raw: str = ''
    text_hint_summary: str = ''
    text_hint_tokens: list[str] = field(default_factory=list)


@dataclass
class CalibrationDecision:
    raw_text: str = ''
    decision: bool = False
    object_valid: bool = False
    family_match: bool = False
    exact_match: bool = False
    score: int = 0
    label: str = ''
    category: str = ''
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
        document_text_instruction: str | None = None,
        transactional_text_instruction: str | None = None,
        client: SwiftVLMCaller | None = None,
    ) -> None:
        self.client = client or SwiftVLMCaller(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            decoding_mode=decoding_mode,
            seed=seed,
            max_pixels=max_pixels,
        )
        query_instruction = query_only_instruction or QUERY_ONLY_SEMANTIC_PROMPT_TEMPLATE
        self.instruction = render_prompt_with_family_config(instruction) if instruction else ''
        self.query_only_instruction = render_prompt_with_family_config(query_instruction)
        self.document_text_instruction = document_text_instruction or DOCUMENT_TEXT_PROMPT
        self.transactional_text_instruction = transactional_text_instruction or TRANSACTIONAL_TEXT_PROMPT

    def infer_query_only(self, query_image_path: str) -> SemanticCue:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text)

    def infer_query_only_with_raw(self, query_image_path: str) -> tuple[SemanticCue, str]:
        raw_text = self.client.generate(query_image_path, instruction=self.query_only_instruction)
        return _parse_semantic_cue(raw_text), raw_text

    def infer(self, support_image_paths: Sequence[str], query_image_path: str) -> SemanticCue:
        image_paths = [*map(str, support_image_paths), str(query_image_path)]
        raw_text = self.client.generate_images(image_paths, instruction=self.instruction)
        return _parse_semantic_cue(raw_text)

    def infer_with_raw(self, support_image_paths: Sequence[str], query_image_path: str) -> tuple[SemanticCue, str]:
        image_paths = [*map(str, support_image_paths), str(query_image_path)]
        raw_text = self.client.generate_images(image_paths, instruction=self.instruction)
        return _parse_semantic_cue(raw_text), raw_text

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
            route_type=semantic.route_type,
            categories=list(semantic.categories),
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
        self.instruction = instruction or ''
        self.reference_instruction = reference_instruction or REFERENCE_MATCH_PROMPT
        self.reference_source = reference_source
        self.support_references: list[SupportReferenceCrop] = []
        if self.mode == 'reference_match' and support_json_path and support_dir:
            self.support_references = _build_support_reference_crops(support_json_path, support_dir, reference_source=self.reference_source)

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
            category=_extract_category_text(raw_text).strip(),
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
            filtered_references = [entry for entry in self.support_references if not allowed_categories or entry.category_name in allowed_categories]
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
        category = _extract_category_text(raw_text).strip()
        return CalibrationDecision(
            raw_text=raw_text,
            decision=bool(category),
            object_valid=bool(category),
            family_match=True,
            exact_match=bool(category),
            score=100 if category else 0,
            label=category,
            category=category,
            reason=_extract_tag(raw_text, 'reason').strip(),
        )


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
    raw_categories = _extract_tag(raw_text, 'categories')
    categories = _normalize_category_list(raw_categories)
    # Prefer explicit <route_type>; fall back to category-family derivation when absent.
    explicit_route = canonicalize_family_name(_extract_tag(raw_text, 'route_type'))
    if explicit_route and explicit_route.lower() not in {'none', 'null', ''}:
        route_type = explicit_route
    elif categories:
        route_type = get_category_family(categories[0])
    else:
        route_type = ''
    family = route_type
    summary = _extract_tag(raw_text, 'summary')
    cue = _extract_tag(raw_text, 'cue')
    null_text = _extract_tag(raw_text, 'null').strip().lower()
    if null_text in NEGATIVE_VALUES:
        null_likely = True
    else:
        null_likely = not categories or not route_type
    # proposal_prompts = VLM cue + summary + category names (strong noun anchors for G-DINO).
    proposal_prompts = _normalize_prompt_list(cue, summary, categories=categories)
    return SemanticCue(
        raw_text=raw_text,
        family=family,
        route_type=route_type,
        route_confidence=_normalize_route_confidence(_extract_tag(raw_text, 'route_confidence')),
        categories=categories,
        summary=summary,
        proposal_prompts=proposal_prompts,
        null_likely=null_likely,
    )


def _extract_tag(text: str, tag: str) -> str:
    match = re.search(rf'<{tag}>(.*?)</{tag}>', text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ''
    return match.group(1).strip()


def _extract_category_text(raw_text: str) -> str:
    """Extract category name tolerating prompt-format drift.

    Handles three observed VLM output patterns:
    - <category>X</category> (well-formed)
    - <X> (tag-as-name drift, e.g. `<pregnancy test box>`)
    - plain X (no tags)
    """
    tagged = _extract_tag(raw_text, 'category')
    if tagged:
        return tagged
    stripped = (raw_text or '').strip()
    if not stripped:
        return ''
    single_tag = re.fullmatch(r'\s*<\s*([^<>/]+?)\s*/?>\s*', stripped)
    if single_tag:
        return single_tag.group(1).strip()
    return stripped


def _extract_bool_tag(text: str, tag: str, default: bool = False) -> bool:
    value = _extract_tag(text, tag).strip().lower()
    if not value:
        return default
    return value in NEGATIVE_VALUES


def _normalize_route_confidence(text: str) -> str:
    normalized = _normalize_phrase(text)
    if normalized in {'high', 'medium', 'low'}:
        return normalized
    return ''


def _normalize_category_list(raw_text: str) -> list[str]:
    items = []
    for value in re.split(r'[,;\n]+', raw_text or ''):
        normalized = canonicalize_family_name(value)
        normalized_phrase = _normalize_phrase(normalized)
        if not normalized_phrase or normalized_phrase in {'none', 'null', 'empty', 'n a'}:
            continue
        if normalized not in items:
            items.append(normalized)
    return items[:4]


def _normalize_prompt_list(
    cue: str,
    *extra_sources: str,
    categories: list[str] | None = None,
) -> list[str]:
    """Build detector-friendly proposal_prompts = VLM cue + summary + category names."""
    items: list[str] = []
    values: list[str] = [cue, *extra_sources]
    for value in values:
        if not value:
            continue
        if value == cue:
            items.extend(part.strip() for part in cue.split(',') if part.strip())
        else:
            items.append(value.strip())
    if categories:
        items.extend(c.strip() for c in categories if c and c.strip())

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
    return ranked[:5]


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
        'receipt', 'bill', 'statement', 'report', 'letter', 'addressed correspondence', 'educational record', 'medical document', 'transcript', 'prescription'
    ])


def _normalize_text_token_list(raw_text: str) -> list[str]:
    tokens = []
    for item in raw_text.split(','):
        normalized = _normalize_phrase(item)
        if not normalized or normalized in {'none', 'unknown', 'unreadable', 'n a'}:
            continue
        if normalized not in tokens:
            tokens.append(normalized)
    return tokens[:8]


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


def finalize_candidate_results(
    segmenter: SamSegmenter | None,
    query_image_path: str,
    candidates: Sequence[DetectionCandidate],
    image_id: int,
    use_sam: bool,
) -> list[dict[str, Any]]:
    return _finalize_candidate_results(segmenter, query_image_path, candidates, image_id, use_sam)


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
    'detect_free_text',
    'finalize_candidate_results',
    'load_support_image_paths',
    'should_run_detection',
]
