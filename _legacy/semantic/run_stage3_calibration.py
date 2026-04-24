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
from semantic.family_config import get_family_categories, set_active_family_config
from semantic.semantic_gdino_sam import (
    DetectionCandidate,
    ProposalCalibrator,
    SamSegmenter,
    SemanticCue,
    _extract_category_text,
    _extract_tag,
    _normalize_text_token_list,
    _save_candidate_crop,
    finalize_candidate_results,
)

DOCUMENT_CATEGORIES = {
    'bills or receipt',
    'bank statement',
    'letters with address',
    'transcript',
    'mortgage or investment report',
    'medical record document',
    'doctors prescription',
    'local newspaper',
}

DOCUMENT_CATEGORY_ALIASES = {
    'receipt': 'bills or receipt',
    'bill': 'bills or receipt',
    'invoice': 'bills or receipt',
    'statement': 'bank statement',
    'bank document': 'bank statement',
    'letter': 'letters with address',
    'addressed letter': 'letters with address',
    'student transcript': 'transcript',
    'mortgage report': 'mortgage or investment report',
    'investment report': 'mortgage or investment report',
    'medical record': 'medical record document',
    'prescription': 'doctors prescription',
    'newspaper': 'local newspaper',
}

# Keyword → subtype rules for document subtype tiebreak. Keywords are extracted
# from the official family_config category_descriptions (not dev-observed),
# which makes this rule paper-safe.
DOCUMENT_SUBTYPE_KEYWORDS: dict[str, list[str]] = {
    'bills or receipt': [
        'total', 'subtotal', 'tax', 'item', 'qty', 'payment', 'cash', 'change',
        'receipt', 'store', 'merchant', 'invoice', 'bill',
    ],
    'bank statement': [
        'account', 'balance', 'statement', 'transaction', 'deposit',
        'withdrawal', 'routing', 'debit', 'credit', 'bank',
    ],
    'letters with address': [
        'dear', 'sincerely', 'recipient', 'sender', 'street', 'address', 'zip',
    ],
    'transcript': [
        'transcript', 'course', 'grade', 'gpa', 'credit', 'semester',
        'academic', 'school', 'student',
    ],
    'mortgage or investment report': [
        'mortgage', 'loan', 'principal', 'interest', 'escrow', 'apr',
        'portfolio', 'investment',
    ],
    'medical record document': [
        'patient', 'diagnosis', 'treatment', 'clinic', 'hospital', 'lab',
        'chart', 'physician',
    ],
    'doctors prescription': [
        'rx', 'prescription', 'medication', 'dosage', 'mg', 'refill',
        'prescriber', 'pharmacy',
    ],
    'local newspaper': [
        'headline', 'column', 'byline', 'section', 'masthead', 'puzzle',
        'newspaper',
    ],
}


def _ocr_keyword_preferred_category(
    text_tokens: list[str],
    text_hint: str,
    shortlist: list[str],
) -> str | None:
    """Return the shortlist category most strongly supported by OCR keywords.

    Conservative rule: returns a category only if
    (a) its keyword hit count is non-zero, AND
    (b) it is either the sole match or dominates (>= 2x) the runner-up.
    Otherwise returns None (VLM decision is kept).

    Keyword lists are derived from official category_descriptions — paper-safe.
    """
    if not shortlist:
        return None
    combined = ' '.join([text_hint or ''] + list(text_tokens or [])).lower()
    if not combined.strip():
        return None
    scores: dict[str, int] = {}
    for cat in shortlist:
        normalized = _normalize_category_name(cat)
        keywords = DOCUMENT_SUBTYPE_KEYWORDS.get(normalized, [])
        hits = sum(1 for kw in keywords if kw in combined)
        if hits > 0:
            scores[cat] = hits
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_cat, top_hits = ranked[0]
    if len(ranked) == 1:
        return top_cat
    _, second_hits = ranked[1]
    if top_hits >= 2 * second_hits:
        return top_cat
    return None

DOCUMENT_REFINE_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'deprecated' / 'semantic_document_refine.txt'
)
DOCUMENT_FORCED_REFINE_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'active' / 'stage3' / 'semantic_document_refine_forced_shortlist.txt'
)
DOCUMENT_BBOX_SELECT_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'deprecated' / 'semantic_document_bbox_select.txt'
)
CANDIDATE_CAPTION_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'deprecated' / 'semantic_candidate_caption.txt'
)
REJECT_GATE_DEFAULT_PROMPT = (
    PROJECT_ROOT / 'prompts' / 'active' / 'stage2_5' / 'semantic_candidate_reject_gate.txt'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 3 calibration/finalization only.')
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--stage1_path', required=True)
    parser.add_argument('--stage2_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--sam_checkpoint', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--family_config', default=None)
    parser.add_argument('--calibration_mode', choices=['legacy', 'reference_match'], default='legacy')
    parser.add_argument('--reference_source', choices=['crop', 'full_image'], default='crop')
    parser.add_argument('--support_dir', default=None)
    parser.add_argument('--support_json', default=None)
    parser.add_argument('--disable_sam', action='store_true')
    parser.add_argument('--proposal_score_threshold', type=float, default=None)
    parser.add_argument(
        '--final_score_threshold',
        type=float,
        default=None,
        help='Deprecated alias for --proposal_score_threshold.',
    )
    parser.add_argument('--classification_top_k', type=int, default=None)
    parser.add_argument('--skip_null_stage3', action='store_true')
    parser.add_argument('--verbose_decisions', action='store_true')
    parser.add_argument('--decision_log_jsonl', default=None)
    parser.add_argument('--save_calibration_raw_text', action='store_true')
    parser.add_argument(
        '--save_candidate_caption',
        action='store_true',
        help='Run a short logging-only caption pass on each Stage-3 candidate crop without affecting decisions.',
    )
    parser.add_argument(
        '--candidate_caption_prompt_path',
        default=str(CANDIDATE_CAPTION_DEFAULT_PROMPT),
    )
    parser.add_argument(
        '--enable_reject_gate',
        action='store_true',
        help='Run a separate pre-mapping VLM reject gate before any category resolver. '
             'Only `invalid` candidates are dropped; `uncertain` candidates continue.',
    )
    parser.add_argument(
        '--reject_gate_prompt_path',
        default=str(REJECT_GATE_DEFAULT_PROMPT),
    )
    parser.add_argument('--enable_document_refine', action='store_true')
    parser.add_argument('--document_refine_prompt_path', default=str(DOCUMENT_REFINE_DEFAULT_PROMPT))
    parser.add_argument(
        '--document_refine_mode',
        choices=['prior_preserving', 'prior_preserving_no_confidence', 'authoritative', 'shortlist_forced', 'shortlist_forced_support', 'bbox_select_top1'],
        default='prior_preserving',
        help='For document-routed candidates, either preserve the Stage-1 shortlist prior or let document refine act as the final subtype resolver within document label space.',
    )
    parser.add_argument(
        '--document_forced_refine_prompt_path',
        default=str(DOCUMENT_FORCED_REFINE_DEFAULT_PROMPT),
    )
    parser.add_argument(
        '--document_bbox_select_prompt_path',
        default=str(DOCUMENT_BBOX_SELECT_DEFAULT_PROMPT),
    )
    parser.add_argument(
        '--enable_document_ocr',
        action='store_true',
        help='Use a single text-aware document refine prompt that explicitly reads visible text fragments before subtype classification.',
    )
    parser.add_argument(
        '--enable_document_prompt_match_fallback',
        action='store_true',
        help='Allow document branch to prefer shortlist categories matched lexically from the candidate prompt.',
    )
    parser.add_argument('--save_document_refine_raw_text', action='store_true')
    parser.add_argument(
        '--branch_dispatch_mode',
        choices=['route', 'shortlist'],
        default='route',
        help='Use Stage-1 route_type or Stage-1 category shortlist membership to dispatch document/non-document Stage-3 branches.',
    )
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
                'score': float(result.get('proposal_score', result.get('score', 0.0))),
                'category_id': category_id,
                'area': float(area or 0.0),
                'bbox': bbox,
                'segmentation': segmentation,
            })
    return submission


def _normalize_category_name(text: str) -> str:
    return ' '.join((text or '').strip().replace('_', ' ').lower().split())


def _resolve_stage1_category_shortlist(stage1_categories: list[str], category_names: list[str]) -> list[str]:
    """Return Stage-1 VLM-narrowed categories filtered by the allowed set.

    Empty when Stage-1 produced no valid category; caller handles fallback.
    """
    normalized_allowed = {
        _normalize_category_name(category): category
        for category in category_names
    }
    resolved: list[str] = []
    seen: set[str] = set()
    for category in stage1_categories or []:
        normalized = _normalize_category_name(category)
        if not normalized:
            continue
        matched = normalized_allowed.get(normalized)
        if matched is None or matched in seen:
            continue
        resolved.append(matched)
        seen.add(matched)
    return resolved


def _candidate_prompt(candidate_record: dict[str, object]) -> str:
    prompt = candidate_record.get('source_prompt') or candidate_record.get('label_text') or 'candidate'
    return str(prompt)


def _reject_reasons(
    *,
    matched_category: str | None,
    proposal_score: float,
    proposal_score_threshold: float,
    decision: object,
    reference_match_mode: bool,
) -> list[str]:
    reasons: list[str] = []
    if not matched_category:
        reasons.append('no_allowed_category_match')
    if proposal_score < proposal_score_threshold:
        reasons.append('below_proposal_score_threshold')
    if reference_match_mode:
        return reasons
    if getattr(decision, 'decision', None) is False:
        reasons.append('decision_false')
    if getattr(decision, 'object_valid', None) is False:
        reasons.append('object_valid_false')
    if getattr(decision, 'family_match', None) is False:
        reasons.append('family_match_false')
    if getattr(decision, 'exact_match', None) is False:
        reasons.append('exact_match_false')
    return reasons


def _format_bbox(values: list[float]) -> str:
    return '[' + ', '.join(f'{value:.1f}' for value in values) + ']'


def _fallback_reference_category(candidate_prompt: str, allowed_categories: list[str]) -> tuple[str | None, str | None]:
    if len(allowed_categories) == 1:
        return allowed_categories[0], 'single_allowed_category'
    normalized_prompt = _normalize_category_name(candidate_prompt)
    if not normalized_prompt:
        return None, None
    matches = [
        category
        for category in allowed_categories
        if _normalize_category_name(category) in normalized_prompt
    ]
    if not matches:
        return None, None
    matches.sort(key=lambda category: len(_normalize_category_name(category)), reverse=True)
    return matches[0], 'candidate_prompt_match'


def _is_document_category(category: str | None) -> bool:
    return _normalize_category_name(category or '') in DOCUMENT_CATEGORIES


def _normalize_document_refine_category(text: str) -> str | None:
    normalized = _normalize_category_name(text)
    if not normalized or normalized == 'unknown':
        return None
    normalized = _normalize_category_name(DOCUMENT_CATEGORY_ALIASES.get(normalized, normalized))
    for category in DOCUMENT_CATEGORIES:
        if _normalize_category_name(category) == normalized:
            return category
    return None


def _normalize_yes_no(text: str) -> bool | None:
    normalized = _normalize_category_name(text)
    if normalized == 'yes':
        return True
    if normalized == 'no':
        return False
    return None


def _normalize_document_confidence(text: str) -> str | None:
    normalized = _normalize_category_name(text)
    if normalized in {'high', 'low', 'none'}:
        return normalized
    return None


def _normalize_evidence(raw_text: str) -> list[str]:
    evidence: list[str] = []
    for item in raw_text.split(','):
        token = ' '.join(item.strip().lower().split())
        if not token or token in {'none', 'unknown', 'unreadable', 'n/a'}:
            continue
        if any(char.isdigit() for char in token) and len(token) > 8:
            continue
        if len(token) > 32:
            continue
        if token not in evidence:
            evidence.append(token)
    return evidence[:8]


def _normalize_short_text(raw_text: str, *, allow_none: bool = True, max_len: int = 80) -> str:
    text = ' '.join((raw_text or '').strip().split())
    if not text:
        return 'none' if allow_none else ''
    if allow_none and text.lower() in {'none', 'unknown', 'n/a'}:
        return 'none'
    return text[:max_len]


def _normalize_reject_gate_decision(text: str) -> str:
    normalized = _normalize_category_name(text)
    if normalized in {'valid', 'invalid', 'uncertain'}:
        return normalized
    return 'uncertain'


def _should_refine_document_candidate(route_type: str, matched_category: str | None) -> bool:
    return _is_document_route(route_type) and _is_document_category(matched_category)


def _is_document_route(route_type: str) -> bool:
    """A route is 'document' if any of its family categories are document subtypes.

    Supports arbitrary taxonomies: the legacy 4-way grouping has a single
    `document` family; a 6-way grouping splits it into `medical document`,
    `financial document`, `other document` — all of which resolve to document
    categories under the hood via family_config.
    """
    normalized = _normalize_category_name(route_type or '')
    if not normalized or normalized == 'none':
        return False
    family_cats = get_family_categories(normalized)
    if not family_cats:
        # Fallback for legacy string match if family lookup missed.
        return normalized == 'document'
    return any(_is_document_category(c) for c in family_cats)


def _shortlist_contains_document(categories: list[str]) -> bool:
    return any(_is_document_category(category) for category in categories)


def _shortlist_contains_non_document(categories: list[str]) -> bool:
    return any(not _is_document_category(category) for category in categories)


def _document_dispatch_enabled(*, route_type: str, category_shortlist: list[str], dispatch_mode: str) -> bool:
    if dispatch_mode == 'route':
        return _is_document_route(route_type)
    return _shortlist_contains_document(category_shortlist)


def _reference_dispatch_enabled(*, route_type: str, category_shortlist: list[str], dispatch_mode: str) -> bool:
    if dispatch_mode == 'route':
        return not _is_document_route(route_type)
    return _shortlist_contains_non_document(category_shortlist) or not category_shortlist


def _pick_shortlist_earlier(
    category_shortlist: list[str],
    *,
    document_category: str | None,
    reference_category: str | None,
) -> tuple[str | None, str | None]:
    ranked = {name: idx for idx, name in enumerate(category_shortlist)}
    candidates: list[tuple[str, str]] = []
    if document_category:
        candidates.append((document_category, 'document_refine'))
    if reference_category:
        candidates.append((reference_category, 'reference_match'))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: ranked.get(item[0], 10**9))
    return candidates[0][0], candidates[0][1]


def _prune_document_candidates_top1(
    kept_candidates: list[dict[str, object]],
    *,
    route_type: str,
    category_shortlist: list[str],
    dispatch_mode: str,
) -> list[dict[str, object]]:
    if dispatch_mode == 'route':
        should_prune = _is_document_route(route_type)
    else:
        should_prune = _shortlist_contains_document(category_shortlist)
    if not should_prune or not kept_candidates:
        return kept_candidates
    top_item = max(
        kept_candidates,
        key=lambda item: (
            float(item.get('selection_score', item['proposal_score'])),
            float(item['proposal_score']),
        ),
    )
    return [top_item]


def _build_document_refine_instruction(base_prompt_text: str, text_aware_mode: bool) -> str:
    if not text_aware_mode:
        return base_prompt_text
    return (
        f"{base_prompt_text}\n\n"
        "Before choosing the subtype, actively read short visible text fragments, headers, or diagnostic words from the crop "
        "and use them together with layout cues in the same decision. Do not output the extracted text separately; only use it "
        "to improve the final <category> and <evidence> fields. Ignore long numeric strings and privacy-sensitive full details."
    )


def _build_document_forced_instruction(
    *,
    base_prompt_text: str,
    allowed_categories: list[str],
    candidate_prompt: str,
    text_aware_mode: bool,
) -> str:
    allowed_text = ', '.join(allowed_categories)
    instruction = (
        f"{base_prompt_text}\n\n"
        f"candidate_prompt: {candidate_prompt}\n"
        f"allowed_categories: {allowed_text}\n\n"
        "Choose exactly one category from `allowed_categories`."
    )
    if text_aware_mode:
        instruction += (
            "\nUse short visible text fragments when possible, but if text is weak, rely on document layout and the "
            "candidate prompt as supporting context."
        )
    return instruction


def _build_document_forced_support_instruction(
    *,
    base_prompt_text: str,
    support_category_names: list[str],
    allowed_categories: list[str],
    candidate_prompt: str,
    text_aware_mode: bool,
) -> str:
    allowed_text = ', '.join(allowed_categories)
    support_lines = [f'{idx}. {name}' for idx, name in enumerate(support_category_names, start=1)]
    instruction = (
        f"{base_prompt_text}\n\n"
        "The images are ordered as: support reference crops first, then one query candidate crop last.\n"
        "Support reference labels in order:\n"
        f"{chr(10).join(support_lines)}\n"
        f"candidate_prompt: {candidate_prompt}\n"
        f"allowed_categories: {allowed_text}\n\n"
        "Choose exactly one category from `allowed_categories` for the final candidate crop."
    )
    if text_aware_mode:
        instruction += (
            "\nUse short visible text fragments when possible, but if text is weak, rely on document layout, "
            "the support references, and the candidate prompt as supporting context."
        )
    return instruction


def _refine_document_category(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
    text_aware_mode: bool = False,
) -> dict[str, object]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        instruction = _build_document_refine_instruction(prompt_text, text_aware_mode)
        raw_text = client.generate(crop_path, instruction=instruction)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    document_confidence = _normalize_document_confidence(_extract_tag(raw_text, 'document_confidence'))
    document_present = _normalize_yes_no(_extract_tag(raw_text, 'document_present'))
    if document_confidence is None:
        if document_present is True:
            document_confidence = 'high'
        elif document_present is False:
            document_confidence = 'none'
    refined_category = _normalize_document_refine_category(_extract_category_text(raw_text))
    evidence = _normalize_evidence(_extract_tag(raw_text, 'evidence'))
    sensitive_region = _normalize_short_text(_extract_tag(raw_text, 'sensitive_region'))
    sensitive_attribute = _normalize_short_text(_extract_tag(raw_text, 'sensitive_attribute'))
    reason = _normalize_short_text(_extract_tag(raw_text, 'reason'))
    if document_confidence == 'low':
        refined_category = None
    if document_confidence == 'none':
        refined_category = None
        evidence = []
        sensitive_region = 'none'
        sensitive_attribute = 'none'
    return {
        'raw_text': raw_text,
        'document_confidence': document_confidence,
        'document_present': document_present,
        'category': refined_category,
        'evidence': evidence,
        'sensitive_region': sensitive_region,
        'sensitive_attribute': sensitive_attribute,
        'reason': reason,
    }


def _forced_document_category(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
    allowed_categories: list[str],
    candidate_prompt: str,
    text_aware_mode: bool = False,
) -> dict[str, object]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        instruction = _build_document_forced_instruction(
            base_prompt_text=prompt_text,
            allowed_categories=allowed_categories,
            candidate_prompt=candidate_prompt,
            text_aware_mode=text_aware_mode,
        )
        raw_text = client.generate(crop_path, instruction=instruction)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    refined_category = _normalize_document_refine_category(_extract_category_text(raw_text))
    normalized_allowed = {_normalize_category_name(name): name for name in allowed_categories}
    normalized_refine = _normalize_category_name(refined_category or '')
    matched_category = normalized_allowed.get(normalized_refine)
    return {
        'raw_text': raw_text,
        'category': matched_category,
    }


def _forced_document_category_with_support(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
    allowed_categories: list[str],
    candidate_prompt: str,
    support_references: list[object],
    text_aware_mode: bool = False,
) -> dict[str, object]:
    filtered_references = [
        entry for entry in support_references
        if getattr(entry, 'category_name', None) in allowed_categories
    ]
    if not filtered_references:
        return _forced_document_category(
            client=client,
            prompt_text=prompt_text,
            query_image_path=query_image_path,
            candidate=candidate,
            allowed_categories=allowed_categories,
            candidate_prompt=candidate_prompt,
            text_aware_mode=text_aware_mode,
        )
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        instruction = _build_document_forced_support_instruction(
            base_prompt_text=prompt_text,
            support_category_names=[str(entry.category_name) for entry in filtered_references],
            allowed_categories=allowed_categories,
            candidate_prompt=candidate_prompt,
            text_aware_mode=text_aware_mode,
        )
        image_paths = [str(entry.crop_path) for entry in filtered_references]
        image_paths.append(crop_path)
        raw_text = client.generate_images(image_paths, instruction=instruction)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    refined_category = _normalize_document_refine_category(_extract_category_text(raw_text))
    normalized_allowed = {_normalize_category_name(name): name for name in allowed_categories}
    normalized_refine = _normalize_category_name(refined_category or '')
    matched_category = normalized_allowed.get(normalized_refine)
    return {
        'raw_text': raw_text,
        'category': matched_category,
    }


def _score_document_bbox_candidate(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
    target_category: str,
) -> dict[str, object]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        instruction = (
            f"{prompt_text}\n\n"
            f"target_category: {target_category}\n"
        )
        raw_text = client.generate(crop_path, instruction=instruction)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    score_text = _extract_tag(raw_text, 'score').strip()
    try:
        score = int(score_text)
    except ValueError:
        score = 0
    score = max(0, min(3, score))
    reason = _normalize_short_text(_extract_tag(raw_text, 'reason'))
    return {
        'raw_text': raw_text,
        'score': score,
        'reason': reason,
    }


def _generate_candidate_caption(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
) -> dict[str, str]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        raw_text = client.generate(crop_path, instruction=prompt_text)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    caption = _normalize_short_text(_extract_tag(raw_text, 'caption'), allow_none=False, max_len=120)
    return {
        'raw_text': raw_text,
        'caption': caption,
    }


def _build_reject_gate_instruction(
    *,
    base_prompt_text: str,
    shortlist: list[str],
    candidate_prompt: str,
    ocr_text_hint: str | None,
    ocr_text_tokens: str | None,
    candidate_caption: str,
) -> str:
    shortlist_text = ', '.join(shortlist) if shortlist else 'none'
    top1 = shortlist[0] if shortlist else 'none'
    lines = [
        base_prompt_text,
        '',
        f'shortlist_top1: {top1}',
        f'shortlist_categories: {shortlist_text}',
        f'candidate_prompt: {candidate_prompt}',
    ]
    normalized_hint = _normalize_short_text(ocr_text_hint or '', max_len=80)
    normalized_tokens = _normalize_short_text(ocr_text_tokens or '', max_len=160)
    if normalized_hint != 'none':
        lines.append(f'stage1_ocr_hint: {normalized_hint}')
    if normalized_tokens != 'none':
        lines.append(f'stage1_ocr_tokens: {normalized_tokens}')
    normalized_caption = _normalize_short_text(candidate_caption or '', allow_none=False, max_len=120)
    if normalized_caption:
        lines.append(f'candidate_caption: {normalized_caption}')
    return '\n'.join(lines)


def _run_reject_gate(
    *,
    client: SwiftVLMCaller,
    prompt_text: str,
    query_image_path: str,
    candidate: DetectionCandidate,
    shortlist: list[str],
    candidate_prompt: str,
    ocr_text_hint: str | None,
    ocr_text_tokens: str | None,
    candidate_caption: str,
) -> dict[str, str]:
    crop_path = _save_candidate_crop(query_image_path, candidate.xyxy)
    try:
        instruction = _build_reject_gate_instruction(
            base_prompt_text=prompt_text,
            shortlist=shortlist,
            candidate_prompt=candidate_prompt,
            ocr_text_hint=ocr_text_hint,
            ocr_text_tokens=ocr_text_tokens,
            candidate_caption=candidate_caption,
        )
        raw_text = client.generate(crop_path, instruction=instruction)
    finally:
        Path(crop_path).unlink(missing_ok=True)
    decision = _normalize_reject_gate_decision(_extract_tag(raw_text, 'decision'))
    reason = _normalize_short_text(_extract_tag(raw_text, 'reason'))
    return {
        'raw_text': raw_text,
        'decision': decision,
        'reason': reason,
    }


def main() -> None:
    args = parse_args()
    if args.proposal_score_threshold is None:
        args.proposal_score_threshold = (
            args.final_score_threshold
            if args.final_score_threshold is not None
            else 0.30
        )
    args.final_score_threshold = args.proposal_score_threshold
    set_active_family_config(args.family_config)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(Path(args.json_path).read_text())
    category_name_to_id = {cat['name'].replace('_', ' '): cat['id'] for cat in dataset.get('categories', [])}
    category_names = list(category_name_to_id.keys())

    stage1_payload = json.loads(Path(args.stage1_path).read_text())
    stage2_payload = json.loads(Path(args.stage2_path).read_text())
    stage1_by_image = {int(record['image_id']): record for record in stage1_payload['records']}
    stage2_records = stage2_payload['records']
    document_refine_prompt = None
    if args.enable_document_refine:
        document_refine_prompt = Path(args.document_refine_prompt_path).read_text().strip()
    document_forced_refine_prompt = None
    if args.enable_document_refine and args.document_refine_mode == 'shortlist_forced':
        document_forced_refine_prompt = Path(args.document_forced_refine_prompt_path).read_text().strip()
    document_bbox_select_prompt = None
    if args.enable_document_refine and args.document_refine_mode == 'bbox_select_top1':
        document_bbox_select_prompt = Path(args.document_bbox_select_prompt_path).read_text().strip()
    candidate_caption_prompt = None
    if args.save_candidate_caption:
        candidate_caption_prompt = Path(args.candidate_caption_prompt_path).read_text().strip()
    reject_gate_prompt = None
    if args.enable_reject_gate:
        reject_gate_prompt = Path(args.reject_gate_prompt_path).read_text().strip()

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
    decision_log_path = Path(args.decision_log_jsonl).resolve() if args.decision_log_jsonl else None
    if decision_log_path is not None:
        decision_log_path.parent.mkdir(parents=True, exist_ok=True)
    decision_log_fh = decision_log_path.open('w') if decision_log_path else None
    outputs: list[dict[str, object]] = []
    progress = tqdm(stage2_records, desc='stage3 calibration', unit='image')
    try:
        for detection_record in progress:
            image_id = int(detection_record['image_id'])
            semantic_record = stage1_by_image[image_id]
            semantic = SemanticCue(
                family=str(semantic_record['semantic_family']),
                route_type=str(semantic_record.get('route_type', '')),
                categories=list(semantic_record.get('semantic_categories', [])),
                proposal_prompts=list(semantic_record['proposal_prompts']),
                null_likely=bool(semantic_record['null_likely']),
                text_hint_summary=str(semantic_record.get('ocr_text_hint', '') or ''),
                text_hint_tokens=_normalize_text_token_list(str(semantic_record.get('ocr_text_tokens', '') or '')),
            )
            category_shortlist = _resolve_stage1_category_shortlist(semantic.categories, category_names)
            if not category_shortlist and not semantic.null_likely:
                category_shortlist = list(category_names)
            document_shortlist = [name for name in category_shortlist if _is_document_category(name)]
            non_document_shortlist = [name for name in category_shortlist if not _is_document_category(name)]
            kept_candidates: list[dict[str, object]] = []
            calibration_logs: list[dict[str, object]] = []

            raw_candidates = list(detection_record.get('proposal_candidates', []))
            if args.skip_null_stage3 and semantic.null_likely:
                outputs.append({
                    'image_id': image_id,
                    'query_image_path': detection_record['query_image_path'],
                    'support_image_paths': detection_record.get('support_image_paths', []),
                    'controller_mode': detection_record['controller_mode'],
                    'semantic_family': semantic.family,
                    'route_type': semantic.route_type,
                    'semantic_categories': semantic.categories,
                    'proposal_prompts': semantic.proposal_prompts,
                    'null_likely': semantic.null_likely,
                    'null_policy': detection_record['null_policy'],
                    'proposal_candidates': raw_candidates,
                    'category_shortlist': [],
                    'calibration_logs': [],
                    'rerank_logs': [],
                    'results': [],
                })
                progress.set_postfix({
                    'image_id': image_id,
                    'selected': 0,
                })
                message = (
                    f"Stage3 image_id={image_id} family={semantic.family} categories={semantic.categories} "
                    f"candidates={len(raw_candidates)} selected=0 skipped_null_stage3=1"
                )
                tqdm.write(message)
                if decision_log_fh is not None:
                    decision_log_fh.write(json.dumps({
                        'image_id': image_id,
                        'event': 'skip_null_stage3',
                        'semantic_family': semantic.family,
                        'route_type': semantic.route_type,
                        'semantic_categories': semantic.categories,
                        'proposal_prompts': semantic.proposal_prompts,
                        'candidate_count': len(raw_candidates),
                    }, ensure_ascii=False) + '\n')
                    decision_log_fh.flush()
                continue
            candidates_for_scoring = (
                raw_candidates[:args.classification_top_k]
                if args.classification_top_k
                else raw_candidates
            )
            for candidate_index, candidate_record in enumerate(candidates_for_scoring):
                candidate_prompt = _candidate_prompt(candidate_record)
                proposal_score = float(candidate_record['score'])
                candidate = DetectionCandidate(
                    score=proposal_score,
                    label_text=candidate_prompt,
                    category_id=-1,
                    xyxy=[float(value) for value in candidate_record['bbox_xyxy']],
                )
                matched_category = None
                stage3_branch = 'reference_match'
                decision = None
                decision_category = None
                decision_label = None
                decision_reason = None
                decision_score = None
                decision_decision = None
                decision_object_valid = None
                decision_family_match = None
                decision_exact_match = None
                category_fallback_reason = None
                document_refine_category = None
                document_refine_evidence: list[str] = []
                document_refine_override = False
                document_refine_raw_text = None
                document_present = None
                document_confidence = None
                document_sensitive_region = 'none'
                document_sensitive_attribute = 'none'
                document_refine_reason = 'none'
                pre_refine_category = None
                bbox_selection_score = None
                bbox_selection_reason = None
                candidate_caption = ''
                candidate_caption_raw_text = None
                reject_gate_decision = None
                reject_gate_reason = 'none'
                reject_gate_raw_text = None
                document_dispatch_shortlist = document_shortlist if args.branch_dispatch_mode == 'shortlist' else category_shortlist
                reference_dispatch_shortlist = (
                    non_document_shortlist
                    if args.branch_dispatch_mode == 'shortlist' and non_document_shortlist
                    else category_shortlist
                )
                if args.save_candidate_caption and candidate_caption_prompt:
                    caption_result = _generate_candidate_caption(
                        client=shared_vlm,
                        prompt_text=candidate_caption_prompt,
                        query_image_path=str(detection_record['query_image_path']),
                        candidate=candidate,
                    )
                    candidate_caption = caption_result['caption']
                    candidate_caption_raw_text = caption_result['raw_text']
                if args.enable_reject_gate and reject_gate_prompt:
                    reject_gate = _run_reject_gate(
                        client=shared_vlm,
                        prompt_text=reject_gate_prompt,
                        query_image_path=str(detection_record['query_image_path']),
                        candidate=candidate,
                        shortlist=category_shortlist,
                        candidate_prompt=candidate_prompt,
                        ocr_text_hint=semantic_record.get('ocr_text_hint'),
                        ocr_text_tokens=semantic_record.get('ocr_text_tokens'),
                        candidate_caption=candidate_caption,
                    )
                    reject_gate_decision = reject_gate['decision']
                    reject_gate_reason = reject_gate['reason']
                    reject_gate_raw_text = reject_gate['raw_text']
                    if reject_gate_decision == 'invalid':
                        stage3_branch = 'reject_gate'
                        accepted = False
                        reject_reasons = ['reject_gate_invalid']
                        log_entry = {
                            'stage3_branch': stage3_branch,
                            'candidate_index': candidate_index,
                            'proposal_score': proposal_score,
                            'candidate_score': proposal_score,
                            'candidate_label_text': str(candidate_record.get('label_text', '')),
                            'candidate_source_prompt': str(candidate_record.get('source_prompt', candidate_prompt)),
                            'candidate_bbox_xyxy': list(candidate.xyxy),
                            'calibration_decision': None,
                            'object_valid': None,
                            'family_match': None,
                            'exact_match': None,
                            'calibration_score': None,
                            'calibration_label': None,
                            'calibration_category': None,
                            'calibration_reason': None,
                            'pre_refine_category': None,
                            'document_confidence': None,
                            'document_refine_category': None,
                            'document_present': None,
                            'document_refine_evidence': [],
                            'document_sensitive_region': 'none',
                            'document_sensitive_attribute': 'none',
                            'document_refine_reason': 'none',
                            'candidate_caption': candidate_caption,
                            'bbox_selection_score': None,
                            'bbox_selection_reason': None,
                            'document_refine_override': False,
                            'matched_category': None,
                            'category_fallback_reason': None,
                            'accepted': accepted,
                            'reject_reasons': reject_reasons,
                            'proposal_score_threshold': args.proposal_score_threshold,
                            'final_score': proposal_score,
                            'reject_gate_decision': reject_gate_decision,
                            'reject_gate_reason': reject_gate_reason,
                        }
                        if args.save_candidate_caption and candidate_caption_raw_text is not None:
                            log_entry['candidate_caption_raw_text'] = candidate_caption_raw_text
                        if reject_gate_raw_text is not None:
                            log_entry['reject_gate_raw_text'] = reject_gate_raw_text
                        calibration_logs.append(log_entry)
                        if args.verbose_decisions:
                            tqdm.write(
                                f"  cand#{candidate_index} REJECT proposal_score={proposal_score:.3f} "
                                f"branch=reject_gate prompt={candidate_prompt!r} bbox={_format_bbox(list(candidate.xyxy))} "
                                f"reasons=reject_gate_invalid gate={reject_gate_reason}"
                            )
                        if decision_log_fh is not None:
                            jsonl_entry = {
                                'image_id': image_id,
                                'event': 'candidate_decision',
                                'semantic_family': semantic.family,
                                'route_type': semantic.route_type,
                                'semantic_categories': semantic.categories,
                                'proposal_prompts': semantic.proposal_prompts,
                                'category_shortlist': category_shortlist,
                                **log_entry,
                            }
                            decision_log_fh.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                            decision_log_fh.flush()
                        continue

                if (
                    _document_dispatch_enabled(
                        route_type=semantic.route_type,
                        category_shortlist=document_dispatch_shortlist,
                        dispatch_mode=args.branch_dispatch_mode,
                    )
                    and args.enable_document_refine
                    and document_refine_prompt
                ):
                    stage3_branch = 'document_refine'
                    if args.document_refine_mode == 'bbox_select_top1':
                        target_category = str(document_dispatch_shortlist[0]) if document_dispatch_shortlist else None
                        matched_category = target_category
                        category_fallback_reason = 'document_route_prior'
                        if target_category and document_bbox_select_prompt:
                            bbox_select = _score_document_bbox_candidate(
                                client=shared_vlm,
                                prompt_text=document_bbox_select_prompt,
                                query_image_path=str(detection_record['query_image_path']),
                                candidate=candidate,
                                target_category=target_category,
                            )
                            bbox_selection_score = int(bbox_select['score'])
                            bbox_selection_reason = str(bbox_select['reason'])
                            document_refine_raw_text = str(bbox_select['raw_text'])
                    elif args.document_refine_mode in {'shortlist_forced', 'shortlist_forced_support'}:
                        forced_allowed = list(document_dispatch_shortlist) if document_dispatch_shortlist else list(DOCUMENT_CATEGORIES)
                        if args.document_refine_mode == 'shortlist_forced_support':
                            document_forced = _forced_document_category_with_support(
                                client=shared_vlm,
                                prompt_text=document_forced_refine_prompt or document_refine_prompt,
                                query_image_path=str(detection_record['query_image_path']),
                                candidate=candidate,
                                allowed_categories=forced_allowed,
                                candidate_prompt=candidate_prompt,
                                support_references=list(calibrator.support_references),
                                text_aware_mode=args.enable_document_ocr,
                            )
                        else:
                            document_forced = _forced_document_category(
                                client=shared_vlm,
                                prompt_text=document_forced_refine_prompt or document_refine_prompt,
                                query_image_path=str(detection_record['query_image_path']),
                                candidate=candidate,
                                allowed_categories=forced_allowed,
                                candidate_prompt=candidate_prompt,
                                text_aware_mode=args.enable_document_ocr,
                            )
                        document_refine_raw_text = str(document_forced['raw_text'])
                        document_refine_category = document_forced['category']
                        matched_category = document_refine_category or (forced_allowed[0] if forced_allowed else None)
                        document_confidence = 'forced_support' if args.document_refine_mode == 'shortlist_forced_support' else 'forced'
                        category_fallback_reason = None if document_refine_category else (
                            'document_forced_support_parse_fallback'
                            if args.document_refine_mode == 'shortlist_forced_support'
                            else 'document_forced_parse_fallback'
                        )
                        # OCR keyword tiebreak: if OCR text evidence dominantly
                        # points to a different shortlist category, override VLM pick.
                        # Keyword list is derived from official category_descriptions.
                        keyword_preferred = _ocr_keyword_preferred_category(
                            semantic.text_hint_tokens,
                            semantic.text_hint_summary,
                            forced_allowed,
                        )
                        if keyword_preferred and keyword_preferred != matched_category:
                            matched_category = keyword_preferred
                            category_fallback_reason = 'ocr_keyword_override'
                    else:
                        document_refine = _refine_document_category(
                            client=shared_vlm,
                            prompt_text=document_refine_prompt,
                            query_image_path=str(detection_record['query_image_path']),
                            candidate=candidate,
                            text_aware_mode=args.enable_document_ocr,
                        )
                        document_confidence = document_refine['document_confidence']
                        document_present = document_refine['document_present']
                        document_refine_category = document_refine['category']
                        document_refine_evidence = list(document_refine['evidence'])
                        document_sensitive_region = str(document_refine['sensitive_region'])
                        document_sensitive_attribute = str(document_refine['sensitive_attribute'])
                        document_refine_reason = str(document_refine['reason'])
                        document_refine_raw_text = str(document_refine['raw_text'])
                        normalized_shortlist = {
                            _normalize_category_name(name): name for name in document_dispatch_shortlist
                        }
                        normalized_refine = _normalize_category_name(document_refine_category or '')
                        if document_dispatch_shortlist and document_confidence != 'none':
                            matched_category = str(document_dispatch_shortlist[0])
                            category_fallback_reason = 'document_route_prior'
                        prompt_matched_category = None
                        prompt_fallback_reason = None
                        if args.enable_document_prompt_match_fallback and document_dispatch_shortlist:
                            prompt_matched_category, prompt_fallback_reason = _fallback_reference_category(
                                candidate_prompt=candidate_prompt,
                                allowed_categories=document_dispatch_shortlist,
                            )
                        if args.document_refine_mode == 'authoritative':
                            if (
                                document_confidence == 'high'
                                and document_refine_category
                                and document_refine_evidence
                            ):
                                matched_category = document_refine_category
                                category_fallback_reason = 'document_authoritative_refine'
                            elif document_confidence == 'low' and document_dispatch_shortlist:
                                matched_category = str(document_dispatch_shortlist[0])
                                category_fallback_reason = 'document_low_confidence_prior'
                            elif prompt_matched_category is not None:
                                matched_category = prompt_matched_category
                                category_fallback_reason = f'document_{prompt_fallback_reason}'
                        elif args.document_refine_mode == 'prior_preserving_no_confidence':
                            if document_dispatch_shortlist:
                                matched_category = str(document_dispatch_shortlist[0])
                                category_fallback_reason = 'document_route_prior'
                            if (
                                document_refine_category
                                and normalized_refine in normalized_shortlist
                            ):
                                matched_category = normalized_shortlist[normalized_refine]
                                category_fallback_reason = None
                            elif prompt_matched_category is not None:
                                matched_category = prompt_matched_category
                                category_fallback_reason = f'document_{prompt_fallback_reason}'
                            elif document_refine_category and document_dispatch_shortlist:
                                category_fallback_reason = 'refine_out_of_shortlist'
                        else:
                            if (
                                document_confidence == 'high'
                                and
                                document_refine_category
                                and document_refine_evidence
                                and normalized_refine in normalized_shortlist
                            ):
                                matched_category = normalized_shortlist[normalized_refine]
                                category_fallback_reason = None
                            elif document_confidence == 'low' and document_dispatch_shortlist:
                                matched_category = str(document_dispatch_shortlist[0])
                                category_fallback_reason = 'document_low_confidence_prior'
                            elif prompt_matched_category is not None:
                                matched_category = prompt_matched_category
                                category_fallback_reason = f'document_{prompt_fallback_reason}'
                            elif (
                                document_confidence == 'high'
                                and document_dispatch_shortlist
                                and document_refine_category
                                and document_refine_evidence
                            ):
                                category_fallback_reason = 'refine_out_of_shortlist'
                else:
                    decision = calibrator.score_candidate(
                        support_image_paths=detection_record.get('support_image_paths', []),
                        query_image_path=str(detection_record['query_image_path']),
                        candidate=candidate,
                        semantic=semantic,
                        allowed_categories=reference_dispatch_shortlist,
                    )
                    decision_category = decision.category
                    decision_label = decision.label
                    decision_reason = decision.reason
                    decision_score = decision.score
                    decision_decision = decision.decision
                    decision_object_valid = decision.object_valid
                    decision_family_match = decision.family_match
                    decision_exact_match = decision.exact_match
                    normalized_allowed = {
                        _normalize_category_name(name): name for name in reference_dispatch_shortlist
                    }
                    normalized_decision = _normalize_category_name(decision.category)
                    if normalized_decision in normalized_allowed:
                        matched_category = normalized_allowed[normalized_decision]
                    if matched_category is None and args.calibration_mode == 'reference_match':
                        matched_category, category_fallback_reason = _fallback_reference_category(
                            candidate_prompt=candidate_prompt,
                            allowed_categories=reference_dispatch_shortlist,
                        )
                    pre_refine_category = matched_category
                if stage3_branch == 'document_refine' or args.calibration_mode == 'reference_match':
                    accepted = bool(matched_category and proposal_score >= args.proposal_score_threshold)
                else:
                    accepted = bool(
                        matched_category
                        and proposal_score >= args.proposal_score_threshold
                        and (decision.decision is not False)
                        and (decision.object_valid is not False)
                        and (decision.family_match is not False)
                        and (decision.exact_match is not False)
                    )
                reject_reasons = [] if accepted else _reject_reasons(
                    matched_category=matched_category,
                    proposal_score=proposal_score,
                    proposal_score_threshold=args.proposal_score_threshold,
                    decision=decision,
                    reference_match_mode=(stage3_branch == 'document_refine' or args.calibration_mode == 'reference_match'),
                )
                log_entry = {
                    'stage3_branch': stage3_branch,
                    'candidate_index': candidate_index,
                    'proposal_score': proposal_score,
                    # Backward-compatible alias for old analysis scripts.
                    'candidate_score': proposal_score,
                    'candidate_label_text': str(candidate_record.get('label_text', '')),
                    'candidate_source_prompt': str(candidate_record.get('source_prompt', candidate_prompt)),
                    'candidate_bbox_xyxy': list(candidate.xyxy),
                    'calibration_decision': decision_decision,
                    'object_valid': decision_object_valid,
                    'family_match': decision_family_match,
                    'exact_match': decision_exact_match,
                    'calibration_score': decision_score,
                    'calibration_label': decision_label,
                    'calibration_category': decision_category,
                    'calibration_reason': decision_reason,
                    'pre_refine_category': pre_refine_category,
                    'document_confidence': document_confidence,
                    'document_refine_category': document_refine_category,
                    'document_present': document_present,
                    'document_refine_evidence': document_refine_evidence,
                    'document_sensitive_region': document_sensitive_region,
                    'document_sensitive_attribute': document_sensitive_attribute,
                    'document_refine_reason': document_refine_reason,
                    'candidate_caption': candidate_caption,
                    'bbox_selection_score': bbox_selection_score,
                    'bbox_selection_reason': bbox_selection_reason,
                    'reject_gate_decision': reject_gate_decision,
                    'reject_gate_reason': reject_gate_reason,
                    'document_refine_override': document_refine_override,
                    'matched_category': matched_category,
                    'category_fallback_reason': category_fallback_reason,
                    'accepted': accepted,
                    'reject_reasons': reject_reasons,
                    'proposal_score_threshold': args.proposal_score_threshold,
                    # Backward-compatible alias. This is not a semantic confidence.
                    'final_score': proposal_score,
                }
                if args.save_calibration_raw_text and decision is not None:
                    log_entry['calibration_raw_text'] = decision.raw_text
                if args.save_document_refine_raw_text and document_refine_raw_text is not None:
                    log_entry['document_refine_raw_text'] = document_refine_raw_text
                if args.save_candidate_caption and candidate_caption_raw_text is not None:
                    log_entry['candidate_caption_raw_text'] = candidate_caption_raw_text
                if reject_gate_raw_text is not None:
                    log_entry['reject_gate_raw_text'] = reject_gate_raw_text
                calibration_logs.append(log_entry)
                if args.verbose_decisions:
                    status = 'ACCEPT' if accepted else 'REJECT'
                    reason_text = ','.join(reject_reasons) if reject_reasons else '-'
                    refine_text = (
                        f" doc_refine={document_refine_category!r} evidence={document_refine_evidence}"
                        if document_refine_category or document_refine_evidence
                        else ''
                    )
                    tqdm.write(
                        f"  cand#{candidate_index} {status} proposal_score={proposal_score:.3f} "
                        f"branch={stage3_branch} "
                        f"prompt={candidate_prompt!r} bbox={_format_bbox(list(candidate.xyxy))} "
                        f"vlm_category={decision_category!r} matched={matched_category!r} "
                        f"fallback={category_fallback_reason or '-'} reasons={reason_text}{refine_text}"
                    )
                if decision_log_fh is not None:
                    jsonl_entry = {
                        'image_id': image_id,
                        'event': 'candidate_decision',
                        'semantic_family': semantic.family,
                        'route_type': semantic.route_type,
                        'semantic_categories': semantic.categories,
                        'proposal_prompts': semantic.proposal_prompts,
                        'category_shortlist': category_shortlist,
                        **log_entry,
                    }
                    decision_log_fh.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')
                    decision_log_fh.flush()
                if not accepted:
                    continue
                kept_candidates.append({
                    'detection': DetectionCandidate(
                        score=proposal_score,
                        label_text=matched_category,
                        category_id=category_name_to_id.get(matched_category, -1),
                        xyxy=list(candidate.xyxy),
                    ),
                    'proposal_score': proposal_score,
                    'selection_score': float(bbox_selection_score if bbox_selection_score is not None else proposal_score),
                    # Backward-compatible alias for old output readers.
                    'detector_score': proposal_score,
                    'calibration_category': decision_category,
                    'matched_category': matched_category,
                    # Backward-compatible alias. This is currently equal to proposal_score.
                    'final_score': proposal_score,
                })

            kept_candidates.sort(key=lambda item: item['proposal_score'], reverse=True)
            kept_candidates = _prune_document_candidates_top1(
                kept_candidates,
                route_type=semantic.route_type,
                category_shortlist=category_shortlist,
                dispatch_mode=args.branch_dispatch_mode,
            )
            finalized = finalize_candidate_results(
                segmenter=segmenter,
                query_image_path=str(detection_record['query_image_path']),
                candidates=[item['detection'] for item in kept_candidates],
                image_id=image_id,
                use_sam=not args.disable_sam,
            )
            for result, item in zip(finalized, kept_candidates):
                result['proposal_score'] = item['proposal_score']
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
                'route_type': semantic.route_type,
                'semantic_categories': semantic.categories,
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
            rejected_count = len(calibration_logs) - len(kept_candidates)
            tqdm.write(
                f"Stage3 image_id={image_id} family={semantic.family} categories={semantic.categories} "
                f"shortlist={category_shortlist} candidates={len(raw_candidates)} "
                f"scored={len(candidates_for_scoring)} selected={len(finalized)} rejected={rejected_count}"
            )
    finally:
        progress.close()
        if decision_log_fh is not None:
            decision_log_fh.close()

    submission_records = build_submission_records(outputs, category_name_to_id)
    (output_dir / 'semantic_pipeline_results.json').write_text(json.dumps(outputs, ensure_ascii=False, indent=2))
    (output_dir / 'query_submission.json').write_text(json.dumps(submission_records, ensure_ascii=False, indent=2))
    (output_dir / 'run_config.json').write_text(json.dumps(vars(args), ensure_ascii=False, indent=2))
    tqdm.write(f"Saved Stage 3 outputs to: {output_dir / 'semantic_pipeline_results.json'}")
    tqdm.write(f"Saved Stage 3 submission to: {output_dir / 'query_submission.json'}")


if __name__ == '__main__':
    main()
