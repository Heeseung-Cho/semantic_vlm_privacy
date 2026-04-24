"""Minimal Stage-3: VLM picks one category from Stage-1 shortlist for each Stage-2 bbox crop.

No support references, no document branch, no OCR override, no reject gate, no SAM.
Skips:
  - images with null_likely=True (Stage-1 null)
  - images where shortlist ∩ allowed_categories is empty (all hallucinated)

Outputs:
  <output_dir>/query_submission.json     - COCO-format detections for mAP eval
  <output_dir>/stage3_decisions.jsonl    - per-candidate decision log
  <output_dir>/detection_metrics.json    - computed via pycocotools
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import re
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from common.vlm import SwiftVLMCaller
from semantic.family_config import get_family_categories, set_active_family_config
from semantic.semantic_gdino_sam import (
    DetectionCandidate,
    _build_reference_match_instruction,
    _build_support_reference_crops,
    _extract_category_text,
    _extract_tag,
    _save_candidate_crop,
)
from tqdm import tqdm


def _annotate_image_with_bboxes(query_image_path: str, candidates: list[dict], out_path: str) -> None:
    """Draw numbered red bboxes on the full query image."""
    with Image.open(query_image_path) as src:
        img = ImageOps.exif_transpose(src).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
    except OSError:
        font = ImageFont.load_default()
    for i, cand in enumerate(candidates, start=1):
        x1, y1, x2, y2 = cand['bbox_xyxy']
        draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
        label = str(i)
        bbox = draw.textbbox((x1, y1 - 32), label, font=font)
        draw.rectangle([bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2], fill='red')
        draw.text((x1, y1 - 32), label, fill='white', font=font)
    img.save(out_path, quality=90)


def _build_per_image_prompt(
    base_prompt: str,
    cue: str,
    candidates: list[dict],
    shortlist: list[str],
    support_category_names: list[str] | None = None,
    ocr_text: str = '',
    ocr_hint: str = '',
    image_size: tuple[int, int] | None = None,
) -> str:
    cand_lines = []
    for i, cand in enumerate(candidates, start=1):
        lt = str(cand.get('source_prompt') or cand.get('label_text') or '')
        x1, y1, x2, y2 = cand['bbox_xyxy']
        if image_size is not None:
            W, H = image_size
            nx1 = int(round(1000 * x1 / max(W, 1)))
            ny1 = int(round(1000 * y1 / max(H, 1)))
            nx2 = int(round(1000 * x2 / max(W, 1)))
            ny2 = int(round(1000 * y2 / max(H, 1)))
            cand_lines.append(
                f'  Box {i}: <box>({nx1},{ny1}),({nx2},{ny2})</box>, '
                f'detector_prompt="{lt}", score={cand["score"]:.3f}'
            )
        else:
            cand_lines.append(f'  Box {i}: detector_prompt="{lt}", score={cand["score"]:.3f}')
    candidate_block = '\n'.join(cand_lines)
    ranked = ', '.join(f'{i+1}. {c}' for i, c in enumerate(shortlist))
    unranked = ', '.join(shortlist)
    filled = base_prompt
    filled = filled.replace('{{cue}}', cue or '(none)')
    filled = filled.replace('{{candidate_block}}', candidate_block)
    filled = filled.replace('{{shortlist_ranked}}', ranked)
    filled = filled.replace('{{shortlist}}', unranked)
    filled = filled.replace('{{ocr_text}}', ocr_text or '(none)')
    filled = filled.replace('{{ocr_hint}}', ocr_hint or '(none)')
    extra = []
    if support_category_names:
        n_sup = len(support_category_names)
        lines = ['Input images in order (the model sees images in this exact order):']
        for i, name in enumerate(support_category_names, start=1):
            lines.append(f'  Image {i}: a labeled example of "{name}" (support reference).')
        lines.append(
            f'  Image {n_sup + 1} (the last image): the query image with red numbered '
            'boxes drawn on it. This is the image you must analyze.'
        )
        lines.append(
            'Compare each numbered box in the last (query) image against the labeled '
            'support examples listed above.'
        )
        extra.append('\n'.join(lines))
    if ocr_text:
        extra.append(f'Scene description of the query image: {ocr_text}')
    allowed_block = '\n'.join(f'- {name}' for name in shortlist)
    sections = [filled]
    if extra:
        sections.append('\n'.join(extra))
    sections.append('Allowed categories:\n' + allowed_block)
    return '\n\n'.join(sections)


def _parse_per_image_decisions(raw_text: str, n_candidates: int, shortlist: list[str]) -> list[dict]:
    """Extract <box id="k"><keep>yes|no</keep><category>X</category></box> for each k in 1..n."""
    allowed_norm = {_normalize(c): c for c in shortlist}
    decisions = []
    for k in range(1, n_candidates + 1):
        # find the <box id="k"> ... </box> block
        pat = rf'<box\s+id\s*=\s*["\']?{k}["\']?\s*>(.*?)</box>'
        m = re.search(pat, raw_text, re.DOTALL | re.IGNORECASE)
        if not m:
            decisions.append({'keep': False, 'category': None, 'parse_ok': False})
            continue
        inner = m.group(1)
        keep_raw = _extract_tag(inner, 'keep').strip().lower()
        cat_raw = _extract_tag(inner, 'category').strip()
        keep = keep_raw in {'yes', 'true', '1'}
        # Always try to resolve category (even when keep=no) so force-keep mode can reuse it.
        matched = None
        cat_norm = _normalize(cat_raw)
        if cat_norm:
            matched = allowed_norm.get(cat_norm)
            if matched is None:
                for c in shortlist:
                    if cat_norm in c.lower() or c.lower() in cat_norm:
                        matched = c; break
        decisions.append({'keep': keep, 'category': matched, 'parse_ok': True, 'raw_keep': keep_raw, 'raw_cat': cat_raw})
    return decisions

DOCUMENT_CATEGORIES_SET = {
    'bank statement', 'bills or receipt', 'business card', 'credit or debit card',
    'doctors prescription', 'letters with address', 'local newspaper',
    'medical record document', 'mortgage or investment report', 'transcript',
}


def _is_document_category(name: str) -> bool:
    return name in DOCUMENT_CATEGORIES_SET


def _is_document_route(route_type: str, shortlist: list[str]) -> bool:
    # A route is 'document' if any of its family categories are document subtypes,
    # OR if the current shortlist has any document category.
    fam_cats = get_family_categories(route_type or '')
    if fam_cats and any(_is_document_category(c) for c in fam_cats):
        return True
    return any(_is_document_category(c) for c in shortlist)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--json_path', required=True, help='COCO GT (category id map)')
    p.add_argument('--stage1_path', required=True)
    p.add_argument('--stage2_path', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--prompt_path', required=True)
    p.add_argument('--llm_model', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    p.add_argument('--llm_seed', type=int, default=None)
    p.add_argument('--llm_max_pixels', type=int, default=448)
    p.add_argument('--proposal_score_threshold', type=float, default=0.0)
    p.add_argument('--support_json', default=None, help='If set with --support_dir, use labeled support crops (L1 mode)')
    p.add_argument('--support_dir', default=None)
    p.add_argument('--reference_source', choices=['crop', 'full_image'], default='crop')
    p.add_argument('--enriched_context', action='store_true',
                   help='Inject Stage-1 cue and Stage-2 label_text into prompt (L0-enriched mode)')
    p.add_argument('--document_ocr', action='store_true',
                   help='For document-routed images, run an OCR VLM pass on each crop and use the doc-ocr prompt.')
    p.add_argument('--dococr_prompt_path', default=None,
                   help='Prompt used for document-routed candidates when --document_ocr is on.')
    p.add_argument('--ocr_prompt_path', default=None,
                   help='Prompt for the OCR pass on each document-routed crop.')
    p.add_argument('--family_config', default=None,
                   help='Family config JSON (needed for document-route detection).')
    p.add_argument('--prefilter_reject', action='store_true',
                   help='Before Stage-3 classification, run a reject-gate VLM pass per candidate crop. Drop candidates where VLM says not an object of this family.')
    p.add_argument('--reject_gate_prompt_path', default=None,
                   help='Prompt for pre-filter reject gate. Uses {{family}} placeholder.')
    p.add_argument('--per_image_mode', action='store_true',
                   help='One VLM call per image: show full image with all candidate bboxes numbered; VLM decides keep/category for each box jointly.')
    p.add_argument('--per_image_prompt_path', default=None,
                   help='Prompt template for per-image mode (uses {{cue}}, {{candidate_block}}, {{shortlist_ranked}})')
    p.add_argument('--max_new_tokens', type=int, default=64,
                   help='VLM generation budget. Per-image mode may need 512+ (multi-box output).')
    p.add_argument('--ocr_doc_route_only', action='store_true',
                   help='In per-image mode, run OCR pass only for document-routed images.')
    p.add_argument('--support_non_doc_only', action='store_true',
                   help='In per-image mode, include support crops only for non-document-routed images.')
    p.add_argument('--per_image_force_keep', action='store_true',
                   help='In per-image mode, ignore VLM <keep> tag and keep every candidate above threshold. Isolates category-only contribution for ablation.')
    p.add_argument('--eval', action='store_true', help='Run pycocotools eval after inference')
    return p.parse_args()


def _build_instruction(base_prompt: str, allowed: list[str]) -> str:
    allowed_block = '\n'.join(f'- {name}' for name in allowed)
    return f'{base_prompt}\n\nAllowed categories:\n{allowed_block}'


def _fill_enriched_placeholders(
    base_prompt: str,
    cue: str,
    label_text: str,
    shortlist: list[str],
    ocr_text: str = '',
    ocr_hint: str = '',
) -> str:
    ranked_str = ', '.join(f'{i+1}. {c}' for i, c in enumerate(shortlist))
    unranked_str = ', '.join(shortlist)
    filled = base_prompt
    filled = filled.replace('{{cue}}', cue or '(none)')
    filled = filled.replace('{{label_text}}', label_text or '(none)')
    filled = filled.replace('{{shortlist_ranked}}', ranked_str)
    filled = filled.replace('{{shortlist_unranked}}', unranked_str)
    filled = filled.replace('{{ocr_text}}', ocr_text or '(none)')
    filled = filled.replace('{{ocr_hint}}', ocr_hint or '(none)')
    return filled


def _build_enriched_instruction(
    base_prompt: str, cue: str, label_text: str, shortlist: list[str],
    ocr_text: str = '', ocr_hint: str = '',
) -> str:
    filled = _fill_enriched_placeholders(base_prompt, cue, label_text, shortlist, ocr_text, ocr_hint)
    allowed_block = '\n'.join(f'- {name}' for name in shortlist)
    return f'{filled}\n\nAllowed categories:\n{allowed_block}'


def _build_enriched_support_instruction(
    base_prompt: str,
    cue: str,
    label_text: str,
    shortlist: list[str],
    support_category_names: list[str],
    ocr_text: str = '',
    ocr_hint: str = '',
) -> str:
    filled = _fill_enriched_placeholders(base_prompt, cue, label_text, shortlist, ocr_text, ocr_hint)
    support_lines = '\n'.join(f'{i+1}. {n}' for i, n in enumerate(support_category_names))
    allowed_block = '\n'.join(f'- {name}' for name in shortlist)
    return (
        f'{filled}\n\n'
        'The images are ordered as: labeled support reference crops first, then the candidate crop last.\n'
        f'Support reference labels in order:\n{support_lines}\n\n'
        f'Allowed categories:\n{allowed_block}'
    )


def _normalize(name: str) -> str:
    return ' '.join((name or '').lower().replace('_', ' ').split())


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(args.json_path).read_text())
    cat_name_to_id = {c['name'].replace('_', ' '): c['id'] for c in gt['categories']}
    allowed_categories = list(cat_name_to_id.keys())
    allowed_norm = {_normalize(n): n for n in allowed_categories}

    stage1 = json.loads(Path(args.stage1_path).read_text())
    stage1_recs = stage1 if isinstance(stage1, list) else stage1.get('records', [])
    stage1_by_img = {int(r['image_id']): r for r in stage1_recs}

    stage2 = json.loads(Path(args.stage2_path).read_text())
    stage2_recs = stage2['records'] if isinstance(stage2, dict) else stage2

    prompt_text = Path(args.prompt_path).read_text().strip()
    dococr_prompt_text = Path(args.dococr_prompt_path).read_text().strip() if args.dococr_prompt_path else None
    ocr_prompt_text = Path(args.ocr_prompt_path).read_text().strip() if args.ocr_prompt_path else None
    reject_prompt_text = Path(args.reject_gate_prompt_path).read_text().strip() if args.reject_gate_prompt_path else None
    per_image_prompt_text = Path(args.per_image_prompt_path).read_text().strip() if args.per_image_prompt_path else None

    if args.family_config:
        set_active_family_config(args.family_config)

    use_support = bool(args.support_json and args.support_dir)
    support_refs = []
    if use_support:
        support_refs = _build_support_reference_crops(
            args.support_json, args.support_dir, reference_source=args.reference_source,
        )
        print(f'[L1] loaded {len(support_refs)} support references')

    client = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )

    decisions_fh = (out_dir / 'stage3_decisions.jsonl').open('w')
    submission: list[dict] = []

    n_skip_null = 0
    n_skip_empty = 0
    n_img_processed = 0
    n_candidates = 0

    try:
        for rec in tqdm(stage2_recs, desc='L0 stage3', unit='image'):
            image_id = int(rec['image_id'])
            s1 = stage1_by_img.get(image_id, {})
            null_likely = bool(s1.get('null_likely', False))
            cats_raw = s1.get('semantic_categories', []) or []
            # intersect with allowed (canonical space form)
            shortlist = []
            for c in cats_raw:
                norm = _normalize(c)
                if norm in allowed_norm and allowed_norm[norm] not in shortlist:
                    shortlist.append(allowed_norm[norm])

            if null_likely:
                n_skip_null += 1
                decisions_fh.write(json.dumps({
                    'image_id': image_id, 'event': 'skip_null_likely',
                    'shortlist': shortlist,
                }) + '\n')
                continue
            if not shortlist:
                n_skip_empty += 1
                decisions_fh.write(json.dumps({
                    'image_id': image_id, 'event': 'skip_empty_shortlist',
                    'stage1_categories': cats_raw,
                }) + '\n')
                continue

            candidates = rec.get('proposal_candidates', []) or []
            if not candidates:
                decisions_fh.write(json.dumps({
                    'image_id': image_id, 'event': 'no_candidates',
                    'shortlist': shortlist,
                }) + '\n')
                continue

            n_img_processed += 1
            query_image_path = str(rec['query_image_path'])

            cue_text = _extract_tag(s1.get('semantic_raw_text', '') or '', 'cue') if args.enriched_context else ''
            route_type = s1.get('route_type', '') or ''
            is_doc = args.document_ocr and _is_document_route(route_type, shortlist)

            # === Per-image mode: one VLM call, all bboxes joint ===
            if args.per_image_mode:
                n_candidates += len(candidates)
                # filter by score threshold
                cands_filt = [c for c in candidates if float(c['score']) >= args.proposal_score_threshold]
                if not cands_filt:
                    decisions_fh.write(json.dumps({
                        'image_id': image_id, 'event': 'no_candidates_after_threshold',
                        'shortlist': shortlist, 'threshold': args.proposal_score_threshold,
                    }) + '\n')
                    continue
                # optional OCR pass on full image
                pi_ocr_text = ''
                pi_ocr_hint = ''
                ocr_allowed = args.document_ocr and ocr_prompt_text and (
                    not args.ocr_doc_route_only or is_doc
                )
                if ocr_allowed:
                    ocr_raw = client.generate(query_image_path, instruction=ocr_prompt_text)
                    # Caption-style description pass: use entire response as description.
                    # (Fall back to <text> tag for backward compat with old OCR prompts.)
                    extracted = _extract_tag(ocr_raw, 'text').strip()
                    pi_ocr_text = extracted if extracted else ocr_raw.strip()
                    pi_ocr_hint = ''  # deprecated — taxonomy-leaky
                # optional support crops
                support_image_paths_pi = []
                support_cat_names_pi = []
                sup_allowed = use_support and (
                    not args.support_non_doc_only or not is_doc
                )
                if sup_allowed:
                    filtered_refs = [r for r in support_refs if r.category_name in shortlist] or support_refs
                    support_image_paths_pi = [r.crop_path for r in filtered_refs]
                    support_cat_names_pi = [r.category_name for r in filtered_refs]
                # Resolve query image size (after EXIF transpose) for norm1000 bbox coords.
                with Image.open(query_image_path) as _src:
                    _img_t = ImageOps.exif_transpose(_src)
                    _img_size = _img_t.size  # (W, H)
                annotated = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
                try:
                    _annotate_image_with_bboxes(query_image_path, cands_filt, annotated)
                    pi_prompt = per_image_prompt_text
                    instruction = _build_per_image_prompt(
                        pi_prompt, cue_text, cands_filt, shortlist,
                        support_category_names=support_cat_names_pi or None,
                        ocr_text=pi_ocr_text, ocr_hint=pi_ocr_hint,
                        image_size=_img_size,
                    )
                    if support_image_paths_pi:
                        # Multi-turn few-shot in-context prompting:
                        # each support crop is given its own user/assistant turn.
                        shots = [
                            (p, f'This is a labeled example of "{n}".')
                            for p, n in zip(support_image_paths_pi, support_cat_names_pi)
                        ]
                        raw = client.generate_few_shot(
                            shots=shots,
                            query_image_path=annotated,
                            instruction=instruction,
                        )
                    else:
                        raw = client.generate(annotated, instruction=instruction)
                finally:
                    Path(annotated).unlink(missing_ok=True)
                decisions = _parse_per_image_decisions(raw, len(cands_filt), shortlist)
                # remap decisions to original candidate list (filtered-out = keep=False)
                filt_set = {id(c) for c in cands_filt}
                dec_iter = iter(decisions)
                decisions_full = []
                for c in candidates:
                    if id(c) in filt_set:
                        decisions_full.append(next(dec_iter))
                    else:
                        decisions_full.append({'keep': False, 'category': None, 'parse_ok': True, 'raw_keep':'(skipped_by_threshold)', 'raw_cat':''})
                decisions = decisions_full
                for cand_idx, (cand, dec) in enumerate(zip(candidates, decisions)):
                    score = float(cand['score'])
                    xyxy = [float(v) for v in cand['bbox_xyxy']]
                    force_keep_applied = False
                    effective_keep = dec['keep']
                    effective_cat = dec['category']
                    # Force-keep: ignore VLM <keep>; keep every candidate that passed threshold.
                    # (Candidates below threshold stay marked as skipped via raw_keep.)
                    if args.per_image_force_keep and dec.get('raw_keep','') != '(skipped_by_threshold)':
                        if not effective_keep:
                            force_keep_applied = True
                        effective_keep = True
                        # If VLM said keep=no and gave no parsable category, fallback to shortlist top.
                        if effective_cat is None:
                            effective_cat = shortlist[0]
                    log = {
                        'image_id': image_id, 'event': 'candidate_decision',
                        'candidate_index': cand_idx,
                        'shortlist': shortlist,
                        'bbox_xyxy': xyxy,
                        'proposal_score': score,
                        'per_image_raw': raw if cand_idx == 0 else '(shared)',
                        'vlm_keep_raw': dec.get('raw_keep',''),
                        'vlm_cat_raw': dec.get('raw_cat',''),
                        'kept': effective_keep,
                        'matched_category': effective_cat,
                        'force_keep_applied': force_keep_applied,
                        'parse_ok': dec['parse_ok'],
                    }
                    decisions_fh.write(json.dumps(log) + '\n')
                    if effective_keep and effective_cat:
                        submission.append({
                            'image_id': image_id,
                            'category_id': cat_name_to_id[effective_cat],
                            'bbox': [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                            'score': score,
                        })
                continue
            # === end per-image mode; below is per-candidate mode ===

            if use_support:
                filtered_refs = [r for r in support_refs if r.category_name in shortlist] or support_refs
                support_image_paths = [r.crop_path for r in filtered_refs]
                support_cat_names = [r.category_name for r in filtered_refs]
                if args.enriched_context:
                    instruction = None  # per-candidate build
                else:
                    instruction = _build_reference_match_instruction(
                        base_instruction=prompt_text,
                        support_references=filtered_refs,
                        allowed_categories=shortlist,
                    )
            else:
                filtered_refs = []
                support_image_paths = []
                support_cat_names = []
                # non-enriched path uses simple _build_instruction; enriched needs per-candidate label_text
                instruction = None  # built per-candidate if enriched
                if not args.enriched_context:
                    instruction = _build_instruction(prompt_text, shortlist)

            for cand_idx, cand in enumerate(candidates):
                n_candidates += 1
                score = float(cand['score'])
                xyxy = [float(v) for v in cand['bbox_xyxy']]
                if score < args.proposal_score_threshold:
                    decisions_fh.write(json.dumps({
                        'image_id': image_id, 'event': 'skip_low_score',
                        'candidate_index': cand_idx, 'score': score,
                    }) + '\n')
                    continue

                label_text = str(cand.get('source_prompt') or cand.get('label_text') or '')

                crop_path = _save_candidate_crop(query_image_path, xyxy)
                try:
                    # Pre-filter reject gate (per-candidate VLM pass)
                    if args.prefilter_reject and reject_prompt_text:
                        family_name = route_type or 'target'
                        rj_prompt = reject_prompt_text.replace('{{family}}', family_name)
                        rj_raw = client.generate(crop_path, instruction=rj_prompt)
                        rj_decision = _extract_tag(rj_raw, 'object').strip().lower()
                        if rj_decision == 'no':
                            decisions_fh.write(json.dumps({
                                'image_id': image_id, 'event': 'prefilter_rejected',
                                'candidate_index': cand_idx, 'bbox_xyxy': xyxy,
                                'reject_raw': rj_raw,
                            }) + '\n')
                            continue

                    ocr_text = ''
                    ocr_hint = ''
                    if is_doc and ocr_prompt_text:
                        ocr_raw = client.generate(crop_path, instruction=ocr_prompt_text)
                        extracted = _extract_tag(ocr_raw, 'text').strip()
                        ocr_text = extracted if extracted else ocr_raw.strip()
                        ocr_hint = ''

                    base = dococr_prompt_text if (is_doc and dococr_prompt_text) else prompt_text
                    if args.enriched_context and use_support:
                        instruction_this = _build_enriched_support_instruction(
                            base, cue_text, label_text, shortlist, support_cat_names,
                            ocr_text=ocr_text, ocr_hint=ocr_hint,
                        )
                    elif args.enriched_context:
                        instruction_this = _build_enriched_instruction(
                            base, cue_text, label_text, shortlist,
                            ocr_text=ocr_text, ocr_hint=ocr_hint,
                        )
                    else:
                        instruction_this = instruction

                    if use_support:
                        image_paths = [*support_image_paths, crop_path]
                        raw = client.generate_images(image_paths, instruction=instruction_this)
                    else:
                        raw = client.generate(crop_path, instruction=instruction_this)
                finally:
                    Path(crop_path).unlink(missing_ok=True)

                parsed = _extract_category_text(raw).strip()
                norm = _normalize(parsed)
                # must match inside shortlist
                shortlist_norm = {_normalize(n): n for n in shortlist}
                matched = shortlist_norm.get(norm)
                fallback = None
                if matched is None:
                    matched = shortlist[0]
                    fallback = 'parse_or_outside_shortlist'

                submission.append({
                    'image_id': image_id,
                    'category_id': cat_name_to_id[matched],
                    'bbox': [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                    'score': score,
                })
                decisions_fh.write(json.dumps({
                    'image_id': image_id, 'event': 'candidate_decision',
                    'candidate_index': cand_idx,
                    'shortlist': shortlist,
                    'bbox_xyxy': xyxy,
                    'proposal_score': score,
                    'vlm_raw': raw,
                    'vlm_parsed': parsed,
                    'matched_category': matched,
                    'fallback_reason': fallback,
                }) + '\n')
    finally:
        decisions_fh.close()

    (out_dir / 'query_submission.json').write_text(json.dumps(submission, indent=2))

    print(f'[L0] images_processed={n_img_processed} '
          f'skipped_null={n_skip_null} skipped_empty_shortlist={n_skip_empty} '
          f'candidates_total={n_candidates} submission_rows={len(submission)}')

    if args.eval:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO(args.json_path)
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(str(out_dir / 'query_submission.json'))
            ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
            ev.evaluate(); ev.accumulate(); ev.summarize()
        stats_names = ['bbox_mAP','bbox_AP50','bbox_AP75','bbox_mAP_small','bbox_mAP_medium','bbox_mAP_large',
                       'bbox_AR1','bbox_AR10','bbox_AR100','bbox_AR_small','bbox_AR_medium','bbox_AR_large']
        summary = {n: float(v) for n, v in zip(stats_names, ev.stats.tolist())}
        (out_dir / 'detection_metrics.json').write_text(json.dumps(summary, indent=2))
        print(f'[L0] mAP={summary["bbox_mAP"]:.4f} AP50={summary["bbox_AP50"]:.4f}')


if __name__ == '__main__':
    main()
