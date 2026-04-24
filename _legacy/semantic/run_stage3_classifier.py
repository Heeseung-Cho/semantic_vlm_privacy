"""Stage-3 VLM classifier for Stage-2 bbox proposals.

Two modes:
  --per_image_mode  : one VLM call per image; the full query image with numbered
                       boxes is shown, and the model emits <box id=k> keep/category
                       decisions jointly (BEST setup).
  (default)         : per-candidate; one VLM call per cropped bbox.

Optional knobs:
  --enriched_context    inject Stage-1 cue + Stage-2 label into prompt
  --document_ocr        add an OCR pass (full image in per-image mode, crop in
                        per-candidate mode) and inject the extracted text/hint
  --ocr_doc_route_only  gate the OCR pass to document-routed images
  --support_*           include labeled support-set reference crops (legacy)
  --prefilter_reject    per-candidate reject gate before classification (legacy)

Skips:
  - images with null_likely=True (Stage-1 null)
  - images whose shortlist ∩ GT-category set is empty

Outputs:
  <output_dir>/query_submission.json     COCO-format detections for mAP eval
  <output_dir>/stage3_decisions.jsonl    per-candidate decision log
  <output_dir>/detection_metrics.json    pycocotools summary (when --eval)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

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

# Categories whose proposals benefit from an OCR pass. Coupled to the
# BIV-Priv-Seg taxonomy: 8 document subtypes + 2 card subtypes that also carry
# printed/embossed text.
DOCUMENT_OCR_CATEGORIES: set[str] = {
    'bank statement', 'bills or receipt', 'business card', 'credit or debit card',
    'doctors prescription', 'letters with address', 'local newspaper',
    'medical record document', 'mortgage or investment report', 'transcript',
}


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


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _candidate_block(candidates: list[dict]) -> str:
    lines = []
    for i, cand in enumerate(candidates, start=1):
        lt = str(cand.get('source_prompt') or cand.get('label_text') or '')
        lines.append(f'  Box {i}: detector_prompt="{lt}", score={cand["score"]:.3f}')
    return '\n'.join(lines)


def _fill_placeholders(
    base_prompt: str,
    shortlist: list[str],
    *,
    cue: str = '',
    label_text: str = '',
    candidate_block: str = '',
    ocr_text: str = '',
    ocr_hint: str = '',
) -> str:
    ranked = ', '.join(f'{i+1}. {c}' for i, c in enumerate(shortlist))
    unranked = ', '.join(shortlist)
    filled = base_prompt
    filled = filled.replace('{{cue}}', cue or '(none)')
    filled = filled.replace('{{label_text}}', label_text or '(none)')
    filled = filled.replace('{{candidate_block}}', candidate_block)
    filled = filled.replace('{{shortlist_ranked}}', ranked)
    filled = filled.replace('{{shortlist_unranked}}', unranked)
    filled = filled.replace('{{shortlist}}', unranked)
    filled = filled.replace('{{ocr_text}}', ocr_text or '(none)')
    filled = filled.replace('{{ocr_hint}}', ocr_hint or '(none)')
    return filled


def _append_allowed_block(prompt: str, shortlist: list[str]) -> str:
    allowed = '\n'.join(f'- {name}' for name in shortlist)
    return f'{prompt}\n\nAllowed categories:\n{allowed}'


def _build_per_image_prompt(
    base_prompt: str,
    cue: str,
    candidates: list[dict],
    shortlist: list[str],
    support_category_names: list[str] | None = None,
    ocr_text: str = '',
    ocr_hint: str = '',
) -> str:
    filled = _fill_placeholders(
        base_prompt, shortlist,
        cue=cue, candidate_block=_candidate_block(candidates),
        ocr_text=ocr_text, ocr_hint=ocr_hint,
    )
    extras = []
    if support_category_names:
        support_lines = '\n'.join(f'  Support {i+1}. {n}' for i, n in enumerate(support_category_names))
        extras.append(
            'The images are ordered as: labeled support reference crops first, then the '
            'query image with numbered boxes last.\nSupport reference labels in order:\n'
            + support_lines
        )
    if ocr_text or ocr_hint:
        extras.append(f'Visible text from the query image (OCR): {ocr_text or "(none)"}')
        extras.append(f'Coarse document type hint from OCR: {ocr_hint or "(none)"}')
    sections = [filled]
    if extras:
        sections.append('\n'.join(extras))
    return _append_allowed_block('\n\n'.join(sections), shortlist)


def _build_per_candidate_prompt(
    base_prompt: str,
    shortlist: list[str],
    *,
    cue: str = '',
    label_text: str = '',
    support_category_names: list[str] | None = None,
    ocr_text: str = '',
    ocr_hint: str = '',
) -> str:
    filled = _fill_placeholders(
        base_prompt, shortlist,
        cue=cue, label_text=label_text,
        ocr_text=ocr_text, ocr_hint=ocr_hint,
    )
    if support_category_names:
        support_lines = '\n'.join(f'{i+1}. {n}' for i, n in enumerate(support_category_names))
        filled = (
            f'{filled}\n\n'
            'The images are ordered as: labeled support reference crops first, then the candidate crop last.\n'
            f'Support reference labels in order:\n{support_lines}'
        )
    return _append_allowed_block(filled, shortlist)


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    return ' '.join((name or '').lower().replace('_', ' ').split())


def _parse_per_image_decisions(raw_text: str, n_candidates: int, shortlist: list[str]) -> list[dict]:
    """Extract <box id="k"><keep>yes|no</keep><category>X</category></box> for each k in 1..n."""
    allowed_norm = {_normalize(c): c for c in shortlist}
    decisions = []
    for k in range(1, n_candidates + 1):
        pat = rf'<box\s+id\s*=\s*["\']?{k}["\']?\s*>(.*?)</box>'
        m = re.search(pat, raw_text, re.DOTALL | re.IGNORECASE)
        if not m:
            decisions.append({'keep': False, 'category': None, 'parse_ok': False})
            continue
        inner = m.group(1)
        keep_raw = _extract_tag(inner, 'keep').strip().lower()
        cat_raw = _extract_tag(inner, 'category').strip()
        keep = keep_raw in {'yes', 'true', '1'}
        matched = None
        if keep:
            cat_norm = _normalize(cat_raw)
            matched = allowed_norm.get(cat_norm)
            if matched is None:
                for c in shortlist:
                    if cat_norm and (cat_norm in c.lower() or c.lower() in cat_norm):
                        matched = c
                        break
        decisions.append({
            'keep': keep, 'category': matched, 'parse_ok': True,
            'raw_keep': keep_raw, 'raw_cat': cat_raw,
        })
    return decisions


def _is_document_category(name: str) -> bool:
    return name in DOCUMENT_OCR_CATEGORIES


def _is_document_route(route_type: str, shortlist: list[str]) -> bool:
    """A route is 'document' if any of its family categories are document subtypes,
    OR if the current shortlist has any document category."""
    fam_cats = get_family_categories(route_type or '')
    if fam_cats and any(_is_document_category(c) for c in fam_cats):
        return True
    return any(_is_document_category(c) for c in shortlist)


# ---------------------------------------------------------------------------
# Config / context objects
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Shared per-run state (prompts, VLM client, support refs, IO)."""
    prompt_text: str
    per_image_prompt_text: str | None
    dococr_prompt_text: str | None
    ocr_prompt_text: str | None
    reject_prompt_text: str | None
    support_refs: list = field(default_factory=list)
    use_support: bool = False
    client: SwiftVLMCaller | None = None
    cat_name_to_id: dict[str, int] = field(default_factory=dict)
    decisions_fh: object = None
    submission: list[dict] = field(default_factory=list)


@dataclass
class ImageContext:
    image_id: int
    query_image_path: str
    shortlist: list[str]
    candidates: list[dict]
    cue: str
    route_type: str
    is_doc_route: bool


def _build_image_context(
    rec: dict,
    stage1_by_img: dict[int, dict],
    allowed_norm: dict[str, str],
    args: argparse.Namespace,
    decisions_fh,
) -> tuple[ImageContext | None, str | None]:
    """Return (ctx, skip_reason). skip_reason ∈ {None, 'null', 'empty', 'no_candidates'}."""
    image_id = int(rec['image_id'])
    s1 = stage1_by_img.get(image_id, {})
    null_likely = bool(s1.get('null_likely', False))
    cats_raw = s1.get('semantic_categories', []) or []
    shortlist: list[str] = []
    for c in cats_raw:
        norm = _normalize(c)
        if norm in allowed_norm and allowed_norm[norm] not in shortlist:
            shortlist.append(allowed_norm[norm])

    if null_likely:
        decisions_fh.write(json.dumps({
            'image_id': image_id, 'event': 'skip_null_likely', 'shortlist': shortlist,
        }) + '\n')
        return None, 'null'
    if not shortlist:
        decisions_fh.write(json.dumps({
            'image_id': image_id, 'event': 'skip_empty_shortlist',
            'stage1_categories': cats_raw,
        }) + '\n')
        return None, 'empty'
    candidates = rec.get('proposal_candidates', []) or []
    if not candidates:
        decisions_fh.write(json.dumps({
            'image_id': image_id, 'event': 'no_candidates', 'shortlist': shortlist,
        }) + '\n')
        return None, 'no_candidates'

    cue_text = _extract_tag(s1.get('semantic_raw_text', '') or '', 'cue') if args.enriched_context else ''
    route_type = s1.get('route_type', '') or ''
    return ImageContext(
        image_id=image_id,
        query_image_path=str(rec['query_image_path']),
        shortlist=shortlist,
        candidates=candidates,
        cue=cue_text,
        route_type=route_type,
        is_doc_route=_is_document_route(route_type, shortlist),
    ), None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

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
    p.add_argument('--max_new_tokens', type=int, default=64,
                   help='VLM generation budget. Per-image mode needs 512+ (multi-box output).')
    p.add_argument('--proposal_score_threshold', type=float, default=0.0)
    p.add_argument('--family_config', default=None,
                   help='Family config JSON (needed for document-route detection).')
    p.add_argument('--enriched_context', action='store_true',
                   help='Inject Stage-1 cue and Stage-2 label_text into the prompt template.')
    p.add_argument('--per_image_mode', action='store_true',
                   help='One VLM call per image: numbered-bbox image; VLM decides keep/category jointly.')
    p.add_argument('--per_image_prompt_path', default=None,
                   help='Prompt template for per-image mode (uses {{cue}}, {{candidate_block}}, {{shortlist_ranked}})')
    p.add_argument('--document_ocr', action='store_true',
                   help='Run an OCR VLM pass (full image in per-image mode, crop in per-candidate).')
    p.add_argument('--ocr_prompt_path', default=None,
                   help='Prompt for the OCR pass; must emit <text>…</text> and <document_hint>…</document_hint>.')
    p.add_argument('--ocr_doc_route_only', action='store_true',
                   help='Run OCR only for document-routed images (applies to both modes).')
    # --- legacy flags (kept for reproducing older ablation runs; BEST does not use) ---
    p.add_argument('--dococr_prompt_path', default=None,
                   help='[legacy] Alternative prompt for document-routed per-candidate calls.')
    p.add_argument('--support_json', default=None,
                   help='[legacy] If set with --support_dir, use labeled support crops.')
    p.add_argument('--support_dir', default=None, help='[legacy]')
    p.add_argument('--reference_source', choices=['crop', 'full_image'], default='crop',
                   help='[legacy] Source for support reference images.')
    p.add_argument('--support_non_doc_only', action='store_true',
                   help='[legacy] In per-image mode, include support crops only for non-doc images.')
    p.add_argument('--prefilter_reject', action='store_true',
                   help='[legacy] Per-candidate reject-gate VLM pass before classification.')
    p.add_argument('--reject_gate_prompt_path', default=None,
                   help='[legacy] Prompt for prefilter reject gate; uses {{family}}.')
    p.add_argument('--eval', action='store_true', help='Run pycocotools eval after inference')
    return p.parse_args()


LEGACY_FLAGS = [
    ('dococr_prompt_path', 'value'),
    ('support_json', 'value'),
    ('support_dir', 'value'),
    ('support_non_doc_only', 'flag'),
    ('prefilter_reject', 'flag'),
    ('reject_gate_prompt_path', 'value'),
]


def _warn_legacy_flags(args: argparse.Namespace) -> None:
    in_use = [name for name, kind in LEGACY_FLAGS
              if (kind == 'flag' and getattr(args, name)) or (kind == 'value' and getattr(args, name))]
    if in_use:
        print(f'[warn] legacy flags in use (kept for old ablations, not used by BEST): {in_use}',
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-image mode processor
# ---------------------------------------------------------------------------

def _process_per_image(ctx: ImageContext, cfg: RunConfig, args: argparse.Namespace) -> int:
    cands_all = ctx.candidates
    cands_filt = [c for c in cands_all if float(c['score']) >= args.proposal_score_threshold]
    if not cands_filt:
        cfg.decisions_fh.write(json.dumps({
            'image_id': ctx.image_id, 'event': 'no_candidates_after_threshold',
            'shortlist': ctx.shortlist, 'threshold': args.proposal_score_threshold,
        }) + '\n')
        return len(cands_all)

    pi_ocr_text = ''
    pi_ocr_hint = ''
    ocr_allowed = (
        args.document_ocr and cfg.ocr_prompt_text
        and (not args.ocr_doc_route_only or ctx.is_doc_route)
    )
    if ocr_allowed:
        ocr_raw = cfg.client.generate(ctx.query_image_path, instruction=cfg.ocr_prompt_text)
        pi_ocr_text = _extract_tag(ocr_raw, 'text').strip()
        pi_ocr_hint = _extract_tag(ocr_raw, 'document_hint').strip()

    support_image_paths = []
    support_cat_names = []
    sup_allowed = cfg.use_support and (not args.support_non_doc_only or not ctx.is_doc_route)
    if sup_allowed:
        filtered_refs = [r for r in cfg.support_refs if r.category_name in ctx.shortlist] or cfg.support_refs
        support_image_paths = [r.crop_path for r in filtered_refs]
        support_cat_names = [r.category_name for r in filtered_refs]

    annotated = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    try:
        _annotate_image_with_bboxes(ctx.query_image_path, cands_filt, annotated)
        instruction = _build_per_image_prompt(
            cfg.per_image_prompt_text, ctx.cue, cands_filt, ctx.shortlist,
            support_category_names=support_cat_names or None,
            ocr_text=pi_ocr_text, ocr_hint=pi_ocr_hint,
        )
        image_paths = [*support_image_paths, annotated]
        if len(image_paths) == 1:
            raw = cfg.client.generate(annotated, instruction=instruction)
        else:
            raw = cfg.client.generate_images(image_paths, instruction=instruction)
    finally:
        Path(annotated).unlink(missing_ok=True)

    decisions = _parse_per_image_decisions(raw, len(cands_filt), ctx.shortlist)
    filt_set = {id(c) for c in cands_filt}
    dec_iter = iter(decisions)
    decisions_full = []
    for c in cands_all:
        if id(c) in filt_set:
            decisions_full.append(next(dec_iter))
        else:
            decisions_full.append({
                'keep': False, 'category': None, 'parse_ok': True,
                'raw_keep': '(skipped_by_threshold)', 'raw_cat': '',
            })

    for cand_idx, (cand, dec) in enumerate(zip(cands_all, decisions_full)):
        score = float(cand['score'])
        xyxy = [float(v) for v in cand['bbox_xyxy']]
        cfg.decisions_fh.write(json.dumps({
            'image_id': ctx.image_id, 'event': 'candidate_decision',
            'candidate_index': cand_idx,
            'shortlist': ctx.shortlist,
            'bbox_xyxy': xyxy,
            'proposal_score': score,
            'per_image_raw': raw if cand_idx == 0 else '(shared)',
            'vlm_keep_raw': dec.get('raw_keep', ''),
            'vlm_cat_raw': dec.get('raw_cat', ''),
            'kept': dec['keep'],
            'matched_category': dec['category'],
            'parse_ok': dec['parse_ok'],
        }) + '\n')
        if dec['keep'] and dec['category']:
            cfg.submission.append({
                'image_id': ctx.image_id,
                'category_id': cfg.cat_name_to_id[dec['category']],
                'bbox': [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
                'score': score,
            })
    return len(cands_all)


# ---------------------------------------------------------------------------
# Per-candidate mode processor
# ---------------------------------------------------------------------------

def _process_per_candidate(ctx: ImageContext, cfg: RunConfig, args: argparse.Namespace) -> int:
    if cfg.use_support:
        filtered_refs = [r for r in cfg.support_refs if r.category_name in ctx.shortlist] or cfg.support_refs
        support_image_paths = [r.crop_path for r in filtered_refs]
        support_cat_names = [r.category_name for r in filtered_refs]
        if args.enriched_context:
            static_instruction = None
        else:
            static_instruction = _build_reference_match_instruction(
                base_instruction=cfg.prompt_text,
                support_references=filtered_refs,
                allowed_categories=ctx.shortlist,
            )
    else:
        support_image_paths = []
        support_cat_names = []
        static_instruction = None if args.enriched_context else _append_allowed_block(cfg.prompt_text, ctx.shortlist)

    seen = 0
    for cand_idx, cand in enumerate(ctx.candidates):
        seen += 1
        score = float(cand['score'])
        xyxy = [float(v) for v in cand['bbox_xyxy']]
        if score < args.proposal_score_threshold:
            cfg.decisions_fh.write(json.dumps({
                'image_id': ctx.image_id, 'event': 'skip_low_score',
                'candidate_index': cand_idx, 'score': score,
            }) + '\n')
            continue
        label_text = str(cand.get('source_prompt') or cand.get('label_text') or '')
        crop_path = _save_candidate_crop(ctx.query_image_path, xyxy)
        try:
            if args.prefilter_reject and cfg.reject_prompt_text:
                family_name = ctx.route_type or 'target'
                rj_prompt = cfg.reject_prompt_text.replace('{{family}}', family_name)
                rj_raw = cfg.client.generate(crop_path, instruction=rj_prompt)
                if _extract_tag(rj_raw, 'object').strip().lower() == 'no':
                    cfg.decisions_fh.write(json.dumps({
                        'image_id': ctx.image_id, 'event': 'prefilter_rejected',
                        'candidate_index': cand_idx, 'bbox_xyxy': xyxy,
                        'reject_raw': rj_raw,
                    }) + '\n')
                    continue

            ocr_text = ''
            ocr_hint = ''
            if args.document_ocr and ctx.is_doc_route and cfg.ocr_prompt_text:
                ocr_raw = cfg.client.generate(crop_path, instruction=cfg.ocr_prompt_text)
                ocr_text = _extract_tag(ocr_raw, 'text').strip()
                ocr_hint = _extract_tag(ocr_raw, 'document_hint').strip()

            use_dococr_prompt = args.document_ocr and ctx.is_doc_route and cfg.dococr_prompt_text
            base = cfg.dococr_prompt_text if use_dococr_prompt else cfg.prompt_text
            if args.enriched_context:
                instruction_this = _build_per_candidate_prompt(
                    base, ctx.shortlist,
                    cue=ctx.cue, label_text=label_text,
                    support_category_names=support_cat_names if cfg.use_support else None,
                    ocr_text=ocr_text, ocr_hint=ocr_hint,
                )
            else:
                instruction_this = static_instruction

            if cfg.use_support:
                raw = cfg.client.generate_images([*support_image_paths, crop_path], instruction=instruction_this)
            else:
                raw = cfg.client.generate(crop_path, instruction=instruction_this)
        finally:
            Path(crop_path).unlink(missing_ok=True)

        parsed = _extract_category_text(raw).strip()
        shortlist_norm = {_normalize(n): n for n in ctx.shortlist}
        matched = shortlist_norm.get(_normalize(parsed))
        fallback = None
        if matched is None:
            matched = ctx.shortlist[0]
            fallback = 'parse_or_outside_shortlist'

        cfg.submission.append({
            'image_id': ctx.image_id,
            'category_id': cfg.cat_name_to_id[matched],
            'bbox': [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]],
            'score': score,
        })
        cfg.decisions_fh.write(json.dumps({
            'image_id': ctx.image_id, 'event': 'candidate_decision',
            'candidate_index': cand_idx,
            'shortlist': ctx.shortlist,
            'bbox_xyxy': xyxy,
            'proposal_score': score,
            'vlm_raw': raw,
            'vlm_parsed': parsed,
            'matched_category': matched,
            'fallback_reason': fallback,
        }) + '\n')
    return seen


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_text_or_none(path: str | None) -> str | None:
    return Path(path).read_text().strip() if path else None


def _run_eval(output_dir: Path, json_path: str) -> None:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(json_path)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(str(output_dir / 'query_submission.json'))
        ev = COCOeval(coco_gt, coco_dt, iouType='bbox')
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    stats_names = [
        'bbox_mAP', 'bbox_AP50', 'bbox_AP75',
        'bbox_mAP_small', 'bbox_mAP_medium', 'bbox_mAP_large',
        'bbox_AR1', 'bbox_AR10', 'bbox_AR100',
        'bbox_AR_small', 'bbox_AR_medium', 'bbox_AR_large',
    ]
    summary = {n: float(v) for n, v in zip(stats_names, ev.stats.tolist())}
    (output_dir / 'detection_metrics.json').write_text(json.dumps(summary, indent=2))
    print(f'[stage3] mAP={summary["bbox_mAP"]:.4f} AP50={summary["bbox_AP50"]:.4f}')


def main() -> None:
    args = parse_args()
    _warn_legacy_flags(args)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = json.loads(Path(args.json_path).read_text())
    cat_name_to_id = {c['name'].replace('_', ' '): c['id'] for c in gt['categories']}
    allowed_norm = {_normalize(n): n for n in cat_name_to_id}

    stage1 = json.loads(Path(args.stage1_path).read_text())
    stage1_recs = stage1 if isinstance(stage1, list) else stage1.get('records', [])
    stage1_by_img = {int(r['image_id']): r for r in stage1_recs}

    stage2 = json.loads(Path(args.stage2_path).read_text())
    stage2_recs = stage2['records'] if isinstance(stage2, dict) else stage2

    if args.family_config:
        set_active_family_config(args.family_config)

    cfg = RunConfig(
        prompt_text=Path(args.prompt_path).read_text().strip(),
        per_image_prompt_text=_load_text_or_none(args.per_image_prompt_path),
        dococr_prompt_text=_load_text_or_none(args.dococr_prompt_path),
        ocr_prompt_text=_load_text_or_none(args.ocr_prompt_path),
        reject_prompt_text=_load_text_or_none(args.reject_gate_prompt_path),
        use_support=bool(args.support_json and args.support_dir),
        cat_name_to_id=cat_name_to_id,
    )
    if cfg.use_support:
        cfg.support_refs = _build_support_reference_crops(
            args.support_json, args.support_dir, reference_source=args.reference_source,
        )
        print(f'[stage3] loaded {len(cfg.support_refs)} support references')

    cfg.client = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )

    n_skip_null = n_skip_empty = n_img_processed = n_candidates = 0
    decisions_path = out_dir / 'stage3_decisions.jsonl'
    cfg.decisions_fh = decisions_path.open('w')
    try:
        for rec in tqdm(stage2_recs, desc='stage3', unit='image'):
            ctx, skip = _build_image_context(rec, stage1_by_img, allowed_norm, args, cfg.decisions_fh)
            if ctx is None:
                if skip == 'null':
                    n_skip_null += 1
                elif skip == 'empty':
                    n_skip_empty += 1
                continue
            n_img_processed += 1
            processor = _process_per_image if args.per_image_mode else _process_per_candidate
            n_candidates += processor(ctx, cfg, args)
    finally:
        cfg.decisions_fh.close()

    (out_dir / 'query_submission.json').write_text(json.dumps(cfg.submission, indent=2))
    print(f'[stage3] images_processed={n_img_processed} '
          f'skipped_null={n_skip_null} skipped_empty_shortlist={n_skip_empty} '
          f'candidates_total={n_candidates} submission_rows={len(cfg.submission)}')

    if args.eval:
        _run_eval(out_dir, args.json_path)


if __name__ == '__main__':
    main()
