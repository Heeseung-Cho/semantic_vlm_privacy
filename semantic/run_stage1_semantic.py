#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller, release_torch_runtime
from common.vlm import _resolve_device_map
from baseline.qwen_gdino_sam import load_support_image_paths
from semantic.family_config import (
    get_active_family_config_path,
    get_category_family,
    get_family_names,
    set_active_family_config,
)
from semantic.semantic_gdino_sam import (
    DOCUMENT_TEXT_PROMPT,
    SemanticController,
    _extract_tag,
    _parse_semantic_cue,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 1 query-only semantic inference.')
    parser.add_argument('--query_dir', required=True)
    parser.add_argument('--json_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--llm_model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm_max_new_tokens', type=int, default=128)
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--family_config', '--family-config', dest='family_config', default=None)
    parser.add_argument('--query_prompt_path', default=None)
    parser.add_argument('--support_query_prompt_path', default=None)
    parser.add_argument('--support_json', default=None)
    parser.add_argument('--support_dir', default=None)
    parser.add_argument('--stage1_mode', choices=['query_only', 'support_query'], default='query_only')
    parser.add_argument('--null_policy', choices=['strict', 'skip', 'ignore'], default='ignore')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image_id', type=int, default=None)
    parser.add_argument('--runtime_stats_jsonl', default=None)
    parser.add_argument('--save_raw_text', action='store_true')
    parser.add_argument('--save_global_caption', action='store_true')
    parser.add_argument('--global_caption_prompt_path', default=None)
    parser.add_argument(
        '--enable_ocr_enrichment',
        action='store_true',
        help='Run a query-image OCR/text-hint pass before Stage 1 and append it to the Stage-1 prompt.',
    )
    parser.add_argument('--cuda_cleanup_interval', type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    set_active_family_config(args.family_config)
    query_prompt_text = None
    if args.query_prompt_path:
        query_prompt_text = Path(args.query_prompt_path).read_text().strip()
    support_query_prompt_text = None
    if args.support_query_prompt_path:
        support_query_prompt_text = Path(args.support_query_prompt_path).read_text().strip()
    support_image_paths: list[str] = []
    if args.stage1_mode == 'support_query':
        if not args.support_json or not args.support_dir:
            raise ValueError('support_query mode requires --support_json and --support_dir')
        support_image_paths = load_support_image_paths(args.support_json, args.support_dir)
        if support_query_prompt_text and '{{support_block}}' in support_query_prompt_text:
            support_payload = json.loads(Path(args.support_json).read_text())
            cat_by_id = {c['id']: c['name'] for c in support_payload.get('categories', [])}
            cat_by_image = {}
            for ann in support_payload.get('annotations', []):
                cat_by_image.setdefault(ann['image_id'], ann['category_id'])
            per_image: list[tuple[int, str, str]] = []
            fam_counts: dict[str, int] = {}
            for i, img in enumerate(support_payload.get('images', []), start=1):
                cat_raw = cat_by_id.get(cat_by_image.get(img['id']), 'unknown')
                cat_display = cat_raw.replace('_', ' ')
                fam = get_category_family(cat_display) or 'unknown'
                per_image.append((i, fam, cat_display))
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
            summary_lines = ['Support image distribution across families:']
            for fam in get_family_names():
                n = fam_counts.get(fam, 0)
                summary_lines.append(f'  - {fam}: {n} support image(s)')
            summary_lines.append(
                'Note: the number of support examples per family is a property of the support set, '
                'not a prior for the query. Choose route_type strictly based on the QUERY image.'
            )
            img_lines = []
            for i, fam, cat_display in per_image:
                img_lines.append(
                    f'  Image {i} [family={fam}]: a labeled example of "{cat_display}" (support reference).'
                )
            img_lines.append(
                f'  Image {len(support_image_paths) + 1} (the last image): the QUERY image. Analyze this one.'
            )
            support_block = '\n'.join(summary_lines + [''] + img_lines)
            support_query_prompt_text = support_query_prompt_text.replace(
                '{{support_block}}', support_block
            )
    global_caption_prompt_text = None
    if args.global_caption_prompt_path:
        global_caption_prompt_text = Path(args.global_caption_prompt_path).read_text().strip()

    dataset = json.loads(Path(args.json_path).read_text())
    images = dataset['images']
    if args.image_id is not None:
        images = [img for img in images if img['id'] == args.image_id]
    if args.limit is not None:
        images = images[:args.limit]

    shared_vlm = SwiftVLMCaller(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        device=args.device,
    )
    controller = SemanticController(
        model_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        decoding_mode=args.llm_decoding_mode,
        seed=args.llm_seed,
        max_pixels=args.llm_max_pixels,
        query_only_instruction=query_prompt_text,
        instruction=support_query_prompt_text,
        client=shared_vlm,
    )
    request_config = getattr(shared_vlm, 'request_config', None)
    runtime_config = {
        'parsed_args': {
            'llm_model': args.llm_model,
            'device': args.device,
            'llm_max_new_tokens': args.llm_max_new_tokens,
            'llm_decoding_mode': args.llm_decoding_mode,
            'llm_seed': args.llm_seed,
            'llm_max_pixels': args.llm_max_pixels,
            'family_config': get_active_family_config_path(),
            'stage1_mode': args.stage1_mode,
        },
        'vlm_caller': {
            'model_path': shared_vlm.model_path,
            'max_new_tokens': shared_vlm.max_new_tokens,
            'decoding_mode': shared_vlm.decoding_mode,
            'seed': shared_vlm.seed,
            'max_pixels': shared_vlm.max_pixels,
            'device': shared_vlm.device,
            'resolved_device_map': _resolve_device_map(shared_vlm.device),
        },
        'family_spec': {
            'config_path': get_active_family_config_path(),
            'family_names': get_family_names(),
        },
        'request_config': {
            'max_tokens': getattr(request_config, 'max_tokens', None),
            'temperature': getattr(request_config, 'temperature', None),
            'top_k': getattr(request_config, 'top_k', None),
            'top_p': getattr(request_config, 'top_p', None),
            'seed': getattr(request_config, 'seed', None),
            'repetition_penalty': getattr(request_config, 'repetition_penalty', None),
        },
    }
    print('[stage1] runtime_config')
    print(pformat(runtime_config, sort_dicts=False))

    outputs: list[dict[str, object]] = []
    runtime_stats_path = Path(args.runtime_stats_jsonl).resolve() if args.runtime_stats_jsonl else None
    runtime_stats_fh = runtime_stats_path.open('w') if runtime_stats_path else None
    progress = tqdm(images, desc='stage1 semantic', unit='image')
    try:
        for index, image_info in enumerate(progress, start=1):
            query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
            image_start = time.perf_counter()
            ocr_hint = ''
            ocr_text = ''
            ocr_raw_text = None
            if args.enable_ocr_enrichment:
                if args.stage1_mode != 'query_only':
                    raise ValueError('--enable_ocr_enrichment currently supports only --stage1_mode query_only')
                ocr_raw_text = shared_vlm.generate(query_image_path, instruction=DOCUMENT_TEXT_PROMPT)
                ocr_text = _extract_tag(ocr_raw_text, 'text').strip()
                ocr_hint = _extract_tag(ocr_raw_text, 'document_hint').strip()

            has_ocr = bool(ocr_text and ocr_text.lower() not in {'', 'none'})
            if has_ocr:
                ocr_context = (
                    '\n\nOCR hint from image (use only if relevant to category selection):\n'
                    f'Type hint: {ocr_hint}\n'
                    f'Visible text: {ocr_text}'
                )
                enriched_instruction = controller.query_only_instruction + ocr_context
                stage1_raw = shared_vlm.generate(query_image_path, instruction=enriched_instruction)
                semantic = _parse_semantic_cue(stage1_raw)
                raw_text = stage1_raw if args.save_raw_text else None
            elif args.save_raw_text:
                if args.stage1_mode == 'support_query':
                    semantic, raw_text = controller.infer_with_raw(support_image_paths, query_image_path)
                else:
                    semantic, raw_text = controller.infer_query_only_with_raw(query_image_path)
            else:
                if args.stage1_mode == 'support_query':
                    semantic = controller.infer(support_image_paths, query_image_path)
                else:
                    semantic = controller.infer_query_only(query_image_path)
                raw_text = None
            # Rule-based detector-channel enrichment: if OCR gave a coarse noun hint
            # (e.g., "receipt", "newspaper", "business card"), append it to
            # proposal_prompts regardless of whether the main VLM saw OCR context.
            if args.enable_ocr_enrichment and ocr_hint and ocr_hint.lower() not in {'', 'none'}:
                if ocr_hint not in semantic.proposal_prompts:
                    semantic.proposal_prompts.append(ocr_hint)
            global_caption = ''
            global_caption_raw_text = None
            if args.save_global_caption:
                if not global_caption_prompt_text:
                    raise ValueError('--save_global_caption requires --global_caption_prompt_path')
                global_caption_raw_text = shared_vlm.generate(query_image_path, instruction=global_caption_prompt_text)
                global_caption = _extract_caption_from_raw(global_caption_raw_text)
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed_sec = time.perf_counter() - image_start
            record = {
                'image_id': image_info['id'],
                'query_image_path': query_image_path,
                'support_image_paths': list(support_image_paths),
                'controller_mode': args.stage1_mode,
                'null_policy': args.null_policy,
                'semantic_family': semantic.family,
                'route_type': semantic.route_type,
                'route_confidence': semantic.route_confidence,
                'semantic_categories': semantic.categories,
                'proposal_prompts': semantic.proposal_prompts,
                'null_likely': semantic.null_likely,
            }
            if raw_text is not None:
                record['semantic_raw_text'] = raw_text
            if args.enable_ocr_enrichment:
                record['ocr_text_hint'] = ocr_hint
                record['ocr_text_tokens'] = ocr_text
                if ocr_raw_text is not None:
                    record['ocr_raw_text'] = ocr_raw_text
            if args.save_global_caption:
                record['global_caption'] = global_caption
                record['global_caption_raw_text'] = global_caption_raw_text or ''
            runtime_stats = {
                'image_index': index,
                'image_id': image_info['id'],
                'elapsed_sec': round(elapsed_sec, 3),
                'controller_mode': args.stage1_mode,
                'support_image_count': len(support_image_paths),
                'semantic_family': semantic.family,
                'route_type': semantic.route_type,
                'route_confidence': semantic.route_confidence,
                'semantic_categories': semantic.categories,
                'null_likely': semantic.null_likely,
                'prompt_count': len(semantic.proposal_prompts),
            }
            if raw_text is not None:
                runtime_stats['raw_text_char_len'] = len(raw_text)
                runtime_stats['raw_text_word_len'] = len(raw_text.split())
            if args.enable_ocr_enrichment:
                runtime_stats['ocr_enrichment_used'] = has_ocr
                runtime_stats['ocr_text_hint'] = ocr_hint
                runtime_stats['ocr_text_token_count'] = len([tok for tok in ocr_text.split(',') if tok.strip()]) if ocr_text else 0
            if args.save_global_caption:
                runtime_stats['global_caption'] = global_caption
                runtime_stats['global_caption_char_len'] = len(global_caption)
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                runtime_stats['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1)
                torch.cuda.reset_peak_memory_stats(args.device)
            outputs.append(record)
            progress.set_postfix({
                'image_id': image_info['id'],
                'route': semantic.route_type or '-',
                'categories': ','.join(semantic.categories[:2]) or semantic.family or '-',
                'null': semantic.null_likely,
                'sec': f'{elapsed_sec:.1f}',
            })
            message = (
                f"Stage1 image_id={image_info['id']} categories={semantic.categories or [semantic.family]} "
                f"route={semantic.route_type or '-'} route_conf={semantic.route_confidence or '-'} "
                f"null={semantic.null_likely} prompts={len(semantic.proposal_prompts)} "
                f"elapsed_sec={elapsed_sec:.1f} mode={args.stage1_mode} support_images={len(support_image_paths)}"
            )
            if args.save_global_caption and global_caption:
                message += f" global_caption='{global_caption[:80]}'"
            if 'gpu_memory_reserved_mb' in runtime_stats:
                message += (
                    f" gpu_alloc_mb={runtime_stats['gpu_memory_allocated_mb']:.1f}"
                    f" gpu_reserved_mb={runtime_stats['gpu_memory_reserved_mb']:.1f}"
                    f" gpu_peak_mb={runtime_stats['gpu_max_memory_allocated_mb']:.1f}"
                )
            tqdm.write(message)
            if runtime_stats_fh is not None:
                runtime_stats_fh.write(json.dumps(runtime_stats) + '\n')
                runtime_stats_fh.flush()
            if args.cuda_cleanup_interval > 0 and index % args.cuda_cleanup_interval == 0:
                release_torch_runtime()
        progress.close()
    finally:
        if runtime_stats_fh is not None:
            runtime_stats_fh.close()

    payload = {
        'config': vars(args),
        'records': outputs,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    tqdm.write(f'Saved Stage 1 outputs to: {output_path}')


def _extract_caption_from_raw(raw_text: str) -> str:
    start_tag = '<caption>'
    end_tag = '</caption>'
    lower = raw_text.lower()
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        return raw_text[start + len(start_tag):end].strip()
    return raw_text.strip()


if __name__ == '__main__':
    main()
