#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.vlm import SwiftVLMCaller, release_torch_runtime
from semantic.semantic_gdino_sam import SemanticController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Stage 1 query-only semantic inference.')
    parser.add_argument('--query-dir', required=True)
    parser.add_argument('--json-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--llm-model', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--llm-max-new-tokens', type=int, default=256)
    parser.add_argument('--llm-decoding-mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm-seed', type=int, default=None)
    parser.add_argument('--llm-max-pixels', type=int, default=448)
    parser.add_argument('--null-policy', choices=['strict', 'skip', 'ignore'], default='ignore')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--image-id', type=int, default=None)
    parser.add_argument('--runtime-stats-jsonl', default=None)
    parser.add_argument('--cuda-cleanup-interval', type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        client=shared_vlm,
    )

    outputs: list[dict[str, object]] = []
    runtime_stats_path = Path(args.runtime_stats_jsonl).resolve() if args.runtime_stats_jsonl else None
    runtime_stats_fh = runtime_stats_path.open('w') if runtime_stats_path else None
    progress = tqdm(images, desc='stage1 semantic', unit='image')
    try:
        for index, image_info in enumerate(progress, start=1):
            query_image_path = str((Path(args.query_dir) / image_info['file_name']).resolve())
            image_start = time.perf_counter()
            semantic = controller.infer_query_only(query_image_path)
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elapsed_sec = time.perf_counter() - image_start
            record = {
                'image_id': image_info['id'],
                'query_image_path': query_image_path,
                'support_image_paths': [],
                'controller_mode': 'query_only',
                'null_policy': args.null_policy,
                'semantic_family': semantic.family,
                'proposal_prompts': semantic.proposal_prompts,
                'null_likely': semantic.null_likely,
            }
            runtime_stats = {
                'image_index': index,
                'image_id': image_info['id'],
                'elapsed_sec': round(elapsed_sec, 3),
                'controller_mode': 'query_only',
                'support_image_count': 0,
                'semantic_family': semantic.family,
                'null_likely': semantic.null_likely,
                'prompt_count': len(semantic.proposal_prompts),
            }
            if torch.cuda.is_available() and str(args.device).startswith('cuda'):
                runtime_stats['gpu_memory_allocated_mb'] = round(torch.cuda.memory_allocated(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_memory_reserved_mb'] = round(torch.cuda.memory_reserved(args.device) / (1024 ** 2), 1)
                runtime_stats['gpu_max_memory_allocated_mb'] = round(torch.cuda.max_memory_allocated(args.device) / (1024 ** 2), 1)
                torch.cuda.reset_peak_memory_stats(args.device)
            outputs.append(record)
            progress.set_postfix({
                'image_id': image_info['id'],
                'family': semantic.family or '-',
                'null': semantic.null_likely,
                'sec': f'{elapsed_sec:.1f}',
            })
            message = (
                f"Stage1 image_id={image_info['id']} family={semantic.family} "
                f"null={semantic.null_likely} prompts={len(semantic.proposal_prompts)} "
                f"elapsed_sec={elapsed_sec:.1f} mode=query_only support_images=0"
            )
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


if __name__ == '__main__':
    main()
