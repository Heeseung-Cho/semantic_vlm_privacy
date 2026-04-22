#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

DEFAULT_QUERY_DIR = WORKSPACE_ROOT / 'data' / 'Biv-priv-seg' / 'query_images'
DEFAULT_JSON_PATH = WORKSPACE_ROOT / 'data' / 'Biv-priv-seg' / 'dev_pseudo_label_3w_coco.json'
DEFAULT_SUPPORT_DIR = WORKSPACE_ROOT / 'data' / 'Biv-priv-seg' / 'support_images'
DEFAULT_SUPPORT_JSON = WORKSPACE_ROOT / 'data' / 'Biv-priv-seg' / 'support_set.json'
DEFAULT_FAMILY_CONFIG = PROJECT_ROOT / 'config' / 'family_category_route4_v1.json'
DEFAULT_STAGE1_PROMPT = PROJECT_ROOT / 'prompts' / 'active' / 'stage1' / 'semantic_query_route4_v1.txt'
DEFAULT_DOCUMENT_PROMPT = PROJECT_ROOT / 'prompts' / 'deprecated' / 'semantic_document_refine_route_v2.txt'
DEFAULT_DOCUMENT_FORCED_PROMPT = PROJECT_ROOT / 'prompts' / 'active' / 'stage3' / 'semantic_document_refine_forced_shortlist.txt'
DEFAULT_GDINO_CONFIG = PROJECT_ROOT / 'configs' / 'grounding_dino_swin-t_finetune_8xb2_20e_viz.py'
DEFAULT_GDINO_CHECKPOINT = WORKSPACE_ROOT / 'challenge' / 'LLM2Seg' / 'checkpoints' / 'groundingdino_swint_ogc_mmdet-822d7e9d.pth'
DEFAULT_SAM_CHECKPOINT = WORKSPACE_ROOT / 'challenge' / 'LLM2Seg' / 'checkpoints' / 'sam_vit_h_4b8939.pth'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the best confirmed Route4 protocol as a dedicated entrypoint.'
    )
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--query_dir', default=str(DEFAULT_QUERY_DIR))
    parser.add_argument('--json_path', default=str(DEFAULT_JSON_PATH))
    parser.add_argument('--support_dir', default=str(DEFAULT_SUPPORT_DIR))
    parser.add_argument('--support_json', default=str(DEFAULT_SUPPORT_JSON))
    parser.add_argument('--llm_model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--family_config', default=str(DEFAULT_FAMILY_CONFIG))
    parser.add_argument('--stage1_prompt_path', default=str(DEFAULT_STAGE1_PROMPT))
    parser.add_argument('--document_refine_prompt_path', default=str(DEFAULT_DOCUMENT_PROMPT))
    parser.add_argument('--gdino_config', default=str(DEFAULT_GDINO_CONFIG))
    parser.add_argument('--gdino_checkpoint', default=str(DEFAULT_GDINO_CHECKPOINT))
    parser.add_argument('--sam_checkpoint', default=str(DEFAULT_SAM_CHECKPOINT))
    parser.add_argument('--stage1_path', default=None)
    parser.add_argument('--stage2_path', default=None)
    parser.add_argument('--llm_max_new_tokens', type=int, default=180)
    parser.add_argument('--llm_max_pixels', type=int, default=448)
    parser.add_argument('--llm_decoding_mode', choices=['deterministic', 'stochastic'], default='deterministic')
    parser.add_argument('--llm_seed', type=int, default=None)
    parser.add_argument('--stage2_box_threshold', type=float, default=0.20)
    parser.add_argument('--stage2_text_threshold', type=float, default=0.10)
    parser.add_argument('--stage2_proposal_nms_iou', type=float, default=0.50)
    parser.add_argument('--stage2_max_candidates', type=int, default=5)
    parser.add_argument('--stage3_proposal_score_threshold', type=float, default=0.0)
    parser.add_argument('--classification_top_k', type=int, default=None)
    parser.add_argument('--enable_document_ocr', action='store_true')
    parser.add_argument(
        '--document_refine_mode',
        choices=['prior_preserving', 'authoritative', 'shortlist_forced'],
        default='shortlist_forced',
    )
    parser.add_argument('--document_forced_refine_prompt_path', default=str(DEFAULT_DOCUMENT_FORCED_PROMPT))
    parser.add_argument(
        '--enable_document_prompt_match_fallback',
        action='store_true',
        help='Re-enable the later lexical prompt-match experiment. Keep this off for the best confirmed protocol.',
    )
    return parser.parse_args()


def _runtime_env(output_root: Path) -> dict[str, str]:
    runtime_root = output_root / '.runtime_env'
    modelscope_cache = runtime_root / 'modelscope'
    hf_home = runtime_root / 'hf_home'
    mpl_config = runtime_root / 'mplconfig'
    tmp_dir = runtime_root / 'tmp'
    for path in (modelscope_cache, hf_home, mpl_config, tmp_dir):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MODELSCOPE_CACHE', str(modelscope_cache))
    env.setdefault('HF_HOME', str(hf_home))
    env.setdefault('MPLCONFIGDIR', str(mpl_config))
    env.setdefault('TMPDIR', str(tmp_dir))
    return env


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print('[best-protocol] running:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_env = _runtime_env(output_root)

    stage1_path = Path(args.stage1_path).resolve() if args.stage1_path else output_root / 'stage1_semantic.json'
    stage2_path = Path(args.stage2_path).resolve() if args.stage2_path else output_root / 'stage2_detection_gdino_ft.json'
    stage3_dir = output_root / 'stage3_best_protocol'

    manifest = {
        'protocol_name': 'route4_best_confirmed_v1',
        'notes': [
            'Captures the current strongest observed workspace policy: Stage-1 route4 prior + recall-heavy Stage-2 + thin Stage-3 document forced-shortlist selection.',
            'Document Stage-3 is intentionally thin here: no confidence tiers, no OCR path, no prompt-match fallback, and no region/attribute reasoning in the final decision.',
            'If the active Stage-1 route4 prompt has changed since the dev-best diagnostic run, pass explicit --stage1_path/--stage2_path artifacts to reproduce the exact upstream inputs.',
        ],
        'paths': {
            'query_dir': str(Path(args.query_dir).resolve()),
            'json_path': str(Path(args.json_path).resolve()),
            'support_dir': str(Path(args.support_dir).resolve()),
            'support_json': str(Path(args.support_json).resolve()),
            'family_config': str(Path(args.family_config).resolve()),
            'stage1_prompt_path': str(Path(args.stage1_prompt_path).resolve()),
            'document_refine_prompt_path': str(Path(args.document_refine_prompt_path).resolve()),
            'gdino_config': str(Path(args.gdino_config).resolve()),
            'gdino_checkpoint': str(Path(args.gdino_checkpoint).resolve()),
            'sam_checkpoint': str(Path(args.sam_checkpoint).resolve()),
            'stage1_path': str(stage1_path),
            'stage2_path': str(stage2_path),
            'stage3_dir': str(stage3_dir),
        },
        'stage2_params': {
            'box_threshold': args.stage2_box_threshold,
            'text_threshold': args.stage2_text_threshold,
            'proposal_nms_iou': args.stage2_proposal_nms_iou,
            'max_candidates': args.stage2_max_candidates,
        },
        'stage3_params': {
            'proposal_score_threshold': args.stage3_proposal_score_threshold,
            'classification_top_k': args.classification_top_k,
            'skip_null_stage3': True,
            'disable_sam': True,
            'enable_document_refine': True,
            'enable_document_ocr': args.enable_document_ocr,
            'document_refine_mode': args.document_refine_mode,
            'enable_document_prompt_match_fallback': args.enable_document_prompt_match_fallback,
            'document_forced_refine_prompt_path': args.document_forced_refine_prompt_path,
        },
    }
    (output_root / 'protocol_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    if args.stage1_path is None:
        stage1_cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'semantic' / 'run_stage1_semantic.py'),
            '--query_dir', str(Path(args.query_dir).resolve()),
            '--json_path', str(Path(args.json_path).resolve()),
            '--output_path', str(stage1_path),
            '--runtime_stats_jsonl', str(output_root / 'stage1_semantic.runtime.jsonl'),
            '--llm_model', args.llm_model,
            '--device', args.device,
            '--llm_max_new_tokens', str(args.llm_max_new_tokens),
            '--llm_decoding_mode', args.llm_decoding_mode,
            '--llm_max_pixels', str(args.llm_max_pixels),
            '--family_config', str(Path(args.family_config).resolve()),
            '--query_prompt_path', str(Path(args.stage1_prompt_path).resolve()),
            '--null_policy', 'skip',
            '--save_raw_text',
        ]
        if args.llm_seed is not None:
            stage1_cmd.extend(['--llm_seed', str(args.llm_seed)])
        _run(stage1_cmd, env=runtime_env)

    if args.stage2_path is None:
        stage2_cmd = [
            sys.executable,
            str(PROJECT_ROOT / 'semantic' / 'run_stage2_detection.py'),
            '--stage1_path', str(stage1_path),
            '--output_path', str(stage2_path),
            '--config_path', str(Path(args.gdino_config).resolve()),
            '--checkpoint_path', str(Path(args.gdino_checkpoint).resolve()),
            '--device', args.device,
            '--box_threshold', str(args.stage2_box_threshold),
            '--text_threshold', str(args.stage2_text_threshold),
            '--proposal_nms_iou', str(args.stage2_proposal_nms_iou),
            '--max_candidates', str(args.stage2_max_candidates),
        ]
        _run(stage2_cmd, env=runtime_env)

    stage3_cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'semantic' / 'run_stage3_calibration.py'),
        '--json_path', str(Path(args.json_path).resolve()),
        '--stage1_path', str(stage1_path),
        '--stage2_path', str(stage2_path),
        '--output_dir', str(stage3_dir),
        '--sam_checkpoint', str(Path(args.sam_checkpoint).resolve()),
        '--llm_model', args.llm_model,
        '--device', args.device,
        '--llm_decoding_mode', args.llm_decoding_mode,
        '--llm_max_pixels', str(args.llm_max_pixels),
        '--family_config', str(Path(args.family_config).resolve()),
        '--calibration_mode', 'reference_match',
        '--reference_source', 'crop',
        '--support_dir', str(Path(args.support_dir).resolve()),
        '--support_json', str(Path(args.support_json).resolve()),
        '--disable_sam',
        '--proposal_score_threshold', str(args.stage3_proposal_score_threshold),
        '--skip_null_stage3',
        '--verbose_decisions',
        '--decision_log_jsonl', str(stage3_dir / 'stage3_decisions.jsonl'),
        '--save_calibration_raw_text',
        '--enable_document_refine',
        '--document_refine_prompt_path', str(Path(args.document_refine_prompt_path).resolve()),
        '--document_refine_mode', args.document_refine_mode,
    ]
    if args.document_forced_refine_prompt_path:
        stage3_cmd.extend(['--document_forced_refine_prompt_path', str(Path(args.document_forced_refine_prompt_path).resolve())])
    if args.enable_document_ocr:
        stage3_cmd.append('--enable_document_ocr')
    if args.llm_seed is not None:
        stage3_cmd.extend(['--llm_seed', str(args.llm_seed)])
    if args.classification_top_k is not None:
        stage3_cmd.extend(['--classification_top_k', str(args.classification_top_k)])
    if args.enable_document_prompt_match_fallback:
        stage3_cmd.append('--enable_document_prompt_match_fallback')
    _run(stage3_cmd, env=runtime_env)

    print(f'[best-protocol] manifest: {output_root / "protocol_manifest.json"}')
    print(f'[best-protocol] stage3 output: {stage3_dir / "semantic_pipeline_results.json"}')


if __name__ == '__main__':
    main()
