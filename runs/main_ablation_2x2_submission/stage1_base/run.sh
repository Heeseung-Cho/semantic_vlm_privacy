#!/usr/bin/env bash
# Base Stage-1 (VLM routing with cue) on QUERY FULL DATA (1056 images).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/choheeseung/workspace/vlm-privacy}"
REPO_ROOT="${PROJECT_ROOT}/challenge_repo"
CODE_DIR="${REPO_ROOT}"
OUT_DIR="${REPO_ROOT}/runs/main_ablation_2x2_submission/stage1_base"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"
python "${CODE_DIR}/semantic/run_stage1_semantic.py" \
  --query_dir "${PROJECT_ROOT}/data/Biv-priv-seg/query_images" \
  --json_path "${PROJECT_ROOT}/data/Biv-priv-seg/query_set_images_info.json" \
  --output_path "${OUT_DIR}/stage1_semantic.json" \
  --runtime_stats_jsonl "${OUT_DIR}/stage1_semantic.runtime.jsonl" \
  --llm_model Qwen/Qwen3-VL-8B-Instruct \
  --device cuda \
  --llm_max_new_tokens 1024 \
  --llm_decoding_mode deterministic \
  --llm_max_pixels 448 \
  --family_config "${CODE_DIR}/config/family_category_route4_v1.json" \
  --query_prompt_path "${CODE_DIR}/prompts/active/semantic_query_route4_v1.txt" \
  --null_policy skip \
  --save_raw_text 2>&1 | tee "${OUT_DIR}/stage1_log.txt"
