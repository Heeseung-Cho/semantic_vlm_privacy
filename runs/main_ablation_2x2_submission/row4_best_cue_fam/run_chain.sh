#!/usr/bin/env bash
# SUBMISSION / Row 4 (BEST): cue + FAM expansion + caption
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/choheeseung/workspace/vlm-privacy}"
REPO_ROOT="${PROJECT_ROOT}/challenge_repo"
CODE_DIR="${REPO_ROOT}"
ROW_DIR="${REPO_ROOT}/runs/main_ablation_2x2_submission/row4_best_cue_fam"
BASE_STAGE1="${REPO_ROOT}/runs/main_ablation_2x2_submission/stage1_base/stage1_semantic.json"
GT_JSON="${PROJECT_ROOT}/data/Biv-priv-seg/query_set_images_info.json"
export PYTHONPATH="${CODE_DIR}:${PYTHONPATH:-}"

mkdir -p "${ROW_DIR}/stage1_fam" "${ROW_DIR}/stage2" "${ROW_DIR}/stage3"

python "${CODE_DIR}/scripts/expand_stage1_to_family.py" \
  --input_path "${BASE_STAGE1}" \
  --output_path "${ROW_DIR}/stage1_fam/stage1_semantic.json" \
  --family_config "${CODE_DIR}/config/family_category_route4_v1.json"

python "${CODE_DIR}/semantic/run_stage2_detection.py" \
  --stage1_path "${ROW_DIR}/stage1_fam/stage1_semantic.json" \
  --output_path "${ROW_DIR}/stage2/stage2_detection_gdino_ft.json" \
  --config_path "${PROJECT_ROOT}/challenge/LLM2Seg/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py" \
  --checkpoint_path "${PROJECT_ROOT}/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth" \
  --device cuda \
  --box_threshold 0.20 --text_threshold 0.10 --proposal_nms_iou 0.50 --max_candidates 5 2>&1 | tee "${ROW_DIR}/stage2/stage2_log.txt"

python "${CODE_DIR}/semantic/run_stage3_minimal.py" \
  --json_path "${GT_JSON}" \
  --stage1_path "${ROW_DIR}/stage1_fam/stage1_semantic.json" \
  --stage2_path "${ROW_DIR}/stage2/stage2_detection_gdino_ft.json" \
  --output_dir "${ROW_DIR}/stage3" \
  --prompt_path "${CODE_DIR}/prompts/active/stage3_l0_enriched.txt" \
  --per_image_prompt_path "${CODE_DIR}/prompts/active/stage3_per_image_norank.txt" \
  --ocr_prompt_path "${CODE_DIR}/prompts/active/semantic_image_description.txt" \
  --family_config "${CODE_DIR}/config/family_category_route4_v1.json" \
  --llm_model Qwen/Qwen3-VL-8B-Instruct \
  --device cuda \
  --llm_decoding_mode deterministic \
  --llm_max_pixels 3584 \
  --proposal_score_threshold 0.40 \
  --enriched_context \
  --per_image_mode \
  --document_ocr \
  --max_new_tokens 1024 2>&1 | tee "${ROW_DIR}/stage3/stage3_log.txt"

mkdir -p "${ROW_DIR}/stage4_sam"
python "${CODE_DIR}/scripts/add_sam_masks.py" \
  --det_path "${ROW_DIR}/stage3/query_submission.json" \
  --gt_path "${GT_JSON}" \
  --image_dir "${PROJECT_ROOT}/data/Biv-priv-seg/query_images" \
  --sam_checkpoint "${PROJECT_ROOT}/challenge/LLM2Seg/checkpoints/sam_vit_h_4b8939.pth" \
  --sam_model_type vit_h \
  --device cuda \
  --output_path "${ROW_DIR}/stage4_sam/query_submission_segm.json" 2>&1 | tee "${ROW_DIR}/stage4_sam/sam_log.txt"
