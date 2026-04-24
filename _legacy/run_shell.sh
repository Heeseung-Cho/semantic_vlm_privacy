## Stage 1
python semantic/run_stage1_semantic.py \
    --query_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/query_images \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/direct_categories_stage1/stage1_semantic.json \
    --runtime_stats_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/direct_categories_stage1/stage1_semantic.runtime.jsonl \
    --llm_model Qwen/Qwen3-VL-4B-Instruct \
    --device cuda \
    --llm_max_new_tokens 160 \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge_repo/config/family_category_direct_v1.json \
    --query_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge_repo/prompts/active/semantic_query_only.txt \
    --null_policy skip \
    --save_raw_text

## Stage 2

  python semantic/run_stage2_detection.py \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/direct_categories_stage1/stage1_semantic.json \
    --config_path /home/choheeseung/workspace/vlm-privacy/challenge_repo/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py \
    --checkpoint_path /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/stage2_direct_categories_gdino_ft_nulldrop_stage2_detection_gdino_ft.json \
    --device cuda \
    --box_threshold 0.20 \
    --text_threshold 0.10 \
    --proposal_nms_iou 0.50 \
    --max_candidates 5


  python semantic/run_stage3_calibration.py \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/direct_categories_stage1/stage1_semantic.json \
    --stage2_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/stage2_direct_categories_gdino_ft_nulldrop_stage2_detection_gdino_ft.json \
    --output_dir /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/stage3_direct_categories_stage1shortlist \
    --sam_checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/sam_vit_h_4b8939.pth \
    --llm_model Qwen/Qwen3-VL-4B-Instruct \
    --device cuda \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge_repo/config/family_category_direct_v1.json \
    --calibration_mode reference_match \
    --reference_source crop \
    --support_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
    --support_json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
    --disable_sam \
    --final_score_threshold 0.0 \
    --skip_null_stage3 \
    --verbose_decisions \
    --decision_log_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/stage3_direct_categories_stage1shortlist_catonly_trace/stage3_decisions.jsonl \
    --save_calibration_raw_text

  python /home/choheeseung/workspace/vlm-privacy/challenge/scripts/visualize_stage3_results.py \
    --stage3-json /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/stage3_direct_categories_stage1shortlist/semantic_pipeline_results.json \
    --pseudo-coco-path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --output-dir /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/stage3_direct_categories_stage1shortlist/viz_thr000 \
    --score-threshold 0.0 \
    --iou-threshold 0.5 \
    --contact-sheet-max-images 128

 python semantic/run_stage3_calibration.py \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/direct_categories_stage1/stage1_semantic.json \
    --stage2_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/stage2_direct_categories_gdino_ft_nulldrop_stage2_detection_gdino_ft.json \
    --output_dir /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/stage3_direct_categories_docrefine \
    --sam_checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/sam_vit_h_4b8939.pth \
    --llm_model Qwen/Qwen3-VL-4B-Instruct \
    --device cuda \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge_repo/config/family_category_direct_v1.json \
    --calibration_mode reference_match \
    --reference_source crop \
    --support_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
    --support_json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
    --disable_sam \
    --final_score_threshold 0.0 \
    --skip_null_stage3 \
    --verbose_decisions \
    --decision_log_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260415/4b/stage3_direct_categories_docrefine/stage3_decisions.jsonl \
    --save_calibration_raw_text \
    --enable_document_refine \
    --document_refine_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge_repo/prompts/active/semantic_document_refine.txt




  python /home/choheeseung/workspace/vlm-privacy/challenge/scripts/run_route4_best_protocol_query1k.py \
    --output_root /home/choheeseung/workspace/vlm-privacy/challenge/results/route4_best_protocol_query1k

  python /home/choheeseung/workspace/vlm-privacy/challenge/scripts/evaluate_detection_coco.py \
    --gt_json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --pred_json /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/4b_route4_doc_forced_support_ocr/query_submission.json \
    --output_json /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/4b_route4_doc_forced_support_ocr/detection_metrics.json


  python /home/choheeseung/workspace/vlm-privacy/challenge/semantic/run_stage1_semantic.py \
    --query_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/query_images \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/4b_route4_stage1_global_caption/stage1_semantic.json \
    --runtime_stats_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/4b_route4_stage1_global_caption/stage1_semantic.runtime.jsonl \
    --llm_model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --llm_max_new_tokens 1024 \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge/config/family_category_route4_v1.json \
    --query_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_query_route4_v1.txt \
    --null_policy skip \
    --save_raw_text \
    --save_global_caption \
    --global_caption_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_stage1_global_caption.txt


  python /home/choheeseung/workspace/vlm-privacy/challenge/semantic/run_stage1_semantic.py \
    --query_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/query_images \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage1_semantic.json \
    --runtime_stats_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage1_semantic.runtime.jsonl \
    --llm_model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --llm_max_new_tokens 180 \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge/config/family_category_route4_v1.json \
    --query_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_query_route4_v1.txt \
    --null_policy skip \
    --save_raw_text \
    --enable_ocr_enrichment

  python /home/choheeseung/workspace/vlm-privacy/challenge/semantic/run_stage3_calibration.py \
    --json_path /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/dev_pseudo_label_3w_coco.json \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage1_semantic.json \
    --stage2_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage2_detection_gdino_ft.json \
    --output_dir /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage3_minforced \
    --sam_checkpoint /home/choheeseung/workspace/vlm-privacy/challenge/LLM2Seg/checkpoints/sam_vit_h_4b8939.pth \
    --llm_model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --family_config /home/choheeseung/workspace/vlm-privacy/challenge/config/family_category_route4_v1.json \
    --calibration_mode reference_match \
    --reference_source crop \
    --support_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
    --support_json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
    --disable_sam \
    --proposal_score_threshold 0.0 \
    --skip_null_stage3 \
    --verbose_decisions \
    --decision_log_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage3_minforced/stage3_decisions.jsonl \
    --save_calibration_raw_text \
    --enable_document_refine \
    --document_refine_mode shortlist_forced \
    --document_refine_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_document_refine_route_v2.txt \
    --document_forced_refine_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_document_refine_forced_shortlist.txt


  python /home/choheeseung/workspace/vlm-privacy/challenge/semantic/run_stage25_reject_gate.py \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage1_semantic.json \
    --stage2_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage2_detection_gdino_ft.json \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage25_rejectgate_prior/stage25_filtered_stage2.json \
    --llm_model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --reject_gate_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_candidate_reject_gate.txt \
    --decision_log_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage25_rejectgate_prior/stage25_reject_gate.jsonl \
    --reject_policy invalid_only \
    --enable_prior_category_filter  


  python /home/choheeseung/workspace/vlm-privacy/challenge/semantic/run_stage25_reject_gate.py \
    --stage1_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage1_semantic.json \
    --stage2_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1_ocr_enriched/stage2_detection_gdino_ft.json \
    --output_path /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage25_rejectgate_v2/stage25_filtered_stage2.json \
    --llm_model Qwen/Qwen3-VL-8B-Instruct \
    --device cuda \
    --llm_decoding_mode deterministic \
    --llm_max_pixels 448 \
    --support_dir /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_images \
    --support_json /home/choheeseung/workspace/vlm-privacy/data/Biv-priv-seg/support_set.json \
    --reference_source crop \
    --reject_gate_prompt_path /home/choheeseung/workspace/vlm-privacy/challenge/prompts/active/semantic_candidate_reject_gate.txt \
    --decision_log_jsonl /home/choheeseung/workspace/vlm-privacy/challenge/results/challenge_repo_dev_pseudo_run/20260420/8b_route4_stage1ocr_stage25_rejectgate_v2/stage25_reject_gate.jsonl \
    --reject_policy invalid_only

