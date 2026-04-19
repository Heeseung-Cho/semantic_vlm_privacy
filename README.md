# semantic_vlm_privacy

Standalone code snapshot for the current privacy challenge pipeline.

Pipeline Summary
1. Stage 1 semantic split
2. Stage 2 Grounding DINO proposal generation
3. Stage 3 support-reference matching on top-k candidate crops
4. Final selection and optional visualization

Key point
- Few-shot support is used only at Stage 3.
- Stage 1 remains query-only in the current strongest no-train configuration.

Stage Details
Stage 1: Semantic split
- Input: query image only
- Output: coarse semantic family, detector-friendly cues, null prior
- Role: produce family prior and detector cues

Stage 2: Grounding DINO proposals
- Input: query image and Stage-1 cue list
- Output: candidate bounding boxes and detector scores
- Role: generate candidate object regions

Stage 3: Few-shot reference matching
- Input: support reference crops with exact labels, top-k candidate crops from Stage 2
- Output: best matching exact category and matching score
- Role: this is where few-shot support is used
- Role: asks which support category the crop is most similar to

Stage 4: Final selection
- Input: detector candidates and Stage-3 matches
- Output: final bbox/category result and optional overlay visualization
- Role: simple score-based final selection

Repository Structure
- semantic/: semantic pipeline and runner
- baseline/: detector-only evaluation and reusable Grounding DINO helpers
- common/: project-owned VLM caller, model loaders, overlay helpers, and text utils
- prompts/active/: active semantic and reference-matching prompts
- configs/: local Grounding DINO config files used by this repo

Main Files
- semantic/semantic_gdino_sam.py
- semantic/run_semantic_gdino_sam_pipeline.py
- baseline/qwen_gdino_sam.py
- baseline/eval_support_gdino_detector.py
- common/vlm.py
- common/model_loaders.py
- common/text_utils.py
- common/overlay_utils.py

Active Prompts
- prompts/active/semantic_query_only.txt
- prompts/active/semantic_support_query.txt
- prompts/active/semantic_rerank.txt
- prompts/active/semantic_document_text.txt
- prompts/active/semantic_transactional_text.txt
- prompts/active/semantic_reference_match.txt

Main Run Modes
1) Query-only semantic probe
python semantic/run_semantic_gdino_sam_pipeline.py --query-dir /path/to/images --json-path /path/to/annotations.json --output-dir /path/to/output/semantic_query_only --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --llm-model /path/to/Qwen3-VL-4B-Instruct --controller-mode query_only --device cuda --disable-sam --save-vis

2) Few-shot reference matching
This is the current main no-train configuration.
python semantic/run_semantic_gdino_sam_pipeline.py --query-dir /path/to/query_images --json-path /path/to/query_annotations.json --support-dir /path/to/support_images --support-json /path/to/support_set.json --output-dir /path/to/output/reference_match --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth --sam-checkpoint /path/to/sam_vit_h_4b8939.pth --llm-model /path/to/Qwen3-VL-4B-Instruct --controller-mode query_only --device cuda --disable-sam --calibration-mode reference_match --classification-top-k 2 --disable-document-text --save-vis

Notes
- This repo does not require an LLM2Seg checkout or runtime import path.
- Datasets, checkpoints, and experiment outputs are excluded.
- The current strongest no-train path is Stage-3 support-reference matching.
- Method/leakage guard checklist:
  - `METHOD_GUARD_CHECKLIST.md`
  - includes wording templates for method section, README, and code design principles

External Dependencies
See THIRD_PARTY.md.

Current Staged Workflow
Run commands inside the `psi` conda environment.

1) Stage 1 query-only direct category cues
python semantic/run_stage1_semantic.py --query_dir /path/to/query_images --json_path /path/to/dev_pseudo_label_3w_coco.json --output_path /path/to/stage1_semantic.json --runtime_stats_jsonl /path/to/stage1_semantic.runtime.jsonl --llm_model Qwen/Qwen3-VL-4B-Instruct --device cuda --llm_max_new_tokens 160 --llm_decoding_mode deterministic --llm_max_pixels 448 --family_config config/family_category_direct_v1.json --query_prompt_path prompts/active/semantic_query_only.txt --null_policy ignore --save_raw_text

2) Stage 2 Grounding DINO candidates with null hard drop and cue provenance
python semantic/run_stage2_detection.py --stage1_path /path/to/stage1_semantic.json --config_path configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py --checkpoint_path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth --output_path /path/to/stage2_detection_gdino_ft.json

3) Stage 3 Stage-1-category-shortlist reference match
python semantic/run_stage3_calibration.py --json_path /path/to/dev_pseudo_label_3w_coco.json --stage1_path /path/to/stage1_semantic.json --stage2_path /path/to/stage2_detection_gdino_ft.json --output_dir /path/to/stage3_output --sam_checkpoint /path/to/sam_vit_h_4b8939.pth --llm_model Qwen/Qwen3-VL-4B-Instruct --device cuda --llm_decoding_mode deterministic --llm_max_pixels 448 --family_config config/family_category_direct_v1.json --calibration_mode reference_match --reference_source crop --support_dir /path/to/support_images --support_json /path/to/support_set.json --disable_sam --proposal_score_threshold 0.0 --classification_top_k 2 --skip_null_stage3

권장 실행 경로: Route4 Best Protocol
- 전용 엔트리포인트:
  - `python semantic/run_route4_best_protocol.py --output_root /path/to/output_root`
- 이 스크립트는 현재 저장소에서 권장하는 Route4 파이프라인을 Stage 1 -> Stage 2 -> Stage 3 순서로 한 번에 실행한다.
- 기본 동작:
  - Stage 1: 이미지 단위로 하나의 route와 소수의 category 후보를 정리한다.
  - Stage 2: Stage-1 cue를 이용해 Grounding DINO proposal을 만든다.
  - Stage 3: support/reference 비교 또는 shortlist 내부 forced 선택으로 최종 category를 정리한다.
- 문서 계열 처리 원칙:
  - 현재 strongest observed 방향은 document Stage-3를 얇게 두는 것이다.
  - 즉 Stage-1 shortlist를 그대로 강하게 쓰고, 문서 crop에서는 shortlist 내부 forced subtype selection만 수행한다.
  - confidence tier, OCR, region/attribute reasoning, prompt-match fallback은 현재 decision path에서 제외하는 것이 더 안정적이었다.
  - 문서 route에서는 이미지별로 최종 문서 후보 1개만 남긴다.
- 설계 의도:
  - Stage 1은 넓은 의미의 semantic prior를 제공한다.
  - Stage 2는 localization을 담당한다.
  - Stage 3는 과도한 reasoning layer가 아니라 proposal-level category resolution과 후보 정리를 담당한다.
  - 즉, route/category prior, localization, 얇은 exact-category resolution을 분리한 구조다.
- 출력:
  - `output_root/stage1_semantic.json`
  - `output_root/stage2_detection_gdino_ft.json`
  - `output_root/stage3_best_protocol/semantic_pipeline_results.json`
  - `output_root/stage3_best_protocol/query_submission.json`
  - `output_root/protocol_manifest.json`
- 이미 만들어둔 Stage-1/Stage-2 결과를 그대로 재사용하고 싶으면 `--stage1_path`, `--stage2_path`를 직접 넘기면 된다.

Score terminology:
- `proposal_score`: Grounding DINO text-conditioned proposal score from Stage 2. Use this for proposal thresholding/ranking.
- `query_submission[].score`: submission-format alias of `proposal_score`.
- `candidate_score`, `detector_score`, and `final_score`: deprecated compatibility aliases. They currently equal `proposal_score` and should not be interpreted as semantic/category confidence.
