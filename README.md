# semantic_vlm_privacy

Standalone code snapshot for the current privacy challenge pipeline.

## Current Structure
- `semantic/`
  - main no-train semantic pipeline
  - `VLM -> G-DINO -> optional rerank -> optional SAM`
- `baseline/`
  - detector-only support evaluator
  - baseline Qwen/G-DINO/SAM utilities
  - GT overlay exporter
- `common/`
  - shared overlay and image helpers
- `prompts/`
  - active semantic prompt files used by the current pipeline
- `configs/`
  - detector config override

## Current Pipeline
1. `VLM semantic controller` reads the query image, or support+query images, and predicts:
   - coarse object family
   - short detector-friendly cue list
   - null prior
2. `Grounding DINO` uses the cue list to generate proposal boxes.
3. `VLM reranking` is optional and is used only when support-conditioned matching is enabled.
4. `SAM` is optional and currently treated as a later refinement stage rather than the main decision module.

## Main Files
- `semantic/semantic_gdino_sam.py`
- `semantic/run_semantic_gdino_sam_pipeline.py`
- `baseline/qwen_gdino_sam.py`
- `baseline/eval_support_gdino_detector.py`
- `baseline/export_support_gt_overlays.py`
- `common/overlay_utils.py`

## Prompt Files
- `prompts/semantic_query_only.txt`
- `prompts/semantic_support_query.txt`
- `prompts/semantic_rerank.txt`

## How To Run
All commands below assume:
- dataset paths and checkpoint paths are updated for the local machine

### 1. Detector-only baseline on support images
This evaluates pretrained or finetuned Grounding DINO on the BIV support set using the GT category name as the prompt.

```bash
python baseline/eval_support_gdino_detector.py   --support-dir /path/to/Biv-priv-seg/support_images   --support-json /path/to/Biv-priv-seg/support_set.json   --output-dir /path/to/output/support_gdino_pretrained   --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py   --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth   --device cuda
```

Outputs:
- `support_eval_summary.json`
- `support_eval_records.json`
- `visualizations/*_gt_pred_overlay.jpg`

### 2. Export GT overlays for support images
This is only for annotation sanity checking.

```bash
python baseline/export_support_gt_overlays.py   --support-dir /path/to/Biv-priv-seg/support_images   --support-json /path/to/Biv-priv-seg/support_set.json   --output-dir /path/to/output/support_gt_overlays   --draw-bbox   --draw-polygon
```

Outputs:
- `manifest.json`
- `*_gt_overlay.jpg`

### 3. Query-only semantic pipeline
This runs the current no-train semantic controller pipeline without support images.
Use this first to inspect `semantic cue -> G-DINO proposals` behavior.

```bash
python semantic/run_semantic_gdino_sam_pipeline.py   --query-dir /path/to/Biv-priv-seg/support_images   --json-path /path/to/Biv-priv-seg/support_set.json   --output-dir /path/to/output/semantic_query_only   --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py   --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth   --sam-checkpoint /path/to/sam_vit_h_4b8939.pth   --llm-model /path/to/Qwen3-VL-4B-Instruct   --controller-mode query_only   --device cuda   --limit 8   --llm-max-pixels 256   --max-candidates 2   --disable-sam   --null-policy strict   --save-vis
```

Outputs:
- `semantic_pipeline_results.json`
- `run_config.json`
- `visualizations/*_semantic_overlay.jpg`

### 4. Support-conditioned semantic pipeline
This enables support-conditioned semantic disambiguation and optional VLM reranking.

```bash
python semantic/run_semantic_gdino_sam_pipeline.py   --query-dir /path/to/Biv-priv-seg/query_images   --json-path /path/to/Biv-priv-seg/dev_set_images_info.json   --support-dir /path/to/Biv-priv-seg/support_images   --support-json /path/to/Biv-priv-seg/support_set.json   --output-dir /path/to/output/semantic_support_query   --config-path /path/to/configs/grounding_dino_swin-t_finetune_8xb2_20e_viz.py   --checkpoint-path /path/to/groundingdino_swint_ogc_mmdet-822d7e9d.pth   --sam-checkpoint /path/to/sam_vit_h_4b8939.pth   --llm-model /path/to/Qwen3-VL-4B-Instruct   --controller-mode support_query   --device cuda   --limit 8   --llm-max-pixels 256   --max-candidates 2   --disable-sam   --null-policy strict   --save-vis
```

Notes:
- Add `--disable-sam` to inspect bbox-only behavior first.
- Remove `--disable-sam` only after the detector-side behavior is stable enough.
- `query_only` currently picks the top detector proposal when no support-conditioned reranking is available.

## Notes
- This repo is meant to run the current semantic-controller experiments standalone.
- This repo no longer requires an LLM2Seg checkout or runtime import path.
- Datasets, checkpoints, and experiment outputs are excluded.

## External dependencies
See `THIRD_PARTY.md`.
