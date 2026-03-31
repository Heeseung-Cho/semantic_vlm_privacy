# privacy-challenge-code

Public code-only snapshot for the privacy challenge experiments.

## Layout
- `challenge/semantic`
  - main semantic pipeline module
  - main semantic pipeline entrypoint
  - semantic pipeline status note
- `challenge/baseline`
  - baseline protocol and detector-only evaluator
- `challenge/manual`
  - Jupyter manual bbox annotation helper and notebook
- `challenge/archive`
  - older or secondary scripts kept out of the main public surface
- `challenge/configs`
  - config overrides
- `challenge/prompts`
  - prompt templates
- `challenge/folds`
  - split definitions

## Main files
- `challenge/semantic/semantic_gdino_sam.py`
- `challenge/semantic/run_semantic_gdino_sam_pipeline.py`
- `challenge/baseline/qwen_gdino_sam.py`
- `challenge/baseline/eval_support_gdino_detector.py`
- `challenge/manual/manual_bbox_annotation.py`
- `challenge/manual/manual_bbox_annotation.ipynb`

## Excluded
- datasets
- model weights and checkpoints
- experiment outputs
- vendored third-party repositories
- internal planning documents

## External dependencies
See `THIRD_PARTY.md`.
