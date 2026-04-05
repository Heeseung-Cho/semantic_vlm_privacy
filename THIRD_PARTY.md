# Third-Party Dependencies

This repository does not bundle datasets, weights, or third-party source trees.

## Expected external layout
- `checkpoints/groundingdino_swint_ogc_mmdet-822d7e9d.pth`
- `checkpoints/sam_vit_h_4b8939.pth`


## Models
Provide local model paths when running scripts, for example:
- Qwen model directory via script arguments such as `--llm-model` or `--model`
- BERT model directory via `BERT_MODEL_DIR` when needed by the config

## Data
All dataset paths are passed at runtime through script arguments .
