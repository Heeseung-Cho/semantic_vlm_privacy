#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LLM2Seg Grounding DINO baseline.")
    parser.add_argument("--config", required=True, help="Path to the training config")
    parser.add_argument("--train-ann", required=True, help="Training COCO annotations")
    parser.add_argument("--train-img-root", required=True, help="Training image root")
    parser.add_argument("--pretrained-checkpoint", required=True, help="Base Grounding DINO checkpoint")
    parser.add_argument("--val-ann", default=None, help="Optional validation COCO annotations")
    parser.add_argument("--val-img-root", default=None, help="Optional validation image root")
    parser.add_argument("--work-dir", default=None, help="Output directory for checkpoints/logs")
    parser.add_argument("--max-epochs", type=int, default=None, help="Optional override for max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional train batch size override")
    parser.add_argument(
        "--unfreeze-language-model",
        action="store_true",
        help="Remove any language_model lr_mult=0 override before training",
    )
    return parser.parse_args()


def load_categories(ann_path: Path) -> tuple[list[str], list[dict]]:
    payload = json.loads(ann_path.read_text())
    categories = sorted(payload["categories"], key=lambda item: item["id"])
    class_names = [cat["name"] for cat in categories]
    return class_names, categories


def normalize_coco_annotations(src_path: Path, dst_path: Path) -> Path:
    payload = json.loads(src_path.read_text())
    updated = False
    normalized_annotations: list[dict[str, Any]] = []
    for new_ann_id, ann in enumerate(payload.get("annotations", []), start=1):
        normalized = dict(ann)
        if normalized.get("id") != new_ann_id:
            normalized["id"] = new_ann_id
            updated = True
        if "iscrowd" not in normalized:
            normalized["iscrowd"] = 0
            updated = True
        if "ignore" not in normalized:
            normalized["ignore"] = 0
            updated = True
        normalized_annotations.append(normalized)
    if updated:
        payload["annotations"] = normalized_annotations
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"Normalized COCO annotations written to: {dst_path}")
        return dst_path
    return src_path


def patch_loader_dataset(loader_cfg, ann_file: str, img_root: str, metainfo: dict) -> None:
    if loader_cfg is None:
        return
    dataset = loader_cfg["dataset"]
    dataset["data_root"] = ""
    dataset["ann_file"] = ann_file
    dataset["data_prefix"] = {"img": img_root}
    dataset["metainfo"] = metainfo


def maybe_unfreeze_language_model(cfg: Config) -> bool:
    optim_wrapper = cfg.get("optim_wrapper")
    if not optim_wrapper:
        return False
    paramwise_cfg = optim_wrapper.get("paramwise_cfg")
    if not paramwise_cfg:
        return False
    custom_keys = paramwise_cfg.get("custom_keys")
    if not custom_keys or "language_model" not in custom_keys:
        return False
    del custom_keys["language_model"]
    return True


def patch_dataset(cfg: Config, train_ann: Path, train_img_root: Path, val_ann: Path | None, val_img_root: Path | None) -> None:
    class_names, _ = load_categories(train_ann)
    metainfo = dict(classes=tuple(class_names), palette=[(220, 20, 60)])

    cfg.class_name = tuple(class_names)
    cfg.num_classes = len(class_names)
    cfg.metainfo = metainfo
    cfg.model["bbox_head"]["num_classes"] = len(class_names)

    train_ann_file = str(train_ann.resolve())
    train_img_dir = str(train_img_root.resolve()) + "/"
    patch_loader_dataset(cfg.get("train_dataloader"), train_ann_file, train_img_dir, metainfo)

    eval_ann = val_ann.resolve() if val_ann else train_ann.resolve()
    eval_img_root = val_img_root.resolve() if val_img_root else train_img_root.resolve()
    eval_ann_file = str(eval_ann)
    eval_img_dir = str(eval_img_root) + "/"
    patch_loader_dataset(cfg.get("val_dataloader"), eval_ann_file, eval_img_dir, metainfo)
    patch_loader_dataset(cfg.get("test_dataloader"), eval_ann_file, eval_img_dir, metainfo)

    for evaluator_name in ("val_evaluator", "test_evaluator"):
        evaluator = cfg.get(evaluator_name)
        if evaluator is not None:
            evaluator["ann_file"] = eval_ann_file


def resolve_work_dir(work_dir: str | None) -> str:
    if work_dir:
        return work_dir
    now = datetime.now()
    return str(
        Path(__file__).resolve().parents[2] / 'results' / 'llm2seg_baseline_train'
        / now.strftime("%Y%m%d")
        / now.strftime("%H%M%S")
    )


def main() -> None:
    args = parse_args()
    register_all_modules(init_default_scope=True)

    work_dir = resolve_work_dir(args.work_dir)
    normalized_dir = Path(work_dir) / "normalized_coco"
    train_ann_path = normalize_coco_annotations(
        Path(args.train_ann).resolve(),
        normalized_dir / "train.json",
    )
    val_ann_path = (
        normalize_coco_annotations(Path(args.val_ann).resolve(), normalized_dir / "val.json")
        if args.val_ann
        else None
    )

    cfg = Config.fromfile(args.config)
    patch_dataset(
        cfg,
        train_ann_path,
        Path(args.train_img_root).resolve(),
        val_ann_path,
        Path(args.val_img_root).resolve() if args.val_img_root else None,
    )
    cfg.load_from = str(Path(args.pretrained_checkpoint).resolve())
    cfg.work_dir = work_dir

    if args.max_epochs is not None:
        cfg.max_epoch = args.max_epochs
        if "train_cfg" in cfg:
            cfg.train_cfg["max_epochs"] = args.max_epochs
        for scheduler in cfg.get("param_scheduler", []):
            if scheduler.get("by_epoch"):
                scheduler["end"] = args.max_epochs

    if args.batch_size is not None and "train_dataloader" in cfg:
        cfg.train_dataloader["batch_size"] = args.batch_size

    language_unfrozen = False
    if args.unfreeze_language_model:
        language_unfrozen = maybe_unfreeze_language_model(cfg)

    print(f"Training with config: {args.config}")
    print(f"Training annotations: {train_ann_path}")
    print(f"Training image root: {Path(args.train_img_root).resolve()}")
    if args.val_ann:
        print(f"Validation annotations: {val_ann_path}")
        print(f"Validation image root: {Path(args.val_img_root).resolve() if args.val_img_root else Path(args.train_img_root).resolve()}")
    print(f"Loading pretrained checkpoint: {cfg.load_from}")
    print(f"Saving outputs to: {cfg.work_dir}")
    print(f"Language model unfreezed at runtime: {language_unfrozen}")

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
