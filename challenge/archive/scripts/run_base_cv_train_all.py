#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run all VizWiz base 3-fold detector training jobs sequentially.')
    parser.add_argument('--config', required=True, help='Training config path')
    parser.add_argument('--splits-dir', required=True, help='Directory containing fold1/fold2/fold3 split JSON files')
    parser.add_argument('--image-root', required=True, help='Base image root')
    parser.add_argument('--pretrained-checkpoint', required=True, help='Base Grounding DINO checkpoint')
    parser.add_argument('--output-root', required=True, help='Root output directory for fold work dirs')
    parser.add_argument('--batch-size', type=int, default=4, help='Train batch size override')
    parser.add_argument('--max-epochs', type=int, default=None, help='Optional max epoch override')
    parser.add_argument('--start-fold', type=int, default=1, choices=[1, 2, 3], help='First fold to run')
    parser.add_argument('--end-fold', type=int, default=3, choices=[1, 2, 3], help='Last fold to run')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_fold > args.end_fold:
        raise ValueError('start-fold must be <= end-fold')

    train_py = Path(__file__).resolve().parent / 'train_llm2seg_baseline.py'
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(args.start_fold, args.end_fold + 1):
        fold_name = f'fold{fold_idx}'
        train_ann = Path(args.splits_dir) / fold_name / 'train.json'
        val_ann = Path(args.splits_dir) / fold_name / 'val.json'
        work_dir = output_root / fold_name

        cmd = [
            sys.executable,
            str(train_py),
            '--config', str(Path(args.config).resolve()),
            '--train-ann', str(train_ann.resolve()),
            '--train-img-root', str(Path(args.image_root).resolve()),
            '--val-ann', str(val_ann.resolve()),
            '--val-img-root', str(Path(args.image_root).resolve()),
            '--pretrained-checkpoint', str(Path(args.pretrained_checkpoint).resolve()),
            '--work-dir', str(work_dir),
            '--batch-size', str(args.batch_size),
        ]
        if args.max_epochs is not None:
            cmd.extend(['--max-epochs', str(args.max_epochs)])

        print(f'\n===== Running {fold_name} =====')
        print('Command:', ' '.join(cmd))
        subprocess.run(cmd, check=True)

    print('\nAll requested folds finished successfully.')


if __name__ == '__main__':
    main()
