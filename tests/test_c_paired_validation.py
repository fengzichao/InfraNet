#!/usr/bin/env python3
"""Exercise paired IR→RGB C training validation with an actual DualDDetect graph.

Synthetic images, labels and random weights only. No paper accuracy claim.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.yolo import Model


def main():
    torch.set_num_threads(1)
    conf = yaml.safe_load((ROOT / 'models/detect/yolov8-c.yaml').read_text())
    conf.update(nc=1, width_multiple=0.25, depth_multiple=0.5)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = Model(conf, ch=3, nc=1).eval()
    model.names = ['person']
    model.nc = 1
    with tempfile.TemporaryDirectory(prefix='infranet_c_val_') as temp:
        base = Path(temp)
        for modality in ('ir', 'rgb'):
            for split in ('train', 'val'):
                images = base / modality / split / 'images'
                labels = base / modality / split / 'labels'
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                for idx in range(2):
                    image = np.random.default_rng(123 + idx + (8 if modality == 'rgb' else 0)).integers(
                        0, 256, size=(64, 64, 3), dtype=np.uint8
                    )
                    assert cv2.imwrite(str(images / f'{idx:04}.png'), image)
                    (labels / f'{idx:04}.txt').write_text('0 0.5 0.5 0.25 0.25\n')
        descriptor = base / 'data.yaml'
        descriptor.write_text(yaml.safe_dump({
            'path': str(base), 'nc': 1, 'names': ['person'],
            'modality_order': ['ir', 'rgb'],
            'train': 'ir/train/images', 'train_rgb': 'rgb/train/images',
            'val': 'ir/val/images', 'val_rgb': 'rgb/val/images',
        }))
        checkpoint = base / 'paired_c.pt'
        torch.save({'model': model, 'ema': None, 'optimizer': None, 'epoch': -1}, checkpoint)
        command = [
            sys.executable, str(ROOT / 'val_paired.py'), '--weights', str(checkpoint),
            '--data', str(descriptor), '--img', '64', '--batch-size', '1',
            '--workers', '0', '--device', 'cpu', '--project', str(base / 'val'),
            '--name', 'smoke', '--exist-ok',
        ]
        environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
                                text=True, timeout=150)
        if result.returncode:
            print((result.stdout + result.stderr)[-6500:])
            raise AssertionError(f'C paired validation failed: {result.returncode}')
        # Training invokes val_paired.run(model=..., compute_loss=...) rather than CLI mode.
        # Test this second interface as well, to prevent silent breaks mid-training.
        from utils.dataloaders import create_dataloader
        from utils.loss_tal_dual import ComputeLoss
        from val_paired import run as paired_validate
        model.hyp = yaml.safe_load((ROOT / 'data/hyps/hyp.scratch-high.yaml').read_text())
        loader = create_dataloader(
            str(base / 'ir/val/images'), str(base / 'rgb/val/images'),
            64, 1, 32, workers=0, rect=True, augment=False, cache=False
        )[0]
        result_values, maps, _ = paired_validate(
            yaml.safe_load(descriptor.read_text()), model=model,
            compute_loss=ComputeLoss(model), dataloader=loader,
            batch_size=1, imgsz=64, half=False, plots=False,
        )
        assert len(maps) == 1 and all(np.isfinite(x) for x in result_values)
        print('PASS: training-time C paired validation also computes auxiliary/main loss')
        print('PASS: C six-channel paired-validation loads checkpoint and completes NMS/metrics')
        print('NO AP CLAIM: fully synthetic evaluation')


if __name__ == '__main__':
    main()
