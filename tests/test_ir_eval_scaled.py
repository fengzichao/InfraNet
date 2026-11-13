#!/usr/bin/env python3

"""Run val.py with a freshly converted tiny random-weight IR-only checkpoint.

This tests checkpoint serialization/load and evaluation *pipeline* only.
It does NOT provide meaningful AP or validate the paper's benchmark results.
All temporary images/weights stay outside the repository in TemporaryDirectory.
"""
from pathlib import Path
import contextlib
import io
import os
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
from convert import convert_model


def main():
    torch.set_num_threads(1)
    source_cfg = yaml.safe_load((ROOT / 'models/detect/yolov8-c.yaml').read_text())
    target_cfg = yaml.safe_load((ROOT / 'models/detect/gelanv8-c.yaml').read_text())
    for cfg in (source_cfg, target_cfg):
        cfg.update(nc=1, width_multiple=0.25, depth_multiple=0.5)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model = Model(source_cfg, ch=3, nc=1).eval()
        converted, n = convert_model(model, target_cfg)
    converted.names = ['person']
    with tempfile.TemporaryDirectory(prefix='infranet_ir_eval_') as d:
        base = Path(d)
        for split in ('train', 'val'):
            (base / split / 'images').mkdir(parents=True)
            (base / split / 'labels').mkdir(parents=True)
            for j in range(2):
                rng = np.random.default_rng(1234 + j)
                image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
                cv2.imwrite(str(base / split / 'images' / f'{j:04}.png'), image)
                (base / split / 'labels' / f'{j:04}.txt').write_text('0 0.5 0.5 0.3 0.3\n')
        data = base / 'data.yaml'
        data.write_text(yaml.safe_dump({'path': str(base), 'train': 'train/images', 'val': 'val/images', 'nc': 1, 'names': ['person']}))
        weights = base / 'test_converted.pt'
        torch.save({'model': converted, 'optimizer': None, 'epoch': -1, 'ema': None}, weights)
        cmd = [sys.executable, str(ROOT / 'val.py'), '--weights', str(weights), '--data', str(data), '--img', '64', '--batch-size', '1', '--workers', '0', '--device', 'cpu', '--project', str(base / 'eval_run'), '--name', 'test', '--exist-ok']
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        result = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=150)
        if result.returncode:
            print(result.stdout[-3500:])
            raise RuntimeError(f'val.py failed: {result.returncode}')
        print('PASS: val.py loaded converted IR-only checkpoint and completed a two-image toy dataset evaluation.')
        print(f'Converted tensor count: {n}')
        print('NO BENCHMARK CLAIM: images, labels and weights are random.')
        for line in result.stdout.splitlines():
            if line.strip().startswith('all') or 'Speed:' in line:
                print('pipeline: ' + line.strip()[:200])


if __name__ == '__main__':
    main()
