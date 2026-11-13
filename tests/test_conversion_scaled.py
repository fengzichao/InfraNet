#!/usr/bin/env python3

"""End-to-end random-weight C conversion test with source vs deployed IR output comparison.

Uses a reduced model for CPU, and does NOT validate paper accuracy / weights.
"""
import io
import contextlib
from pathlib import Path
import sys
import yaml
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.yolo import Model
from convert import convert_model, transfer_weights
torch.set_num_threads(1)
cfg = yaml.safe_load((ROOT / 'models/detect/yolov8-c.yaml').read_text())
cfg.update(width_multiple=0.25, depth_multiple=0.5, nc=1)
target_cfg = yaml.safe_load((ROOT / 'models/detect/gelanv8-c.yaml').read_text())
target_cfg.update(width_multiple=0.25, depth_multiple=0.5, nc=1)
print('Constructing scaled training and IR-only graphs...', flush=True)
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    source = Model(cfg, ch=3, nc=1).eval()
    converted, count = convert_model(source, target_cfg)
assert count > 100, count
assert not any((layer.f == -4 for layer in converted.model))
assert converted.model[-1].__class__.__name__ == 'DDetect'
print(f'PASS: strict tensor transfer ({count} tensors), no RGB jumps in deployment graph')
with torch.no_grad():
    ir = torch.randn(1, 3, 64, 64)
    rgb = torch.randn_like(ir)
    dual = source(torch.cat((ir, rgb), dim=1))
    one = converted(ir)
assert isinstance(dual, tuple) and isinstance(dual[0], list)
torch.testing.assert_close(one[0], dual[0][1], rtol=2e-05, atol=2e-05)
print('PASS: output of converted 3-channel IR model matches IR head of the six-channel source.')
with torch.no_grad():
    dual_other = source(torch.cat((ir, torch.randn_like(rgb)), dim=1))
torch.testing.assert_close(dual[0][1], dual_other[0][1], rtol=0, atol=0)
print('PASS: source IR prediction unchanged when RGB input changes.')
print('LIMITATIONS: synthetic random weights, scaled architecture; no full checkpoint or dataset mAP.')
