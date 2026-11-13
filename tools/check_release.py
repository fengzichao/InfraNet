#!/usr/bin/env python3
"""Validate the C-only repository scope, syntax, and training-to-deployment topology."""
import ast
import re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in ROOT.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
for path in [*ROOT.rglob('*.yaml'), *ROOT.rglob('*.yml')]:
    yaml.safe_load(path.read_text(encoding='utf-8'))

models = sorted(path.name for path in (ROOT / 'models/detect').glob('*.yaml'))
assert models == ['gelanv8-c.yaml', 'yolov8-c.yaml'], models
assert not (ROOT / 'detect_dual.py').exists()
assert not (ROOT / 'train.py').exists()
assert not (ROOT / 'val_dual.py').exists()
assert "default=ROOT / 'data/llvip.yaml'" in (ROOT / 'detect.py').read_text(encoding='utf-8')
assert not list((ROOT / 'data').glob('*_e.yaml'))
assert sum(1 for path in ROOT.rglob('*.md')) == 1
assert (ROOT / 'README.md').exists() and (ROOT / 'LICENSE').exists()
training = yaml.safe_load((ROOT / 'models/detect/yolov8-c.yaml').read_text(encoding='utf-8'))
deployment = yaml.safe_load((ROOT / 'models/detect/gelanv8-c.yaml').read_text(encoding='utf-8'))
train_layers = training['backbone'] + training['head']
deploy_layers = deployment['backbone'] + deployment['head']
assert len(train_layers) == 43 and len(deploy_layers) == 23
assert [layer[2] for layer in train_layers[1:23]] == [layer[2] for layer in deploy_layers[:22]]
assert [i for i, layer in enumerate(train_layers) if layer[2] == 'QualityAwareCBFuse'] == [34, 37, 40]
assert train_layers[-1][2] == 'DualDDetect'
assert deploy_layers[-1][2] == 'DDetect'
assert not any(layer[0] == -4 for layer in deploy_layers)
assert '--source' in (ROOT / 'convert.py').read_text(encoding='utf-8')
assert "default='models/detect/yolov8-c.yaml'" in (ROOT / 'train_dual.py').read_text(encoding='utf-8')
assert "default=ROOT / 'data/llvip.yaml'" in (ROOT / 'train_dual.py').read_text(encoding='utf-8')
assert "validate_modality_order(data, 'c', require=True)" in (ROOT / 'val_paired.py').read_text(encoding='utf-8')
assert "validate_modality_order(data_dict, 'c', require=True)" in (ROOT / 'train_dual.py').read_text(encoding='utf-8')
for path in ROOT.rglob('*.py'):
    if path.name == "check_release.py":
        continue
    content = path.read_text(encoding='utf-8')
    if re.search(r'(?<![A-Za-z0-9])(?:/data[0-9]+/|/home/[^/]+/|[A-Z]:\\Users\\)', content):
        raise AssertionError(f'Potential private path in {path}')
print('PASS: C-only scope, syntax, IR-only deployment graph, QualGate sites, and path hygiene')
print('NOT VERIFIED: published benchmark accuracy and original training provenance')
