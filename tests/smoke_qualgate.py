#!/usr/bin/env python3

"""Isolated smoke test of the unchanged QualityAwareCBFuse class definition.

Loads the class AST from models/common.py, not an independently reimplemented gate.
This does NOT prove that model or paper metrics are reproduced.
"""
from pathlib import Path
import ast
import torch
from torch import nn
import torch.nn.functional as F
repo = Path(__file__).resolve().parents[1]
source = (repo / 'models/common.py').read_text(encoding='utf-8')
node = next((n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'QualityAwareCBFuse'))
mod = ast.Module(body=[node], type_ignores=[])
namespace = dict(torch=torch, nn=nn, F=F)
exec(compile(ast.fix_missing_locations(mod), str(repo / 'models/common.py'), 'exec'), namespace)
Gate = namespace['QualityAwareCBFuse']
torch.manual_seed(0)
gate = Gate(idx=[0, 0]).train()
ir1 = [torch.randn(2, 32, 16, 16, requires_grad=True)]
ir2 = [torch.randn(2, 48, 8, 8, requires_grad=True)]
rgb = torch.randn(2, 64, 16, 16, requires_grad=True)
out = gate([ir1, ir2, rgb])
assert out.shape == rgb.shape, out.shape
assert torch.isfinite(out).all()
assert gate.quality_estimator is not None
optim = torch.optim.SGD(gate.parameters(), lr=0.01)
loss = out.square().mean()
loss.backward()
assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
assert any((p.grad is not None for p in gate.quality_estimator.parameters()))
optim.step()
print('PASS: actual QualityAwareCBFuse source — tensor shape, finite output, gradients, optimizer step')
print('NOTE: isolated module only; backbone, full inference, checkpoint loading and metric validation untested')
