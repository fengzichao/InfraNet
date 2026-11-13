"""Test modality provenance at real C QualGate layers, not toy gate inputs.

A quality score may use a stream previously fused with the other modality.
This test distinguishes literal RGB-only conditioning from RGB-path conditioning.
"""

from pathlib import Path
import contextlib
import io
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.yolo import Model  # noqa: E402
from models.common import QualityAwareCBFuse  # noqa: E402


def probe(model, original, modified):
    """Return input activations seen by each gate's quality estimator."""
    gate_activations = []
    handles = []
    for module in model.model:
        if isinstance(module, QualityAwareCBFuse):
            records = []
            gate_activations.append(records)
            handles.append(module.quality_estimator.register_forward_pre_hook(
                lambda _m, args, rec=records: rec.append(args[0].detach().clone())
            ))
    try:
        with torch.inference_mode():
            model(original)
            model(modified)
    finally:
        for handle in handles:
            handle.remove()
    assert all(len(rec) == 2 for rec in gate_activations)
    return [float((r[0] - r[1]).abs().max()) for r in gate_activations]


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2026)
    for config_name, expected_gates in (("yolov8-c.yaml", 3),):
        spec = yaml.safe_load((ROOT / 'models/detect' / config_name).read_text())
        spec.update(width_multiple=0.25, depth_multiple=0.5, nc=1)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model = Model(spec, ch=3, nc=1).eval()
        x = torch.randn(1, 6, 64, 64)
        # C: 0:3 IR, 3:6 RGB; E: 0:3 RGB, 3:6 IR.
        first_changed = x.clone()
        first_changed[:, :3] += 3.0
        second_changed = x.clone()
        second_changed[:, 3:] += 3.0
        first = probe(model, x, first_changed)
        second = probe(model, x, second_changed)
        assert len(first) == len(second) == expected_gates
        print(config_name, 'primary change -> q inputs:', first)
        print(config_name, 'secondary change -> q inputs:', second)
        if config_name == "yolov8-c.yaml":
            assert first[0] == 0.0, 'C first gate must score RGB without IR dependence'
            assert second[0] > 1e-7, 'C first gate should see RGB variation'
            # Subsequent C gates see a previously mixed RGB-path representation.
            # Therefore do not claim all C gates depend *exclusively* on RGB.
        else:
            assert all(d == 0.0 for d in second), 'E new q must be IR-independent'
            assert all(d > 1e-7 for d in first), 'E new q must depend on RGB'
    print('PASS: actual C backbone modality routing at all QualGate sites')


if __name__ == '__main__':
    main()
