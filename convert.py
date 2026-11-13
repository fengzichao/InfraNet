#!/usr/bin/env python3
"""Convert trusted InfraNet-IR paired-training checkpoints to 3-channel IR-only detectors.

Source: YOLOv8-style C2f `yolov8-c.yaml` with three QualGate sites and DualDDetect.
Target: `gelanv8-c.yaml` with only the IR backbone/head.

All target state entries must match one source entry exactly in shape and name mapping.
Loading legacy PyTorch full-model checkpoints is unsafe for untrusted files.
"""

import argparse
import re
from pathlib import Path

import torch

from models.yolo import Model

_MODEL_KEY = re.compile(r"^model\.(\d+)\.(.+)$")
_DEFAULT_TARGET = "models/detect/gelanv8-c.yaml"


def source_key(target_key: str) -> str:
    """Map destination layer indices and main detection head weights to source keys."""
    match = _MODEL_KEY.fullmatch(target_key)
    if match is None:
        raise ValueError(f"Unexpected target key: {target_key}")
    index, suffix = int(match.group(1)), match.group(2)
    if index < 22:
        source_index = index + 1  # Training model has initial Silence (layer 0).
    elif index == 22:
        source_index = 42  # Training DualDDetect (auxiliary + main IR heads).
        for source_prefix, target_prefix in (
            ("cv2.", "cv4."),
            ("cv3.", "cv5."),
            ("dfl.", "dfl2."),
        ):
            if suffix.startswith(source_prefix):
                suffix = target_prefix + suffix[len(source_prefix):]
                break
    else:
        raise ValueError(f"Unsupported IR-only destination layer: {index}")
    return f"model.{source_index}.{suffix}"


def validate_architectures(source, target) -> None:
    """Reject non-C checkpoints/YAMLs even if some tensor shapes happen to match."""
    if len(source.model) != 43 or len(target.model) != 23:
        raise ValueError('Expected C training (43 layers) and IR deployment (23 layers) graphs')
    if source.model[-1].__class__.__name__ != 'DualDDetect':
        raise ValueError('Expected DualDDetect source training head')
    if target.model[-1].__class__.__name__ != 'DDetect':
        raise ValueError('Expected DDetect target deployment head')
    gate_indices = [i for i, layer in enumerate(source.model)
                    if layer.__class__.__name__ == 'QualityAwareCBFuse']
    if gate_indices != [34, 37, 40]:
        raise ValueError(f'Unexpected QualGate locations: {gate_indices}')
    if any(layer.f == -4 for layer in target.model):
        raise ValueError('IR-only deployment graph must not read a second input stream')
    if source.model[-1].nc != target.model[-1].nc:
        raise ValueError('Source and target class counts differ')


def transfer_weights(source, target) -> int:
    """Strictly match every destination tensor; no silent partial loading."""
    source_state, target_state = source.state_dict(), target.state_dict()
    mapped = {}
    errors = []
    for name, tensor in target_state.items():
        source_name = source_key(name)
        if source_name not in source_state:
            errors.append(f'Missing source: {source_name} for {name}')
        elif source_state[source_name].shape != tensor.shape:
            errors.append(f'Shape mismatch: {source_name} -> {name}')
        else:
            mapped[name] = source_state[source_name].detach().to(
                device='cpu', dtype=tensor.dtype
            ).clone()
    if errors:
        raise ValueError(f'Conversion aborted ({len(errors)} unmatched tensors):\n' +
                         '\n'.join(errors[:25]))
    target.load_state_dict(mapped, strict=True)
    target.names = getattr(source, 'names', [str(i) for i in range(target.yaml['nc'])])
    target.nc = int(getattr(source, 'nc', target.yaml['nc']))
    return len(mapped)


def convert_model(source, target_cfg=_DEFAULT_TARGET):
    """Construct a standalone IR-only detector and copy verified C weights."""
    if source.model[-1].__class__.__name__ != 'DualDDetect':
        raise ValueError('Source must have the C paired-training DualDDetect head')
    target = Model(target_cfg, ch=3, nc=int(source.model[-1].nc)).cpu().float()
    validate_architectures(source, target)
    return target.eval(), transfer_weights(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Trusted InfraNet-IR C checkpoint')
    parser.add_argument('--target-cfg', default=_DEFAULT_TARGET)
    parser.add_argument('--output', required=True, help='Converted IR-only .pt path')
    parser.add_argument('--fp16', action='store_true', help='Save a half precision model')
    args = parser.parse_args()
    print('SECURITY: Only load checkpoints from trusted sources.')
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=False)
    source = checkpoint.get('ema') if isinstance(checkpoint, dict) else checkpoint
    if source is None and isinstance(checkpoint, dict):
        source = checkpoint.get('model')
    if source is None:
        raise ValueError('Missing source model or EMA in checkpoint')
    source = source.float().cpu().eval()
    converted, count = convert_model(source, args.target_cfg)
    if args.fp16:
        converted.half()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model': converted,
        'ema': None,
        'optimizer': None,
        'epoch': -1,
        'opt': None,
        'conversion': {
            'source_checkpoint': Path(args.source).name,
            'target_yaml': str(args.target_cfg),
            'strict_state_entries': count,
            'input_modality': 'infrared-only',
        },
    }, output)
    print(f'PASS: copied {count} state entries; saved pure IR checkpoint to {output}')


if __name__ == '__main__':
    main()
