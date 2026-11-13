#!/usr/bin/env python3

"""Test the actual convert.py CLI on a scaled C network, with a temporary checkpoint.
Only synthetic weights; original benchmark checkpoint is unavailable.
"""
from pathlib import Path
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.yolo import Model


def main():
    torch.set_num_threads(1)
    cfg = yaml.safe_load((ROOT / 'models/detect/yolov8-c.yaml').read_text())
    tgt = yaml.safe_load((ROOT / 'models/detect/gelanv8-c.yaml').read_text())
    for c in (cfg, tgt):
        c.update(width_multiple=0.25, depth_multiple=0.5, nc=1)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        m = Model(cfg, ch=3, nc=1).eval()
    with tempfile.TemporaryDirectory(prefix='infranet_convert_cli_') as d:
        base = Path(d)
        torch.save({'model': m, 'ema': None}, base / 'source.pt')
        (base / 'target.yaml').write_text(yaml.safe_dump(tgt))
        cmd = [sys.executable, str(ROOT / 'convert.py'),  '--source', str(base / 'source.pt'), '--target-cfg', str(base / 'target.yaml'), '--output', str(base / 'out.pt')]
        out = subprocess.run(cmd, cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '1'}, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if out.returncode:
            print(out.stdout[-3500:])
            raise RuntimeError('convert.py CLI failed')
        ckpt = torch.load(base / 'out.pt', map_location='cpu', weights_only=False)
        net = ckpt['model']
        assert ckpt['conversion']['strict_state_entries'] > 100
        assert net.model[-1].__class__.__name__ == 'DDetect'
        assert not any((x.f == -4 for x in net.model))
        with torch.no_grad():
            ir = torch.rand(1, 3, 64, 64)
            dual = m(torch.cat((ir, torch.rand_like(ir)), dim=1))[0][1]
            ir_only = net.eval()(ir)[0]
        torch.testing.assert_close(ir_only, dual, rtol=2e-05, atol=2e-05)
        print('PASS: convert.py CLI produces loadable IR-only model with strict tensor mapping and prediction parity.')
        print('LIMITATION: scaled model and random weights only')


if __name__ == '__main__':
    main()
