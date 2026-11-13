#!/usr/bin/env python3
"""End-to-end CPU smoke test: paired training -> best checkpoint -> IR conversion -> evaluation.

Uses a scaled C graph and synthetic images. It never claims benchmark accuracy.
"""
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


def run_command(args, timeout=150):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                               env=env, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=timeout)
    if completed.returncode:
        raise AssertionError(f"Command failed ({completed.returncode}): {args}\n"
                             + completed.stdout[-7000:])
    return completed.stdout


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="infranet_training_smoke_") as d:
        base = Path(d)
        for modality in ("ir", "rgb"):
            for split, n in (("train", 4), ("val", 2)):
                image_dir = base / modality / split / "images"
                label_dir = base / modality / split / "labels"
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                for i in range(n):
                    seed = i + (100 if modality == "rgb" else 0)
                    image = np.random.default_rng(seed).integers(
                        0, 256, size=(96, 96, 3), dtype=np.uint8)
                    assert cv2.imwrite(str(image_dir / f"{i:04d}.png"), image)
                    (label_dir / f"{i:04d}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        data = {
            "path": str(base), "names": ["person"], "nc": 1,
            "modality_order": ["ir", "rgb"],
            "train": "ir/train/images", "train_rgb": "rgb/train/images",
            "val": "ir/val/images", "val_rgb": "rgb/val/images",
        }
        data_path = base / "data.yaml"
        data_path.write_text(yaml.safe_dump(data))

        training_cfg = yaml.safe_load((ROOT / "models/detect/yolov8-c.yaml").read_text())
        inference_cfg = yaml.safe_load((ROOT / "models/detect/gelanv8-c.yaml").read_text())
        for cfg in (training_cfg, inference_cfg):
            cfg.update(nc=1, width_multiple=0.25, depth_multiple=0.5)
        training_yaml, inference_yaml = base / "train.yaml", base / "infer.yaml"
        training_yaml.write_text(yaml.safe_dump(training_cfg))
        inference_yaml.write_text(yaml.safe_dump(inference_cfg))

        run_dir = base / "runs" / "smoke"
        run_command([
            ROOT / "train_dual.py", "--cfg", training_yaml, "--data", data_path,
            "--hyp", ROOT / "data/hyps/hyp.scratch-high.yaml", "--weights", "",
            "--epochs", "1", "--img", "64", "--batch-size", "2",
            "--workers", "0", "--device", "cpu", "--noplots",
            "--noautoanchor", "--project", base / "runs", "--name", "smoke",
            "--exist-ok",
        ])
        best = run_dir / "weights/best.pt"
        assert best.is_file() and (run_dir / "results.csv").is_file()
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        assert checkpoint["model"].model[-1].__class__.__name__ == "DualDDetect"

        converted = base / "ir_only.pt"
        run_command([
            ROOT / "convert.py", "--source", best,
            "--target-cfg", inference_yaml, "--output", converted,
        ])
        assert converted.is_file()
        out = torch.load(converted, map_location="cpu", weights_only=False)
        assert out["model"].model[-1].__class__.__name__ == "DDetect"
        assert out["conversion"]["input_modality"] == "infrared-only"
        run_command([
            ROOT / "val.py", "--weights", converted, "--data", data_path,
            "--img", "64", "--batch-size", "1", "--workers", "0",
            "--device", "cpu", "--project", base / "val", "--name", "smoke",
            "--exist-ok",
        ])
        print("PASS: paired C training, saved best.pt, conversion, and IR-only evaluation")
        print("NO BENCHMARK CLAIM: one epoch, scaled model, synthetic images and labels")


if __name__ == "__main__":
    main()
