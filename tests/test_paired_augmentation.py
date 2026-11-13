"""Check that C-only paired augmentations preserve the IR image under RGB HSV."""

from unittest.mock import patch
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataloaders import LoadImagesAndLabels


def run():
    dataset = LoadImagesAndLabels.__new__(LoadImagesAndLabels)
    dataset.indices = [0]
    dataset.hyp = {
        "mosaic": 0.0,
        "mixup": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "flipud": 0.0,
        "fliplr": 0.0,
    }
    dataset.augment = True
    dataset.mosaic = False
    dataset.rect = False
    dataset.img_size = 64
    dataset.labels = [np.array([[0, 0.5, 0.5, 0.4, 0.4]], dtype=np.float32)]
    dataset.im_files = ["ir/example.png"]
    dataset.im_files_2 = ["rgb/example.png"]

    ir = np.full((64, 64, 3), 35, dtype=np.uint8)
    rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    dataset.load_image = lambda _: (ir.copy(), (64, 64), (64, 64))
    dataset.load_image_2 = lambda _: (rgb.copy(), (64, 64), (64, 64))
    calls = []

    def simulate_hsv(image, **kwargs):
        calls.append((image.copy(), kwargs))
        image[:] = 220

    with patch("utils.dataloaders.augment_hsv", side_effect=simulate_hsv):
        ir_tensor, rgb_tensor, labels, _, _ = dataset[0]

    assert len(calls) == 1, f"Expected one HSV call on RGB only, got {len(calls)}"
    assert torch.all(ir_tensor == 35), "IR image was changed by photometric perturbation"
    assert torch.all(rgb_tensor == 220), "RGB image missed photometric perturbation"
    assert labels.shape == (1, 6)
    print("PASS: paired C loader applies RGB-only HSV; IR is unchanged")


if __name__ == "__main__":
    run()
