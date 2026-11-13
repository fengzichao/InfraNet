"""Ensure corrupt primary images do not shift paired secondary indices."""
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.dataloaders import LoadImagesAndLabels


def main():
    with tempfile.TemporaryDirectory() as temp:
        base = Path(temp)
        for stream in ('ir', 'rgb'):
            for part in ('images', 'labels'):
                (base / stream / part).mkdir(parents=True)
            for name in ('000', '001', '002'):
                destination = base / stream / 'images' / f'{name}.png'
                if stream == 'ir' and name == '001':
                    destination.write_bytes(b'not a valid png')
                else:
                    pixels = np.full((64, 64, 3), int(name) + 30, dtype=np.uint8)
                    assert cv2.imwrite(str(destination), pixels)
                (base / stream / 'labels' / f'{name}.txt').write_text('0 0.5 0.5 0.25 0.25\n')
        ds = LoadImagesAndLabels(
            str(base / 'ir' / 'images'), str(base / 'rgb' / 'images'),
            img_size=64, batch_size=1, augment=False, stride=32, prefix='test: '
        )
        assert len(ds) == 2, len(ds)
        stems = [(Path(a).stem, Path(b).stem) for a,b in zip(ds.im_files, ds.im_files_2)]
        assert stems == [('000','000'), ('002','002')], stems
        for idx in range(2):
            ir, rgb, labels, *_ = ds[idx]
            assert ir.shape == rgb.shape == (3, 64, 64)
        print('PASS secondary pairs remain aligned when corrupted primary image is dropped')

if __name__ == '__main__':
    main()
