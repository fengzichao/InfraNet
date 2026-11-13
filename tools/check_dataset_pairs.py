#!/usr/bin/env python3

"""Validate paired image stems and primary-stream YOLO labels; no alignment test."""
import argparse
from pathlib import Path
import yaml
import cv2
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def resolve(root, value):
    return [(root / Path(p)).resolve() for p in (value if isinstance(value, list) else [value])]


def find_files(dirs):
    out = []
    for p in dirs:
        if not p.is_dir():
            raise FileNotFoundError(f'Missing image folder: {p}')
        out += [x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXT]
    return sorted(out, key=lambda p: (p.stem, p.suffix))


def check(conf):
    cfg = yaml.safe_load(conf.read_text(encoding='utf-8'))
    root = (Path(__file__).resolve().parents[1] / cfg.get('path', '')).resolve()
    nc = cfg['nc']
    names = cfg['names']
    order = cfg.get('modality_order', ['primary', 'secondary'])
    if order != ['ir', 'rgb']:
        raise ValueError(f'InfraNet-IR C requires modality_order: [ir, rgb], got {order!r}')
    if len(names) != nc:
        raise ValueError(f'nc={nc} but names={len(names)}')
    for split in ('train', 'val'):
        primary = find_files(resolve(root, cfg[split]))
        secondary = find_files(resolve(root, cfg[f'{split}_rgb']))
        primary_stems = [p.stem for p in primary]
        secondary_stems = [p.stem for p in secondary]
        if primary_stems != secondary_stems:
            raise ValueError(f'{split}: paired stems differ: primary={len(primary)} secondary={len(secondary)}')
        if len(set(primary_stems)) != len(primary_stems):
            raise ValueError(f'{split}: duplicate image basenames detected')
        for a, b in zip(primary, secondary):
            first = cv2.imread(str(a), cv2.IMREAD_UNCHANGED)
            second = cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
            if first is None or second is None:
                raise ValueError(f'{split}: unreadable RGB/IR pair: {a}, {b}')
            if first.shape[:2] != second.shape[:2]:
                raise ValueError(f'{split}: dimension mismatch {a}: {first.shape[:2]} vs {b}: {second.shape[:2]}')
        missing = []
        nlabels = 0
        wrong = []
        for image in primary:
            txt = Path(str(image).replace('/images/', '/labels/')).with_suffix('.txt')
            if not txt.is_file():
                missing.append(str(txt))
                continue
            for line in txt.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = line.split()
                try:
                    if len(row) != 5:
                        raise ValueError('expected five fields')
                    cls = int(row[0])
                    coords = [float(q) for q in row[1:]]
                    if not 0 <= cls < nc or any((not 0 <= v <= 1 for v in coords)):
                        raise ValueError('class/coords out of range')
                    nlabels += 1
                except Exception as e:
                    wrong.append((str(txt), str(e)))
        print(f'{split}: primary={order[0]} ({len(primary)}), secondary={order[1]} ({len(secondary)}), boxes={nlabels}, missing labels={len(missing)}, invalid lines={len(wrong)}')
        if missing:
            print('  first missing:', missing[:3])
        if wrong:
            print('  first invalid:', wrong[:3])
        if missing or wrong:
            raise ValueError(f'{split}: label validation failed')
    print('PASS: pair names, dimensions and YOLO label syntax checked; geometric registration NOT verified')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    args = ap.parse_args()
    check(args.data)
