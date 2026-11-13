"""Verify all public C-only dataset configurations and reject inverted streams."""
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.modality_contract import validate_modality_order


def main():
    for name in ('llvip',):
        path = ROOT / 'data' / f'{name}.yaml'
        data = yaml.safe_load(path.read_text(encoding='utf-8'))
        assert data['modality_order'] == ['ir', 'rgb']
        assert validate_modality_order(data, 'c', require=True)
        assert all(data.get(k) for k in ('train', 'train_rgb', 'val', 'val_rgb'))
    for bad in (['rgb', 'ir'], ['rgb', 'rgb']):
        try:
            validate_modality_order({'modality_order': bad}, 'c', require=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Accepted invalid stream order: {bad}')
    try:
        validate_modality_order({}, 'c', require=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Accepted missing modality declaration')
    print('PASS: C-only LLVIP data template and negative-order validation')


if __name__ == '__main__':
    main()
