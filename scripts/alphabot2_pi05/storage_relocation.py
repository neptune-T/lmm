"""Explicit storage-only relocation; the frozen training config hash stays intact."""
from copy import deepcopy
from pathlib import Path

KEYS = frozenset({'out', 'init_checkpoint', 'train_data', 'val_data'})


def relocate(config, mapping):
    if not isinstance(mapping, dict) or not set(mapping) <= KEYS:
        raise ValueError('Storage map may only change out/init_checkpoint/train_data/val_data')
    result = deepcopy(config)
    for key, change in mapping.items():
        if (not isinstance(change, dict) or set(change) != {'from', 'to'}
                or change['from'] != config[key] or not Path(change['to']).is_absolute()):
            raise ValueError(f'Invalid storage relocation: {key}')
        result[key] = change['to']
    return result
