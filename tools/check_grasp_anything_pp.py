"""Check manifest loading, actual dense targets and batched collation on CPU."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data.grasp_anything_language_data import GraspAnythingLanguageDataset, language_collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--samples-per-split', type=int, default=8)
    parser.add_argument('--input-size', type=int, default=224)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        parser.error('--samples-per-split must be positive')
    torch.set_num_threads(1)
    summary = {}
    scenes = {}
    for split in ('train', 'val', 'test_seen', 'test_unseen'):
        dataset = GraspAnythingLanguageDataset(args.manifest, split=split, segmentation=True,
                                              mask_target='part', output_size=args.input_size)
        scenes[split] = {r['scene_id'] for r in dataset.samples}
        indices = np.linspace(0, len(dataset) - 1, min(args.samples_per_split, len(dataset)), dtype=int)
        samples = [dataset[int(index)] for index in indices]
        for sample in samples:
            if not all(torch.isfinite(t).all() for t in (sample['image'], sample['target_mask'], *sample['grasp_targets'].values())):
                raise ValueError('Nonfinite tensors')
            if not sample['target_mask'].any() or not sample['grasp_targets']['pos'].any():
                raise ValueError('Empty segmentation/grasp target')
        batch = language_collate(samples[:2])
        assert batch['image'].shape[1:] == (3, args.input_size, args.input_size)
        assert batch['target_mask'].shape[1:] == (1, args.input_size, args.input_size)
        summary[split] = {'samples': len(dataset), 'scenes': len(scenes[split]),
                          'checked_samples': len(samples), 'batch_shape': list(batch['image'].shape)}
        print(json.dumps({split: summary[split]}), flush=True)
    keys = list(scenes)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            if scenes[a] & scenes[b]:
                raise ValueError(f'Scene leakage: {a}/{b}')
    print('PASS: real data, dense targets, collation and scene separation', flush=True)


if __name__ == '__main__':
    main()
