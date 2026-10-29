"""Explicit image/instruction/grasp/mask mapping, without filename heuristics."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.dataset_processing import grasp
from .grasp_anything_data import GraspAnythingDataset
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnythingLanguageDataset(GraspAnythingDataset):
    def __init__(self, file_path, split='train', output_size=224,
                 segmentation=False, mask_target='object', normalization='legacy'):
        # Reuse label rasterization/tensor conversion, bypass legacy file indexing.
        LanguageGraspDatasetBase.__init__(
            self, output_size=output_size, include_rgb=True, include_depth=False)
        if mask_target not in ('object', 'part', 'union'):
            raise ValueError('mask_target must be object, part or union')
        if normalization not in ('legacy', 'imagenet'):
            raise ValueError('normalization must be legacy or imagenet')
        self.segmentation = segmentation
        self.mask_target = mask_target
        self.normalization = normalization
        manifest = Path(file_path).resolve()
        self.samples = []
        ids = set()
        scene_splits = {}
        with manifest.open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in ('sample_id', 'scene_id', 'instruction', 'rgb_path',
                            'grasp_path', 'annotation_height', 'annotation_width', 'split'):
                    if key not in row or row[key] is None:
                        raise ValueError(f'Line {line_no}: missing {key}')
                if row['sample_id'] in ids:
                    raise ValueError(f'Duplicate sample_id: {row["sample_id"]}')
                ids.add(row['sample_id'])
                previous = scene_splits.setdefault(row['scene_id'], row['split'])
                if previous != row['split']:
                    raise ValueError(f'Scene occurs in multiple splits: {row["scene_id"]}')
                if not isinstance(row['instruction'], str) or not row['instruction'].strip():
                    raise ValueError(f'Line {line_no}: empty instruction')
                for key in ('annotation_height', 'annotation_width'):
                    if not isinstance(row[key], int) or row[key] <= 0:
                        raise ValueError(f'Line {line_no}: invalid {key}')
                if row['split'] != split:
                    continue
                paths = ['rgb_path', 'grasp_path']
                if segmentation:
                    paths.append(f'{mask_target}_mask_path')
                for key in paths:
                    if not row.get(key):
                        raise ValueError(f'{row["sample_id"]}: missing {key}')
                    path = Path(row[key])
                    row[key] = str(path if path.is_absolute() else manifest.parent / path)
                    if not Path(row[key]).is_file():
                        raise FileNotFoundError(row[key])
                self.samples.append(row)
        if not self.samples:
            raise ValueError(f'No samples for split {split}')
        self.grasp_files = [r['grasp_path'] for r in self.samples]
        self.length = len(self.samples)

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        row = self.samples[int(idx)]
        boxes = grasp.GraspRectangles.load_from_grasp_anything_file(
            row['grasp_path'], indices=row.get('grasp_indices'))
        if not boxes.grs:
            raise ValueError(f'{row["sample_id"]}: empty positive grasps')
        scale = np.array([self.output_size / row['annotation_height'],
                          self.output_size / row['annotation_width']])
        for box in boxes:
            box.points = box.points.astype(np.float64) * scale
        return boxes

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        row = self.samples[int(idx)]
        rgb = cv2.imread(row['rgb_path'], cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f'Cannot decode {row["rgb_path"]}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.output_size, self.output_size), interpolation=cv2.INTER_LINEAR)
        if not normalise:
            return rgb
        rgb = rgb.astype(np.float32) / 255.0
        if self.normalization == 'legacy':
            rgb -= rgb.mean()
        else:
            rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
                [0.229, 0.224, 0.225], dtype=np.float32)
        return rgb.transpose(2, 0, 1)

    def get_prompts(self, idx):
        return '', self.samples[idx]['instruction']

    def get_mask(self, idx):
        row = self.samples[idx]
        mask = np.load(row[f'{self.mask_target}_mask_path'], allow_pickle=False)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ValueError(f'{row["sample_id"]}: mask must be a finite 2D array')
        instance = row.get('mask_instance_id')
        if self.mask_target == 'object' and instance is not None:
            if not isinstance(instance, int) or instance <= 0:
                raise ValueError('mask_instance_id must be a positive integer')
            mask = mask == instance
        elif self.mask_target == 'union':
            mask = mask > 0
        elif not np.isin(mask, [0, 1]).all():
            raise ValueError(f'{row["sample_id"]}: target mask must be binary; '
                             'object instance maps require mask_instance_id')
        mask = cv2.resize(mask.astype(np.float32), (self.output_size, self.output_size),
                          interpolation=cv2.INTER_NEAREST)
        if not mask.any():
            raise ValueError(f'{row["sample_id"]}: empty target mask after resize')
        return self.numpy_to_torch(mask)

    def __getitem__(self, idx):
        # No geometric augmentation: spatial instructions retain their meaning.
        x, targets, _, _, _, _, query = super().__getitem__(idx)
        row = self.samples[idx]
        result = {
            'image': x, 'instruction': query,
            'grasp_targets': dict(zip(('pos', 'cos', 'sin', 'width'), targets)),
            'metadata': {'sample_id': row['sample_id'], 'scene_id': row['scene_id'],
                         'index': idx},
        }
        if self.segmentation:
            result['target_mask'] = self.get_mask(idx)
        return result


def language_collate(samples):
    batch = {
        'image': torch.stack([s['image'] for s in samples]),
        'instruction': [s['instruction'] for s in samples],
        'grasp_targets': {k: torch.stack([s['grasp_targets'][k] for s in samples])
                          for k in ('pos', 'cos', 'sin', 'width')},
        'metadata': [s['metadata'] for s in samples],
    }
    if any('target_mask' in s for s in samples):
        batch['target_mask'] = torch.stack([s['target_mask'] for s in samples])
    return batch
