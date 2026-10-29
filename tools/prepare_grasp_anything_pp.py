"""Validate extracted Grasp-Anything++ and build a portable manifest for this repo."""
import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import hashlib
import io
import json
from pathlib import Path
import pickle
import random
import sys
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.dataset_processing.grasp import GraspRectangles, _grasp_anything_format


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Instruction/split pickle must contain plain data')


def read_plain(path):
    return PlainUnpickler(io.BytesIO(path.read_bytes())).load()


def assign_validation_scenes(scenes, fraction, seed):
    scenes = set(scenes)
    if not 0 < fraction < 1 or len(scenes) < 2:
        raise ValueError('Need at least two scenes and 0 < validation fraction < 1')
    ordered = sorted(scenes, key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    count = max(1, min(len(ordered) - 1, int(round(len(ordered) * fraction))))
    return set(ordered[:count])


def validate_grasps(values, height, width):
    if not isinstance(values, torch.Tensor) or values.ndim != 2 or values.shape[1] != 6:
        raise ValueError('grasp_shape_not_Nx6_tensor')
    a = values.detach().cpu().numpy()
    if a.shape[0] == 0:
        raise ValueError('empty_grasps')
    valid = (np.isfinite(a).all(1) & (a[:, 1] >= 0) & (a[:, 1] < width)
             & (a[:, 2] >= 0) & (a[:, 2] < height)
             & (a[:, 3] >= 2) & (a[:, 4] >= 2)
             & (a[:, 3] <= 2 * width) & (a[:, 4] <= 2 * height)
             & (np.abs(a[:, 5]) <= 360))
    indices = np.flatnonzero(valid).tolist()
    if not indices:
        if np.isfinite(a).all() and (np.abs(a[:, 1:]) <= 1).all():
            raise ValueError('unverified_subpixel_grasp_units')
        raise ValueError('no_valid_pixel_grasps')
    return a, indices


def check_mask(path, shape, output_size):
    a = np.load(path, allow_pickle=False)
    if a.ndim != 2 or a.shape != shape:
        raise ValueError('mask_shape_mismatch')
    if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError('mask_not_binary')
    if not a.any():
        raise ValueError('empty_mask')
    small = cv2.resize(a.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST)
    if not small.any():
        raise ValueError('mask_vanished_at_output_resolution')
    return a


def portable(path, output):
    import os
    return Path(os.path.relpath(path, output)).as_posix()


def make_preview(rows, output, size=256):
    chosen = []
    for split in ('train', 'val', 'test_seen', 'test_unseen'):
        candidates = [r for r in rows if r['split'] == split]
        for r in candidates[:5]:
            chosen.append(r)
    width, height = size + 24, size + 104
    sheet = Image.new('RGB', (5 * width, 4 * height), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(chosen):
        rgb = Image.open(output / row['rgb_path']).convert('RGB').resize((size, size))
        mask = np.load(output / row['part_mask_path'], allow_pickle=False)
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)
        array = np.asarray(rgb).copy()
        array[mask] = (0.6 * array[mask] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        tile = Image.fromarray(array)
        td = ImageDraw.Draw(tile)
        raw = torch.load(output / row['grasp_path'], map_location='cpu', weights_only=True)
        for index in row['grasp_indices'][:5]:
            box = _grasp_anything_format(raw[index].tolist())
            xy = [(float(p[1] * size / row['annotation_width']),
                   float(p[0] * size / row['annotation_height'])) for p in box.points]
            td.line(xy + [xy[0]], fill='red', width=2)
        x, y = (i % 5) * width + 12, (i // 5) * height + 12
        sheet.paste(tile, (x, y))
        label = row['split'] + '\n' + '\n'.join(textwrap.wrap(row['instruction'], 40)[:4])
        draw.multiline_text((x, y + size + 5), label, fill='black', spacing=3)
    sheet.save(output / 'preview_20_samples.jpg', quality=90)


def build(source, output, split_root, image_root, val_fraction=0.1, seed=123, output_size=224, workers=4):
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite dataset: {output}')
    source_groups = ('train_seen', 'test_seen', 'test_unseen')
    split_files = {'train_seen': 'train/seen.obj', 'test_seen': 'test/seen.obj', 'test_unseen': 'test/unseen.obj'}
    official = {g: set(read_plain(split_root / split_files[g])) for g in source_groups}
    official_scenes = {g: {v.rsplit('_', 1)[0] for v in ids} for g, ids in official.items()}
    if official_scenes['test_seen'] & official_scenes['test_unseen']:
        raise ValueError('Official test seen/unseen share scenes; resolve before building manifest')
    test_scenes = official_scenes['test_seen'] | official_scenes['test_unseen']
    removed_train_scenes = official_scenes['train_seen'] & test_scenes
    exclusions, candidates, image_sizes = [], [], {}
    object_mask_cache, optional_mask_problems = {}, Counter()
    source_counts, invalid_rows_removed = Counter(), 0
    stats_lock = threading.Lock()
    def check_one(group, entry):
        nonlocal invalid_rows_removed
        sample = entry['sample_id']
        try:
            scene, object_string, part_string = sample.rsplit('_', 2)
            object_key = scene + '_' + object_string
            if object_key not in official[group]:
                raise ValueError('sample_not_in_official_split')
            if group == 'train_seen' and scene in test_scenes:
                raise ValueError('train_scene_reserved_for_test')
            paths = {kind: source / group / folder / (sample + suffix)
                     for kind, folder, suffix in (
                         ('grasp', 'grasp_label_positive', '.pt'),
                         ('instruction', 'grasp_instructions', '.pkl'),
                         ('part_mask', 'part_mask', '.npy'))}
            rgb = image_root / ('image_' + group) / (scene + '.jpg')
            if any(not p.is_file() for p in (*paths.values(), rgb)):
                raise ValueError('missing_required_file')
            instruction = read_plain(paths['instruction'])
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError('instruction_not_nonempty_string')
            if scene not in image_sizes:
                with Image.open(rgb) as img:
                    img.load()
                    if img.mode != 'RGB':
                        raise ValueError('image_not_rgb')
                    image_sizes[scene] = (img.height, img.width)
            shape = image_sizes[scene]
            # The existing GA parser uses pixel coordinates, degrees and a 416 canvas.
            if shape != (416, 416):
                raise ValueError('unverified_annotation_canvas')
            check_mask(paths['part_mask'], shape, output_size)
            raw = torch.load(paths['grasp'], map_location='cpu', weights_only=True)
            array, indices = validate_grasps(raw, *shape)
            boxes = GraspRectangles([_grasp_anything_format(array[i].tolist()) for i in indices])
            boxes.scale(output_size / 416.0)
            quality, _, _ = boxes.draw((output_size, output_size))
            if not quality.any():
                raise ValueError('empty_rasterized_grasp_target')
            with stats_lock:
                invalid_rows_removed += len(array) - len(indices)
            object_mask = source / group / 'mask' / (object_key + '.npy')
            cache_key = (group, object_key)
            if cache_key not in object_mask_cache:
                try:
                    check_mask(object_mask, shape, output_size)
                    object_mask_cache[cache_key] = object_mask
                except (ValueError, OSError) as exc:
                    with stats_lock:
                        if cache_key not in object_mask_cache:
                            optional_mask_problems[str(exc) if isinstance(exc, ValueError) else 'missing_object_mask'] += 1
                        object_mask_cache[cache_key] = None
            object_mask = object_mask_cache[cache_key]
            scene_description = source / group / 'scene_description' / (scene + '.pkl')
            candidates.append({
                'sample_id': sample, 'scene_id': scene, 'object_id': int(object_string),
                'part_id': int(part_string), 'instruction_id': sample,
                'instruction': instruction.strip(), 'rgb_path': portable(rgb, output),
                'grasp_path': portable(paths['grasp'], output),
                'grasp_indices': indices, 'grasp_format': 'score_x_y_w_h_theta_pixels_degrees',
                'part_mask_path': portable(paths['part_mask'], output),
                'object_mask_path': portable(object_mask, output) if object_mask else None,
                'scene_description_path': portable(scene_description, output) if scene_description.is_file() else None,
                'annotation_height': 416, 'annotation_width': 416,
                'source_split': group, 'seen_status': 'unseen' if group == 'test_unseen' else 'seen',
            })
        except (ValueError, OSError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
            exclusions.append({'sample_id': sample, 'source_split': group,
                               'reason': str(exc), 'exception': type(exc).__name__})

    if workers < 1:
        raise ValueError('workers must be positive')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for group in source_groups:
            with (source / group / 'annotation_inventory.csv').open(encoding='utf-8-sig', newline='') as stream:
                inventory = list(csv.DictReader(stream))
            source_counts[group] = len(inventory)
            entries = iter(inventory)
            pending = deque()
            for _ in range(workers * 2):
                entry = next(entries, None)
                if entry is None:
                    break
                pending.append(pool.submit(check_one, group, entry))
            number = 0
            while pending:
                pending.popleft().result()
                number += 1
                entry = next(entries, None)
                if entry is not None:
                    pending.append(pool.submit(check_one, group, entry))
                if number % 500 == 0 or number == len(inventory):
                    print(f'{group}: checked {number}/{len(inventory)}, accepted so far={len(candidates)}, excluded={len(exclusions)}', flush=True)
    train_scenes = {r['scene_id'] for r in candidates if r['source_split'] == 'train_seen'}
    val_scenes = assign_validation_scenes(train_scenes, val_fraction, seed)
    scene_assignments, sample_ids = {}, set()
    for row in candidates:
        row['split'] = ('val' if row['scene_id'] in val_scenes else 'train') if row['source_split'] == 'train_seen' else row['source_split']
        prior = scene_assignments.setdefault(row['scene_id'], row['split'])
        if prior != row['split'] or row['sample_id'] in sample_ids:
            raise ValueError('Internal scene or sample leakage')
        sample_ids.add(row['sample_id'])
    candidates.sort(key=lambda r: (r['split'], r['sample_id']))
    exclusions.sort(key=lambda r: (r['source_split'], r['sample_id']))
    counts = {split: {'samples': sum(r['split'] == split for r in candidates),
                      'scenes': len({r['scene_id'] for r in candidates if r['split'] == split})}
              for split in ('train', 'val', 'test_seen', 'test_unseen')}
    if not all(c['samples'] for c in counts.values()):
        raise ValueError('One of the required splits has no valid samples')
    output.mkdir(parents=True)
    manifest = output / 'manifest.jsonl'
    with manifest.open('w', encoding='utf-8', newline='\n') as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (output / 'excluded_samples.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
        for row in exclusions:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {
        'schema': 'graspmamba-language-v1', 'mask_target': 'part', 'seed': seed,
        'validation_scene_fraction': val_fraction, 'checked_output_size': output_size,
        'source_inventory_samples': dict(source_counts), 'splits': counts,
        'excluded_samples': len(exclusions), 'exclusion_reasons': dict(Counter(r['reason'] for r in exclusions)),
        'removed_train_scenes_shared_with_test': sorted(removed_train_scenes),
        'invalid_grasp_rows_removed_from_retained_samples': invalid_rows_removed,
        'optional_object_mask_problems': dict(optional_mask_problems),
        'minimum_grasp_side_pixels': 2, 'validation_workers': workers,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'source_split_sha256': {g: hashlib.sha256((split_root / f).read_bytes()).hexdigest() for g, f in split_files.items()},
        'scene_overlap_between_output_splits': 0,
        'unannotated_source_objects': {g: len((source / g / 'missing_positive_object_ids.txt').read_text().splitlines()) for g in source_groups},
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    make_preview(candidates, output)
    (output / 'README.md').write_text("""# Grasp-Anything++ prepared dataset

The manifest references existing selected images/annotations; it does not duplicate them.
Keep this directory beside grasp-anything and grasp-anything-pp-selected when moving data.
Use mask-target part. Object masks are optional and null when missing/invalid.
Validation uses 10% of eligible training scenes (seed 123 by default).
Official test lists are retained; training scenes shared with test are excluded.
Pixel/degrees grasp format follows the repository parser. Invalid/subpixel annotations
are rejected, not guessed to be normalized. Grasp sides below 2 source pixels are rejected.
Only validated grasp_indices are rasterized; original annotations are unchanged.
See report.json and excluded_samples.jsonl for coverage and cleaning details.
The sample preview is a diagnostic, not proof of correct semantic labels for all records.

From the repository root:

```powershell
python train_network.py --dataset grasp-anything-language --dataset-path data/grasp-anything-pp-ready/manifest.jsonl --segmentation --mask-target part --grasp-loss smooth_l1 --input-size 224 --batch-size 8 --lr 0.001 --description pp_part_seg
python evaluate_manifest.py --manifest data/grasp-anything-pp-ready/manifest.jsonl --model-path CHECKPOINT --mask-target part
python tools/check_grasp_anything_pp.py --manifest data/grasp-anything-pp-ready/manifest.jsonl
```

This prepares the released subset; it does not certify the paper's complete data protocol.
""", encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO / 'data/grasp-anything-pp-selected')
    parser.add_argument('--output', type=Path, default=REPO / 'data/grasp-anything-pp-ready')
    parser.add_argument('--split-root', type=Path, default=REPO / 'split/grasp-anything++')
    parser.add_argument('--image-root', type=Path, default=REPO / 'data/grasp-anything')
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output-size', type=int, default=224)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    build(args.source.resolve(), args.output.resolve(), args.split_root.resolve(), args.image_root.resolve(), args.val_fraction, args.seed, args.output_size, args.workers)


if __name__ == '__main__':
    main()
