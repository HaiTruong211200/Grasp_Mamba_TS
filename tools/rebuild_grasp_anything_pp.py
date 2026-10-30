#!/usr/bin/env python3
"""Rebuild the original Grasp-Anything++ split subset from ZIP archives.

Writes a NEW output folder: per-source-split files, a combined loader layout,
reproducible train/val ID lists and reports. Does not modify existing datasets,
source ZIPs, split files, training code or model parameters.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
from extract_grasp_anything_pp_tests import (
    CORE, EXTENSIONS, copy_images, file_crc, read_ids, scan_zip, write_payload,
)

SPLITS = {'train_seen': 'train/seen.obj', 'test_seen': 'test/seen.obj',
          'test_unseen': 'test/unseen.obj'}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_ids(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(item + '\n' for item in values), encoding='utf-8')


def sample_fields(sample):
    scene, obj, part = sample.rsplit('_', 2)
    return scene, scene + '_' + obj, obj, part


def present_fields(sample, coverage):
    scene, object_id, _, _ = sample_fields(sample)
    return {
        'positive': sample in coverage['grasp_label_positive'],
        'instruction': sample in coverage['grasp_instructions'],
        'part_mask': sample in coverage['part_mask'],
        'rgb': scene in coverage['image'],
        'scene_description': scene in coverage['scene_description'],
        'object_mask': object_id in coverage['mask'],
    }


def build_indices(output, official, coverage, args):
    pools, ignored, rgb_mapping_errors = {}, {}, {}
    for split in SPLITS:
        raw = sorted(coverage[split]['grasp_label_positive'])
        # Mirror the baseline filename filter literally, without repairing its limitations.
        pools[split] = [sample for sample in raw if (sample + '.pt')[:-5] in official[split]]
        ignored[split] = sorted(set(raw) - set(pools[split]))
        rgb_mapping_errors[split] = [sample for sample in pools[split]
                                    if len(sample_fields(sample)[2]) != 1 or len(sample_fields(sample)[3]) != 1]
        write_ids(output / 'indices' / (split + '_raw_positive.txt'), raw)
        write_ids(output / 'indices' / (split + '_ignored_by_legacy_filter.txt'), ignored[split])
    train_pool = pools['train_seen']
    offset = int(len(train_pool) * args.ds_rotate)
    train_pool = train_pool[offset:] + train_pool[:offset]
    indices = list(range(len(train_pool)))
    numpy_version = None
    if args.ds_shuffle:
        try:
            import numpy as np
        except ImportError as error:
            raise RuntimeError('--ds-shuffle needs NumPy to reproduce the trainer shuffle exactly') from error
        np.random.RandomState(args.random_seed).shuffle(indices)
        numpy_version = np.__version__
    boundary = math.floor(args.train_fraction * len(train_pool))
    train = [train_pool[index] for index in indices[:boundary]]
    val = [train_pool[index] for index in indices[boundary:]]
    groups = {'train_seen': train_pool, 'train': train, 'val': val,
              'test_seen': pools['test_seen'], 'test_unseen': pools['test_unseen']}
    for name, samples in groups.items():
        write_ids(output / 'indices' / (name + '.txt'), samples)
    scene_sets = {name: {sample[:64] for sample in samples} for name, samples in groups.items()}
    overlap = {}
    names = ('train', 'val', 'test_seen', 'test_unseen')
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            overlap[first + '__' + second] = sorted(scene_sets[first] & scene_sets[second])
    return {
        'policy': 'baseline_sample_index_split', 'train_fraction': args.train_fraction,
        'ds_rotate': args.ds_rotate, 'ds_shuffle': args.ds_shuffle,
        'random_seed': args.random_seed, 'numpy_version_if_shuffled': numpy_version,
        'counts': {name: {'samples': len(samples), 'distinct_scenes': len(scene_sets[name])}
                   for name, samples in groups.items()},
        'ignored_by_legacy_id_filter': ignored,
        'legacy_rgb_filename_mapping_errors': rgb_mapping_errors,
        'scene_overlap_ids': overlap,
        'note': 'Audit lists only. The current trainer recomputes its own indices; use matching split/shuffle/rotation arguments.',
    }


def write_split_reports(output, official, coverage, inventory):
    summaries, missing_companions = {}, []
    for split in SPLITS:
        directory = output / 'selected' / split
        directory.mkdir(parents=True, exist_ok=True)
        positive = sorted(coverage[split]['grasp_label_positive'])
        objects = {sample_fields(sample)[1] for sample in positive}
        needed_scenes = {sample[:64] for sample in positive}
        missing_objects = sorted(official[split] - objects)
        write_ids(directory / 'requested_object_ids.txt', sorted(official[split]))
        write_ids(directory / 'missing_positive_object_ids.txt', missing_objects)
        write_ids(directory / 'missing_required_rgb_scene_ids.txt', sorted(needed_scenes - coverage[split]['image']))
        rows = []
        for sample in positive:
            scene, object_id, _, _ = sample_fields(sample)
            fields = present_fields(sample, coverage[split])
            required = ('positive', 'instruction', 'part_mask', 'rgb', 'scene_description')
            missing = [name for name in required if not fields[name]]
            if missing:
                missing_companions.append({'source_split': split, 'sample_id': sample, 'missing': missing})
            rows.append({'sample_id': sample, 'scene_id': scene, 'object_id': object_id,
                'positive_grasp_path': 'grasp_label_positive/' + sample + '.pt',
                'instruction_path': 'grasp_instructions/' + sample + '.pkl',
                'part_mask_path': 'part_mask/' + sample + '.npy',
                'rgb_path': 'image/' + scene + '.jpg',
                'scene_description_path': 'scene_description/' + scene + '.pkl',
                'object_mask_path': 'mask/' + object_id + '.npy' if fields['object_mask'] else '',
                'complete_required_files': not missing,
                'passes_legacy_id_filter': (sample + '.pt')[:-5] in official[split]})
        columns = ['sample_id', 'scene_id', 'object_id', 'positive_grasp_path', 'instruction_path',
                   'part_mask_path', 'rgb_path', 'scene_description_path', 'object_mask_path',
                   'complete_required_files', 'passes_legacy_id_filter']
        with (directory / 'annotation_inventory.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        provenance_columns = ['kind', 'sample_id', 'object_id', 'scene_id', 'archive', 'entry', 'output', 'bytes', 'crc32', 'action']
        with (directory / 'zip_inventory.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=provenance_columns)
            writer.writeheader()
            writer.writerows(inventory[split])
        summaries[split] = {
            'official_object_ids': len(official[split]),
            'official_distinct_scenes': len({item[:64] for item in official[split]}),
            'objects_with_positive_labels': len(objects),
            'missing_positive_object_ids_count': len(missing_objects),
            'positive_part_samples': len(positive), 'positive_distinct_scenes': len(needed_scenes),
            'rgb_found_for_positive_scenes': len(needed_scenes & coverage[split]['image']),
            'missing_rgb_for_positive_scenes': sorted(needed_scenes - coverage[split]['image']),
            'samples_missing_required_companions': sum(not row['complete_required_files'] for row in rows),
            'files_found_by_kind': {kind: len(value) for kind, value in coverage[split].items()},
        }
    with (output / 'missing_companions.jsonl').open('w', encoding='utf-8') as stream:
        for row in missing_companions:
            stream.write(json.dumps(row) + '\n')
    return summaries, missing_companions


def link_payload(source, destination, mode):
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        if source.stat().st_size != destination.stat().st_size or file_crc(source) != file_crc(destination):
            raise FileExistsError(f'Different existing output file: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'hardlink':
        try:
            os.link(source, destination)
        except OSError as error:
            raise OSError('Hardlink failed; rerun with --link-mode copy or choose a filesystem supporting hardlinks') from error
    else:
        write_payload(destination, lambda: source.open('rb'), source.stat().st_size, file_crc(source))


def build_loader_layout(output, coverage, mode, include_negative):
    for split in SPLITS:
        selected = output / 'selected' / split
        for sample in sorted(coverage[split]['grasp_label_positive']):
            scene, object_id, _, _ = sample_fields(sample)
            for kind in CORE:
                name = sample + EXTENSIONS[kind]
                link_payload(selected / kind / name, output / 'dataset/grasp-anything-pp' / kind / name, mode)
            for kind, name in [('image', scene + '.jpg'), ('scene_description', scene + '.pkl')]:
                link_payload(selected / kind / name, output / 'dataset/grasp-anything' / kind / name, mode)
            if object_id in coverage[split]['mask']:
                name = object_id + '.npy'
                link_payload(selected / 'mask' / name, output / 'dataset/grasp-anything/mask' / name, mode)
        if include_negative:
            for sample in sorted(coverage[split]['grasp_label_negative']):
                name = sample + '.pt'
                link_payload(selected / 'grasp_label_negative' / name,
                             output / 'dataset/grasp-anything-pp/grasp_label_negative' / name, mode)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pp-archives-dir', type=Path, required=True)
    parser.add_argument('--base-archives-dir', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, help='Extracted RGB root; alternative/fallback to image.zip')
    parser.add_argument('--split-root', type=Path, default=REPO / 'split/grasp-anything++')
    parser.add_argument('--output', type=Path, default=REPO / 'data/grasp-anything-pp-rebuilt')
    parser.add_argument('--link-mode', choices=['hardlink', 'copy'], default='hardlink')
    parser.add_argument('--train-fraction', type=float, default=.9)
    parser.add_argument('--ds-rotate', type=float, default=0.)
    parser.add_argument('--ds-shuffle', action='store_true')
    parser.add_argument('--random-seed', type=int, default=123)
    parser.add_argument('--include-negative', action='store_true')
    parser.add_argument('--scan-only', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1 or not math.isfinite(args.ds_rotate):
        parser.error('train-fraction must be between 0 and 1; ds-rotate must be finite')
    return args


def main():
    args = parse_args()
    output = args.output.resolve()
    split_paths = {name: args.split_root / relative for name, relative in SPLITS.items()}
    official = {name: read_ids(path) for name, path in split_paths.items()}
    pp_kinds = list(CORE) + (['grasp_label_negative'] if args.include_negative else [])
    required_archives = [args.pp_archives_dir / (kind + '.zip') for kind in pp_kinds]
    required_archives.append(args.base_archives_dir / 'scene_description.zip')
    for path in required_archives:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.images_dir is not None and not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    image_zip = args.base_archives_dir / 'image.zip'
    if not image_zip.is_file() and args.images_dir is None:
        raise FileNotFoundError('Provide --images-dir or put image.zip in --base-archives-dir')
    base_kinds = [kind for kind in ('mask', 'scene_description', 'image')
                  if (args.base_archives_dir / (kind + '.zip')).is_file()]
    source_archives = list(dict.fromkeys(required_archives + [args.base_archives_dir / (kind + '.zip') for kind in base_kinds]))
    archive_metadata = [{'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
                         'mtime_ns': path.stat().st_mtime_ns} for path in source_archives]
    split_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in split_paths.items()}
    selection = {'schema': 'grasp-anything-pp-rebuild-v1',
                 'object_ids': {name: sorted(ids) for name, ids in official.items()},
                 'split_sha256': split_hashes, 'archives': archive_metadata,
                 'images_dir': str(args.images_dir.resolve()) if args.images_dir else None}
    marker = output / '_selection.json'
    if marker.is_file():
        if json.loads(marker.read_text(encoding='utf-8')) != selection:
            raise FileExistsError('Output belongs to another source/split selection; choose a new --output folder')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty; choose a new --output folder')
    output.mkdir(parents=True, exist_ok=True)
    write_json(marker, selection)
    selected = output / 'selected'
    coverage = {split: {kind: set() for kind in EXTENSIONS} for split in SPLITS}
    inventory = {split: [] for split in SPLITS}
    scans = []
    def scan(kind, path, target_objects, allowed_samples=None):
        counts = scan_zip(path, kind, target_objects, selected, coverage, inventory,
                          args.scan_only, args.quiet, allowed_samples=allowed_samples)
        scans.append({'kind': kind, 'path': str(path.resolve()), **counts})
        write_json(output / 'scan_progress.json', {'completed_archives': scans, 'finished': False})
    # Positive labels are the anchor. All companion files are joined to those samples.
    scan('grasp_label_positive', args.pp_archives_dir / 'grasp_label_positive.zip', official)
    positives = {split: set(coverage[split]['grasp_label_positive']) for split in SPLITS}
    objects = {split: {sample_fields(sample)[1] for sample in samples} for split, samples in positives.items()}
    for kind in ('grasp_instructions', 'part_mask'):
        scan(kind, args.pp_archives_dir / (kind + '.zip'), objects, allowed_samples=positives)
    if args.include_negative:
        scan('grasp_label_negative', args.pp_archives_dir / 'grasp_label_negative.zip', official)
    for kind in base_kinds:
        scan(kind, args.base_archives_dir / (kind + '.zip'), objects)
    if args.images_dir:
        # Prefer the full raw image/ folder over prior per-split image copies.
        image_source = args.images_dir / 'image'
        if not image_source.is_dir():
            image_source = args.images_dir
        copy_images(image_source, objects, selected, coverage, inventory, args.scan_only)
    summaries, missing = write_split_reports(output, official, coverage, inventory)
    indices = build_indices(output, official, coverage, args)
    object_overlap = {}
    groups = list(SPLITS)
    for i, first in enumerate(groups):
        for second in groups[i + 1:]:
            object_overlap[first + '__' + second] = sorted(official[first] & official[second])
    blocking = []
    if missing:
        blocking.append(f'{len(missing)} positive samples lack required companion files')
    if any(not coverage[split]['grasp_label_positive'] for split in SPLITS):
        blocking.append('At least one source split has no positive samples')
    if any(not indices['counts'][name]['samples'] for name in ('train', 'val')):
        blocking.append('The chosen train/validation ratio produces an empty partition')
    if any(indices['legacy_rgb_filename_mapping_errors'].values()):
        blocking.append('Current baseline RGB filename mapping cannot read some object IDs; see report')
    report = {'schema': selection['schema'], 'scan_only': args.scan_only,
              'dataset_ready': False, 'blocking_issues': blocking,
              'source_split_sha256': split_hashes, 'source_archives': archive_metadata,
              'source_splits': summaries, 'baseline_indices': indices,
              'official_object_overlap_ids': object_overlap, 'archive_scans': scans,
              'checks': {'zip_crc_when_extracted': not args.scan_only,
                         'required_file_join': True, 'decoded_numeric_or_semantic_validation': False},
              'dataset_paths': {'dataset_path': str(output / 'dataset/grasp-anything'),
                                'add_file_path': str(output / 'dataset/grasp-anything-pp')}}
    write_json(output / 'report.json', report)
    write_json(output / 'scan_progress.json', {'completed_archives': scans, 'finished': True})
    print(json.dumps(summaries, indent=2), flush=True)
    print('Audit train/val counts:', indices['counts'], flush=True)
    if args.scan_only:
        print('Scan only: metadata written; no payloads or loader layout were produced.', flush=True)
        return 0
    if blocking:
        print('Dataset layout NOT published; fix the reported issues and rerun:', blocking, flush=True)
        return 2
    build_loader_layout(output, coverage, args.link_mode, args.include_negative)
    report['dataset_ready'] = True
    write_json(output / 'report.json', report)
    print('Dataset ready:', report['dataset_paths'], flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
