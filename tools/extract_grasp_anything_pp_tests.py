#!/usr/bin/env python3
"""Extract official Grasp-Anything++ test IDs into separate seen/unseen folders.

Only matching files are extracted. Existing dataset folders and source archives
are never modified. Opens one ZIP at a time; large ZIP indexes may require RAM.
"""
import argparse
import csv
import gc
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import re
import shutil
import tempfile
import time
import zipfile
import zlib

REPO = Path(__file__).resolve().parents[1]
OBJECT = re.compile(r'^[0-9a-f]{64}_\d+$')
PART = re.compile(r'^([0-9a-f]{64}_\d+)_\d+$')
SCENE = re.compile(r'^[0-9a-f]{64}$')
CORE = ('grasp_label_positive', 'grasp_instructions', 'part_mask')
EXTENSIONS = {'grasp_label_positive': '.pt', 'grasp_label_negative': '.pt',
              'grasp_instructions': '.pkl', 'part_mask': '.npy',
              'mask': '.npy', 'scene_description': '.pkl', 'image': '.jpg'}


class PlainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError('Split file must contain plain IDs, not Python objects')


def read_ids(path):
    value = PlainUnpickler(io.BytesIO(path.read_bytes())).load()
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(f'Expected a list of object IDs: {path}')
    if any(not isinstance(item, str) or not OBJECT.fullmatch(item) for item in value):
        raise ValueError(f'Invalid scene_object ID in {path}')
    return set(value)


def file_crc(path):
    checksum = 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xffffffff


def write_payload(destination, source_opener, expected_size, expected_crc):
    """Atomic copy; an existing different file is never overwritten."""
    if destination.exists():
        if destination.stat().st_size != expected_size or file_crc(destination) != expected_crc:
            raise FileExistsError(f'Output differs from source; use a new output folder: {destination}')
        return 'existing_verified'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + '.',
                                         suffix='.partial', delete=False) as output:
            temporary = Path(output.name)
            with source_opener() as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        # ZipExtFile also checks the source CRC when fully read.
        if temporary.stat().st_size != expected_size or file_crc(temporary) != expected_crc:
            raise ValueError(f'Incomplete or corrupt source data: {destination.name}')
        temporary.replace(destination)
        return 'extracted'
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def entry_identity(kind, name):
    path = PurePosixPath(name.replace('\\', '/'))
    if path.suffix.lower() != EXTENSIONS[kind]:
        return None
    stem = path.stem
    if kind in CORE or kind == 'grasp_label_negative':
        match = PART.fullmatch(stem)
        if match:
            return {'sample': stem, 'object': match[1], 'scene': stem[:64], 'name': path.name}
    elif kind == 'mask' and OBJECT.fullmatch(stem):
        return {'sample': None, 'object': stem, 'scene': stem[:64], 'name': path.name}
    elif kind in ('image', 'scene_description') and SCENE.fullmatch(stem):
        return {'sample': None, 'object': None, 'scene': stem, 'name': path.name}
    return None


def scan_zip(path, kind, selected, output, coverage, inventory, scan_only, quiet, allowed_samples=None):
    print(f'Opening {path} (one archive at a time)...', flush=True)
    start = time.perf_counter()
    counts = {'matched_entries': 0, 'extracted': 0, 'existing_verified': 0}
    selected_scenes = {split: {item[:64] for item in ids} for split, ids in selected.items()}
    archive = zipfile.ZipFile(path)
    try:
        entries = archive.infolist()
        print(f'  {len(entries):,} ZIP entries; scanning {kind}', flush=True)
        iterator = entries
        if not quiet:
            try:
                from tqdm import tqdm
                iterator = tqdm(entries, desc=kind, unit='entry', mininterval=1)
            except ImportError:
                pass
        for number, info in enumerate(iterator, 1):
            if info.is_dir():
                continue
            identity = entry_identity(kind, info.filename)
            if identity is None:
                continue
            matches = [split for split, ids in selected.items()
                       if (identity['object'] in ids if identity['object'] is not None
                           else identity['scene'] in selected_scenes[split])]
            if allowed_samples is not None and identity['sample'] is not None:
                matches = [split for split in matches if identity['sample'] in allowed_samples[split]]
            if not matches:
                continue
            counts['matched_entries'] += 1
            for split in matches:
                destination = output / split / kind / identity['name']
                action = 'scan_only'
                if not scan_only:
                    action = write_payload(destination, lambda info=info: archive.open(info),
                                           info.file_size, info.CRC)
                    counts[action] += 1
                coverage[split][kind].add(identity['sample'] or identity['object'] or identity['scene'])
                inventory[split].append({'kind': kind, 'sample_id': identity['sample'] or '',
                    'object_id': identity['object'] or '', 'scene_id': identity['scene'],
                    'archive': str(path), 'entry': info.filename,
                    'output': str(destination.relative_to(output)), 'bytes': info.file_size,
                    'crc32': f'{info.CRC:08x}', 'action': action})
    finally:
        archive.close()
        # Release the full multi-million-entry index before opening another ZIP.
        del archive
        if 'entries' in locals():
            del entries
        if 'iterator' in locals():
            del iterator
        gc.collect()
    counts['elapsed_seconds'] = round(time.perf_counter() - start, 3)
    print(f'  Done: {counts}', flush=True)
    return counts


def copy_images(image_root, selected, output, coverage, inventory, scan_only):
    crc_cache = {}
    for split, ids in selected.items():
        for scene in sorted({item[:64] for item in ids}):
            if scene in coverage[split]['image']:
                continue
            candidates = [image_root / f'image_{split}' / (scene + '.jpg'),
                          image_root / 'image' / (scene + '.jpg'),
                          image_root / split / 'image' / (scene + '.jpg'),
                          image_root / (scene + '.jpg')]
            source = next((path for path in candidates if path.is_file()), None)
            if source is None:
                continue
            destination = output / split / 'image' / source.name
            action = 'scan_only'
            size = source.stat().st_size
            checksum = ''
            if not scan_only:
                if source not in crc_cache:
                    crc_cache[source] = file_crc(source)
                checksum = crc_cache[source]
                action = write_payload(destination, lambda source=source: source.open('rb'), size, checksum)
            coverage[split]['image'].add(scene)
            inventory[split].append({'kind': 'image', 'sample_id': '', 'object_id': '',
                'scene_id': scene, 'archive': '', 'entry': str(source),
                'output': str(destination.relative_to(output)), 'bytes': size,
                'crc32': f'{checksum:08x}' if checksum != '' else '', 'action': action})


def make_report(output, official, selected, coverage, inventory, archives, scan_only):
    report = {'scan_only': scan_only, 'archives': archives, 'splits': {}}
    for split, ids in selected.items():
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        scenes = {item[:64] for item in ids}
        positive = coverage[split]['grasp_label_positive']
        found_objects = {item.rsplit('_', 1)[0] for item in positive}
        aligned = positive & coverage[split]['grasp_instructions'] & coverage[split]['part_mask']
        available = {sample for sample in aligned if sample[:64] in coverage[split]['image']}
        entry = {'official_object_ids': len(official[split]), 'requested_object_ids': len(ids),
                 'positive_object_ids_found': len(found_objects),
                 'missing_positive_object_ids': sorted(ids - found_objects),
                 'positive_part_samples': len(positive), 'aligned_part_samples': len(aligned),
                 'available_samples_with_rgb': len(available),
                 'requested_scenes': len(scenes), 'images_found': len(coverage[split]['image']),
                 'missing_image_scene_ids': sorted(scenes - coverage[split]['image']),
                 'positive_samples_missing_instruction': sorted(positive - coverage[split]['grasp_instructions']),
                 'positive_samples_missing_part_mask': sorted(positive - coverage[split]['part_mask']),
                 'files_found_by_kind': {kind: len(values) for kind, values in coverage[split].items()}}
        report['splits'][split] = entry
        for field in ('missing_positive_object_ids', 'missing_image_scene_ids'):
            (directory / (field + '.txt')).write_text(''.join(item + '\n' for item in entry[field]), encoding='utf-8')
        with (directory / 'inventory.csv').open('w', encoding='utf-8', newline='') as stream:
            fields = ['kind', 'sample_id', 'object_id', 'scene_id', 'archive', 'entry', 'output', 'bytes', 'crc32', 'action']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(inventory[split])
        print(f'{split}: {len(found_objects)}/{len(ids)} object IDs have positive labels; '
              f'{len(ids - found_objects)} missing; {len(positive)} positive part samples; '
              f'{len(coverage[split]["image"])}/{len(scenes)} images', flush=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pp-archives-dir', type=Path, required=True,
                        help='Directory containing Grasp-Anything++ ZIPs')
    parser.add_argument('--base-archives-dir', type=Path,
                        help='Optional original Grasp-Anything mask/scene_description/image ZIPs')
    parser.add_argument('--images-dir', type=Path,
                        help='Optional extracted RGB root, containing image/ or image_test_seen/ etc.')
    parser.add_argument('--split-root', type=Path, default=REPO / 'split/grasp-anything++/test')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--only-missing', action='store_true',
                        help='Search only official objects without existing ++ positive labels')
    parser.add_argument('--existing-labels', type=Path, default=REPO / 'data/grasp-anything-pp/grasp_label_positive')
    parser.add_argument('--include-negative', action='store_true')
    parser.add_argument('--scan-only', action='store_true', help='Write reports but do not extract payload files')
    parser.add_argument('--quiet', action='store_true', help='Disable tqdm progress bars')
    return parser.parse_args()


def main():
    args = parse_args()
    for optional_dir in (args.base_archives_dir, args.images_dir):
        if optional_dir is not None and not optional_dir.is_dir():
            raise FileNotFoundError(optional_dir)
    official = {f'test_{name}': read_ids(args.split_root / (name + '.obj')) for name in ('seen', 'unseen')}
    selected = {split: set(ids) for split, ids in official.items()}
    if args.only_missing:
        if not args.existing_labels.is_dir():
            raise FileNotFoundError(args.existing_labels)
        present = set()
        for path in args.existing_labels.glob('*.pt'):
            match = PART.fullmatch(path.stem)
            if match:
                present.add(match[1])
        selected = {split: ids - present for split, ids in selected.items()}
    output = (args.output or REPO / 'data' / ('grasp-anything-pp-tests-missing' if args.only_missing else 'grasp-anything-pp-tests-from-zips')).resolve()
    kinds = list(CORE) + (['grasp_label_negative'] if args.include_negative else [])
    plan = [(args.pp_archives_dir / (kind + '.zip'), kind) for kind in kinds]
    for path, kind in plan:
        if not path.is_file():
            raise FileNotFoundError(f'Missing Grasp-Anything++ archive: {path}')
    if args.base_archives_dir:
        for kind in ('mask', 'scene_description', 'image'):
            path = args.base_archives_dir / (kind + '.zip')
            if path.is_file():
                plan.append((path, kind))
    selection = {split: sorted(ids) for split, ids in selected.items()}
    marker = output / '_selection.json'
    if marker.is_file():
        if json.loads(marker.read_text(encoding='utf-8')) != selection:
            raise FileExistsError('Output belongs to another ID selection; choose a new --output folder')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Output is not empty and has no selection marker; choose a new --output folder')
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(selection, indent=2) + '\n', encoding='utf-8')
    coverage = {split: {kind: set() for kind in EXTENSIONS} for split in selected}
    inventory = {split: [] for split in selected}
    print('Requested objects:', {split: len(ids) for split, ids in selected.items()}, flush=True)
    print('Output:', output, flush=True)
    archives = []
    if not any(selected.values()):
        make_report(output, official, selected, coverage, inventory, archives, args.scan_only)
        print('No missing object IDs to search.', flush=True)
        return
    for path, kind in plan:
        counts = scan_zip(path, kind, selected, output, coverage, inventory, args.scan_only, args.quiet)
        archives.append({'path': str(path), 'kind': kind, **counts})
    if args.images_dir:
        copy_images(args.images_dir, selected, output, coverage, inventory, args.scan_only)
    make_report(output, official, selected, coverage, inventory, archives, args.scan_only)
    print(f'Report: {output / "report.json"}', flush=True)

if __name__ == '__main__':
    main()
