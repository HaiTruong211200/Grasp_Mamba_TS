import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def prepare(repo, image_source, selected):
    base = repo / 'data/grasp-anything'
    plus = repo / 'data/grasp-anything-pp'
    sources = {}
    def add(src, dst):
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst in sources and sources[dst] != src:
            if sources[dst].read_bytes() != src.read_bytes():
                raise ValueError(f'Conflicting sources for {dst}')
        else:
            sources[dst] = src
    counts = {}
    for split in ('train_seen', 'test_seen', 'test_unseen'):
        folder = selected / split
        labels = sorted((folder / 'grasp_label_positive').glob('*.pt'))
        if not labels:
            raise ValueError(f'No labels: {folder}')
        counts[split] = len(labels)
        for label in labels:
            scene, obj, part = label.stem.rsplit('_', 2)
            image = next((p for p in (image_source / f'image_{split}' / f'{scene}.jpg', image_source / 'image' / f'{scene}.jpg') if p.is_file()), None)
            if image is None:
                raise FileNotFoundError(f'Image for {label.stem}')
            add(label, plus / 'grasp_label_positive' / label.name)
            for name, ext in (('grasp_instructions', '.pkl'), ('part_mask', '.npy')):
                add(folder / name / (label.stem + ext), plus / name / (label.stem + ext))
            add(folder / 'scene_description' / f'{scene}.pkl', base / 'scene_description' / f'{scene}.pkl')
            add(image, base / 'image' / image.name)
            # Preserve the existing prepared manifest's relative image paths too.
            add(image, base / f'image_{split}' / image.name)
        for mask in (folder / 'mask').glob('*.npy'):
            add(mask, base / 'mask' / mask.name)
    # Validate existing destinations before adding files. Never overwrite data.
    for dst, src in sources.items():
        if dst.exists() and not os.path.samefile(src, dst):
            if src.read_bytes() != dst.read_bytes():
                raise FileExistsError(f'Different file already exists: {dst}')
    created = 0
    for dst, src in sources.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.link(src, dst)
            created += 1
    report = {'source_split_label_counts': counts, 'created_hardlinks': created,
              'base': str(base), 'plus': str(plus),
              'files': {str(p.relative_to(repo)): sum(1 for _ in p.iterdir()) for p in [base/'image', base/'scene_description', base/'mask', plus/'grasp_label_positive', plus/'grasp_instructions', plus/'part_mask']},
              'note': 'Selected released annotations only; original loader chooses seen split and trainer splits 90/10. Hardlinks are regular files sharing storage; source files were preserved.'}
    (plus / 'baseline_layout_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arrange selected ++ annotations for the original grasp-anywhere loader using hardlinks on the same volume.')
    parser.add_argument('--image-source', type=Path, required=True, help='Folder containing image/ or image_train_seen/, image_test_seen/, image_test_unseen/')
    parser.add_argument('--selected', type=Path, default=REPO/'data/grasp-anything-pp-selected')
    args = parser.parse_args()
    prepare(REPO, args.image_source.resolve(), args.selected.resolve())
