"""Download only the two pretrained snapshots used by the current GraspMamba."""
import argparse
import json
from pathlib import Path

MODELS = ('nvidia/MambaVision-L-1K', 'google/siglip-base-patch16-224')
PATTERNS = ['*.json', '*.py', '*.safetensors', '*.model', 'LICENSE*', 'README.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='Print plan, without downloads or optional imports')
    parser.add_argument('--cache-dir', type=Path, help='HF hub cache folder; set HF_HOME globally to reuse from model code')
    parser.add_argument('--output', type=Path, default=Path('pretrained_models.json'))
    parser.add_argument('--max-workers', type=int, default=4)
    args = parser.parse_args()
    if args.list:
        print(json.dumps({'models': MODELS, 'allow_patterns': PATTERNS}, indent=2))
        return
    from huggingface_hub import snapshot_download
    snapshots = []
    for model_id in MODELS:
        snapshot = Path(snapshot_download(model_id, allow_patterns=PATTERNS,
                        cache_dir=str(args.cache_dir) if args.cache_dir else None,
                        max_workers=args.max_workers))
        if not (snapshot / 'model.safetensors').is_file():
            raise FileNotFoundError(f'Missing model.safetensors in {snapshot}')
        snapshots.append({'model_id': model_id, 'snapshot_path': str(snapshot), 'revision': snapshot.name,
                          'weights_bytes': (snapshot / 'model.safetensors').stat().st_size})
        print(f'Cached {model_id} at {snapshot}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshots, indent=2), encoding='utf-8')
    print(f'Model inventory: {args.output}')


if __name__ == '__main__':
    main()
