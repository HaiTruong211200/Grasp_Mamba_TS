#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/grasp-mamba}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Environment missing. Run bash setup_env.sh inside Linux/Ubuntu WSL2 first." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"
export USE_TF=0
export USE_FLAX=0

# Same trainer/model as the repository; defaults use the prepared ++ part dataset.
# Start at batch size 1 for the local RTX 4060 Laptop (8 GB); measure actual VRAM.
exec python train_network.py \
    --network graspmamba \
    --dataset grasp-anything-language \
    --dataset-path "${DATASET_PATH:-data/grasp-anything-pp-ready/manifest.jsonl}" \
    --segmentation \
    --mask-target part \
    --grasp-loss smooth_l1 \
    --normalization imagenet \
    --input-size 224 \
    --batch-size "${BATCH_SIZE:-1}" \
    --num-workers "${NUM_WORKERS:-0}" \
    --epochs "${EPOCHS:-50}" \
    --batches-per-epoch "${BATCHES_PER_EPOCH:-1000}" \
    --lr "${LR:-0.001}" \
    --description "${DESCRIPTION:-pp_local}" \
    "$@"
