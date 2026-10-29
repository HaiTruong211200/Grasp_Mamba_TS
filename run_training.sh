#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Original Grasp-Anything++ loader baseline; dataset paths are local to this repo.
# Edit defaults here, set environment variables, or pass CLI flags from a notebook.
# Explicit flags in "$@" are appended last and override the defaults below.
NETWORK="${NETWORK:-graspmamba}"
INPUT_SIZE="${INPUT_SIZE:-224}"
USE_DEPTH="${USE_DEPTH:-0}"
USE_RGB="${USE_RGB:-1}"
USE_DROPOUT="${USE_DROPOUT:-1}"
DROPOUT_PROB="${DROPOUT_PROB:-0.1}"
CHANNEL_SIZE="${CHANNEL_SIZE:-32}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.25}"
DATASET="${DATASET:-grasp-anywhere}"
DATASET_PATH="${DATASET_PATH:-data/grasp-anything}"
SPLIT="${SPLIT:-0.9}"
DS_SHUFFLE="${DS_SHUFFLE:-0}"
DS_ROTATE="${DS_ROTATE:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
BATCHES_PER_EPOCH="${BATCHES_PER_EPOCH:-1000}"
OPTIM="${OPTIM:-adam}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
DESCRIPTION="${DESCRIPTION:-training_grasp_mamba}"
LOGDIR="${LOGDIR:-logs/}"
VIS="${VIS:-0}"
FORCE_CPU="${FORCE_CPU:-0}"
RANDOM_SEED="${RANDOM_SEED:-123}"
SEEN="${SEEN:-1}"
ADD_FILE_PATH="${ADD_FILE_PATH:-data/grasp-anything-pp}"
GRASP_LOSS="${GRASP_LOSS:-mse}"
NORMALIZATION="${NORMALIZATION:-legacy}"
MASK_TARGET="${MASK_TARGET:-part}"
export USE_TF="${USE_TF:-0}"
export USE_FLAX="${USE_FLAX:-0}"

# Array preserves arguments containing spaces (dataset/log paths, descriptions).
options=(
    --network "$NETWORK"
    --input-size "$INPUT_SIZE"
    --use-depth "$USE_DEPTH"
    --use-rgb "$USE_RGB"
    --use-dropout "$USE_DROPOUT"
    --dropout-prob "$DROPOUT_PROB"
    --channel-size "$CHANNEL_SIZE"
    --iou-threshold "$IOU_THRESHOLD"
    --dataset "$DATASET"
    --dataset-path "$DATASET_PATH"
    --split "$SPLIT"
    --ds-rotate "$DS_ROTATE"
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
    --epochs "$EPOCHS"
    --batches-per-epoch "$BATCHES_PER_EPOCH"
    --optim "$OPTIM"
    --lr "$LEARNING_RATE"
    --description "$DESCRIPTION"
    --logdir "$LOGDIR"
    --random-seed "$RANDOM_SEED"
    --seen "$SEEN"
    --add-file-path "$ADD_FILE_PATH"
    --grasp-loss "$GRASP_LOSS"
    --normalization "$NORMALIZATION"
    --mask-target "$MASK_TARGET"
)
[[ "$DS_SHUFFLE" == "1" ]] && options+=(--ds-shuffle)
[[ "$VIS" == "1" ]] && options+=(--vis)
[[ "$FORCE_CPU" == "1" ]] && options+=(--cpu)

# Add --segmentation for the auxiliary branch; omit it for the baseline.
exec "${PYTHON_BIN:-python3}" "$ROOT/train_network.py" "${options[@]}" "$@"
