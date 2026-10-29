#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Full GraspMamba requires Linux/WSL2 + NVIDIA CUDA; run this inside Ubuntu WSL2." >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/grasp-mamba}"
export MAX_JOBS="${MAX_JOBS:-2}"
export USE_TF=0
export USE_FLAX=0

"$PYTHON_BIN" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10-3.12 for this profile"'
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
# Keep an existing matched GPU torch/torchvision install in this private venv.
if ! python -c 'import torch, torchvision; assert torch.cuda.is_available()' >/dev/null 2>&1; then
    python -m pip install --no-cache-dir -r requirements-torch-cu124.txt
fi
python -m pip install --no-cache-dir -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation --no-deps -r requirements-cuda.txt
python tools/check_notebook_environment.py
printf '\nEnvironment ready: %s\nRun: bash run.sh\n' "$VENV_DIR"
