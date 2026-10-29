# GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning

### Installation
Set up the docker environment, build the docker container for the first time:
```bash
cd GraspMamba/docker
sudo docker build --network host -t graspmamba .
```
Then 
```bash
cd GraspMamba/docker
sudo bash run_container.sh
```
If the code has a problem with triton version, please uninstall triton and install again:
```bash
pip uninstall triton
pip install triton
```

## Grasp-anything Datasets
The dataset can be accessed via [this link](https://airvlab.github.io/grasp-anything/docs/download/). 

### Training model 
Please modify the dataset path based on your setup and other configurations in the [run_training.sh](/run_training.sh) file. Then:
```bash
bash run_training.sh
```
### Evaluate model 

```bash
python3 evaluate_test.py --dataset grasp-anything --dataset-path </your dataset path> --seen <1 or 0>  --model-path </your model path> 
```

## Testing
For testing procedure, by comparing with other grasping methods, please follow the detailed instruction via [LGD](https://github.com/Fsoft-AIC/LGD). 

## Acknowledgement
Our codebase is developed based on [LGD](https://github.com/Fsoft-AIC/LGD) and [MambaVision](https://github.com/NVlabs/MambaVision). Please give us the motivation by citing our work:

```bibtex
@misc{nguyen2024graspmambamambabasedlanguagedrivengrasp,
      title={GraspMamba: A Mamba-based Language-driven Grasp Detection Framework with Hierarchical Feature Learning}, 
      author={Huy Hoang Nguyen and An Vuong and Anh Nguyen and Ian Reid and Minh Nhat Vu},
      year={2024},
      eprint={2409.14403},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.14403}, 
}

```

## Auxiliary target segmentation

The existing GraspMamba now supports an optional segmentation head and BCE + Dice
supervision. See [SEGMENTATION.md](SEGMENTATION.md) for the manifest schema,
training commands, inference API, tests, and validation limits.

## Prepared Grasp-Anything++ subset

`tools/prepare_grasp_anything_pp.py` validates the selected annotations and writes
`data/grasp-anything-pp-ready/manifest.jsonl`. It reserves official tests, excludes
training scenes shared with tests, and takes validation scenes from the remaining
training pool. Invalid annotation files/rows are recorded in the dataset report.
Files are referenced through relative paths rather than copied again.

```powershell
python tools/prepare_grasp_anything_pp.py --workers 4
python tools/check_grasp_anything_pp.py --manifest data/grasp-anything-pp-ready/manifest.jsonl
python train_network.py --dataset grasp-anything-language --dataset-path data/grasp-anything-pp-ready/manifest.jsonl --segmentation --mask-target part --grasp-loss smooth_l1 --input-size 224 --batch-size 8 --lr 0.001 --description pp_part_seg
python evaluate_manifest.py --manifest data/grasp-anything-pp-ready/manifest.jsonl --model-path CHECKPOINT --mask-target part
```

Use an environment with the model's PyTorch/CUDA/MambaVision dependencies for
training. Data preparation and contract checks can run on CPU. The prepared
subset is not a claim to reproduce the paper's complete dataset protocol.

## Colab / Kaggle without Docker

Use [NOTEBOOK_SETUP.md](NOTEBOOK_SETUP.md) and
[the starter notebook](notebooks/graspmamba_setup.ipynb). Install
`requirements-notebook.txt` after CUDA torch, then `requirements-cuda.txt` with
`--no-build-isolation --no-deps`. The required pretrained models are
`nvidia/MambaVision-L-1K` and `google/siglip-base-patch16-224`;
`tools/download_pretrained.py` downloads only their required snapshots.
Run the environment and GPU smoke checks before training. This profile has not
been validated end to end on hosted GPU in this workspace.

## Minimal local setup and run scripts

The full CUDA model uses Linux; on this Windows machine use an Ubuntu WSL2
distribution. Git Bash alone does not supply the required Linux CUDA extensions.
The existing model/trainer are unchanged by these setup/run scripts.

Inside Ubuntu WSL2, install Python 3.10-3.12 with venv support and provide a
compatible CUDA development toolkit if extension compilation is necessary.
`nvidia-smi` must see the Windows NVIDIA GPU from WSL. No Linux display driver
should be installed over the Windows-provided WSL GPU driver.

```bash
cd /mnt/d/Grasp_Mamba_TS
bash setup_env.sh
# First do a short run; then remove these overrides for the configured full run.
EPOCHS=1 BATCHES_PER_EPOCH=10 bash run.sh
```

The private venv defaults to `$HOME/.venvs/grasp-mamba`, avoiding the nearly-full
D: repo drive. Use the same `VENV_DIR` override for setup and run if changing it.
The setup installs the existing requirements and verifies the CUDA extension.
Pretrained models download automatically when the original model is constructed.
A working GPU environment still needs to be verified locally; 8 GB VRAM does not
guarantee every training configuration fits. The run script defaults to batch 1.
