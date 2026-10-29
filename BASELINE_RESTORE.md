# Baseline with only path options and progress display

Baseline reference: `3b62f66`. Run with `run_training.v1.sh`.

The only allowed source extensions are:

1. `--vision-model` and `--text-model` select Hugging Face IDs or complete local
   pretrained directories; defaults remain the original two model IDs. SigLIP
   model and processor load from the same text directory.
2. The original script now appends `"$@"`, so existing `--dataset`,
   `--dataset-path` and `--add-file-path` options can override script defaults.
   Paths containing spaces must be quoted. Original script defaults stay intact.
3. A manual tqdm display tracks completed optimizer steps and current loss.
   It does not change loop bounds, batches consumed, LR ordering, RNG seeds,
   loss aggregation or optimizer behavior. The original loop performs
   batches_per_epoch - 1 optimizer updates for a nonempty loader; the progress
   total reflects that original behavior, rather than changing it.

```python
DATASET_ROOT = REPO / 'data/grasp-anything'
ANNOTATION_ROOT = REPO / 'data/grasp-anything-pp'
VISION_MODEL = 'nvidia/MambaVision-L-1K'
TEXT_MODEL = 'google/siglip-base-patch16-224'
!MPLBACKEND=Agg bash run_training.v1.sh --dataset grasp-anywhere --dataset-path "{DATASET_ROOT}" --add-file-path "{ANNOTATION_ROOT}" --vision-model "{VISION_MODEL}" --text-model "{TEXT_MODEL}"
```

`grasp-anywhere` is the original ++ loader. It retains original split filtering,
90/10 train/validation sampling, RGB normalization, rotation/zoom and object text
from scene_description. No manifest loader or segmentation is enabled.
The original Windows/single-digit filename limitations are deliberately retained.
Run this original loader on Linux with the expected dataset filenames.

Model forward/fusion/upsampling, text mode, original compute_loss, train loop,
warmup ordering, evaluation/IoU, post-processing and scheduler are unchanged.
Original CUDA assumptions and runtime/version issues (including original W&B
calls and scheduler arguments) remain. Changing pretrained paths is intended
for the same MambaVision-L and SigLIP architecture, not different variants.

The earlier versions are backed up in `.baseline_restore_backup/`. Dataset
files are unchanged. Verification compares behavior against original source
and does not certify a full pretrained CUDA training run on Colab/Kaggle.
Added segmentation/manifest files remain inactive and their extension tests
are not applicable to this original baseline API.

## Select object queries or grasp instructions

The explicitly requested text extension is available through `--text-source`:

- `object` (default): keep the original `(prompt, queries[obj_id])` from
  scene_description. The model receives the object query, as upstream.
- `instruction`: read `<scene_id>_<object_id>_<part_id>.pkl` from
  grasp_instructions and return `('', instruction)`. This mode does not read
  scene_description and does not concatenate scene text with instructions.
  Missing files or empty/non-string instructions raise an error; there is no
  silent fallback to object queries.

Both options use the original grasp-anywhere loader, labels, split, augmentation,
loss, optimizer/warmup and validation logic. The notebook exposes TEXT_SOURCE.
`evaluate_test.py` accepts the same switch for its dataset. Keep the selected
source identical during training and evaluation. This adds source selection
only; unrelated original evaluation/runtime limitations remain.

```bash
bash run_training.v1.sh --dataset grasp-anywhere --dataset-path data/grasp-anything --add-file-path data/grasp-anything-pp --text-source object --description baseline_object
bash run_training.v1.sh --dataset grasp-anywhere --dataset-path data/grasp-anything --add-file-path data/grasp-anything-pp --text-source instruction --description baseline_instruction
```
