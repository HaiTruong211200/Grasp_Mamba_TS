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
calls) remain. Changing pretrained paths is intended
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

## PyTorch scheduler compatibility

Removed only `verbose=True` from ReduceLROnPlateau because the current runtime
rejects that keyword. Its mode=max, factor=0.5, patience=5 and scheduler.step(iou)
remain unchanged; this does not change the LR reduction rule.

## Optional segmentation on the same original ++ loader

Both baseline and segmentation now use `--dataset grasp-anywhere` and the same
folder paths. Do not use the older grasp-anything-language manifest trainer.
Add `--segmentation --lambda-seg 0.5 --lambda-dice 1` to the existing v1 command.
Use `--lambda-dice 0` for BCE-only auxiliary supervision. `--text-source` remains
object/instruction; keep it identical in both runs.

The additional mask is `<add-file-path>/part_mask/<scene>_<object>_<part>.npy`,
a finite binary 2D array matching the source RGB size. Original seven batch
fields are unchanged; segmentation appends an eighth mask tensor. Mask rotation,
central zoom crop and resizing reuse the sampled RGB augmentation parameters,
with nearest interpolation. A target cropped completely out of view becomes
background for that augmented sample; missing/invalid source masks raise errors.
No sample filtering or alternate split is introduced.

The original grasp loss is unchanged. With the flag enabled:
`total = grasp_MSE + lambda_seg * (BCE + lambda_dice * Dice)`.
The optional head reads the final fused 96-channel feature; masks never enter
the grasp forward path. Default inference returns four grasp maps. Additional
head initialization preserves CPU RNG state. Baseline does not read masks or
instantiate the head. Original loop bounds, warmup, augmentation, text mode,
upsampling and 15-grasp/width-150 IoU validation remain intact.

`evaluate_test.py` accepts the same segmentation and text flags for the matching
checkpoint. Its constructor now uses the actual Grasp_Mamba signature rather
than unsupported old GR-ConvNet arguments; IoU logic is unchanged. Load a
segmentation checkpoint with --segmentation, or omit it for a baseline checkpoint.
The standalone old manifest/evaluate_manifest experiment is not this pipeline.

This integration was tested with small CPU encoders/data, not a full pretrained
Kaggle CUDA training run. Original unrelated runtime/version issues remain.

## Checkpoint retention

Each training run retains up to five state_dict checkpoints with the highest
validation IoU, ranked using the full-precision score. It considers every epoch;
when full, a score no better than the fifth-best is not saved. Ties keep the
earlier epoch. Replaced checkpoints from this run are removed after a successful
new save. A failed write preserves the previous checkpoints. Filenames contain
the epoch and IoU and use .pt; checkpoint contents remain plain state_dict.
Scheduler, validation and training calculations are unchanged.
