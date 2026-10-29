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

## Diagnose training time

`tools/profile_training.py` compares a few baseline/segmentation batches and
reports data wait, host-to-GPU transfer, forward/loss, scalar logging,
backward/clipping, optimizer and total milliseconds, plus peak GPU memory.
Stop the training cell first. Pass the same paths, batch size, workers and
pretrained locations as the real run. It profiles a newly loaded model without
saving checkpoints. CUDA synchronization adds instrumentation overhead; compare
stage proportions and baseline/segmentation ratios, not only absolute timing.
Use --data-only to measure data loading without weights/CUDA.

Local CPU measurements on eight extracted samples showed about 116 ms/sample
without masks vs 147 ms/sample with masks, not an 8x difference. Mask rotation
accounted for about 21 ms of a 34 ms mask call. The square-image quarter-turn
path now uses np.rot90, checked pixel-for-pixel against the previous transform.
Other angles/rectangular inputs keep the original implementation. No baseline
RGB/grasp augmentation, train loop, loss weights or evaluation rules changed.
The reported 10-to-80 minute Kaggle slowdown still requires runtime measurements
to distinguish data starvation, GPU compute/memory or a changed environment.

## Explicit official seen/unseen test splits

`evaluate_test.py --eval-split test_seen` selects the IDs from
`split/grasp-anything++/test/seen.obj`. Use --eval-split test_unseen for the
matching official unseen file. An explicit eval-split overrides legacy --seen.
Without the new flag, legacy --seen 1 still selects train/seen.obj and --seen 0
selects test/unseen.obj. Training passes no eval-split, so train/validation
selection and all optimization behavior remain unchanged. Evaluation continues
to use the original five grasp candidates, angle gate and IoU threshold.

## Test all retained checkpoints

`evaluate_checkpoints.py --checkpoint-dir <run-folder> --dataset-path <base>
--add-file-path <plus> --output-dir <writable-output>` evaluates every retained
checkpoint on both official test_seen and test_unseen. It accepts both current
.pt filenames and legacy extensionless epoch_*_iou_* files. Model and loader
are reused across checkpoints of the same type. Segmentation is detected from
state_dict keys; no SEG_FLAG is needed. Pretrained IDs/paths, text source, input
size and loss weights default to commandline_args.json when available and can
be overridden explicitly. The raw state_dict is loaded strictly on the matching
architecture. The original evaluator's five-grasp, angle and IoU protocol is
reused; training/checkpoint retention is not modified.

Outputs checkpoint_results.csv and checkpoint_results.json include validation
accuracy parsed from filename, seen/unseen correct/total/accuracy and harmonic
mean H=2*S*U/(S+U). CSV accuracy columns are percentages; JSON accuracy fields
are fractions. Every row is evaluated; test scores do not determine checkpoint
selection/order. Results are written after each completed checkpoint so earlier
rows survive an interrupted run. A partial JSON has complete=false. Default
output directory is evaluation_results/<run-folder-name> under the current
working directory; --output-dir overrides it. This evaluator needs the current
loader/evaluate_test.py with explicit official split selection.

## Automatic checkpoint folder selection

Use --logs-dir <repo>/logs instead of --checkpoint-dir. The evaluator scans
completed epoch_*_iou_* checkpoint files and chooses the parent folder of the
most recently written one, independent of the run's name or date format.
Temporary checkpoints and folders with only logs are ignored. It prints the
selected folder and reads its commandline_args.json as before. Without an
explicit --output-dir, CSV/JSON go to evaluation_results/<selected-run-name>.
An explicit --checkpoint-dir remains available to select a particular run.

## Unreadable checkpoints and interrupted evaluation

Bulk evaluation now tests up to five readable checkpoints by saved validation
score (--max-checkpoints overrides). If a checkpoint cannot be deserialized
because it is truncated/corrupt/missing, the script records its load error in
JSON and attempts the next candidate. Model architecture, dataset, CUDA-memory
and evaluation errors still propagate; they are not silently skipped. Original
checkpoint files are never changed. Legacy folders may contain more than five
models because retention applies to the active training run, not old files.

Add --resume with the same settings/output directory to reuse completed rows.
Resume verifies the evaluation protocol/settings and file size/mtime. For an
older report without fingerprints, only files not modified since that report
are reused. Validation rank remains the only selection criterion. JSON records
candidate_count, checkpoint_count, completed_count, errors, finished and complete.
If load errors prevent reaching the selected target, results are explicitly incomplete;
an all-failed run reports an error after writing its load-error report.

## Configure checkpoint counts

`bash run_training.v1.sh ... --max-checkpoints 10` retains the ten best
validation checkpoints produced by that training run. Default is five.
`evaluate_checkpoints.py ... --max-checkpoints 10` evaluates up to ten readable
checkpoints; this controls test count separately and does not delete model files.
