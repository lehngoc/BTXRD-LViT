# W1 - Person B Reconstruction / Evaluation Audit

## Scope

Person B owns Week 1 inference, evaluation, threshold calibration, and UQ prototype.
Person A has finished data/training audit, so this report focuses on:

- sliding-window reconstruction;
- average probability aggregation;
- synthetic reconstruction test;
- validation probability-map tooling;
- threshold-sweep tooling;
- entropy and overlap-disagreement prototype outputs.

## B1-B3 Status

### Sliding-window settings

Configured in `configs/train_unet_patch384.yaml`:

```text
patch_size = 384
stride = 192
overlap = 50%
merge = average_probability
batch_size = 4
threshold = 0.5
```

### Code audit

Implementation: `src/inference/sliding_window.py`

Verified by reading and smoke testing:

- window starts cover each spatial axis and anchor the last window to the image border;
- every valid source pixel has coverage count >= 1;
- images smaller than patch size are padded before inference and cropped back after aggregation;
- output probability map is cropped back to original source-image shape;
- aggregation is average probability: `prob_sum / count_sum`;
- unsupported merge modes raise an error instead of silently changing behavior;
- no per-patch thresholding is used before aggregation.

### Synthetic reconstruction test

Added:

```text
src/inference/smoke_test_sliding_window_reconstruction.py
```

The synthetic test checks:

- coordinate mapping back to full image;
- overlap averaging;
- coverage count;
- border handling when image size is not divisible by stride;
- crop-back for images smaller than `384x384`;
- overlap variance from `sum(p)`, `sum(p^2)`, and `count`.

Command run:

```bash
python src/inference/smoke_test_sliding_window_reconstruction.py
```

Result:

```text
Sliding-window synthetic reconstruction test passed.
```

Additional smoke test:

```bash
python src/training/smoke_test_patch_pipeline.py --config configs/train_unet_patch384.yaml --samples-per-split 2
```

Result:

```text
BTXRD patch UNet pipeline smoke test passed.
```

## B4-B7 Tooling Added

### Save validation probability maps

`evaluate_full_images()` now supports saving full-image probability maps:

```text
--save-probability-dir w1_validation_probability_maps
```

Maps are saved as:

```text
w1_validation_probability_maps/<image_id>.npy
```

### Entropy prototype

Added `predictive_entropy(p)`:

```text
H(p) = -p log(p) - (1-p) log(1-p)
```

Evaluator option:

```text
--save-entropy-dir w1_entropy_prototype
```

### Overlap-disagreement prototype

Sliding-window inference now can save:

```text
count
sum_probability
sum_probability_squared
variance
```

Evaluator option:

```text
--save-overlap-stats-dir w1_overlap_disagreement_prototype
--save-disagreement-dir w1_overlap_disagreement_prototype_maps
```

The current disagreement map is overlap variance:

```text
Var(p) = E[p^2] - E[p]^2
```

### Offline threshold sweep

Added:

```text
src/inference/sweep_probability_maps.py
```

It consumes saved `.npy` probability maps and produces:

- threshold-level summary CSV;
- optional per-image metrics CSV;
- tumor Dice / IoU / Precision / Recall;
- tumor median Dice;
- tumor Dice=0 case count;
- normal FP image rate;
- normal predicted area ratio.

## Current Blocker

The workspace currently has no usable old checkpoint artifact:

```text
experiments/E2_unet_patch384_preprocessed_pos060/best.pt
```

Because of this, B4-B6 cannot be completed on this machine yet:

- re-run Patch384 v1 checkpoint on validation;
- save real validation probability maps;
- run real threshold sweep;
- build real failure gallery.

## Commands To Run Once Checkpoint Exists

Run validation and save probability/UQ prototype maps:

```bash
python src/inference/evaluate_unet_sliding_window.py \
  --config configs/train_unet_patch384.yaml \
  --checkpoint experiments/E2_unet_patch384_preprocessed_pos060/best.pt \
  --split val \
  --output w1_val_sliding_metrics_thr050.json \
  --save-probability-dir w1_validation_probability_maps \
  --save-overlap-stats-dir w1_overlap_disagreement_prototype \
  --save-entropy-dir w1_entropy_prototype \
  --save-disagreement-dir w1_overlap_disagreement_prototype_maps
```

Run threshold sweep:

```bash
python src/inference/sweep_probability_maps.py \
  --manifest data/exports/btxrd_preprocessed/val.csv \
  --root-dir . \
  --probability-dir w1_validation_probability_maps \
  --output w1_threshold_sweep.csv \
  --per-image-output w1_threshold_sweep_per_image.csv \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
  --min-fp-area-ratio 0.001
```

Note: current config uses `min_fp_area_ratio = 0.001`. If the group wants the stricter
canonical rule mentioned elsewhere in the plan, rerun the sweep with:

```text
--min-fp-area-ratio 0.0001
```

## Week 1 Person B Conclusion So Far

Sliding-window reconstruction and average-probability aggregation are not the
primary suspected bug based on code audit and synthetic testing.

Current status:

```text
B1 reconstruction audit: DONE
B2 aggregation verification: DONE
B3 synthetic unit test: DONE
B4 validation probability maps: BLOCKED, missing checkpoint
B5 threshold sweep: TOOL READY, waiting for probability maps
B6 failure gallery: BLOCKED, waiting for real predictions
B7 entropy/disagreement prototype: TOOL READY, waiting for real predictions
```
