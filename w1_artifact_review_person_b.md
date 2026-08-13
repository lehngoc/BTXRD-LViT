# W1 - Artifact Review for Person B

## Artifacts Found

### UNet Patch384

Folder:

```text
E2_unet_patch384_preprocessed_pos060/
```

Available:

- `best.pt`
- `last.pt`
- `history.csv`
- `best_summary.json`
- `val_sliding_epoch_001/010/.../070.json`

Checkpoint status:

```text
best.pt = epoch 60
last.pt = epoch 70
best selection metric = full-image sliding-window validation tumor Dice
```

Best validation at threshold `0.5`:

```text
tumor Dice      = 0.2910
tumor Precision = 0.2799
tumor Recall    = 0.5268
normal FP rate  = 0.9823
normal area     = 0.0613
```

Interpretation:

UNet is still strongly foreground-biased at threshold `0.5`. Training longer from
70 to 100 is not the best next action unless there is spare compute, because
full-image validation already peaked at epoch 60 and epoch 70 is slightly worse.

### LViT-T Patch384 Text

Folder:

```text
E4_lvit_t_patch384_text_artifacts/
```

Available:

- `best.pt`
- `last.pt`
- `history.csv`
- `best_summary.json`
- `val_sliding_metrics.json`
- `test_sliding_metrics.json`
- `val_sliding_epoch_060/070/080/090/100.json`

Checkpoint status:

```text
best.pt = epoch 20
last.pt = epoch 100
best selection metric = full-image text-conditioned sliding-window validation tumor Dice
```

Best validation at threshold `0.5`:

```text
tumor Dice      = 0.3642
tumor Precision = 0.4498
tumor Recall    = 0.4615
normal FP rate  = 0.7766
normal area     = 0.0149
```

Last checkpoint validation at threshold `0.5`:

```text
tumor Dice      = 0.2872
tumor Precision = 0.6583
tumor Recall    = 0.2630
normal FP rate  = 0.3759
normal area     = 0.0043
```

Interpretation:

LViT-T trained to 100 epochs, but the selected best checkpoint is epoch 20.
Later checkpoints become more conservative: precision improves and normal FP
drops, but tumor recall and Dice fall. For W1 evidence, use `best.pt` first and
then run threshold sweep on validation probability maps.

## Should UNet Continue From 70 to 100?

Recommendation:

```text
Do not prioritize continuing UNet to 100 right now.
```

Reason:

- UNet full-image validation tumor Dice peaks at epoch 60.
- Epoch 70 is already lower than epoch 60.
- Normal FP remains extremely high at both epoch 60 and epoch 70.
- W1's main missing evidence is threshold sweep and failure analysis, not more
  training.

If there is idle compute, continuing UNet to 100 is acceptable as a completeness
run, but it should not block LViT evaluation/reporting.

## What To Run Next

Priority order:

```text
1. Run LViT-T validation inference again and save probability maps.
2. Sweep thresholds on LViT-T validation probability maps.
3. Run UNet validation inference and save probability maps.
4. Sweep thresholds on UNet validation probability maps.
5. Compare LViT-T vs UNet on validation only.
6. Build failure gallery from validation predictions.
```

Use validation for all Week 1 decisions. Do not use `test_sliding_metrics.json`
to pick threshold or decide model direction.

## LViT-T Evaluation Commands

From repo root:

```bash
python src/inference/evaluate_lvit_t_sliding_window.py \
  --config configs/train_lvit_t_patch384.yaml \
  --checkpoint E4_lvit_t_patch384_text_artifacts/best.pt \
  --split val \
  --output w1_lvit_t_val_sliding_metrics_thr050.json \
  --save-probability-dir w1_lvit_t_validation_probability_maps \
  --save-overlap-stats-dir w1_lvit_t_overlap_disagreement_prototype \
  --save-entropy-dir w1_lvit_t_entropy_prototype \
  --save-disagreement-dir w1_lvit_t_overlap_disagreement_maps
```

Then sweep:

```bash
python src/inference/sweep_probability_maps.py \
  --manifest data/exports/btxrd_preprocessed/val.csv \
  --root-dir . \
  --probability-dir w1_lvit_t_validation_probability_maps \
  --output w1_lvit_t_threshold_sweep.csv \
  --per-image-output w1_lvit_t_threshold_sweep_per_image.csv \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
  --min-fp-area-ratio 0.001
```
## UNet Evaluation Commands

```bash
python src/inference/evaluate_unet_sliding_window.py \
  --config configs/train_unet_patch384.yaml \
  --checkpoint E2_unet_patch384_preprocessed_pos060/best.pt \
  --split val \
  --output w1_unet_val_sliding_metrics_thr050.json \
  --save-probability-dir w1_unet_validation_probability_maps \
  --save-overlap-stats-dir w1_unet_overlap_disagreement_prototype \
  --save-entropy-dir w1_unet_entropy_prototype \
  --save-disagreement-dir w1_unet_overlap_disagreement_maps
```

Then sweep:

```bash
python src/inference/sweep_probability_maps.py \
  --manifest data/exports/btxrd_preprocessed/val.csv \
  --root-dir . \
  --probability-dir w1_unet_validation_probability_maps \
  --output w1_unet_threshold_sweep.csv \
  --per-image-output w1_unet_threshold_sweep_per_image.csv \
  --thresholds 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
  --min-fp-area-ratio 0.001
```
