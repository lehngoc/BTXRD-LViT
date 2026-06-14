# E2/E3 LViT 224x224 Plan

This branch adds a clean LViT phase after the completed UNet baseline.

## Goal

Answer Q2:

```text
Does text_lvit_prompt help LViT segmentation compared with the no-text LViT baseline?
```

Main experiments:

```text
E2 = LViT-TW no-text
E3 = LViT-T with text_lvit_prompt
```

## Experiment Contract

E2 and E3 must use the same controlled setup:

```text
data = preprocessed BTXRD
split = seed42
resize = 224x224
train = tumor_only=true
eval = full test tumor+normal
augmentation = same
loss = same
optimizer/scheduler = same
epochs = same
best checkpoint = best val tumor dice
threshold sweep = 0.3, 0.4, 0.5, 0.6, 0.7
```

The only intended difference is:

```text
E2: no text
E3: text_lvit_prompt
```

## Scope Control

This branch should not become another UNet branch:

- Do not add more UNet notebooks or scripts.
- Do not turn `train_unet.py` into a mixed UNet/LViT trainer.
- Do not copy a full legacy LViT repository into this repo.
- Reuse the BTXRD dataset, metric schema, and output conventions already established here.

## Deferred Work

The following are useful, but should wait until E2/E3 full-label runs are complete:

```text
E2-50 / E3-50
raw LViT
normal-aware LViT
negative-prompt ablations
patch/high-resolution phase
```

## Required Final Comparison

The 224x224 phase should end with this table completed:

| Model | Data | Text | Threshold | Tumor Dice | Tumor IoU | Precision | Recall | Normal FP rate | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| E1as UNet | Preprocessed | No | 0.6 | 0.5612 | 0.4540 | 0.6630 | 0.6505 | 0.3156 | CNN baseline |
| E2 LViT-TW | Preprocessed | No | best | TBD | TBD | TBD | TBD | TBD | image-only LViT |
| E3 LViT-T | Preprocessed | Yes | best | TBD | TBD | TBD | TBD | TBD | text-conditioned LViT |
