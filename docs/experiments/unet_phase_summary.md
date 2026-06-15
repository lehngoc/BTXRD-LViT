# UNet Phase Summary

This document freezes the completed UNet phase before the LViT 224x224 experiments.

## Completed Data Pipeline

The BTXRD data pipeline is complete for controlled 224x224 experiments:

- Raw LabelMe annotations were converted to binary segmentation masks.
- X-ray border and marker preprocessing was applied while preserving tumor masks.
- Preprocessed image and mask pairs were validated.
- Text annotations were generated from structured metadata, including `text_lvit_prompt`.
- Seed 42 train/validation/test splits were created.
- Training CSV manifests were exported under `data/exports/btxrd_preprocessed/`.

Current export summary:

```text
total rows: 3746
train/val/test: 2622 / 562 / 562
normal/tumor: 1879 / 1867
benign/malignant: 1525 / 342
missing preprocessed images: 0
missing preprocessed masks: 0
```

## Q1 Conclusion

The UNet experiments support the following interpretation:

```text
Clean recipe: preprocessed > raw.
Strong recipe: preprocessed ~= raw.
Preprocessing did not damage the data; the earlier performance gap was mostly driven by recipe/model capacity.
```

## Q3 Conclusion

The final UNet baseline is:

```text
E1as_strong_preprocessed_tumor_only
```

Main operating point:

```text
threshold = 0.6
```

Low false-positive alternative:

```text
threshold = 0.7
```

The E1bs strong normal-aware run is kept as an ablation. It improved over the clean normal-aware baseline, but it did not outperform E1as on the Dice/normal-FP tradeoff.

## Final UNet Table

| Run | Threshold | Tumor Dice | Normal FP image rate | Normal pred area ratio | Decision |
|---|---:|---:|---:|---:|---|
| E1as strong preprocessed tumor-only | 0.6 | 0.5612 | 0.3156 | 0.00331 | Final UNet operating point |
| E1as strong preprocessed tumor-only | 0.7 | 0.5576 | 0.2872 | 0.00257 | Stricter low-FP option |
| E1bs strong normal-aware preprocessed | 0.7 | 0.5327 | 0.2943 | 0.00343 | Ablation, not final |

## Hand-Off To LViT Phase

The next phase should answer Q2 only:

```text
Does text conditioning help LViT compared with the no-text LViT baseline?
```

The UNet code and results should remain stable. New LViT code should live in separate model/training files and reuse only the shared dataset, losses, metrics, and exported manifests.
