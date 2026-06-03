# BTXRD-LViT
Clean BTXRD preprocessing and text annotation pipeline for lightweight vision-language bone tumor segmentation.

## Motivation

The original LViT pipeline resizes medical images to a fixed low resolution, which may discard fine-grained clinical details in high-resolution X-ray radiographs. BTXRD provides high-resolution bone tumor radiographs, expert segmentation masks, bounding boxes, and structured metadata.

This repository first focuses on dataset preparation before developing high-resolution lightweight vision-language segmentation models.

## Current Goals

1. Convert BTXRD LabelMe annotations to binary segmentation masks.
2. Remove irrelevant X-ray border artifacts while preserving native image resolution.
3. Generate structured text annotations from `dataset.csv`.
4. Create reproducible train/validation/test splits.
5. Export cleaned data for LViT, UNet, and future patch-based models.

## Planned Experiments

- Raw BTXRD + UNet baseline
- Raw BTXRD + LViT-TW
- Raw BTXRD + LViT-T
- Cleaned BTXRD + LViT-TW
- Cleaned BTXRD + LViT-T
- Cleaned BTXRD + negative text prompts
- High-resolution patch-based segmentation

## Repository Structure

```text
configs/        Configuration files
data/           Local dataset directory, ignored by Git
src/            Source code
scripts/        Runnable shell scripts
notebooks/      Data inspection notebooks
docs/           Project documentation
experiments/    Local experiment outputs, ignored by Git
external/       External reference repositories
