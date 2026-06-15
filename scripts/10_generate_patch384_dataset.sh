#!/usr/bin/env bash
set -euo pipefail

python src/export/extract_patches.py \
  --manifest data/exports/btxrd_preprocessed/train.csv \
  --output-root data/exports/patches_jitter \
  --patch-size 384 \
  --jitter-fraction 0.5 \
  --positive-crops-per-lesion 3 \
  --hard-negatives-per-lesion 1 \
  --random-negatives-per-image 0.5 \
  --large-lesion-grid-stride-ratio 0.75 \
  --max-large-lesion-grid-patches 6 \
  --target-positive-ratio 0.6 \
  --clean-output

python src/export/extract_patches.py \
  --manifest data/exports/btxrd_preprocessed/val.csv \
  --output-root data/exports/patches_jitter_val \
  --patch-size 384 \
  --jitter-fraction 0.5 \
  --positive-crops-per-lesion 3 \
  --hard-negatives-per-lesion 1 \
  --random-negatives-per-image 0.5 \
  --large-lesion-grid-stride-ratio 0.75 \
  --max-large-lesion-grid-patches 6 \
  --target-positive-ratio 0.6 \
  --clean-output

python src/export/extract_patches.py \
  --manifest data/exports/btxrd_preprocessed/test.csv \
  --output-root data/exports/patches_jitter_test \
  --patch-size 384 \
  --jitter-fraction 0.5 \
  --positive-crops-per-lesion 3 \
  --hard-negatives-per-lesion 1 \
  --random-negatives-per-image 0.5 \
  --large-lesion-grid-stride-ratio 0.75 \
  --max-large-lesion-grid-patches 6 \
  --target-positive-ratio 0.6 \
  --clean-output
