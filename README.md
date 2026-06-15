# BTXRD-LViT

BTXRD preprocessing workspace for building a clean X-ray segmentation dataset and preparing text-conditioned training manifests for LViT-style experiments.

## Motivation

The original LViT pipeline resizes medical images to a fixed low resolution, which may discard fine-grained clinical details in high-resolution X-ray radiographs. BTXRD provides high-resolution bone tumor radiographs, expert segmentation masks, bounding boxes, and structured metadata.

This repository currently focuses on cleaning the BTXRD dataset, generating text prompts from metadata, creating reproducible splits, and exporting training CSVs before model training.

## BTXRD Data Pipeline Status

Completed pipeline pieces:

1. Audited the raw BTXRD metadata, image files, and annotation files.
2. Converted LabelMe polygon annotations to binary segmentation masks.
3. Generated visual mask previews for manual quality control.
4. Cropped low-information X-ray borders, including black, white, and gray frame artifacts.
5. Removed safe L/R marker artifacts and isolated bright marker blobs while preserving tumor masks.
6. Validated that preprocessed image and mask sizes match and that tumor mask pixels are preserved.
7. Generated text annotations from structured metadata, including LViT-style prompts, short prompts, anatomy-aware prompts, segmentation target text, and negative prompts for normal cases.
8. Created reproducible train/validation/test splits with seed 42.
9. Exported training manifests for all/train/val/test rows.

Current local export summary:

```text
total rows: 3746
train/val/test: 2622 / 562 / 562
normal/tumor: 1879 / 1867
benign/malignant: 1525 / 342
missing preprocessed images: 0
missing preprocessed masks: 0
```

## Current Preprocessing Pipeline

Run the full BTXRD data-cleaning and export pipeline on Windows:

```bat
scripts\01_prepare_btxrd.bat
```

Equivalent command sequence:

```bash
python src/preprocessing/audit_btxrd_dataset.py
python src/preprocessing/convert_labelme_to_mask.py
python src/preprocessing/visualize_masks.py
python src/preprocessing/preprocess_btxrd_data.py --clean-output
python src/preprocessing/validate_preprocessed_data.py
python src/preprocessing/visualize_preprocessed_data.py
python src/preprocessing/generate_text_annotations.py
python src/preprocessing/make_splits.py
python src/export/export_training_dataset.py
```

The Unix helper currently runs the preprocessing/visualization steps only:

```bash
scripts/01_prepare_btxrd.sh
```

Final generated outputs are written locally to:

```text
data/processed/masks/
data/processed/images_preprocessed/
data/processed/masks_preprocessed/
data/processed/text_annotations.csv
data/processed/splits/
data/processed/reports/
data/processed/visual_checks/
data/exports/btxrd_preprocessed/
```

## E1 UNet Baseline Pipeline

The first model pipeline is a clean image-only UNet baseline on preprocessed BTXRD resized to `224x224`.

Smoke test the dataset, mask loading, UNet forward pass, and backward pass:

```bash
python src/training/smoke_test_model_pipeline.py --config configs/train_unet_baseline.yaml
```

Train the baseline:

```bash
python src/training/train_unet.py --config configs/train_unet_baseline.yaml
```

Or on Windows:

```bat
scripts\02_train_unet_baseline.bat
```

Evaluate the best checkpoint on the test split:

```bash
python src/training/evaluate_unet.py \
  --config configs/train_unet_baseline.yaml \
  --checkpoint experiments/E1_unet_preprocessed_224_weighted_loss/best.pt \
  --split test
```

The UNet pipeline reads:

```text
data/exports/btxrd_preprocessed/train.csv
data/exports/btxrd_preprocessed/val.csv
data/exports/btxrd_preprocessed/test.csv
```

Training outputs are written to:

```text
experiments/E1_unet_preprocessed_224_weighted_loss/
  config.json
  history.csv
  best.pt
  last.pt
  best_summary.json
  test_metrics.json
```

Metrics are reported separately for all cases, tumor cases, and normal cases. Normal-case metrics include predicted mask area ratio and false-positive image rate.

The current E1 config uses `positive_weight: 20.0` for BCE and computes DiceLoss on tumor samples only. This keeps normal cases in training for false-positive control while preventing empty-mask normal cases from dominating the Dice objective.

E1 is split into two UNet baselines:

```text
E1a: preprocessed 224x224 tumor-only UNet
E1b: preprocessed 224x224 normal-aware UNet
```

Use E1a to compare segmentation behavior against the historical tumor-only E0 run. Use E1b to study normal false positives.

For normal cases, use `normal_pred_area_ratio` and `normal_fp_image_rate` as the main false-positive metrics. `normal_precision` and `normal_recall` are logged for completeness, but they are not very interpretable when the ground-truth mask is empty.

The dataset reader normalizes Windows-style paths from exported CSV files, so manifests containing paths such as `data\processed\images_preprocessed\IMG000001.jpg` can also run on Linux/Kaggle. Set `data.root_dir` in the config when the dataset root is not the repository root.

For 50% label experiments, note that the current `label_fraction < 1` behavior keeps all normal cases and samples tumor cases only. Treat this as a tumor-labeled fraction setup, not as a 50% sample of the entire train split.

### Kaggle GPU T4 Notebook

Use this notebook for the full E1 run on Kaggle instead of training on local CPU:

```text
notebooks/E1b_unet_normal_aware_224_kaggle.ipynb
```

Kaggle setup:

1. Enable GPU T4 in Notebook Settings.
2. Attach a Kaggle Dataset containing `data/exports/btxrd_preprocessed/`, `data/processed/images_preprocessed/`, and `data/processed/masks_preprocessed/`.
3. Make the repo available under `/kaggle/working/BTXRD-LViT`, or edit `REPO_ROOT` in the notebook.
4. If auto-detection cannot find the attached data, edit `DATA_ROOT` in the notebook.
5. Run the notebook cells in order: smoke test, train, evaluate `val` and `test`, then zip artifacts.

Kaggle outputs are written to:

```text
/kaggle/working/experiments/E1_unet_preprocessed_224_weighted_loss/
/kaggle/working/E1_unet_preprocessed_224_weighted_loss_artifacts.zip
```

For E1a tumor-only training on Kaggle, use:

```text
notebooks/E1a_unet_tumor_only_224_kaggle.ipynb
```

For E1b normal-aware training on Kaggle, use:

```text
notebooks/E1b_unet_normal_aware_224_kaggle.ipynb
```

E1a writes metrics-only artifacts to:

```text
/kaggle/working/experiments/E1a_unet_preprocessed_224_tumor_only/
/kaggle/working/E1a_unet_tumor_only_metrics_only.zip
```

To visualize UNet predictions from a checkpoint:

```bash
python src/training/visualize_unet_predictions.py \
  --config configs/train_unet_baseline.yaml \
  --checkpoint experiments/E1_unet_preprocessed_224_weighted_loss/best.pt \
  --split val
```

## Git Tracking Notes

Version these files because they define reproducible preprocessing behavior:

```text
configs/preprocess.yaml
configs/splits/btxrd_split_seed42.csv
docs/
scripts/
src/
README.md
requirements.txt
data/processed/reports/btxrd_audit_details.csv
data/processed/reports/btxrd_audit_summary.json
data/processed/reports/mask_conversion_report.csv
data/**/.gitkeep
```

Keep these local, or publish them through dataset/artifact storage instead of normal Git:

```text
data/raw/images/
data/raw/Annotations/
data/processed/images_preprocessed/
data/processed/masks/
data/processed/masks_preprocessed/
data/processed/splits/
data/processed/text_annotations.csv
data/processed/visual_checks/
data/processed/reports/*generated_after_full_pipeline*
data/exports/btxrd_preprocessed/
experiments/
logs/
runs/
wandb/
*.pth
*.pt
*.ckpt
```

Note: `data/raw/dataset.csv` is currently tracked in Git even though `.gitignore` ignores new `data/raw/*.csv` files. Keep it tracked only if the metadata is allowed to be public and is needed for reproducibility.

## Repository Structure

```text
configs/
  preprocess.yaml
  train_unet_baseline.yaml
  train_unet_tumor_only.yaml
  splits/
    btxrd_split_seed42.csv
data/
  exports/
    .gitkeep
    btxrd_preprocessed/        # local generated training manifests
  processed/
    masks/                     # local generated LabelMe masks
    reports/                   # versioned audit reports plus local generated reports
    splits/                    # local generated split CSVs
    images_preprocessed/       # local generated cleaned images
    masks_preprocessed/        # local generated cleaned masks
    text_annotations.csv       # local generated text prompts
    visual_checks/             # local QC previews
  raw/
    Annotations/               # local raw LabelMe annotations
    images/                    # local raw X-ray images
    dataset.csv                # tracked metadata CSV
docs/
  dataset_pipeline.md
notebooks/
  E1a_unet_tumor_only_224_kaggle.ipynb
  E1b_unet_normal_aware_224_kaggle.ipynb
scripts/
  01_prepare_btxrd.bat
  01_prepare_btxrd.sh
  02_train_unet_baseline.bat
  02_train_unet_baseline.sh
  03_evaluate_unet_baseline.bat
  03_evaluate_unet_baseline.sh
  04_train_unet_tumor_only.bat
  04_train_unet_tumor_only.sh
  05_evaluate_unet_tumor_only.bat
  05_evaluate_unet_tumor_only.sh
src/
  export/
    export_training_dataset.py
  data/
    btxrd_dataset.py
  models/
    unet.py
  preprocessing/
    audit_btxrd_dataset.py
    convert_labelme_to_mask.py
    crop_xray_border.py
    generate_text_annotations.py
    make_splits.py
    preprocess_btxrd_data.py
    remove_xray_markers.py
    validate_preprocessed_data.py
    visualize_masks.py
    visualize_preprocessed_data.py
  training/
    evaluate_unet.py
    losses.py
    metrics.py
    smoke_test_model_pipeline.py
    train_unet.py
    utils.py
    visualize_unet_predictions.py
.gitignore
README.md
requirements.txt
```

## Planned Experiments

- Raw BTXRD + UNet baseline
- Raw BTXRD + LViT-TW
- Raw BTXRD + LViT-T
- Cleaned BTXRD + LViT-TW
- Cleaned BTXRD + LViT-T
- Cleaned BTXRD + negative text prompts
- High-resolution patch-based segmentation

## Phase 3 UNet Patch384 Main Baseline

This baseline trains on sampled `384x384` patches while keeping full images
split at image level.

The patch size selection, patch-generation procedure, and generated-dataset
statistics are documented in Vietnamese at
[`docs/patch_size_384_analysis_vi.md`](docs/patch_size_384_analysis_vi.md).

The dataset is split at image level before patch generation, preventing source
images and their patches from leaking across train, validation, and test.
Current Patch384 outputs contain `12,830` patches with a `59.99%` positive
ratio. Patch size 384 contains `78.91%` of lesion bounding boxes in full and
`71.18%` with a 20% context allowance.

Generate the three patch splits:

```bash
scripts/10_generate_patch384_dataset.sh
```

Reproduce the bounding-box and generated-patch reports:

```bash
python src/analysis/analyze_lesion_bbox.py --min-component-area 10
python src/analysis/summarize_patch_dataset.py
python src/training/smoke_test_patch_pipeline.py --config configs/train_unet_patch384.yaml
```

Generated reports are written locally to
`data/processed/reports/lesion_bbox_analysis/` and
`data/processed/reports/patch384_dataset_analysis/`.

Train the Patch384 baseline:

```bash
python src/training/train_unet_patch.py --config configs/train_unet_patch384.yaml
```

During training, patch validation is logged every epoch as a fast proxy. The
main checkpoint selection metric is full-image validation Dice from
sliding-window inference on `data/exports/btxrd_preprocessed/val.csv`.

Default full-image validation settings:

```text
patch_size = 384
stride = 192
overlap = 50%
merge = average_probability
padding = reflect bottom/right only for images smaller than 384
inference_batch_size = 4
threshold = 0.5
post_processing = none
full_val_interval = 5
full_val_patience = 20 validation checks
```

Evaluate the selected checkpoint with the same sliding-window settings:

```bash
python src/inference/evaluate_unet_sliding_window.py --config configs/train_unet_patch384.yaml --checkpoint experiments/E2_unet_patch384_preprocessed_pos060/best.pt --split val
python src/inference/evaluate_unet_sliding_window.py --config configs/train_unet_patch384.yaml --checkpoint experiments/E2_unet_patch384_preprocessed_pos060/best.pt --split test
```

Use patch-level metrics only for training diagnostics. Report main validation
and test results from full-image sliding-window evaluation.
