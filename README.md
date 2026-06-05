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
scripts/
  01_prepare_btxrd.bat
  01_prepare_btxrd.sh
src/
  export/
    export_training_dataset.py
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
