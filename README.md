# BTXRD-LViT
BTXRD preprocessing workspace for building a clean X-ray segmentation dataset.

## Motivation

The original LViT pipeline resizes medical images to a fixed low resolution, which may discard fine-grained clinical details in high-resolution X-ray radiographs. BTXRD provides high-resolution bone tumor radiographs, expert segmentation masks, bounding boxes, and structured metadata.

This repository currently focuses on cleaning the BTXRD dataset before model training.

## Current Goals

1. Convert BTXRD LabelMe annotations to binary segmentation masks.
2. Crop low-information X-ray borders, including black, white, and gray frame artifacts.
3. Remove safe L/R marker artifacts and isolated bright marker blobs without touching tumor masks.
4. Validate that image/mask sizes match and tumor mask pixels are preserved.
5. Generate visual previews for manual quality control.

## Current Preprocessing Pipeline

Run the full BTXRD data-cleaning pipeline:

```bash
python src/preprocessing/convert_labelme_to_mask.py
python src/preprocessing/preprocess_btxrd_data.py --clean-output
python src/preprocessing/validate_preprocessed_data.py
python src/preprocessing/visualize_preprocessed_data.py
```

On Windows, the same flow is available through:

```bat
scripts\01_prepare_btxrd.bat
```

Final cleaned outputs are written to:

```text
data/processed/images_preprocessed/
data/processed/masks_preprocessed/
data/processed/reports/preprocess_pipeline_report.csv
data/processed/reports/preprocess_validation_report.csv
data/processed/visual_checks/preprocess_pipeline/
```

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
src/            Source code, including BTXRD preprocessing scripts
scripts/        Runnable shell scripts
notebooks/      Data inspection notebooks
docs/           Project documentation
experiments/    Local experiment outputs, ignored by Git
external/       External reference repositories
