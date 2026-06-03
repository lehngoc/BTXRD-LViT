# BTXRD Dataset Pipeline

## Input

```text
data/raw/images/
data/raw/Annotations/
data/raw/dataset.csv
```

### Step 1: Convert LabelMe JSON to masks

Tumor images have polygon annotations. Normal images should receive empty masks.

### Step 2: Crop X-ray borders

The goal is to remove irrelevant black borders, markers, and scanner background while preserving native image resolution.

### Step 3: Generate text annotations

Text prompts are generated from structured metadata in dataset.csv.

### Step 4: Create splits

Train/validation/test splits should be reproducible and stratified by tumor status.