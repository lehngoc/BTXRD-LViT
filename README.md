# BTXRD-LViT

Clean BTXRD segmentation workspace for LViT 224x224 and Patch384 experiments.

This branch intentionally keeps the active code focused on:

```text
E2 = LViT-TW no-text on preprocessed BTXRD 224x224
E3 = LViT-T with text_lvit_prompt on preprocessed BTXRD 224x224
E4 = LViT-T with text_lvit_prompt trained on Patch384 and evaluated by full-image sliding-window
```

The branch also keeps the UNet Patch384 baseline active for high-resolution
comparison.

## Current Branch Scope

The 224x224 E2/E3 phase answers Q2:

```text
Does text_lvit_prompt help LViT compared with the no-text LViT baseline?
```

Controlled setup:

```text
data = preprocessed BTXRD
split = seed42
resize = 224x224
train = tumor_only=true
eval = full test tumor+normal
threshold sweep = 0.3, 0.4, 0.5, 0.6, 0.7
```

The main E2/E3 configs use `transformer_heads: 4` to match the original LViT config. Earlier H8 runs can be kept as exploratory artifacts, but the H4 runs are the closer-to-original baseline.

The only intended experiment difference is:

```text
E2: no text
E3: text_lvit_prompt
```

Model-port policy:

```text
Architecture: port the original LViT Double-U CNN/ViT design as closely as practical.
Text encoder: use HuggingFace BERT as a maintained replacement for the original bert_embedding dependency.
Text interface: preserve the original [B, 10, 768] tensor shape.
Output API: return raw logits; BTXRD losses/metrics apply sigmoid centrally.
Training/eval: keep the BTXRD protocol for fair E1as/E2/E3 comparison.
Image input: use the original LViT E2/E3 recipe, `mean=[0,0,0]` and
`std=[1,1,1]`, which is equivalent to RGB scaled to `[0,1]`.
```

## Completed Previous Work

The data pipeline and UNet phase are frozen in docs:

```text
docs/dataset_pipeline.md
docs/experiments/unet_phase_summary.md
docs/experiments/e2e3_lvit_224_plan.md
```

Final UNet baseline:

```text
E1as_strong_preprocessed_tumor_only
threshold = 0.6
tumor_dice = 0.5612
normal_fp_image_rate = 0.3156
normal_pred_area_ratio = 0.00331
```

E1bs strong normal-aware is kept as an ablation in the docs, not as the final UNet.

## Run E2

Local smoke test:

```bash
python -m src.training.train_lvit_tw \
  --config configs/train_lvit_tw_preprocessed_224.yaml \
  --epochs 1 \
  --max-train-samples 4 \
  --max-val-samples 4 \
  --max-test-samples 4
```

Full run:

```bash
python -m src.training.train_lvit_tw \
  --config configs/train_lvit_tw_preprocessed_224.yaml
```

Kaggle runner:

```text
notebooks/E2_lvit_tw_preprocessed_224_kaggle.ipynb
```

## Run E3

E3 requires HuggingFace `transformers` for the default BERT text encoder:

```bash
pip install -r requirements.txt
```

Local smoke test:

```bash
python -m src.training.train_lvit_t \
  --config configs/train_lvit_t_preprocessed_224.yaml \
  --epochs 1 \
  --max-train-samples 4 \
  --max-val-samples 4 \
  --max-test-samples 4
```

Full run:

```bash
python -m src.training.train_lvit_t \
  --config configs/train_lvit_t_preprocessed_224.yaml
```

Kaggle runner:

```text
notebooks/E3_lvit_t_preprocessed_224_kaggle.ipynb
```

## Outputs

Both E2 and E3 write the same artifact schema:

```text
experiments/<run_name>/
  best.pt
  last.pt
  history.csv
  best_summary.json
  val_metrics.json
  test_metrics.json
  test_metrics_thr30.json
  test_metrics_thr40.json
  test_metrics_thr50.json
  test_metrics_thr60.json
  test_metrics_thr70.json
  test_threshold_sweep_metrics.json
  config.json
```

## Repository Shape

Active LViT files:

```text
configs/train_lvit_tw_preprocessed_224.yaml
configs/train_lvit_t_preprocessed_224.yaml
notebooks/E2_lvit_tw_preprocessed_224_kaggle.ipynb
notebooks/E3_lvit_t_preprocessed_224_kaggle.ipynb
src/models/lvit_tw.py
src/models/lvit_t.py
src/training/train_lvit_tw.py
src/training/train_lvit_t.py
configs/train_lvit_t_patch384.yaml
notebooks/E4_lvit_t_patch384_text_kaggle.ipynb
src/training/train_lvit_t_patch.py
src/inference/evaluate_lvit_t_sliding_window.py
```

Shared infrastructure:

```text
src/data/btxrd_dataset.py
src/training/losses.py
src/training/metrics.py
src/training/utils.py
src/preprocessing/
src/export/
configs/preprocess.yaml
configs/splits/btxrd_split_seed42.csv
```

Deferred work:

- E2-50 / E3-50
- Raw LViT
- Normal-aware LViT
- Negative-prompt ablations
- LViT-TW Patch384 no-text comparison

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

## Phase 4 Text-Conditioned LViT-T Patch384

E4 trains LViT-T on the generated `384x384` patch dataset with
`text_lvit_prompt`. The prompt describes the source full image and is reused
for every sampled patch and every sliding window from that image.

The main checkpoint is selected by full-image, text-conditioned sliding-window
validation Dice rather than patch-level Dice. During sliding inference, BERT
features are computed once per source image and reused across all its windows.

Default settings:

```text
model = LViT-T, transformer_heads = 4
text_encoder = frozen bert-base-uncased
input = same as E2/E3: image_mean=[0,0,0], image_std=[1,1,1]
patch_size = 384
stride = 192
training_batch_size = 1
gradient_accumulation_steps = 4
mixed_precision = true
learning_rate = 1e-4
gradient_clip_norm = 1.0
full_val_interval = 10
full_val_patience = 5
sliding_window_batch_size = 1
merge = average_probability
```

Smoke test and train:

```bash
python src/training/smoke_test_lvit_t_patch_pipeline.py --config configs/train_lvit_t_patch384.yaml
python src/training/train_lvit_t_patch.py --config configs/train_lvit_t_patch384.yaml --auto-resume
```

Full-image validation and test:

```bash
python src/inference/evaluate_lvit_t_sliding_window.py --config configs/train_lvit_t_patch384.yaml --checkpoint experiments/E4_lvit_t_patch384_text_h4/best.pt --split val
python src/inference/evaluate_lvit_t_sliding_window.py --config configs/train_lvit_t_patch384.yaml --checkpoint experiments/E4_lvit_t_patch384_text_h4/best.pt --split test
```

Kaggle runner:

```text
notebooks/E4_lvit_t_patch384_text_kaggle.ipynb
```
