# BTXRD-LViT

Clean BTXRD segmentation workspace for the E2/E3 LViT 224x224 phase.

This branch intentionally keeps the active code focused on:

```text
E2 = LViT-TW no-text on preprocessed BTXRD 224x224
E3 = LViT-T with text_lvit_prompt on preprocessed BTXRD 224x224
```

The completed UNet phase is documented, but its training code, configs, and notebooks are not active in this branch.

## Current Branch Scope

This branch answers Q2:

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
- High-resolution patch-based segmentation
