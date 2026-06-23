# Normal-aware loss ablation

This ablation keeps the UNet-Strong 224×224 pipeline fixed and compares only
the training loss.  Selection uses validation only.  The test split is used
only after loss, checkpoint, and threshold are locked.

## Screening

Run the six 30-epoch candidates from the repository root:

```powershell
Get-ChildItem configs/loss_ablation/*.yaml | ForEach-Object {
  python -m src.training.train_unet --config $_.FullName --overwrite
}
python -m src.training.aggregate_loss_ablation `
  --runs-root experiments/loss_ablation/screening --stage screening
```

The generated `screening_loss_ablation.csv` contains the selected epoch and
threshold for every loss.  `eligible` and `reason` compare each challenger
against the selected BCE+Dice baseline.

## Full stage

Use only BCE+Dice and the one or two eligible challengers.  Give every run a
separate seed and output directory; checkpoint/threshold calibration is
therefore independent per seed.

```powershell
$seeds = 42, 52, 62, 72, 82
foreach ($seed in $seeds) {
  python -m src.training.train_unet `
    --config configs/loss_ablation/unet_bce_dice_05.yaml `
    --epochs 200 --seed $seed `
    --output-dir "experiments/loss_ablation/full/bce_dice_05/seed$seed" --overwrite
}
```

Repeat the same command for each selected challenger.  Aggregate validation
artifacts before test:

```powershell
python -m src.training.aggregate_loss_ablation `
  --runs-root experiments/loss_ablation/full --stage full
```

## Locked test and qualitative figures

After the winner is fixed from the full-stage validation summary, evaluate its
five selected checkpoints and the five BCE+Dice checkpoints.  `evaluate_unet`
uses `best_summary.json`'s `selected_threshold` automatically unless a manual
threshold override is passed.

```powershell
python -m src.training.evaluate_unet `
  --config configs/loss_ablation/unet_bce_dice_05.yaml `
  --checkpoint experiments/loss_ablation/full/bce_dice_05/seed42/best.pt `
  --split test
```

`visualize_unet_predictions` also reads that saved threshold automatically.
Test metrics and grids are reporting artifacts only and must not alter the
selected loss, checkpoint, or threshold.
