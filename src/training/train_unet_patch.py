from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data import BTXRDPatchSegmentationDataset
from src.models import UNet
from src.training.losses import BCEDiceLoss
from src.training.metrics import SegmentationMetricAccumulator
from src.training.utils import append_history_row, get_device, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet on pre-generated BTXRD patches.")
    parser.add_argument("--config", default="configs/train_unet_patch384.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_dataset(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split_name: str,
    max_samples: int | None,
) -> BTXRDPatchSegmentationDataset:
    return BTXRDPatchSegmentationDataset(
        csv_path=data_cfg[f"{split_name}_csv"],
        expected_size=train_cfg["image_size"],
        include_text=True,
        text_column=train_cfg.get("text_column", "text_lvit_prompt"),
        max_samples=max_samples,
        root_dir=data_cfg.get("root_dir", "."),
    )


def build_loader(dataset: BTXRDPatchSegmentationDataset, cfg: dict[str, Any], train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    threshold: float = 0.5,
    min_fp_area_ratio: float = 0.001,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    metrics = SegmentationMetricAccumulator(threshold=threshold, min_fp_area_ratio=min_fp_area_ratio)
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        is_positive = batch["is_positive"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, masks, tumor=is_positive)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        metrics.update(logits.detach(), masks.detach(), is_positive.detach())

    output = metrics.compute()
    output["loss"] = total_loss / max(total_samples, 1)
    return output


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    metric_cfg = cfg.get("metrics", {})

    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.device is not None:
        train_cfg["device"] = args.device

    set_seed(train_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, output_dir / "config.json")

    train_ds = build_dataset(data_cfg, train_cfg, "train", args.max_train_samples)
    val_ds = build_dataset(data_cfg, train_cfg, "val", args.max_val_samples)
    train_loader = build_loader(train_ds, train_cfg, train=True)
    val_loader = build_loader(val_ds, train_cfg, train=False)

    model = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)
    criterion = BCEDiceLoss(
        bce_weight=train_cfg.get("bce_weight", 1.0),
        dice_weight=train_cfg.get("dice_weight", 1.0),
        positive_weight=train_cfg.get("positive_weight"),
        dice_on_tumor_only=train_cfg.get("dice_on_positive_patches_only", True),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    best_val_dice = -1.0
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    history_path = output_dir / "history.csv"

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
        )

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        append_history_row(history_path, row)

        val_dice = val_metrics["tumor_dice"]
        improved = val_dice > best_val_dice
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
            save_json(
                {"best_epoch": best_epoch, "best_val_positive_patch_dice": best_val_dice},
                output_dir / "best_summary.json",
            )
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_positive_patch_dice={val_metrics['tumor_dice']:.4f} "
            f"val_negative_patch_fp_rate={val_metrics['normal_fp_image_rate']:.4f} "
            f"best={best_val_dice:.4f}@{best_epoch}"
        )

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
