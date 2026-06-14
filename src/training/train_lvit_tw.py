from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data import BTXRDSegmentationDataset
from src.models import LViTTW
from src.training.losses import BCEDiceLoss, LegacyWeightedDiceBCELoss
from src.training.metrics import SegmentationMetricAccumulator
from src.training.utils import append_history_row, get_device, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E2 LViT-TW no-text on BTXRD.")
    parser.add_argument("--config", default="configs/train_lvit_tw_preprocessed_224.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_dataset(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split_name: str,
    max_samples: int | None,
    include_text: bool,
) -> BTXRDSegmentationDataset:
    label_fraction = train_cfg.get("label_fraction", 1.0) if split_name == "train" else 1.0
    tumor_only_key = "train_tumor_only" if split_name == "train" else "eval_tumor_only"

    return BTXRDSegmentationDataset(
        csv_path=data_cfg[f"{split_name}_csv"],
        image_size=train_cfg["image_size"],
        image_mean=tuple(train_cfg.get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(train_cfg.get("image_std", (0.229, 0.224, 0.225))),
        include_text=include_text,
        text_column=train_cfg.get("text_column", "text_lvit_prompt"),
        max_samples=max_samples,
        label_fraction=label_fraction,
        label_seed=train_cfg.get("seed", 42),
        tumor_only=data_cfg.get(tumor_only_key, False),
        augment=split_name == "train" and train_cfg.get("augmentation", {}).get("enabled", False),
        legacy_augment=split_name == "train" and train_cfg.get("augmentation", {}).get("type", "") == "legacy_lvit",
        root_dir=data_cfg.get("root_dir", "."),
    )


def build_loader(dataset: BTXRDSegmentationDataset, cfg: dict[str, Any], train: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=train,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def build_criterion(train_cfg: dict[str, Any]) -> torch.nn.Module:
    loss_type = train_cfg.get("loss_type", "legacy_weighted_dice_bce")

    if loss_type == "bce_dice":
        return BCEDiceLoss(
            bce_weight=train_cfg.get("bce_weight", 1.0),
            dice_weight=train_cfg.get("dice_weight", 1.0),
            positive_weight=train_cfg.get("positive_weight"),
            dice_on_tumor_only=train_cfg.get("dice_on_tumor_only", False),
        )

    if loss_type == "legacy_weighted_dice_bce":
        return LegacyWeightedDiceBCELoss(
            bce_weight=train_cfg.get("bce_weight", 0.5),
            dice_weight=train_cfg.get("dice_weight", 0.5),
            foreground_weight=train_cfg.get("foreground_weight", 0.3),
            background_weight=train_cfg.get("background_weight", 0.7),
        )

    raise ValueError(f"Unsupported loss_type: {loss_type}")


def build_optimizer(model: torch.nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    lr = train_cfg["learning_rate"]
    weight_decay = train_cfg.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]):
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler_name = scheduler_cfg.get("name", "none")

    if scheduler_name in (None, "none"):
        return None

    if scheduler_name == "cosine_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get("t_0", 10),
            T_mult=scheduler_cfg.get("t_mult", 1),
            eta_min=scheduler_cfg.get("eta_min", 1e-4),
        )

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return LViTTW(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 48),
        transformer_depth=model_cfg.get("transformer_depth", 2),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        transformer_dropout=model_cfg.get("transformer_dropout", 0.0),
    )


def forward_model(model: torch.nn.Module, batch: dict[str, Any], device: torch.device, use_text: bool) -> torch.Tensor:
    images = batch["image"].to(device)
    if use_text:
        return model(images, text=list(batch["text"]))
    return model(images)


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    use_text: bool,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
    threshold: float = 0.5,
    min_fp_area_ratio: float = 0.001,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    metrics = SegmentationMetricAccumulator(threshold=threshold, min_fp_area_ratio=min_fp_area_ratio)
    total_loss = 0.0
    total_samples = 0
    accumulation_steps = max(int(accumulation_steps), 1)

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, leave=False), start=1):
        masks = batch["mask"].to(device)
        tumor = batch["tumor"].to(device)

        with torch.set_grad_enabled(train):
            logits = forward_model(model, batch, device, use_text=use_text)
            loss = criterion(logits, masks, tumor=tumor)

            if train:
                (loss / accumulation_steps).backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = masks.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        metrics.update(logits.detach(), masks.detach(), tumor.detach())

    output = metrics.compute()
    output["loss"] = total_loss / max(total_samples, 1)
    return output


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_text: bool,
    threshold: float,
    min_fp_area_ratio: float,
) -> dict[str, float]:
    model.eval()
    metrics = SegmentationMetricAccumulator(threshold=threshold, min_fp_area_ratio=min_fp_area_ratio)
    for batch in tqdm(loader, leave=False):
        masks = batch["mask"].to(device)
        tumor = batch["tumor"].to(device)
        logits = forward_model(model, batch, device, use_text=use_text)
        metrics.update(logits, masks, tumor)
    return metrics.compute()


def save_threshold_sweep(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_text: bool,
    thresholds: list[float],
    min_fp_area_ratio: float,
    output_dir: Path,
) -> None:
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = evaluate_loader(
            model,
            loader,
            device,
            use_text=use_text,
            threshold=threshold,
            min_fp_area_ratio=min_fp_area_ratio,
        )
        metrics["threshold"] = threshold
        save_json(metrics, output_dir / f"test_metrics_thr{int(threshold * 100):02d}.json")
        rows.append(
            {
                "threshold": threshold,
                "tumor_dice": metrics["tumor_dice"],
                "tumor_iou": metrics["tumor_iou"],
                "tumor_precision": metrics["tumor_precision"],
                "tumor_recall": metrics["tumor_recall"],
                "normal_count": metrics["normal_count"],
                "normal_pred_area_ratio": metrics["normal_pred_area_ratio"],
                "normal_fp_image_rate": metrics["normal_fp_image_rate"],
            }
        )

    with (output_dir / "test_threshold_sweep_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def train_from_config(
    cfg: dict[str, Any],
    args: argparse.Namespace,
    model: torch.nn.Module,
    use_text: bool,
) -> None:
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
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

    train_ds = build_dataset(data_cfg, train_cfg, "train", args.max_train_samples, include_text=use_text)
    val_ds = build_dataset(data_cfg, train_cfg, "val", args.max_val_samples, include_text=use_text)
    test_ds = build_dataset(data_cfg, train_cfg, "test", args.max_test_samples, include_text=use_text)
    train_loader = build_loader(train_ds, train_cfg, train=True)
    val_loader = build_loader(val_ds, train_cfg, train=False)
    test_loader = build_loader(test_ds, train_cfg, train=False)

    model = model.to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

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
            use_text=use_text,
            optimizer=optimizer,
            accumulation_steps=train_cfg.get("accumulation_steps", 1),
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_text=use_text,
            optimizer=None,
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
        )
        if scheduler is not None:
            scheduler.step()

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
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if improved:
            best_val_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
            save_json({"best_epoch": best_epoch, "best_val_tumor_dice": best_val_dice}, output_dir / "best_summary.json")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_tumor_dice={val_metrics['tumor_dice']:.4f} "
            f"best={best_val_dice:.4f}@{best_epoch}"
        )

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement.")
            break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    primary_threshold = metric_cfg.get("threshold", 0.5)
    min_fp_area_ratio = metric_cfg.get("min_fp_area_ratio", 0.001)
    val_metrics = evaluate_loader(model, val_loader, device, use_text, primary_threshold, min_fp_area_ratio)
    test_metrics = evaluate_loader(model, test_loader, device, use_text, primary_threshold, min_fp_area_ratio)
    val_metrics.update({"split": "val", "threshold": primary_threshold, "checkpoint": str(output_dir / "best.pt")})
    test_metrics.update({"split": "test", "threshold": primary_threshold, "checkpoint": str(output_dir / "best.pt")})
    save_json(val_metrics, output_dir / "val_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")

    thresholds = [float(value) for value in metric_cfg.get("threshold_sweep", [0.3, 0.4, 0.5, 0.6, 0.7])]
    save_threshold_sweep(model, test_loader, device, use_text, thresholds, min_fp_area_ratio, output_dir)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg, args, build_model(cfg), use_text=False)


if __name__ == "__main__":
    main()
