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
from src.inference.sliding_window import evaluate_full_images
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
    parser.add_argument("--max-full-val-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from, usually experiments/.../last.pt.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from output_dir/last.pt if it exists.",
    )
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
    sw_cfg = cfg.get("sliding_window", {})

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

    best_full_val_dice = -1.0
    best_epoch = 0
    start_epoch = 1
    patience = train_cfg.get("early_stopping_patience", 0)
    epochs_without_improvement = 0
    history_path = output_dir / "history.csv"
    full_val_interval = int(train_cfg.get("full_val_interval", 5))

    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        candidate = output_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_full_val_dice = float(checkpoint.get("best_full_val_dice", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(
            f"Resumed from {resume_path} at epoch {start_epoch}. "
            f"best_full_val={best_full_val_dice:.4f}@{best_epoch}"
        )

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
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

        full_val_metrics = None
        should_run_full_val = full_val_interval > 0 and (
            epoch == 1 or epoch % full_val_interval == 0 or epoch == train_cfg["epochs"]
        )
        if should_run_full_val:
            full_val_metrics = evaluate_full_images(
                model=model,
                manifest_csv=data_cfg.get("val_full_csv", data_cfg["val_csv"]),
                root_dir=data_cfg.get("root_dir", "."),
                device=device,
                patch_size=sw_cfg.get("patch_size", train_cfg["image_size"]),
                stride=sw_cfg.get("stride", 192),
                batch_size=sw_cfg.get("batch_size", train_cfg["batch_size"]),
                threshold=metric_cfg.get("threshold", 0.5),
                min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
                max_images=args.max_full_val_images,
            )
            save_json(full_val_metrics, output_dir / f"val_sliding_epoch_{epoch:03d}.json")

        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"patch_val_{key}": value for key, value in val_metrics.items()})
        full_val_history_keys = [
            "tumor_dice",
            "tumor_iou",
            "tumor_precision",
            "tumor_recall",
            "normal_pred_area_ratio",
            "normal_fp_image_rate",
            "avg_windows_per_image",
            "seconds_per_image",
        ]
        for key in full_val_history_keys:
            row[f"full_val_{key}"] = (
                full_val_metrics[key]
                if full_val_metrics is not None and key in full_val_metrics
                else ""
            )
        append_history_row(history_path, row)

        full_val_dice = full_val_metrics["tumor_dice"] if full_val_metrics is not None else None
        improved = full_val_dice is not None and full_val_dice > best_full_val_dice
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "patch_val_metrics": val_metrics,
            "full_val_metrics": full_val_metrics,
            "best_full_val_dice": best_full_val_dice,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }

        if improved:
            best_full_val_dice = float(full_val_dice)
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint["best_full_val_dice"] = best_full_val_dice
            checkpoint["best_epoch"] = best_epoch
            checkpoint["epochs_without_improvement"] = epochs_without_improvement
            torch.save(checkpoint, output_dir / "best.pt")
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_full_val_tumor_dice": best_full_val_dice,
                    "selection_metric": "full-image sliding-window val tumor_dice",
                    "patch_val_tumor_dice_at_best": val_metrics["tumor_dice"],
                },
                output_dir / "best_summary.json",
            )
        elif full_val_metrics is not None:
            epochs_without_improvement += 1
            checkpoint["epochs_without_improvement"] = epochs_without_improvement

        torch.save(checkpoint, output_dir / "last.pt")

        full_val_text = "not_run"
        if full_val_metrics is not None:
            full_val_text = (
                f"{full_val_metrics['tumor_dice']:.4f} "
                f"normal_fp={full_val_metrics['normal_fp_image_rate']:.4f}"
            )
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"patch_val_positive_dice={val_metrics['tumor_dice']:.4f} "
            f"patch_val_negative_fp={val_metrics['normal_fp_image_rate']:.4f} "
            f"full_val_tumor_dice={full_val_text} "
            f"best_full_val={best_full_val_dice:.4f}@{best_epoch}"
        )

        if patience and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
