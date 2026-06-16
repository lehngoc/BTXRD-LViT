from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import evaluate_full_images
from src.training.metrics import SegmentationMetricAccumulator
from src.training.patch_pipeline import (
    FULL_VAL_HISTORY_KEYS,
    build_patch_dataset,
    build_patch_loader,
    get_full_val_patience,
    load_model_checkpoint,
    should_run_full_validation,
)
from src.training.train_lvit_t import build_model
from src.training.train_lvit_tw import build_criterion, build_optimizer, build_scheduler
from src.training.utils import append_history_row, get_device, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text-conditioned LViT-T on BTXRD Patch384.")
    parser.add_argument("--config", default="configs/train_lvit_t_patch384.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-full-val-images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--auto-resume", action="store_true")
    return parser.parse_args()


def run_patch_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    gradient_clip_norm: float | None = None,
    threshold: float = 0.5,
    min_fp_area_ratio: float = 0.001,
    phase: str = "train",
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
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        is_positive = batch["is_positive"].to(device)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
                logits = model(images, text=list(batch["text"]))
                loss = criterion(logits, masks, tumor=is_positive)

            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                patch_ids = [str(value) for value in batch.get("patch_id", [])]
                shown_patch_ids = ", ".join(patch_ids[:4]) if patch_ids else "unknown"
                raise RuntimeError(
                    f"Non-finite {phase} output at step {step}: "
                    f"loss={float(loss.detach().cpu())}, patch_id={shown_patch_ids}"
                )

            if train:
                scaled_loss = loss / accumulation_steps
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                if step % accumulation_steps == 0 or step == len(loader):
                    if gradient_clip_norm is not None and gradient_clip_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

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

    train_ds = build_patch_dataset(data_cfg, train_cfg, "train", args.max_train_samples, include_text=True)
    val_ds = build_patch_dataset(data_cfg, train_cfg, "val", args.max_val_samples, include_text=True)
    train_loader = build_patch_loader(train_ds, train_cfg, device, shuffle=True)
    val_loader = build_patch_loader(val_ds, train_cfg, device, shuffle=False)

    model = build_model(cfg).to(device)
    criterion = build_criterion(train_cfg)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    mixed_precision = bool(train_cfg.get("mixed_precision", False))
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")

    best_full_val_dice = -1.0
    best_epoch = 0
    start_epoch = 1
    full_val_checks_without_improvement = 0
    full_val_patience = get_full_val_patience(train_cfg)
    full_val_interval = int(train_cfg.get("full_val_interval", 5))
    history_path = output_dir / "history.csv"

    resume_path = Path(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        candidate = output_dir / "last.pt"
        if candidate.exists():
            resume_path = candidate

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {resume_path}")
        checkpoint = load_model_checkpoint(model, resume_path, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_full_val_dice = float(checkpoint.get("best_full_val_dice", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        full_val_checks_without_improvement = int(
            checkpoint.get("full_val_checks_without_improvement", 0)
        )
        print(
            f"Resumed from {resume_path} at epoch {start_epoch}. "
            f"best_full_val={best_full_val_dice:.4f}@{best_epoch}"
        )

    for epoch in range(start_epoch, train_cfg["epochs"] + 1):
        train_metrics = run_patch_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            accumulation_steps=train_cfg.get("accumulation_steps", 1),
            scaler=scaler if scaler.is_enabled() else None,
            mixed_precision=mixed_precision,
            gradient_clip_norm=train_cfg.get("gradient_clip_norm"),
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
            phase="train",
        )
        val_metrics = run_patch_epoch(
            model,
            val_loader,
            criterion,
            device,
            mixed_precision=mixed_precision,
            threshold=metric_cfg.get("threshold", 0.5),
            min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
            phase="val",
        )
        if scheduler is not None:
            scheduler.step()

        full_val_metrics = None
        if should_run_full_validation(epoch, train_cfg["epochs"], full_val_interval):
            full_val_metrics = evaluate_full_images(
                model=model,
                manifest_csv=data_cfg["val_full_csv"],
                root_dir=data_cfg.get("root_dir", "."),
                device=device,
                patch_size=sw_cfg.get("patch_size", train_cfg["image_size"]),
                stride=sw_cfg.get("stride", 192),
                batch_size=sw_cfg.get("batch_size", 1),
                threshold=metric_cfg.get("threshold", 0.5),
                min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
                merge=sw_cfg.get("merge", "average_probability"),
                max_images=args.max_full_val_images,
                text_column=train_cfg.get("text_column", "text_lvit_prompt"),
                image_mean=tuple(train_cfg.get("image_mean", (0.485, 0.456, 0.406))),
                image_std=tuple(train_cfg.get("image_std", (0.229, 0.224, 0.225))),
                mixed_precision=mixed_precision,
            )
            save_json(full_val_metrics, output_dir / f"val_sliding_epoch_{epoch:03d}.json")

        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"patch_val_{key}": value for key, value in val_metrics.items()})
        for key in FULL_VAL_HISTORY_KEYS:
            row[f"full_val_{key}"] = full_val_metrics.get(key, "") if full_val_metrics else ""
        append_history_row(history_path, row)

        full_val_dice = full_val_metrics["tumor_dice"] if full_val_metrics is not None else None
        improved = full_val_dice is not None and full_val_dice > best_full_val_dice
        if improved:
            best_full_val_dice = float(full_val_dice)
            best_epoch = epoch
            full_val_checks_without_improvement = 0
        elif full_val_metrics is not None:
            full_val_checks_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "config": cfg,
            "patch_val_metrics": val_metrics,
            "full_val_metrics": full_val_metrics,
            "best_full_val_dice": best_full_val_dice,
            "best_epoch": best_epoch,
            "full_val_checks_without_improvement": full_val_checks_without_improvement,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_full_val_tumor_dice": best_full_val_dice,
                    "selection_metric": "full-image text-conditioned sliding-window val tumor_dice",
                    "patch_val_tumor_dice_at_best": val_metrics["tumor_dice"],
                },
                output_dir / "best_summary.json",
            )

        full_val_text = "not_run"
        if full_val_metrics is not None:
            full_val_text = (
                f"{full_val_metrics['tumor_dice']:.4f} "
                f"normal_fp={full_val_metrics['normal_fp_image_rate']:.4f}"
            )
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"patch_val_positive_dice={val_metrics['tumor_dice']:.4f} "
            f"full_val_tumor_dice={full_val_text} "
            f"best_full_val={best_full_val_dice:.4f}@{best_epoch}"
        )

        if full_val_patience and full_val_checks_without_improvement >= full_val_patience:
            print(
                f"Early stopping at epoch {epoch} after "
                f"{full_val_checks_without_improvement} full-image validation checks "
                "without improvement."
            )
            break


if __name__ == "__main__":
    main()
