from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data import BTXRDSegmentationDataset
from src.training.evaluate_unet import selected_threshold_from_summary
from src.models import UNet
from src.training.utils import get_device, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize UNet predictions for BTXRD.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_unet_baseline.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tumor", type=int, default=16)
    parser.add_argument("--max-normal", type=int, default=16)
    return parser.parse_args()


def unnormalize_image(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def sample_stats(prob: torch.Tensor, mask: torch.Tensor, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).float()
    mask = (mask > 0).float()

    tp = float((pred * mask).sum().item())
    fp = float((pred * (1 - mask)).sum().item())
    fn = float(((1 - pred) * mask).sum().item())
    eps = 1e-7

    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "pred_area_ratio": float(pred.mean().item()),
        "target_area_ratio": float(mask.mean().item()),
    }


def make_overlay(image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    gt_color = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    pred_color = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    gt = mask > 0
    pr = pred > 0
    overlay[gt] = 0.55 * overlay[gt] + 0.45 * gt_color
    overlay[pr] = 0.55 * overlay[pr] + 0.45 * pred_color

    return overlay.clip(0, 1)


def save_grid(rows: list[dict], output_path: Path, title: str) -> None:
    if not rows:
        print(f"No rows to save for {title}")
        return

    ncols = 4
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes_arr = np.array(axes).reshape(-1)

    for ax, row in zip(axes_arr, rows):
        ax.imshow(row["overlay"])
        ax.set_title(
            f"{row['image_id']}\n"
            f"dice={row['dice']:.3f} pred={row['pred_area_ratio']:.3f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes_arr[len(rows):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    metric_cfg = cfg.get("metrics", {})
    saved_threshold = selected_threshold_from_summary(args.checkpoint)
    threshold = args.threshold if args.threshold is not None else (
        saved_threshold if saved_threshold is not None else metric_cfg.get("threshold", 0.5)
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / "visual_checks"

    device = get_device(args.device)
    dataset = BTXRDSegmentationDataset(
        csv_path=data_cfg[f"{args.split}_csv"],
        image_size=train_cfg["image_size"],
        image_mean=tuple(train_cfg.get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(train_cfg.get("image_std", (0.229, 0.224, 0.225))),
        include_text=True,
        text_column=train_cfg.get("text_column", "text_lvit_prompt"),
        max_samples=args.max_samples,
        tumor_only=data_cfg.get("tumor_only", False),
        root_dir=data_cfg.get("root_dir", "."),
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tumor_rows = []
    normal_rows = []
    mean = dataset.image_mean
    std = dataset.image_std

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)

        for idx in range(images.shape[0]):
            mask = masks[idx, 0].detach().cpu()
            prob = probs[idx, 0].detach().cpu()
            pred = (prob >= threshold).float()
            stats = sample_stats(prob, mask, threshold)
            image = unnormalize_image(batch["image"][idx], mean, std)
            overlay = make_overlay(image, mask.numpy(), pred.numpy())

            row = {
                "image_id": batch["image_id"][idx],
                "tumor": int(batch["tumor"][idx].item()),
                "overlay": overlay,
                **stats,
            }

            if row["tumor"] == 1:
                tumor_rows.append(row)
            else:
                normal_rows.append(row)

    tumor_rows = sorted(tumor_rows, key=lambda row: row["dice"])
    normal_rows = sorted(normal_rows, key=lambda row: row["pred_area_ratio"], reverse=True)

    selected_tumor = []
    if tumor_rows:
        k = min(args.max_tumor // 3, len(tumor_rows))
        selected_tumor.extend(tumor_rows[:k])
        mid_start = max((len(tumor_rows) - k) // 2, 0)
        selected_tumor.extend(tumor_rows[mid_start:mid_start + k])
        selected_tumor.extend(tumor_rows[-k:])
        selected_tumor = selected_tumor[:args.max_tumor]

    selected_normal = normal_rows[:args.max_normal]

    save_grid(selected_tumor, output_dir / f"{args.split}_tumor_predictions.png", f"{args.split} tumor predictions")
    save_grid(selected_normal, output_dir / f"{args.split}_normal_false_positives.png", f"{args.split} normal predictions")
    print(f"Saved visual checks to {output_dir}")


if __name__ == "__main__":
    main()
