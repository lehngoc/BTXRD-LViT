from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data import BTXRDSegmentationDataset
from src.models import UNet
from src.training.losses import BCEDiceLoss
from src.training.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the BTXRD model pipeline.")
    parser.add_argument("--config", default="configs/train_unet_baseline.yaml")
    parser.add_argument("--samples-per-split", type=int, default=8)
    return parser.parse_args()


def check_dataset(
    csv_path: str,
    root_dir: str,
    image_size: int,
    samples: int,
    tumor_only: bool,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
) -> None:
    dataset = BTXRDSegmentationDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        max_samples=samples,
        tumor_only=tumor_only,
    )
    assert len(dataset) > 0, f"Empty dataset: {csv_path}"

    for idx in range(min(samples, len(dataset))):
        sample = dataset[idx]
        assert tuple(sample["image"].shape) == (3, image_size, image_size)
        assert tuple(sample["mask"].shape) == (1, image_size, image_size)

        unique_mask_values = set(torch.unique(sample["mask"]).tolist())
        assert unique_mask_values.issubset({0.0, 1.0}), unique_mask_values

        if tumor_only:
            assert int(sample["tumor"].item()) == 1


def check_forward_backward(
    train_csv: str,
    root_dir: str,
    image_size: int,
    batch_size: int,
    tumor_only: bool,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
) -> None:
    dataset = BTXRDSegmentationDataset(
        csv_path=train_csv,
        root_dir=root_dir,
        image_size=image_size,
        image_mean=image_mean,
        image_std=image_std,
        max_samples=max(batch_size, 2),
        tumor_only=tumor_only,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    model = UNet(base_channels=8)
    criterion = BCEDiceLoss()
    logits = model(batch["image"])
    loss = criterion(logits, batch["mask"])
    loss.backward()

    assert logits.shape == batch["mask"].shape
    assert torch.isfinite(loss), loss


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    image_size = int(cfg["training"]["image_size"])
    image_mean = tuple(cfg["training"].get("image_mean", (0.485, 0.456, 0.406)))
    image_std = tuple(cfg["training"].get("image_std", (0.229, 0.224, 0.225)))
    root_dir = cfg["data"].get("root_dir", ".")
    tumor_only = cfg["data"].get("tumor_only", cfg["training"].get("tumor_only", False))

    for split in ["train", "val", "test"]:
        check_dataset(
            cfg["data"][f"{split}_csv"],
            root_dir,
            image_size,
            args.samples_per_split,
            tumor_only,
            image_mean,
            image_std,
        )

    check_forward_backward(
        cfg["data"]["train_csv"],
        root_dir,
        image_size,
        batch_size=min(2, cfg["training"]["batch_size"]),
        tumor_only=tumor_only,
        image_mean=image_mean,
        image_std=image_std,
    )

    print("BTXRD model pipeline smoke test passed.")


if __name__ == "__main__":
    main()
