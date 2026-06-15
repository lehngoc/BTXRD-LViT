from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data import BTXRDPatchSegmentationDataset
from src.models import UNet


def build_patch_dataset(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split: str,
    max_samples: int | None = None,
) -> BTXRDPatchSegmentationDataset:
    return BTXRDPatchSegmentationDataset(
        csv_path=data_cfg[f"{split}_csv"],
        expected_size=train_cfg["image_size"],
        include_text=False,
        max_samples=max_samples,
        root_dir=data_cfg.get("root_dir", "."),
    )


def build_patch_loader(
    dataset: BTXRDPatchSegmentationDataset,
    train_cfg: dict[str, Any],
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=shuffle,
        num_workers=train_cfg.get("num_workers", 0),
        pin_memory=device.type == "cuda",
    )


def build_unet(model_cfg: dict[str, Any], device: torch.device) -> UNet:
    model_name = str(model_cfg.get("name", "unet")).lower()
    if model_name != "unet":
        raise ValueError(f"Unsupported model for the patch pipeline: {model_name}")

    return UNet(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 32),
    ).to(device)


def load_unet_checkpoint(
    model: UNet,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
