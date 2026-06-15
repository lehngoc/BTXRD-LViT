from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data import BTXRDPatchSegmentationDataset
from src.models import UNet


FULL_VAL_HISTORY_KEYS = [
    "tumor_dice",
    "tumor_iou",
    "tumor_precision",
    "tumor_recall",
    "normal_pred_area_ratio",
    "normal_fp_image_rate",
    "avg_windows_per_image",
    "seconds_per_image",
]


def should_run_full_validation(epoch: int, total_epochs: int, interval: int) -> bool:
    return interval > 0 and (epoch == 1 or epoch % interval == 0 or epoch == total_epochs)


def get_full_val_patience(train_cfg: dict[str, Any]) -> int:
    return int(
        train_cfg.get(
            "full_val_patience",
            train_cfg.get("early_stopping_patience", 0),
        )
    )


def build_patch_dataset(
    data_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split: str,
    max_samples: int | None = None,
    include_text: bool = False,
) -> BTXRDPatchSegmentationDataset:
    return BTXRDPatchSegmentationDataset(
        csv_path=data_cfg[f"{split}_csv"],
        expected_size=train_cfg["image_size"],
        image_mean=tuple(train_cfg.get("image_mean", (0.485, 0.456, 0.406))),
        image_std=tuple(train_cfg.get("image_std", (0.229, 0.224, 0.225))),
        include_text=include_text,
        text_column=train_cfg.get("text_column", "text_lvit_prompt"),
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


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
