from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import window_starts
from src.training.losses import BCEDiceLoss
from src.training.patch_pipeline import build_patch_dataset, build_patch_loader, build_unet
from src.training.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the BTXRD patch UNet pipeline.")
    parser.add_argument("--config", default="configs/train_unet_patch384.yaml")
    parser.add_argument("--samples-per-split", type=int, default=8)
    return parser.parse_args()


def check_dataset(data_cfg: dict, train_cfg: dict, split: str, samples: int) -> None:
    dataset = build_patch_dataset(data_cfg, train_cfg, split, samples)
    assert len(dataset) > 0, f"Empty dataset: {data_cfg[f'{split}_csv']}"
    image_size = int(train_cfg["image_size"])
    for idx in range(min(samples, len(dataset))):
        sample = dataset[idx]
        assert tuple(sample["image"].shape) == (3, image_size, image_size)
        assert tuple(sample["mask"].shape) == (1, image_size, image_size)
        assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})
        assert int(sample["tumor"].item()) == int(sample["is_positive"].item())
        assert "text" not in sample


def check_forward_backward(data_cfg: dict, train_cfg: dict) -> None:
    dataset = build_patch_dataset(data_cfg, train_cfg, "train", max_samples=2)
    batch = next(
        iter(build_patch_loader(dataset, train_cfg, torch.device("cpu"), shuffle=False))
    )
    model = build_unet(
        {"name": "unet", "in_channels": 3, "out_channels": 1, "base_channels": 8},
        torch.device("cpu"),
    )
    logits = model(batch["image"])
    loss = BCEDiceLoss()(logits, batch["mask"], tumor=batch["is_positive"])
    loss.backward()
    assert logits.shape == batch["mask"].shape
    assert torch.isfinite(loss)


def check_sliding_window_grid() -> None:
    assert window_starts(384, patch_size=384, stride=192) == [0]
    assert window_starts(500, patch_size=384, stride=192) == [0, 116]
    assert window_starts(768, patch_size=384, stride=192) == [0, 192, 384]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    for split in ["train", "val", "test"]:
        check_dataset(cfg["data"], cfg["training"], split, args.samples_per_split)
    check_forward_backward(cfg["data"], cfg["training"])
    check_sliding_window_grid()
    print("BTXRD patch UNet pipeline smoke test passed.")


if __name__ == "__main__":
    main()
