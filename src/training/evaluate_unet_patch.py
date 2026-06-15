from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.metrics import SegmentationMetricAccumulator
from src.training.patch_pipeline import (
    build_patch_dataset,
    build_patch_loader,
    build_unet,
    load_unet_checkpoint,
)
from src.training.utils import get_device, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a UNet checkpoint on BTXRD patches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_unet_patch384.yaml")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    metric_cfg = cfg.get("metrics", {})
    threshold = args.threshold if args.threshold is not None else metric_cfg.get("threshold", 0.5)
    device = get_device(args.device)

    dataset = build_patch_dataset(data_cfg, train_cfg, args.split)
    loader = build_patch_loader(dataset, train_cfg, device, shuffle=False)

    model = build_unet(model_cfg, device)
    load_unet_checkpoint(model, args.checkpoint, device)
    model.eval()

    metrics = SegmentationMetricAccumulator(
        threshold=threshold,
        min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
    )
    for batch in tqdm(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        is_positive = batch["is_positive"].to(device)
        metrics.update(model(images), masks, is_positive)

    result = metrics.compute()
    result["split"] = args.split
    result["checkpoint"] = str(args.checkpoint)
    result["threshold"] = threshold
    result["metric_group_note"] = "tumor=positive patch; normal=negative patch"

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.split}_patch_metrics.json"
    save_json(result, output_path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
