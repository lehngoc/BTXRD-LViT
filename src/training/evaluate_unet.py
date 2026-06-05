from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.data import BTXRDSegmentationDataset
from src.models import UNet
from src.training.metrics import SegmentationMetricAccumulator
from src.training.utils import get_device, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/train_unet_baseline.yaml")
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
    dataset = BTXRDSegmentationDataset(
        csv_path=data_cfg[f"{args.split}_csv"],
        image_size=train_cfg["image_size"],
        include_text=True,
        text_column=train_cfg.get("text_column", "text_lvit_prompt"),
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

    metrics = SegmentationMetricAccumulator(
        threshold=threshold,
        min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
    )

    for batch in tqdm(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        tumor = batch["tumor"].to(device)
        logits = model(images)
        metrics.update(logits, masks, tumor)

    result = metrics.compute()
    result["split"] = args.split
    result["checkpoint"] = str(args.checkpoint)
    result["threshold"] = threshold

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.split}_metrics.json"
    save_json(result, output_path)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
