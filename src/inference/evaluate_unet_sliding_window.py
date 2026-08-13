from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import evaluate_full_images
from src.training.patch_pipeline import build_unet, load_unet_checkpoint
from src.training.utils import get_device, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UNet with full-image sliding-window inference.")
    parser.add_argument("--config", default="configs/train_unet_patch384.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--save-pred-dir", default=None)
    parser.add_argument("--save-probability-dir", default=None)
    parser.add_argument("--save-overlap-stats-dir", default=None)
    parser.add_argument("--save-entropy-dir", default=None)
    parser.add_argument("--save-disagreement-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    metric_cfg = cfg.get("metrics", {})
    sw_cfg = cfg.get("sliding_window", {})
    device = get_device(args.device)

    model = build_unet(model_cfg, device)
    load_unet_checkpoint(model, args.checkpoint, device)
    model.eval()

    manifest_key = f"{args.split}_full_csv"
    manifest_csv = data_cfg.get(manifest_key, data_cfg.get(f"{args.split}_csv"))
    result = evaluate_full_images(
        model=model,
        manifest_csv=manifest_csv,
        root_dir=data_cfg.get("root_dir", "."),
        device=device,
        patch_size=args.patch_size or sw_cfg.get("patch_size", train_cfg["image_size"]),
        stride=args.stride or sw_cfg.get("stride", 192),
        batch_size=sw_cfg.get("batch_size", train_cfg["batch_size"]),
        threshold=args.threshold if args.threshold is not None else metric_cfg.get("threshold", 0.5),
        min_fp_area_ratio=metric_cfg.get("min_fp_area_ratio", 0.001),
        merge=sw_cfg.get("merge", "average_probability"),
        max_images=args.max_images,
        save_pred_dir=args.save_pred_dir,
        save_probability_dir=args.save_probability_dir,
        save_overlap_stats_dir=args.save_overlap_stats_dir,
        save_entropy_dir=args.save_entropy_dir,
        save_disagreement_dir=args.save_disagreement_dir,
    )
    result["split"] = args.split
    result["checkpoint"] = str(args.checkpoint)

    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent / f"{args.split}_sliding_metrics.json"
    save_json(result, output_path)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
