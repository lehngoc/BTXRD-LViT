from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import (
    aggregate_group_metrics,
    compute_sample_metrics,
    resolve_path,
)


def parse_thresholds(value: str) -> list[float]:
    if value:
        return [float(item) for item in value.split(",")]
    return [round(0.30 + 0.05 * idx, 2) for idx in range(13)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds over saved full-image probability maps.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root-dir", default=".")
    parser.add_argument("--probability-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-image-output", default=None)
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--min-fp-area-ratio", type=float, default=0.001)
    return parser.parse_args()


def load_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    if mask.shape != target_shape:
        mask_tensor = torch.from_numpy(mask[None, None].astype(np.float32))
        mask = (
            F.interpolate(mask_tensor, size=target_shape, mode="nearest")
            .numpy()[0, 0]
            .astype(np.uint8)
        )
    return mask


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(resolve_path(args.manifest, args.root_dir))
    probability_dir = Path(args.probability_dir)
    thresholds = parse_thresholds(args.thresholds)
    rows_by_threshold: list[dict[str, float | str]] = []
    per_image_rows: list[dict[str, float | str | int]] = []

    cache: list[tuple[str, bool, np.ndarray, np.ndarray]] = []
    for item in manifest.itertuples(index=False):
        image_id = str(item.image_id)
        probability_path = probability_dir / f"{Path(image_id).stem}.npy"
        if not probability_path.exists():
            raise FileNotFoundError(f"Missing probability map: {probability_path}")
        probability = np.load(probability_path).astype(np.float32)
        mask = load_mask(resolve_path(item.mask_path, args.root_dir), probability.shape)
        cache.append((image_id, int(item.tumor) == 1, probability, mask))

    for threshold in thresholds:
        sample_rows = []
        for image_id, is_tumor, probability, mask in cache:
            metrics = compute_sample_metrics(
                probability=probability,
                target_mask=mask,
                is_tumor=is_tumor,
                threshold=threshold,
                min_fp_area_ratio=args.min_fp_area_ratio,
            )
            row = {
                "image_id": image_id,
                "is_tumor": int(is_tumor),
                "threshold": threshold,
                **metrics,
            }
            sample_rows.append(row)
            per_image_rows.append(row)

        summary = aggregate_group_metrics(sample_rows)
        tumor_dice_values = [row["dice"] for row in sample_rows if row["is_tumor"]]
        normal_area_values = [row["pred_area_ratio"] for row in sample_rows if not row["is_tumor"]]
        summary.update(
            {
                "threshold": threshold,
                "tumor_median_dice": float(np.median(tumor_dice_values)) if tumor_dice_values else 0.0,
                "tumor_dice_zero_cases": int(sum(float(value) <= 1e-7 for value in tumor_dice_values)),
                "normal_pred_area_ratio_median": float(np.median(normal_area_values)) if normal_area_values else 0.0,
                "normal_pred_area_ratio_max": float(np.max(normal_area_values)) if normal_area_values else 0.0,
            }
        )
        rows_by_threshold.append(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_by_threshold).to_csv(output_path, index=False)
    if args.per_image_output:
        per_image_path = Path(args.per_image_output)
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_image_rows).to_csv(per_image_path, index=False)
    print(f"Saved threshold sweep to {output_path}")


if __name__ == "__main__":
    main()
