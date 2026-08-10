from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image


NORMAL_FP_AREA_RATIO_THRESHOLD = 1e-4


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def compute_full_image_metrics(
    probability: np.ndarray,
    target_mask: np.ndarray,
    *,
    source_is_tumor: bool,
    threshold: float,
    normal_fp_area_ratio_threshold: float = NORMAL_FP_AREA_RATIO_THRESHOLD,
) -> dict[str, float]:
    """Score one full-image prediction after GT-independent inference."""
    if probability.shape != target_mask.shape:
        raise ValueError("Prediction and target mask must have the same full-image shape.")
    prediction = probability >= threshold
    target = target_mask > 0
    if source_is_tumor and not target.any():
        raise ValueError("tumor=1 requires a non-empty GT mask for canonical evaluation.")
    if not source_is_tumor and target.any():
        raise ValueError("tumor=0 requires an empty GT mask for canonical evaluation.")
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, ~target).sum())
    fn = float(np.logical_and(~prediction, target).sum())
    area_ratio = float(prediction.mean())
    return {
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "pred_area_ratio": area_ratio,
        "fp_image": 0.0
        if source_is_tumor
        else float(area_ratio > normal_fp_area_ratio_threshold),
    }


@dataclass
class FullImageMetricAccumulator:
    threshold: float
    normal_fp_area_ratio_threshold: float = NORMAL_FP_AREA_RATIO_THRESHOLD
    groups: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: {"tumor": [], "normal": []}
    )

    def update(self, probability: np.ndarray, target_mask: np.ndarray, *, source_is_tumor: bool) -> None:
        row = compute_full_image_metrics(
            probability,
            target_mask,
            source_is_tumor=source_is_tumor,
            threshold=self.threshold,
            normal_fp_area_ratio_threshold=self.normal_fp_area_ratio_threshold,
        )
        self.groups["tumor" if source_is_tumor else "normal"].append(row)

    def compute(self) -> dict[str, float]:
        output: dict[str, float] = {}
        for group, rows in self.groups.items():
            output[f"{group}_count"] = float(len(rows))
            if group == "tumor":
                for metric in ("dice", "iou", "precision", "recall"):
                    output[f"tumor_{metric}"] = (
                        float(sum(row[metric] for row in rows) / len(rows)) if rows else 0.0
                    )
            else:
                output["normal_pred_area_ratio"] = (
                    float(sum(row["pred_area_ratio"] for row in rows) / len(rows)) if rows else 0.0
                )
                output["normal_fp_image_rate"] = (
                    float(sum(row["fp_image"] for row in rows) / len(rows)) if rows else 0.0
                )
        return output


def evaluate_full_image_manifest(
    manifest_path: str | Path,
    *,
    predict_probability: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    normal_fp_area_ratio_threshold: float = NORMAL_FP_AREA_RATIO_THRESHOLD,
) -> dict[str, float]:
    """Enforce the full-image contract: predict from image only, then score GT."""
    manifest = pd.read_csv(manifest_path)
    required = {"image_path", "mask_path", "tumor"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Full-image manifest is missing columns: {sorted(missing)}")
    accumulator = FullImageMetricAccumulator(
        threshold=threshold,
        normal_fp_area_ratio_threshold=normal_fp_area_ratio_threshold,
    )
    for row in manifest.itertuples(index=False):
        image = np.asarray(Image.open(row.image_path).convert("RGB"))
        probability = predict_probability(image)  # no GT, bbox, or ROI enters inference
        mask = np.asarray(Image.open(row.mask_path).convert("L"))
        accumulator.update(probability, mask, source_is_tumor=bool(int(row.tumor)))
    return accumulator.compute()
