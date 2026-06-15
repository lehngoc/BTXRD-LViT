from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_TENSOR = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD_TENSOR = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
AVERAGE_PROBABILITY = "average_probability"
GROUP_METRIC_NAMES = [
    "dice",
    "iou",
    "precision",
    "recall",
    "pred_area_ratio",
    "target_area_ratio",
]


def resolve_path(path_value: str | Path, root_dir: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def window_starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def normalize_patch(
    patch: np.ndarray,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    patch = patch.astype(np.float32) / 255.0
    tensor = torch.from_numpy(patch).permute(2, 0, 1)
    mean_tensor = (
        IMAGENET_MEAN_TENSOR
        if mean == IMAGENET_MEAN
        else torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    )
    std_tensor = (
        IMAGENET_STD_TENSOR
        if std == IMAGENET_STD
        else torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    )
    return (tensor - mean_tensor) / std_tensor


def pad_image_to_patch(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)

    mode = "reflect" if height > 1 and width > 1 else "edge"
    if image.ndim == 3:
        padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    else:
        padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    return padded, (pad_h, pad_w)


@torch.no_grad()
def predict_sliding_window(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 384,
    stride: int = 192,
    batch_size: int = 4,
    merge: str = AVERAGE_PROBABILITY,
    text_prompt: str | None = None,
    image_mean: tuple[float, float, float] = IMAGENET_MEAN,
    image_std: tuple[float, float, float] = IMAGENET_STD,
    mixed_precision: bool = False,
) -> tuple[np.ndarray, int]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if merge != AVERAGE_PROBABILITY:
        raise ValueError(
            f"Unsupported sliding-window merge method: {merge}. "
            f"Only {AVERAGE_PROBABILITY} is implemented."
        )

    original_h, original_w = image.shape[:2]
    padded_image, _ = pad_image_to_patch(image, patch_size)
    height, width = padded_image.shape[:2]
    y_starts = window_starts(height, patch_size, stride)
    x_starts = window_starts(width, patch_size, stride)
    windows = [(x, y) for y in y_starts for x in x_starts]

    prob_sum = np.zeros((height, width), dtype=np.float32)
    count_sum = np.zeros((height, width), dtype=np.float32)
    text_features: torch.Tensor | None = None

    model.eval()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch = [
            normalize_patch(
                padded_image[y : y + patch_size, x : x + patch_size],
                mean=image_mean,
                std=image_std,
            )
            for x, y in batch_windows
        ]
        images = torch.stack(batch, dim=0).to(device)
        with torch.autocast(device_type=device.type, enabled=mixed_precision and device.type == "cuda"):
            if text_prompt is None:
                logits = model(images)
            else:
                if text_features is None:
                    if not hasattr(model, "prepare_text_features"):
                        raise TypeError("Text-conditioned sliding-window inference requires prepare_text_features().")
                    text_features = model.prepare_text_features([text_prompt], images[:1])
                logits = model(images, text=text_features.expand(images.shape[0], -1, -1))
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        for prob, (x, y) in zip(probs, batch_windows):
            prob_sum[y : y + patch_size, x : x + patch_size] += prob
            count_sum[y : y + patch_size, x : x + patch_size] += 1.0

    merged = prob_sum / np.maximum(count_sum, 1e-6)
    return merged[:original_h, :original_w], len(windows)


def compute_sample_metrics(
    probability: np.ndarray,
    target_mask: np.ndarray,
    is_tumor: bool,
    threshold: float,
    min_fp_area_ratio: float,
    eps: float = 1e-7,
) -> dict[str, float]:
    pred = probability >= threshold
    target = target_mask > 0
    tp = float(np.logical_and(pred, target).sum())
    fp = float(np.logical_and(pred, ~target).sum())
    fn = float(np.logical_and(~pred, target).sum())
    pred_area = float(pred.sum())
    target_area = float(target.sum())
    total_pixels = float(pred.size)

    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "pred_area_ratio": pred_area / total_pixels,
        "target_area_ratio": target_area / total_pixels,
        "fp_image": 0.0 if is_tumor else float((pred_area / total_pixels) >= min_fp_area_ratio),
    }


def aggregate_group_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    groups = {"all": [], "tumor": [], "normal": []}
    for row in rows:
        groups["all"].append(row)
        groups["tumor" if row["is_tumor"] else "normal"].append(row)

    output: dict[str, float] = {}
    for group_name, group_rows in groups.items():
        prefix = f"{group_name}_"
        output[f"{prefix}count"] = float(len(group_rows))
        if not group_rows:
            for metric in [*GROUP_METRIC_NAMES, "fp_image_rate"]:
                output[f"{prefix}{metric}"] = 0.0
            continue
        for metric in GROUP_METRIC_NAMES:
            output[f"{prefix}{metric}"] = float(sum(row[metric] for row in group_rows) / len(group_rows))
        output[f"{prefix}fp_image_rate"] = float(sum(row["fp_image"] for row in group_rows) / len(group_rows))
    return output


@torch.no_grad()
def evaluate_full_images(
    model: torch.nn.Module,
    manifest_csv: str | Path,
    root_dir: str | Path,
    device: torch.device,
    patch_size: int,
    stride: int,
    batch_size: int,
    threshold: float,
    min_fp_area_ratio: float,
    merge: str = AVERAGE_PROBABILITY,
    max_images: int | None = None,
    save_pred_dir: str | Path | None = None,
    text_column: str | None = None,
    image_mean: tuple[float, float, float] = IMAGENET_MEAN,
    image_std: tuple[float, float, float] = IMAGENET_STD,
    mixed_precision: bool = False,
) -> dict[str, Any]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    if not 0.0 <= min_fp_area_ratio <= 1.0:
        raise ValueError(
            f"min_fp_area_ratio must be between 0 and 1, got {min_fp_area_ratio}"
        )
    if max_images is not None and max_images < 0:
        raise ValueError(f"max_images must be non-negative, got {max_images}")

    manifest_path = resolve_path(manifest_csv, root_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing full-image manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    if text_column is not None and text_column not in df.columns:
        raise ValueError(f"Missing text column in {manifest_path}: {text_column}")
    if max_images is not None:
        df = df.head(max_images)

    pred_dir = Path(save_pred_dir) if save_pred_dir else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    for item in tqdm(df.itertuples(index=False), total=len(df), leave=False):
        image_path = resolve_path(item.image_path, root_dir)
        mask_path = resolve_path(item.mask_path, root_dir)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if image.shape[:2] != mask.shape[:2]:
            mask_tensor = torch.from_numpy(mask[None, None].astype(np.float32))
            mask = (
                F.interpolate(mask_tensor, size=image.shape[:2], mode="nearest")
                .numpy()[0, 0]
                .astype(np.uint8)
            )

        text_prompt = None
        if text_column is not None:
            text_value = getattr(item, text_column)
            text_prompt = "" if pd.isna(text_value) else str(text_value)

        probability, window_count = predict_sliding_window(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
            merge=merge,
            text_prompt=text_prompt,
            image_mean=image_mean,
            image_std=image_std,
            mixed_precision=mixed_precision,
        )
        is_tumor = int(item.tumor) == 1
        metrics = compute_sample_metrics(
            probability=probability,
            target_mask=mask,
            is_tumor=is_tumor,
            threshold=threshold,
            min_fp_area_ratio=min_fp_area_ratio,
        )
        metrics.update(
            {
                "image_id": str(item.image_id),
                "is_tumor": is_tumor,
                "window_count": int(window_count),
                "image_h": int(image.shape[0]),
                "image_w": int(image.shape[1]),
            }
        )
        rows.append(metrics)

        if pred_dir is not None:
            pred_mask = (probability >= threshold).astype(np.uint8) * 255
            Image.fromarray(pred_mask).save(pred_dir / f"{Path(str(item.image_id)).stem}_pred.png")

    elapsed = time.perf_counter() - start_time
    output = aggregate_group_metrics(rows)
    output.update(
        {
            "manifest": str(manifest_path),
            "patch_size": int(patch_size),
            "stride": int(stride),
            "overlap_ratio": float(max(0.0, 1.0 - stride / patch_size)),
            "merge": merge,
            "padding": "reflect_bottom_right_for_images_smaller_than_patch",
            "inference_batch_size": int(batch_size),
            "threshold": float(threshold),
            "post_processing": "none",
            "text_column": text_column,
            "text_conditioning": text_column is not None,
            "mixed_precision": bool(mixed_precision and device.type == "cuda"),
            "input_normalization": {
                "mean": list(image_mean),
                "std": list(image_std),
            },
            "images": int(len(rows)),
            "avg_windows_per_image": float(sum(row["window_count"] for row in rows) / max(len(rows), 1)),
            "total_inference_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(len(rows), 1)),
        }
    )
    return output
