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


def resolve_path(path_value: str | Path, root_dir: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return Path(root_dir) / path


def window_starts(length: int, patch_size: int, stride: int) -> list[int]:
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
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
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
) -> tuple[np.ndarray, int]:
    original_h, original_w = image.shape[:2]
    padded_image, _ = pad_image_to_patch(image, patch_size)
    height, width = padded_image.shape[:2]
    y_starts = window_starts(height, patch_size, stride)
    x_starts = window_starts(width, patch_size, stride)
    windows = [(x, y) for y in y_starts for x in x_starts]

    prob_sum = np.zeros((height, width), dtype=np.float32)
    count_sum = np.zeros((height, width), dtype=np.float32)

    model.eval()
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        batch = [
            normalize_patch(padded_image[y : y + patch_size, x : x + patch_size])
            for x, y in batch_windows
        ]
        images = torch.stack(batch, dim=0).to(device)
        logits = model(images)
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
            for metric in ["dice", "iou", "precision", "recall", "pred_area_ratio", "target_area_ratio", "fp_image_rate"]:
                output[f"{prefix}{metric}"] = 0.0
            continue
        for metric in ["dice", "iou", "precision", "recall", "pred_area_ratio", "target_area_ratio"]:
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
    max_images: int | None = None,
    save_pred_dir: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = resolve_path(manifest_csv, root_dir)
    df = pd.read_csv(manifest_path)
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

        probability, window_count = predict_sliding_window(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            stride=stride,
            batch_size=batch_size,
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
            "merge": "average_probability",
            "threshold": float(threshold),
            "images": int(len(rows)),
            "avg_windows_per_image": float(sum(row["window_count"] for row in rows) / max(len(rows), 1)),
            "total_inference_seconds": float(elapsed),
            "seconds_per_image": float(elapsed / max(len(rows), 1)),
        }
    )
    return output
