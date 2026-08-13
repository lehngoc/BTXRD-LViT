from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import (
    AVERAGE_PROBABILITY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    pad_image_to_patch,
    predict_sliding_window_stats,
    window_starts,
)


class PatchMeanModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0:1] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
        prob = red.mean(dim=(2, 3), keepdim=True).clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(prob).expand(-1, -1, images.shape[2], images.shape[3])


def build_image(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    red = ((xx * 3 + yy * 5) % 200 + 20).astype(np.uint8)
    green = ((xx * 7) % 200 + 20).astype(np.uint8)
    blue = ((yy * 11) % 200 + 20).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def expected_patch_mean_merge(
    image: np.ndarray,
    patch_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original_h, original_w = image.shape[:2]
    padded, _ = pad_image_to_patch(image, patch_size)
    height, width = padded.shape[:2]
    prob_sum = np.zeros((height, width), dtype=np.float32)
    prob_sq_sum = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    for y in window_starts(height, patch_size, stride):
        for x in window_starts(width, patch_size, stride):
            patch_prob = float(padded[y : y + patch_size, x : x + patch_size, 0].mean() / 255.0)
            prob_sum[y : y + patch_size, x : x + patch_size] += patch_prob
            prob_sq_sum[y : y + patch_size, x : x + patch_size] += patch_prob * patch_prob
            count[y : y + patch_size, x : x + patch_size] += 1.0

    probability = prob_sum / np.maximum(count, 1e-6)
    variance = np.maximum(prob_sq_sum / np.maximum(count, 1e-6) - probability * probability, 0.0)
    return (
        probability[:original_h, :original_w],
        count[:original_h, :original_w],
        variance[:original_h, :original_w],
    )


def check_case(height: int, width: int, patch_size: int = 384, stride: int = 192) -> None:
    image = build_image(height, width)
    model = PatchMeanModel()
    stats = predict_sliding_window_stats(
        model=model,
        image=image,
        device=torch.device("cpu"),
        patch_size=patch_size,
        stride=stride,
        batch_size=3,
        merge=AVERAGE_PROBABILITY,
    )
    expected_prob, expected_count, expected_var = expected_patch_mean_merge(image, patch_size, stride)

    assert stats.probability.shape == (height, width)
    assert stats.count.shape == (height, width)
    assert float(stats.count.min()) >= 1.0
    assert np.allclose(stats.probability, expected_prob, atol=1e-6)
    assert np.array_equal(stats.count, expected_count)
    assert np.allclose(stats.variance, expected_var, atol=1e-6)


def main() -> None:
    check_case(640, 704)
    check_case(200, 300)
    check_case(384, 384)
    print("Sliding-window synthetic reconstruction test passed.")


if __name__ == "__main__":
    main()
