from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.evaluation.full_image_metrics import compute_full_image_metrics, evaluate_full_image_manifest


class FullImageContractTests(unittest.TestCase):
    def test_normal_fp_uses_strictly_greater_than_threshold(self) -> None:
        probability = np.zeros((100, 100), dtype=np.float32)
        probability.flat[0] = 1.0  # exactly 1 / 10,000 = 1e-4
        metrics = compute_full_image_metrics(
            probability, np.zeros((100, 100), dtype=np.uint8), source_is_tumor=False, threshold=0.5
        )
        self.assertEqual(metrics["fp_image"], 0.0)
        probability.flat[1] = 1.0
        metrics = compute_full_image_metrics(
            probability, np.zeros((100, 100), dtype=np.uint8), source_is_tumor=False, threshold=0.5
        )
        self.assertEqual(metrics["fp_image"], 1.0)

    def test_inference_receives_only_image_before_gt_is_read_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, mask_path = root / "image.png", root / "mask.png"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
            Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(mask_path)
            manifest = root / "val.csv"
            pd.DataFrame([{"image_path": str(image_path), "mask_path": str(mask_path), "tumor": 0}]).to_csv(manifest, index=False)
            calls = []

            def prediction_function(image: np.ndarray) -> np.ndarray:
                calls.append(image.shape)
                return np.zeros(image.shape[:2], dtype=np.float32)

            result = evaluate_full_image_manifest(manifest, predict_probability=prediction_function, threshold=0.5)
            self.assertEqual(calls, [(4, 4, 3)])
            self.assertEqual(result["normal_fp_image_rate"], 0.0)
            self.assertEqual(set(result), {"tumor_count", "tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall", "normal_count", "normal_pred_area_ratio", "normal_fp_image_rate"})

    def test_empty_prediction_for_non_empty_tumor_is_zero_for_all_tumor_metrics(self) -> None:
        metrics = compute_full_image_metrics(
            np.zeros((4, 4), dtype=np.float32),
            np.pad(np.ones((1, 1), dtype=np.uint8), ((0, 3), (0, 3))),
            source_is_tumor=True,
            threshold=0.5,
        )
        for metric in ("precision", "recall", "dice", "iou"):
            self.assertEqual(metrics[metric], 0.0)

    def test_diagnosis_mask_inconsistency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tumor=0"):
            compute_full_image_metrics(
                np.zeros((2, 2), dtype=np.float32),
                np.array([[1, 0], [0, 0]], dtype=np.uint8),
                source_is_tumor=False,
                threshold=0.5,
            )
        with self.assertRaisesRegex(ValueError, "tumor=1"):
            compute_full_image_metrics(
                np.zeros((2, 2), dtype=np.float32),
                np.zeros((2, 2), dtype=np.uint8),
                source_is_tumor=True,
                threshold=0.5,
            )


if __name__ == "__main__":
    unittest.main()
