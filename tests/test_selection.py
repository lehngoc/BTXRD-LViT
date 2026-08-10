from __future__ import annotations

import unittest

from src.training.selection import challenger_eligibility, select_normal_aware


class SelectionTests(unittest.TestCase):
    def test_selects_lowest_normal_fp_within_dice_tolerance(self) -> None:
        selected = select_normal_aware([
            {"threshold": 0.3, "tumor_dice": 0.62, "tumor_recall": 0.70, "normal_fp_image_rate": 0.30, "normal_pred_area_ratio": 0.01},
            {"threshold": 0.6, "tumor_dice": 0.58, "tumor_recall": 0.65, "normal_fp_image_rate": 0.10, "normal_pred_area_ratio": 0.002},
            {"threshold": 0.7, "tumor_dice": 0.50, "tumor_recall": 0.50, "normal_fp_image_rate": 0.01, "normal_pred_area_ratio": 0.001},
        ])
        self.assertEqual(selected["threshold"], 0.6)

    def test_recall_guard_marks_challenger_ineligible(self) -> None:
        eligible, reason = challenger_eligibility(
            {"tumor_dice": 0.60, "tumor_recall": 0.45},
            {"tumor_dice": 0.62, "tumor_recall": 0.60},
        )
        self.assertFalse(eligible)
        self.assertIn("Recall", reason)

if __name__ == "__main__":
    unittest.main()
