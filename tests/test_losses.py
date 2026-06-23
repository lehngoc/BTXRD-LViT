from __future__ import annotations

import unittest

import torch

from src.training.losses import BCEDiceLoss, DiceLoss, FocalBCEDiceLoss, TverskyLoss


class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        self.losses = [
            DiceLoss(),
            BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
            FocalBCEDiceLoss(),
            TverskyLoss(alpha_fp=0.7, beta_fn=0.3),
        ]
        self.tumor = torch.tensor([0])

    def test_normal_false_positive_is_penalized(self) -> None:
        targets = torch.zeros((1, 1, 8, 8))
        empty_logits = torch.full_like(targets, -12.0)
        false_positive_logits = torch.full_like(targets, 4.0)
        for criterion in self.losses:
            self.assertLess(
                criterion(empty_logits, targets, tumor=self.tumor).item(),
                criterion(false_positive_logits, targets, tumor=self.tumor).item(),
                type(criterion).__name__,
            )

    def test_tumor_correct_prediction_is_preferred_to_empty_prediction(self) -> None:
        targets = torch.zeros((1, 1, 8, 8))
        targets[:, :, 2:6, 2:6] = 1
        correct_logits = torch.full_like(targets, -8.0)
        correct_logits[:, :, 2:6, 2:6] = 8.0
        empty_logits = torch.full_like(targets, -12.0)
        tumor = torch.tensor([1])
        for criterion in self.losses:
            self.assertLess(
                criterion(correct_logits, targets, tumor=tumor).item(),
                criterion(empty_logits, targets, tumor=tumor).item(),
                type(criterion).__name__,
            )

    def test_losses_have_finite_gradients(self) -> None:
        targets = torch.zeros((2, 1, 8, 8))
        targets[1, :, 2:6, 2:6] = 1
        tumor = torch.tensor([0, 1])
        for criterion in self.losses:
            logits = torch.randn_like(targets, requires_grad=True)
            loss = criterion(logits, targets, tumor=tumor)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item(), type(criterion).__name__)
            self.assertTrue(torch.isfinite(logits.grad).all().item(), type(criterion).__name__)


if __name__ == "__main__":
    unittest.main()
