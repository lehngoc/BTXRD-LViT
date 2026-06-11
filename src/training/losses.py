from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, sample_mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        loss = 1 - dice

        if sample_mask is not None:
            sample_mask = sample_mask.to(device=loss.device, dtype=torch.bool).flatten()
            if not sample_mask.any():
                return loss.new_tensor(0.0)
            loss = loss[sample_mask]

        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        positive_weight: float | None = None,
        dice_on_tumor_only: bool = False,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.positive_weight = positive_weight
        self.dice_on_tumor_only = dice_on_tumor_only

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tumor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_weight = None
        if self.positive_weight is not None:
            pos_weight = logits.new_tensor(float(self.positive_weight))

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        dice_sample_mask = None

        if self.dice_on_tumor_only:
            if tumor is None:
                raise ValueError("tumor labels are required when dice_on_tumor_only=True")
            dice_sample_mask = tumor.to(device=logits.device) == 1

        dice_loss = self.dice(logits, targets, sample_mask=dice_sample_mask)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class LegacyWeightedDiceBCELoss(nn.Module):
    """Probability-space weighted Dice+BCE loss used for UNet-Strong comparisons."""

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        foreground_weight: float = 0.3,
        background_weight: float = 0.7,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.smooth = smooth

    def _weighted_bce(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = probs.flatten()
        targets = targets.flatten()
        loss = F.binary_cross_entropy(probs, targets, reduction="none")
        pos = (targets > 0.5).float()
        neg = (targets < 0.5).float()
        pos_count = pos.sum().clamp_min(1e-12)
        neg_count = neg.sum().clamp_min(1e-12)

        return (
            self.foreground_weight * pos * loss / pos_count
            + self.background_weight * neg * loss / neg_count
        ).sum()

    def _weighted_dice(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.shape[0]
        probs = probs.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)
        weights = targets.detach() * (self.background_weight - self.foreground_weight) + self.foreground_weight
        probs = weights * probs
        targets = weights * targets

        intersection = (probs * targets).sum(dim=1)
        denominator = (probs * probs).sum(dim=1) + (targets * targets).sum(dim=1)
        dice_loss = 1 - (2 * intersection + self.smooth) / (denominator + self.smooth)

        return dice_loss.view(batch_size).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        tumor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del tumor
        probs = torch.sigmoid(logits).clamp(min=1e-6, max=1 - 1e-6)
        bce = self._weighted_bce(probs, targets)
        dice = self._weighted_dice(probs, targets)

        return self.bce_weight * bce + self.dice_weight * dice
