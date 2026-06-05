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
