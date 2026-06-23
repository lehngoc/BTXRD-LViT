from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SegmentationMetricAccumulator:
    threshold: float = 0.5
    eps: float = 1e-7
    min_fp_area_ratio: float = 0.001
    groups: dict[str, list[dict[str, float]]] = field(
        default_factory=lambda: {"all": [], "tumor": [], "normal": []}
    )

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, tumor: torch.Tensor) -> None:
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()
        targets = (targets > 0).float()
        tumor = tumor.detach().cpu().long()

        for idx in range(preds.shape[0]):
            pred = preds[idx]
            target = targets[idx]
            is_tumor = int(tumor[idx].item()) == 1

            stats = self._compute_sample_stats(pred, target, is_tumor)
            self.groups["all"].append(stats)
            self.groups["tumor" if is_tumor else "normal"].append(stats)

    def _compute_sample_stats(self, pred: torch.Tensor, target: torch.Tensor, is_tumor: bool) -> dict[str, float]:
        pred = pred.detach().float().cpu()
        target = target.detach().float().cpu()

        tp = float((pred * target).sum().item())
        fp = float((pred * (1 - target)).sum().item())
        fn = float(((1 - pred) * target).sum().item())
        pred_area = float(pred.sum().item())
        target_area = float(target.sum().item())
        total_pixels = float(pred.numel())

        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        pred_area_ratio = pred_area / total_pixels
        fp_image = 0.0 if is_tumor else float(pred_area_ratio > self.min_fp_area_ratio)

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "pred_area_ratio": pred_area_ratio,
            "target_area_ratio": target_area / total_pixels,
            "fp_image": fp_image,
        }

    def compute(self) -> dict[str, float]:
        result: dict[str, float] = {}

        for group_name, rows in self.groups.items():
            prefix = f"{group_name}_"
            result[f"{prefix}count"] = float(len(rows))

            if not rows:
                for metric in ["dice", "iou", "precision", "recall", "pred_area_ratio", "target_area_ratio", "fp_image_rate"]:
                    result[f"{prefix}{metric}"] = 0.0
                continue

            for metric in ["dice", "iou", "precision", "recall", "pred_area_ratio", "target_area_ratio"]:
                result[f"{prefix}{metric}"] = sum(row[metric] for row in rows) / len(rows)

            result[f"{prefix}fp_image_rate"] = sum(row["fp_image"] for row in rows) / len(rows)

        return result
