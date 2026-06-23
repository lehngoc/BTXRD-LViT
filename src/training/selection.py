from __future__ import annotations

from typing import Any, Iterable


SELECTION_RULE = "normal-aware: dice-within-tolerance, then normal-fp, normal-area, dice, recall"


def select_normal_aware(
    rows: Iterable[dict[str, Any]],
    dice_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Select one threshold or checkpoint without looking at test data.

    Rows must contain tumor Dice/Recall and normal false-positive metrics.
    All rows within ``dice_tolerance`` of the best tumor Dice remain eligible;
    the remaining normal-aware tie-breakers then determine the selection.
    """

    candidates = [dict(row) for row in rows]
    if not candidates:
        raise ValueError("cannot select from an empty collection")

    max_dice = max(float(row["tumor_dice"]) for row in candidates)
    eligible = [row for row in candidates if float(row["tumor_dice"]) >= max_dice - dice_tolerance]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["normal_fp_image_rate"]),
            float(row["normal_pred_area_ratio"]),
            -float(row["tumor_dice"]),
            -float(row["tumor_recall"]),
            abs(float(row.get("threshold", 0.5)) - 0.5),
            float(row.get("threshold", 0.5)),
            int(row.get("epoch", 0)),
        ),
    )
    selected["selection_reference_best_tumor_dice"] = max_dice
    selected["selection_dice_tolerance"] = dice_tolerance
    selected["selection_rule"] = SELECTION_RULE
    return selected


def challenger_eligibility(
    challenger: dict[str, Any],
    baseline: dict[str, Any],
    dice_tolerance: float = 0.05,
    recall_tolerance: float = 0.10,
) -> tuple[bool, str]:
    dice_drop = float(baseline["tumor_dice"]) - float(challenger["tumor_dice"])
    recall_drop = float(baseline["tumor_recall"]) - float(challenger["tumor_recall"])
    reasons: list[str] = []
    if dice_drop > dice_tolerance:
        reasons.append(f"Tumor Dice drop {dice_drop:.4f} > {dice_tolerance:.4f}")
    if recall_drop > recall_tolerance:
        reasons.append(f"Tumor Recall drop {recall_drop:.4f} > {recall_tolerance:.4f}")
    return not reasons, "; ".join(reasons) if reasons else "eligible"
