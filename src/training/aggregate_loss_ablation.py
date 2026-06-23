from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.training.selection import challenger_eligibility


VAL_METRICS = ["tumor_dice", "tumor_iou", "tumor_precision", "tumor_recall", "normal_fp_image_rate", "normal_pred_area_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate normal-aware loss-ablation run artifacts.")
    parser.add_argument("--runs-root", required=True, help="Root containing loss/seed*/best_summary.json artifacts.")
    parser.add_argument("--stage", choices=["screening", "full"], required=True)
    parser.add_argument("--baseline-loss", default="bce_dice_05")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dice-tolerance", type=float, default=0.05)
    parser.add_argument("--recall-tolerance", type=float, default=0.10)
    return parser.parse_args()


def load_records(runs_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for summary_path in sorted(runs_root.rglob("best_summary.json")):
        run_dir = summary_path.parent
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if "selected_epoch" not in summary:
            continue
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        experiment = config.get("experiment", {})
        training = config["training"]
        metrics = summary.get("val_metrics", {})
        record: dict[str, Any] = {
            "run_dir": str(run_dir),
            "loss": experiment.get("loss_id", experiment.get("name", run_dir.parent.name)),
            "seed": int(training["seed"]),
            "selected_epoch": summary["selected_epoch"],
            "selected_threshold": summary["selected_threshold"],
            "selection_rule": summary.get("selection_rule", ""),
        }
        record.update({f"val_{name}": metrics.get(name) for name in VAL_METRICS})
        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            with test_path.open("r", encoding="utf-8") as f:
                test_metrics = json.load(f)
            record.update({f"test_{name}": test_metrics.get(name) for name in VAL_METRICS})
        records.append(record)
    return records


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no run artifacts found")
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def loss_summary(rows: list[dict[str, Any]], baseline_loss: str, dice_tolerance: float, recall_tolerance: float) -> dict[str, Any]:
    by_loss: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_loss[str(row["loss"])].append(row)

    baseline_rows = by_loss.get(baseline_loss, [])
    if not baseline_rows:
        raise ValueError(f"baseline loss '{baseline_loss}' was not found")
    baseline_means = {
        metric: mean(float(row[f"val_{metric}"]) for row in baseline_rows)
        for metric in VAL_METRICS
    }
    baseline_by_seed = {int(row["seed"]): row for row in baseline_rows}
    result: dict[str, Any] = {"baseline_loss": baseline_loss, "losses": {}}
    for loss, group in sorted(by_loss.items()):
        means = {metric: mean(float(row[f"val_{metric}"]) for row in group) for metric in VAL_METRICS}
        matched_baselines = [baseline_by_seed.get(int(row["seed"])) for row in group]
        if loss == baseline_loss:
            recall_violations = 0
        else:
            recall_violations = sum(
                baseline is not None
                and float(row["val_tumor_recall"]) < float(baseline["val_tumor_recall"]) - recall_tolerance
                for row, baseline in zip(group, matched_baselines)
            )
        eligible = (
            loss == baseline_loss
            or (
                means["tumor_dice"] >= baseline_means["tumor_dice"] - dice_tolerance
                and means["tumor_recall"] >= baseline_means["tumor_recall"] - recall_tolerance
                and recall_violations < 3
                and all(baseline is not None for baseline in matched_baselines)
            )
        )
        result["losses"][loss] = {
            "runs": len(group),
            "mean_validation": means,
            "std_validation": {
                metric: stdev(float(row[f"val_{metric}"]) for row in group) if len(group) > 1 else 0.0
                for metric in VAL_METRICS
            },
            "eligible_for_winner": eligible,
            "recall_guard_violations": recall_violations,
            "matched_baseline_seeds": sum(baseline is not None for baseline in matched_baselines),
        }
    ranked_challengers = [
        {"loss": loss, **details}
        for loss, details in result["losses"].items()
        if loss not in {baseline_loss, "dice"} and details["eligible_for_winner"]
    ]
    ranked_overall = [
        {"loss": loss, **details}
        for loss, details in result["losses"].items()
        if loss != "dice" and details["eligible_for_winner"]
    ]
    rank_key = lambda item: (
        item["mean_validation"]["normal_fp_image_rate"],
        item["mean_validation"]["normal_pred_area_ratio"],
        -item["mean_validation"]["tumor_dice"],
        -item["mean_validation"]["tumor_recall"],
    )
    ranked_challengers.sort(key=rank_key)
    ranked_overall.sort(key=rank_key)
    result["ranked_challengers"] = [item["loss"] for item in ranked_challengers]
    result["best_challenger"] = ranked_challengers[0]["loss"] if ranked_challengers else None
    result["validation_winner_overall"] = ranked_overall[0]["loss"] if ranked_overall else baseline_loss
    return result


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir) if args.output_dir else runs_root
    rows = load_records(runs_root)
    by_loss_seed = {(str(row["loss"]), int(row["seed"])): row for row in rows}

    for row in rows:
        if row["loss"] == args.baseline_loss:
            row["eligible"] = True
            row["reason"] = "baseline"
            continue
        baseline = by_loss_seed.get((args.baseline_loss, int(row["seed"])))
        if baseline is None:
            row["eligible"] = False
            row["reason"] = f"missing baseline for seed {row['seed']}"
            continue
        challenger_metrics = {name: row[f"val_{name}"] for name in VAL_METRICS}
        baseline_metrics = {name: baseline[f"val_{name}"] for name in VAL_METRICS}
        eligible, reason = challenger_eligibility(
            challenger_metrics,
            baseline_metrics,
            dice_tolerance=args.dice_tolerance,
            recall_tolerance=args.recall_tolerance,
        )
        row["eligible"] = eligible
        row["reason"] = reason
        for metric in VAL_METRICS:
            row[f"delta_vs_baseline_val_{metric}"] = float(row[f"val_{metric}"]) - float(baseline[f"val_{metric}"])
            if f"test_{metric}" in row and f"test_{metric}" in baseline:
                row[f"delta_vs_baseline_test_{metric}"] = float(row[f"test_{metric}"]) - float(baseline[f"test_{metric}"])

    summary = loss_summary(rows, args.baseline_loss, args.dice_tolerance, args.recall_tolerance)
    write_csv(rows, output_dir / f"{args.stage}_loss_ablation.csv")
    with (output_dir / f"{args.stage}_loss_ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
