from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_SPLITS = {
    "train": {
        "metadata": Path("data/exports/patches_jitter/patches_384/metadata.csv"),
        "manifest": Path("data/exports/btxrd_preprocessed/train.csv"),
    },
    "val": {
        "metadata": Path("data/exports/patches_jitter_val/patches_384/metadata.csv"),
        "manifest": Path("data/exports/btxrd_preprocessed/val.csv"),
    },
    "test": {
        "metadata": Path("data/exports/patches_jitter_test/patches_384/metadata.csv"),
        "manifest": Path("data/exports/btxrd_preprocessed/test.csv"),
    },
}
DEFAULT_OUTPUT_DIR = Path("data/processed/reports/patch384_dataset_analysis")
REQUIRED_COLUMNS = [
    "patch_id",
    "patch_path",
    "mask_path",
    "image_id",
    "tumor",
    "is_positive",
    "patch_kind",
    "mask_coverage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and validate the generated BTXRD Patch384 dataset."
    )
    parser.add_argument("--train-metadata", type=Path, default=DEFAULT_SPLITS["train"]["metadata"])
    parser.add_argument("--val-metadata", type=Path, default=DEFAULT_SPLITS["val"]["metadata"])
    parser.add_argument("--test-metadata", type=Path, default=DEFAULT_SPLITS["test"]["metadata"])
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_SPLITS["train"]["manifest"])
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_SPLITS["val"]["manifest"])
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_SPLITS["test"]["manifest"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def numeric_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(clean.min()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p10": float(clean.quantile(0.10)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p90": float(clean.quantile(0.90)),
        "max": float(clean.max()),
    }


def load_report(metadata_path: Path) -> dict[str, Any]:
    report_path = metadata_path.parent / "report.json"
    if not report_path.exists():
        return {}
    with report_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_existing_files(metadata_path: Path, values: pd.Series) -> int:
    root = metadata_path.parent
    return int(sum((root / Path(str(value).replace("\\", "/"))).exists() for value in values))


def validate_columns(df: pd.DataFrame, metadata_path: Path) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {metadata_path}: {missing}")


def load_mask_metrics(metadata_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    root = metadata_path.parent
    rows = []
    for item in df.itertuples(index=False):
        mask_path = root / Path(str(item.mask_path).replace("\\", "/"))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            rows.append({"patch_id": item.patch_id, "mask_touches_border": False})
            continue
        foreground = mask > 0
        touches_border = bool(
            foreground[0, :].any()
            or foreground[-1, :].any()
            or foreground[:, 0].any()
            or foreground[:, -1].any()
        )
        rows.append({"patch_id": item.patch_id, "mask_touches_border": touches_border})
    return pd.DataFrame(rows)


def make_mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    foreground = mask > 0
    if foreground.any():
        red = np.zeros_like(output)
        red[:, :, 2] = 255
        output[foreground] = (0.65 * output[foreground] + 0.35 * red[foreground]).astype(np.uint8)
    return output


def save_empty_positive_previews(output_dir: Path, empty_df: pd.DataFrame) -> list[str]:
    preview_dir = output_dir / "positive_named_empty_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for item in empty_df.itertuples(index=False):
        source_image = cv2.imread(str(Path(str(item.source_image_path).replace("\\", "/"))))
        source_mask = cv2.imread(
            str(Path(str(item.source_mask_path).replace("\\", "/"))),
            cv2.IMREAD_GRAYSCALE,
        )
        patch_root = Path(str(item.metadata_path)).parent
        patch_image = cv2.imread(str(patch_root / Path(str(item.patch_path).replace("\\", "/"))))
        patch_mask = cv2.imread(
            str(patch_root / Path(str(item.mask_path).replace("\\", "/"))),
            cv2.IMREAD_GRAYSCALE,
        )
        if any(value is None for value in [source_image, source_mask, patch_image, patch_mask]):
            continue

        source_overlay = make_mask_overlay(source_image, source_mask)
        crop_x1 = int(item.crop_x)
        crop_y1 = int(item.crop_y)
        crop_x2 = crop_x1 + int(item.crop_w)
        crop_y2 = crop_y1 + int(item.crop_h)
        cv2.rectangle(source_overlay, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 0, 255), 5)
        if not pd.isna(item.bbox_x):
            bbox_x1 = int(item.bbox_x)
            bbox_y1 = int(item.bbox_y)
            bbox_x2 = bbox_x1 + int(item.bbox_w)
            bbox_y2 = bbox_y1 + int(item.bbox_h)
            cv2.rectangle(source_overlay, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), (0, 255, 0), 5)

        patch_overlay = make_mask_overlay(patch_image, patch_mask)
        source_h, source_w = source_overlay.shape[:2]
        scale = min(900 / max(source_w, 1), 700 / max(source_h, 1), 1.0)
        source_preview = cv2.resize(
            source_overlay,
            (max(1, int(source_w * scale)), max(1, int(source_h * scale))),
        )
        patch_preview = cv2.resize(patch_overlay, (384, 384))
        canvas_h = max(source_preview.shape[0], patch_preview.shape[0]) + 55
        canvas_w = source_preview.shape[1] + patch_preview.shape[1]
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        canvas[55 : 55 + source_preview.shape[0], : source_preview.shape[1]] = source_preview
        canvas[55 : 55 + patch_preview.shape[0], source_preview.shape[1] :] = patch_preview
        title = f"{item.dataset_split} | {item.patch_kind} | empty mask | red=crop green=bbox"
        cv2.putText(canvas, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        output_path = preview_dir / f"{item.patch_id}.jpg"
        cv2.imwrite(str(output_path), canvas)
        paths.append(str(output_path))

    return paths


def build_positive_coverage_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    positive_df = df[df["is_positive"].astype(int) == 1]
    for (split, patch_kind), group in positive_df.groupby(["dataset_split", "patch_kind"]):
        bbox_coverage = pd.to_numeric(group["bbox_coverage"], errors="coerce").dropna()
        rows.append(
            {
                "split": split,
                "patch_kind": patch_kind,
                "count": int(len(group)),
                "mean_mask_coverage": float(group["mask_coverage"].mean()),
                "median_mask_coverage": float(group["mask_coverage"].median()),
                "p10_mask_coverage": float(group["mask_coverage"].quantile(0.10)),
                "mean_bbox_coverage": float(bbox_coverage.mean()),
                "median_bbox_coverage": float(bbox_coverage.median()),
                "bbox_coverage_lt_50_count": int((bbox_coverage < 0.5).sum()),
                "bbox_coverage_lt_50_ratio": float((bbox_coverage < 0.5).mean()),
            }
        )
    for patch_kind, group in positive_df.groupby("patch_kind"):
        bbox_coverage = pd.to_numeric(group["bbox_coverage"], errors="coerce").dropna()
        rows.append(
            {
                "split": "all",
                "patch_kind": patch_kind,
                "count": int(len(group)),
                "mean_mask_coverage": float(group["mask_coverage"].mean()),
                "median_mask_coverage": float(group["mask_coverage"].median()),
                "p10_mask_coverage": float(group["mask_coverage"].quantile(0.10)),
                "mean_bbox_coverage": float(bbox_coverage.mean()),
                "median_bbox_coverage": float(bbox_coverage.median()),
                "bbox_coverage_lt_50_count": int((bbox_coverage < 0.5).sum()),
                "bbox_coverage_lt_50_ratio": float((bbox_coverage < 0.5).mean()),
            }
        )
    return rows


def build_positive_foreground_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    positive_df = df[df["is_positive"].astype(int) == 1]
    groups = [(split, group) for split, group in positive_df.groupby("dataset_split")]
    groups.append(("all", positive_df))
    for split, group in groups:
        area_summary = numeric_summary(group["mask_area"])
        coverage_summary = numeric_summary(group["mask_coverage"])
        rows.append(
            {
                "split": split,
                "positive_patches": int(len(group)),
                **{f"mask_area_{key}": value for key, value in area_summary.items()},
                **{f"mask_coverage_{key}": value for key, value in coverage_summary.items()},
                "mask_coverage_lt_1pct_count": int((group["mask_coverage"] < 0.01).sum()),
                "mask_coverage_lt_1pct_ratio": float((group["mask_coverage"] < 0.01).mean()),
                "mask_coverage_lt_5pct_count": int((group["mask_coverage"] < 0.05).sum()),
                "mask_coverage_lt_5pct_ratio": float((group["mask_coverage"] < 0.05).mean()),
                "mask_touches_border_count": int(group["mask_touches_border"].sum()),
                "mask_touches_border_ratio": float(group["mask_touches_border"].mean()),
            }
        )
    return rows


def summarize_split(
    split: str,
    metadata_path: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing patch metadata: {metadata_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")

    df = pd.read_csv(metadata_path)
    manifest_df = pd.read_csv(manifest_path)
    validate_columns(df, metadata_path)

    source_rows = df[["image_id", "tumor"]].drop_duplicates()
    source_tumor = int(source_rows.loc[source_rows["tumor"].astype(int) == 1, "image_id"].nunique())
    source_normal = int(source_rows.loc[source_rows["tumor"].astype(int) == 0, "image_id"].nunique())
    manifest_tumor = int((manifest_df["tumor"].astype(int) == 1).sum())
    manifest_normal = int((manifest_df["tumor"].astype(int) == 0).sum())
    positive_df = df[df["is_positive"].astype(int) == 1]
    report = load_report(metadata_path)

    image_files = count_existing_files(metadata_path, df["patch_path"])
    mask_files = count_existing_files(metadata_path, df["mask_path"])
    total_patches = int(len(df))
    positive_patches = int(df["is_positive"].astype(int).sum())
    negative_patches = total_patches - positive_patches
    report_matches_metadata = bool(
        report
        and report.get("total_patches") == total_patches
        and report.get("positive_patches") == positive_patches
        and report.get("negative_patches") == negative_patches
    )

    summary = {
        "split": split,
        "manifest_rows": int(len(manifest_df)),
        "manifest_tumor_images": manifest_tumor,
        "manifest_normal_images": manifest_normal,
        "source_images_in_patches": int(source_rows["image_id"].nunique()),
        "source_tumor_images_in_patches": source_tumor,
        "source_normal_images_in_patches": source_normal,
        "source_image_coverage": float(source_rows["image_id"].nunique() / max(len(manifest_df), 1)),
        "tumor_source_coverage": float(source_tumor / max(manifest_tumor, 1)),
        "normal_source_coverage": float(source_normal / max(manifest_normal, 1)),
        "total_patches": total_patches,
        "positive_patches": positive_patches,
        "negative_patches": negative_patches,
        "positive_ratio": float(positive_patches / max(total_patches, 1)),
        "patches_per_manifest_image": float(total_patches / max(len(manifest_df), 1)),
        "existing_patch_images": image_files,
        "existing_patch_masks": mask_files,
        "all_patch_files_exist": bool(image_files == total_patches and mask_files == total_patches),
        "report_matches_metadata": report_matches_metadata,
        "all_patch_mask_coverage": numeric_summary(df["mask_coverage"]),
        "positive_patch_mask_coverage": numeric_summary(positive_df["mask_coverage"]),
    }

    kind_rows = []
    for patch_kind, group in df.groupby("patch_kind"):
        kind_rows.append(
            {
                "split": split,
                "patch_kind": str(patch_kind),
                "count": int(len(group)),
                "positive_count": int(group["is_positive"].astype(int).sum()),
                "negative_count": int((group["is_positive"].astype(int) == 0).sum()),
                "mean_mask_coverage": float(group["mask_coverage"].mean()),
            }
        )

    label_rows = []
    for (patch_kind, is_positive), group in df.groupby(["patch_kind", "is_positive"]):
        label_rows.append(
            {
                "split": split,
                "patch_kind": str(patch_kind),
                "is_positive": int(is_positive),
                "count": int(len(group)),
            }
        )

    tagged_df = df.copy()
    tagged_df["dataset_split"] = split
    tagged_df["metadata_path"] = str(metadata_path)
    tagged_df = tagged_df.merge(load_mask_metrics(metadata_path, tagged_df), on="patch_id", how="left")
    return tagged_df, summary, kind_rows, label_rows


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    split_paths = {
        "train": (args.train_metadata, args.train_manifest),
        "val": (args.val_metadata, args.val_manifest),
        "test": (args.test_metadata, args.test_manifest),
    }

    all_frames = []
    split_summaries = []
    kind_rows = []
    label_rows = []
    split_patch_image_ids: dict[str, set[str]] = {}
    split_manifest_image_ids: dict[str, set[str]] = {}

    for split, (metadata_path, manifest_path) in split_paths.items():
        df, summary, split_kind_rows, split_label_rows = summarize_split(
            split, metadata_path, manifest_path
        )
        all_frames.append(df)
        split_summaries.append(summary)
        kind_rows.extend(split_kind_rows)
        label_rows.extend(split_label_rows)
        split_patch_image_ids[split] = set(df["image_id"].astype(str))
        manifest_df = pd.read_csv(manifest_path)
        split_manifest_image_ids[split] = set(manifest_df["image_id"].astype(str))

    all_df = pd.concat(all_frames, ignore_index=True)
    total_patches = int(len(all_df))
    total_positive = int(all_df["is_positive"].astype(int).sum())
    total_negative = total_patches - total_positive
    total_manifest_rows = int(sum(item["manifest_rows"] for item in split_summaries))

    for patch_kind, group in all_df.groupby("patch_kind"):
        kind_rows.append(
            {
                "split": "all",
                "patch_kind": str(patch_kind),
                "count": int(len(group)),
                "positive_count": int(group["is_positive"].astype(int).sum()),
                "negative_count": int((group["is_positive"].astype(int) == 0).sum()),
                "mean_mask_coverage": float(group["mask_coverage"].mean()),
            }
        )
    for (patch_kind, is_positive), group in all_df.groupby(["patch_kind", "is_positive"]):
        label_rows.append(
            {
                "split": "all",
                "patch_kind": str(patch_kind),
                "is_positive": int(is_positive),
                "count": int(len(group)),
            }
        )

    patch_overlap = {
        "train_val": sorted(split_patch_image_ids["train"] & split_patch_image_ids["val"]),
        "train_test": sorted(split_patch_image_ids["train"] & split_patch_image_ids["test"]),
        "val_test": sorted(split_patch_image_ids["val"] & split_patch_image_ids["test"]),
    }
    manifest_overlap = {
        "train_val": sorted(split_manifest_image_ids["train"] & split_manifest_image_ids["val"]),
        "train_test": sorted(split_manifest_image_ids["train"] & split_manifest_image_ids["test"]),
        "val_test": sorted(split_manifest_image_ids["val"] & split_manifest_image_ids["test"]),
    }
    proposal_kind = (
        all_df["proposed_patch_kind"].fillna(all_df["patch_kind"])
        if "proposed_patch_kind" in all_df.columns
        else all_df["patch_kind"]
    )
    positive_named_empty = all_df[
        proposal_kind.astype(str).str.startswith("positive")
        & (all_df["is_positive"].astype(int) == 0)
    ].copy()
    positive_named_empty["diagnosis"] = np.where(
        pd.to_numeric(positive_named_empty["bbox_coverage"], errors="coerce") > 0,
        "crop_intersects_bbox_rectangle_but_contains_no_foreground",
        "crop_does_not_intersect_bbox_rectangle",
    )
    positive_df = all_df[all_df["is_positive"].astype(int) == 1]
    positive_coverage_rows = build_positive_coverage_rows(all_df)
    positive_foreground_rows = build_positive_foreground_rows(all_df)

    combined = {
        "manifest_rows": total_manifest_rows,
        "source_images_in_patches": int(all_df["image_id"].nunique()),
        "total_patches": total_patches,
        "positive_patches": total_positive,
        "negative_patches": total_negative,
        "positive_ratio": float(total_positive / max(total_patches, 1)),
        "patches_per_manifest_image": float(total_patches / max(total_manifest_rows, 1)),
        "positive_named_but_empty_patches": int(len(positive_named_empty)),
        "manifest_tumor_images": int(sum(item["manifest_tumor_images"] for item in split_summaries)),
        "manifest_normal_images": int(sum(item["manifest_normal_images"] for item in split_summaries)),
        "source_tumor_images_in_patches": int(
            sum(item["source_tumor_images_in_patches"] for item in split_summaries)
        ),
        "source_normal_images_in_patches": int(
            sum(item["source_normal_images_in_patches"] for item in split_summaries)
        ),
        "normal_source_coverage": float(
            sum(item["source_normal_images_in_patches"] for item in split_summaries)
            / max(sum(item["manifest_normal_images"] for item in split_summaries), 1)
        ),
        "all_patch_files_exist": bool(all(item["all_patch_files_exist"] for item in split_summaries)),
        "all_reports_match_metadata": bool(
            all(item["report_matches_metadata"] for item in split_summaries)
        ),
        "cross_split_patch_source_image_overlap_counts": {
            key: len(values) for key, values in patch_overlap.items()
        },
        "cross_split_manifest_image_overlap_counts": {
            key: len(values) for key, values in manifest_overlap.items()
        },
        "no_cross_split_source_image_leakage": bool(
            all(len(values) == 0 for values in manifest_overlap.values())
            and all(len(values) == 0 for values in patch_overlap.values())
        ),
        "all_patch_mask_coverage": numeric_summary(all_df["mask_coverage"]),
        "positive_patch_mask_coverage": numeric_summary(
            positive_df["mask_coverage"]
        ),
        "positive_patch_mask_area": numeric_summary(positive_df["mask_area"]),
        "positive_patches_mask_coverage_lt_1pct": int((positive_df["mask_coverage"] < 0.01).sum()),
        "positive_patches_mask_coverage_lt_1pct_ratio": float(
            (positive_df["mask_coverage"] < 0.01).mean()
        ),
        "positive_patches_mask_coverage_lt_5pct": int((positive_df["mask_coverage"] < 0.05).sum()),
        "positive_patches_mask_coverage_lt_5pct_ratio": float(
            (positive_df["mask_coverage"] < 0.05).mean()
        ),
        "positive_patches_mask_touches_border": int(positive_df["mask_touches_border"].sum()),
        "positive_patches_mask_touches_border_ratio": float(
            positive_df["mask_touches_border"].mean()
        ),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_summaries).drop(
        columns=["all_patch_mask_coverage", "positive_patch_mask_coverage"]
    ).to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame(kind_rows).to_csv(output_dir / "patch_kind_summary.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output_dir / "patch_kind_label_summary.csv", index=False)
    positive_named_empty.to_csv(output_dir / "positive_named_empty_patches.csv", index=False)
    pd.DataFrame(positive_coverage_rows).to_csv(
        output_dir / "positive_patch_coverage_by_kind.csv", index=False
    )
    pd.DataFrame(positive_foreground_rows).to_csv(
        output_dir / "positive_patch_foreground_summary.csv", index=False
    )
    empty_preview_paths = save_empty_positive_previews(output_dir, positive_named_empty)

    summary = {
        "split_summaries": split_summaries,
        "combined": combined,
        "cross_split_patch_source_image_overlaps": patch_overlap,
        "cross_split_manifest_image_overlaps": manifest_overlap,
        "positive_patch_coverage_by_kind": positive_coverage_rows,
        "positive_patch_foreground_summary": positive_foreground_rows,
        "positive_named_empty_preview_paths": empty_preview_paths,
    }
    with (output_dir / "patch_dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(summary), file, indent=2)

    columns = [
        "split",
        "manifest_rows",
        "source_images_in_patches",
        "total_patches",
        "positive_patches",
        "negative_patches",
        "positive_ratio",
    ]
    print("=== BTXRD Patch384 dataset summary ===")
    print(pd.DataFrame(split_summaries)[columns].to_string(index=False))
    print(f"\nTotal patches: {total_patches}")
    print(f"Positive ratio: {combined['positive_ratio']:.4f}")
    print(f"All patch files exist: {combined['all_patch_files_exist']}")
    print(f"Reports match metadata: {combined['all_reports_match_metadata']}")
    print(f"No cross-split image leakage: {combined['no_cross_split_source_image_leakage']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
