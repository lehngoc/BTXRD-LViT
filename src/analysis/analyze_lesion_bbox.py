from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path("data/exports/btxrd_preprocessed/all.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/reports/lesion_bbox_analysis")
DEFAULT_PATCH_SIZES = [224, 320, 384, 512]
PERCENTILES = [25, 50, 75, 90, 95, 99]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze BTXRD lesion mask bounding boxes and patch-size tradeoffs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_PATCH_SIZES,
        help="Candidate square patch sizes to compare.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip PNG histogram generation.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Pixel connectivity used to split multi-tumor masks into lesion instances.",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=1,
        help="Minimum connected-component area in pixels to keep as a lesion instance.",
    )
    return parser.parse_args()


def resolve_manifest_path(path_value: str | Path, root_dir: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute() or path.exists():
        return path
    return root_dir / path


def load_gray(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def compute_bbox(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    return {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_w": x2 - x1,
        "bbox_h": y2 - y1,
    }


def compute_component_bboxes(
    mask: np.ndarray,
    connectivity: int = 8,
    min_component_area: int = 1,
) -> list[dict[str, int]]:
    binary_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=connectivity,
    )
    components = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_component_area:
            continue

        components.append(
            {
                "component_label": int(label_id),
                "bbox_x1": x,
                "bbox_y1": y,
                "bbox_x2": x + w,
                "bbox_y2": y + h,
                "bbox_w": w,
                "bbox_h": h,
                "mask_area": area,
                "component_centroid_x": float(centroids[label_id][0]),
                "component_centroid_y": float(centroids[label_id][1]),
            }
        )

    components.sort(key=lambda item: (item["bbox_y1"], item["bbox_x1"]))
    return components


def numeric_summary(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        output = {
            "count": 0.0,
            "min": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "max": 0.0,
        }
        output.update({f"p{p}": 0.0 for p in PERCENTILES})
        return output

    output = {
        "count": float(len(clean)),
        "min": float(clean.min()),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=0)),
        "median": float(clean.median()),
        "max": float(clean.max()),
    }
    output.update({f"p{p}": float(np.percentile(clean, p)) for p in PERCENTILES})
    return output


def build_distribution_summary(stats_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = [
        "bbox_w",
        "bbox_h",
        "bbox_max_side",
        "bbox_area",
        "bbox_area_ratio",
        "mask_area",
        "mask_area_ratio",
        "bbox_w_ratio",
        "bbox_h_ratio",
    ]
    return {metric: numeric_summary(stats_df[metric]) for metric in metrics}


def summarize_group(stats_df: pd.DataFrame, group_col: str) -> list[dict[str, Any]]:
    if group_col not in stats_df.columns or stats_df.empty:
        return []

    rows = []
    for group_value, group_df in stats_df.groupby(group_col, dropna=False):
        row = {
            group_col: "" if pd.isna(group_value) else str(group_value),
            "count": int(len(group_df)),
            "bbox_max_side_p50": float(np.percentile(group_df["bbox_max_side"], 50)),
            "bbox_max_side_p90": float(np.percentile(group_df["bbox_max_side"], 90)),
            "bbox_max_side_p95": float(np.percentile(group_df["bbox_max_side"], 95)),
            "mask_area_ratio_median": float(group_df["mask_area_ratio"].median()),
        }
        rows.append(row)

    return sorted(rows, key=lambda item: item["count"], reverse=True)


def estimate_patch_count(width: int, height: int, patch_size: int, overlap: float) -> int:
    if width <= 0 or height <= 0:
        return 0

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    count_x = 1 if width <= patch_size else math.ceil((width - patch_size) / stride) + 1
    count_y = 1 if height <= patch_size else math.ceil((height - patch_size) / stride) + 1
    return int(count_x * count_y)


def build_patch_candidates(stats_df: pd.DataFrame, image_df: pd.DataFrame, patch_sizes: list[int]) -> pd.DataFrame:
    rows = []
    base_cost = None

    for patch_size in patch_sizes:
        bbox_max_side = stats_df["bbox_max_side"]
        margins = (patch_size - bbox_max_side) / 2.0
        patches_25 = [
            estimate_patch_count(int(row.image_w), int(row.image_h), patch_size, 0.25)
            for row in image_df.itertuples(index=False)
        ]
        patches_50 = [
            estimate_patch_count(int(row.image_w), int(row.image_h), patch_size, 0.50)
            for row in image_df.itertuples(index=False)
        ]
        avg_patches_25 = float(np.mean(patches_25)) if patches_25 else 0.0
        avg_patches_50 = float(np.mean(patches_50)) if patches_50 else 0.0
        compute_cost = avg_patches_50 * float(patch_size * patch_size)

        if base_cost is None:
            base_cost = compute_cost if compute_cost > 0 else 1.0

        rows.append(
            {
                "patch_size": int(patch_size),
                "lesions_fit_full_bbox_ratio": float((bbox_max_side <= patch_size).mean()),
                "lesions_fit_80pct_ratio": float((bbox_max_side <= patch_size * 0.8).mean()),
                "median_context_margin_px": float(np.median(margins)),
                "p90_context_margin_px": float(np.percentile(margins, 90)),
                "estimated_patches_per_image_overlap_25": avg_patches_25,
                "estimated_patches_per_image_overlap_50": avg_patches_50,
                "relative_pixel_cost_vs_224": float((patch_size / 224.0) ** 2),
                "relative_compute_cost_vs_224": float(compute_cost / base_cost),
            }
        )

    return pd.DataFrame(rows)


def recommend_patch_size(candidates_df: pd.DataFrame) -> dict[str, Any]:
    recommended = None
    reason = ""

    non_upper_bound = candidates_df[candidates_df["patch_size"] < 512]
    near_practical = non_upper_bound[
        (non_upper_bound["lesions_fit_full_bbox_ratio"] >= 0.70)
        & (non_upper_bound["lesions_fit_80pct_ratio"] >= 0.60)
    ]
    if not near_practical.empty:
        recommended = int(near_practical.iloc[-1]["patch_size"])
        reason = (
            "Largest sub-512 candidate with practical bbox coverage and context. "
            "Use overlap and stitching for larger lesions."
        )
    else:
        fit_75 = candidates_df[candidates_df["lesions_fit_full_bbox_ratio"] >= 0.75]
        if not fit_75.empty:
            first_fit = int(fit_75.iloc[0]["patch_size"])
            if first_fit >= 512 and not non_upper_bound.empty:
                recommended = int(non_upper_bound.iloc[-1]["patch_size"])
                reason = (
                    "512 is the first candidate crossing 75% bbox fit, but runtime is a constraint; "
                    "start with the largest sub-512 candidate and keep 512 as an upper-bound comparison."
                )
            else:
                recommended = first_fit
                reason = (
                    "Smallest candidate fitting at least 75% of lesion bounding boxes; "
                    "use overlap/stitching for larger lesions."
                )
        else:
            recommended = int(candidates_df.iloc[0]["patch_size"])
            reason = "No candidate fits 75% of lesion bounding boxes; start small and rely on multi-patch stitching."

    if recommended >= 512:
        reason += " Treat 512 as an upper-bound experiment because runtime is a constraint."

    return {
        "recommended_patch_size": recommended,
        "reason": reason,
        "decision_rule": (
            "Prefer the smallest practical patch. Lesions larger than a patch are expected "
            "to be recovered by jitter, overlap, and stitched full-image inference."
        ),
    }


def write_summary_csv(path: Path, distribution: dict[str, dict[str, float]]) -> None:
    rows = []
    for metric, values in distribution.items():
        row = {"metric": metric}
        row.update(values)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_plots(stats_df: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is not available; skipping plots.")
        return []

    plots = [
        ("bbox_w", "bbox_width_hist.png", "Lesion bbox width"),
        ("bbox_h", "bbox_height_hist.png", "Lesion bbox height"),
        ("bbox_max_side", "bbox_max_side_hist.png", "Lesion bbox max side"),
        ("mask_area_ratio", "mask_area_ratio_hist.png", "Mask area ratio"),
        ("bbox_area_ratio", "bbox_area_ratio_hist.png", "BBox area ratio"),
    ]
    saved = []

    for column, filename, title in plots:
        plt.figure(figsize=(8, 5))
        plt.hist(stats_df[column], bins=40, color="#2f6f8f", edgecolor="white")
        plt.title(title)
        plt.xlabel(column)
        plt.ylabel("count")
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150)
        plt.close()
        saved.append(str(output_path))

    return saved


def analyze_manifest(
    manifest_path: Path,
    root_dir: Path,
    connectivity: int,
    min_component_area: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(manifest_path)
    required = ["image_id", "image_path", "mask_path", "tumor"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {manifest_path}: {missing}")

    lesion_records = []
    image_records = []
    counters = {
        "total_rows": int(len(df)),
        "tumor_rows": int((df["tumor"].astype(int) == 1).sum()),
        "normal_rows": int((df["tumor"].astype(int) == 0).sum()),
        "missing_images": 0,
        "missing_masks": 0,
        "unreadable_images": 0,
        "unreadable_masks": 0,
        "tumor_empty_masks": 0,
        "tumor_masks_with_no_kept_components": 0,
        "tumor_images_with_multiple_components": 0,
        "normal_non_empty_masks": 0,
        "size_mismatch_rows": 0,
    }

    for _, row in df.iterrows():
        image_path = resolve_manifest_path(row["image_path"], root_dir)
        mask_path = resolve_manifest_path(row["mask_path"], root_dir)
        image_exists = image_path.exists()
        mask_exists = mask_path.exists()

        if not image_exists:
            counters["missing_images"] += 1
        if not mask_exists:
            counters["missing_masks"] += 1

        image = load_gray(image_path) if image_exists else None
        mask = load_gray(mask_path) if mask_exists else None

        if image_exists and image is None:
            counters["unreadable_images"] += 1
        if mask_exists and mask is None:
            counters["unreadable_masks"] += 1

        image_h = int(image.shape[0]) if image is not None else 0
        image_w = int(image.shape[1]) if image is not None else 0
        mask_h = int(mask.shape[0]) if mask is not None else 0
        mask_w = int(mask.shape[1]) if mask is not None else 0

        if image is not None and mask is not None and (image_h != mask_h or image_w != mask_w):
            counters["size_mismatch_rows"] += 1

        is_tumor = int(row["tumor"]) == 1
        mask_area = int((mask > 0).sum()) if mask is not None else 0

        if mask is not None and not is_tumor and mask_area > 0:
            counters["normal_non_empty_masks"] += 1

        components: list[dict[str, int]] = []
        if is_tumor and mask is not None and mask_area > 0:
            components = compute_component_bboxes(
                mask,
                connectivity=connectivity,
                min_component_area=min_component_area,
            )
            if len(components) > 1:
                counters["tumor_images_with_multiple_components"] += 1

        image_area = max(1, image_w * image_h)

        image_records.append(
            {
                "image_id": str(row["image_id"]),
                "split": str(row.get("split", "")),
                "image_w": image_w,
                "image_h": image_h,
                "mask_w": mask_w,
                "mask_h": mask_h,
                "tumor": int(row["tumor"]),
                "lesion_instances_in_image": len(components),
            }
        )

        if not is_tumor:
            continue

        if mask is None or mask_area == 0:
            counters["tumor_empty_masks"] += 1
            continue

        if not components:
            counters["tumor_masks_with_no_kept_components"] += 1
            continue

        for instance_index, component in enumerate(components, start=1):
            bbox_area = int(component["bbox_w"] * component["bbox_h"])

            lesion_records.append(
                {
                    "image_id": str(row["image_id"]),
                    "lesion_instance_id": f"{Path(str(row['image_id'])).stem}_lesion_{instance_index:02d}",
                    "lesion_instance_index": instance_index,
                    "lesion_instances_in_image": len(components),
                    "component_label": component["component_label"],
                    "split": str(row.get("split", "")),
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "image_w": image_w,
                    "image_h": image_h,
                    "mask_w": mask_w,
                    "mask_h": mask_h,
                    "bbox_x1": component["bbox_x1"],
                    "bbox_y1": component["bbox_y1"],
                    "bbox_x2": component["bbox_x2"],
                    "bbox_y2": component["bbox_y2"],
                    "bbox_w": component["bbox_w"],
                    "bbox_h": component["bbox_h"],
                    "bbox_max_side": int(max(component["bbox_w"], component["bbox_h"])),
                    "bbox_area": bbox_area,
                    "bbox_area_ratio": float(bbox_area / image_area),
                    "mask_area": component["mask_area"],
                    "full_mask_area": mask_area,
                    "mask_area_ratio": float(component["mask_area"] / image_area),
                    "full_mask_area_ratio": float(mask_area / image_area),
                    "bbox_center_x": float((component["bbox_x1"] + component["bbox_x2"]) / 2.0),
                    "bbox_center_y": float((component["bbox_y1"] + component["bbox_y2"]) / 2.0),
                    "component_centroid_x": component["component_centroid_x"],
                    "component_centroid_y": component["component_centroid_y"],
                    "bbox_w_ratio": float(component["bbox_w"] / max(1, image_w)),
                    "bbox_h_ratio": float(component["bbox_h"] / max(1, image_h)),
                    "diagnosis_group": str(row.get("diagnosis_group", "")),
                    "region_text": str(row.get("region_text", "")),
                    "anatomy_text": str(row.get("anatomy_text", "")),
                    "tumor_type_text": str(row.get("tumor_type_text", "")),
                }
            )

    return pd.DataFrame(lesion_records), pd.DataFrame(image_records), counters


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir
    manifest_path = resolve_manifest_path(args.manifest, root_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Analyze BTXRD lesion bbox statistics ===")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_dir}")

    stats_df, image_df, counters = analyze_manifest(
        manifest_path,
        root_dir,
        connectivity=args.connectivity,
        min_component_area=args.min_component_area,
    )
    if stats_df.empty:
        raise RuntimeError("No non-empty lesion instances found; cannot analyze lesion bounding boxes.")

    distribution = build_distribution_summary(stats_df)
    candidates_df = build_patch_candidates(stats_df, image_df, sorted(set(args.patch_sizes)))
    recommendation = recommend_patch_size(candidates_df)
    plot_paths = [] if args.skip_plots else save_plots(stats_df, output_dir)

    stats_path = output_dir / "lesion_bbox_stats.csv"
    summary_json_path = output_dir / "lesion_bbox_summary.json"
    summary_csv_path = output_dir / "lesion_bbox_summary.csv"
    candidates_path = output_dir / "patch_size_candidates.csv"

    stats_df.to_csv(stats_path, index=False)
    write_summary_csv(summary_csv_path, distribution)
    candidates_df.to_csv(candidates_path, index=False)

    summary = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "counts": {
            **counters,
            "valid_tumor_bbox_rows": int(len(stats_df)),
            "valid_lesion_instance_rows": int(len(stats_df)),
            "max_lesion_instances_per_image": int(image_df["lesion_instances_in_image"].max()),
        },
        "component_settings": {
            "connectivity": int(args.connectivity),
            "min_component_area": int(args.min_component_area),
            "stats_unit": "connected_component_lesion_instance",
        },
        "distribution": distribution,
        "by_split": summarize_group(stats_df, "split"),
        "by_diagnosis_group": summarize_group(stats_df, "diagnosis_group"),
        "by_region_text": summarize_group(stats_df, "region_text"),
        "by_anatomy_text": summarize_group(stats_df, "anatomy_text"),
        "patch_size_candidates": candidates_df.to_dict(orient="records"),
        "recommendation": recommendation,
        "plots": plot_paths,
    }

    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print("\n=== Lesion BBox Summary ===")
    print(f"Total rows:             {counters['total_rows']}")
    print(f"Tumor rows:             {counters['tumor_rows']}")
    print(f"Valid lesion instances: {len(stats_df)}")
    print(f"Tumor empty masks:      {counters['tumor_empty_masks']}")
    print(f"Multi-lesion images:    {counters['tumor_images_with_multiple_components']}")
    print(f"Normal non-empty masks: {counters['normal_non_empty_masks']}")
    print(f"Size mismatches:        {counters['size_mismatch_rows']}")

    print("\n=== Patch Size Candidates ===")
    print(candidates_df.to_string(index=False))
    print(
        "\nRecommended patch size: "
        f"{recommendation['recommended_patch_size']} "
        f"({recommendation['reason']})"
    )

    print("\nOutput files:")
    print(f"- {stats_path}")
    print(f"- {summary_csv_path}")
    print(f"- {summary_json_path}")
    print(f"- {candidates_path}")
    for plot_path in plot_paths:
        print(f"- {plot_path}")


if __name__ == "__main__":
    main()
