from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


DEFAULT_MANIFEST = Path("data/exports/btxrd_preprocessed/train.csv")
DEFAULT_OUTPUT_ROOT = Path("data/exports/patches_jitter")
TEXT_COLUMNS = [
    "diagnosis_group",
    "diagnosis_text",
    "tumor_type_text",
    "tumor_subtype_text",
    "region_text",
    "view_text",
    "anatomy_text",
    "segmentation_target_text",
    "text_diagnosis_only",
    "text_anatomy_aware",
    "text_short",
    "text_lvit_prompt",
    "negative_prompt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate BTXRD patch dataset using lesion-size-relative random jitter "
            "and save visual quality-control previews."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root-dir", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--patch-size", type=int, default=384)
    parser.add_argument(
        "--jitter-fraction",
        type=float,
        default=0.6,
        help="Patch-center jitter as a fraction of lesion bbox width/height.",
    )
    parser.add_argument("--positive-crops-per-lesion", type=int, default=5)
    parser.add_argument("--hard-negatives-per-lesion", type=int, default=3)
    parser.add_argument("--random-negatives-per-image", type=float, default=2.0)
    parser.add_argument(
        "--large-lesion-grid-stride-ratio",
        type=float,
        default=0.5,
        help="Grid stride for lesions larger than a patch, relative to patch size.",
    )
    parser.add_argument(
        "--max-large-lesion-grid-patches",
        type=int,
        default=0,
        help="Maximum grid patches per large lesion; 0 keeps all grid patches.",
    )
    parser.add_argument(
        "--target-positive-ratio",
        type=float,
        default=0.5,
        help="Add extra empty random-negative patches until this positive ratio is reached.",
    )
    parser.add_argument(
        "--max-balance-negative-attempts-multiplier",
        type=int,
        default=30,
        help="Maximum random attempts per extra balancing negative patch.",
    )
    parser.add_argument("--min-component-area", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--preview-samples", type=int, default=48)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def resolve_path(path_value: str | Path, root_dir: Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return root_dir / path


def ensure_clean_output(output_dir: Path, clean_output: bool) -> None:
    if not clean_output or not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def pad_to_patch(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    pad_h = max(0, patch_size - height)
    pad_w = max(0, patch_size - width)
    if pad_h == 0 and pad_w == 0:
        return image, mask

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    border_mode = cv2.BORDER_REFLECT_101 if height > 1 and width > 1 else cv2.BORDER_REPLICATE
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_mode)
    padded_mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_image, padded_mask


def crop_from_center(
    image: np.ndarray,
    mask: np.ndarray,
    center_x: float,
    center_y: float,
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    image, mask = pad_to_patch(image, mask, patch_size)
    height, width = image.shape[:2]
    x1 = int(round(center_x - patch_size / 2))
    y1 = int(round(center_y - patch_size / 2))
    x1 = max(0, min(x1, width - patch_size))
    y1 = max(0, min(y1, height - patch_size))
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2], x1, y1


def component_bboxes(mask: np.ndarray, min_component_area: int) -> list[dict[str, int]]:
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
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
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area": int(w * h),
                "mask_area": area,
                "bbox_maxside": int(max(w, h)),
                "center_x": float(x + w / 2),
                "center_y": float(y + h / 2),
            }
        )
    return components


def bbox_intersection_area(
    crop_x: int,
    crop_y: int,
    patch_size: int,
    bbox: dict[str, int],
) -> int:
    x1 = max(crop_x, int(bbox["bbox_x"]))
    y1 = max(crop_y, int(bbox["bbox_y"]))
    x2 = min(crop_x + patch_size, int(bbox["bbox_x"] + bbox["bbox_w"]))
    y2 = min(crop_y + patch_size, int(bbox["bbox_y"] + bbox["bbox_h"]))
    return max(0, x2 - x1) * max(0, y2 - y1)


def jittered_centers(
    bbox: dict[str, int],
    patch_size: int,
    jitter_fraction: float,
    positive_crops_per_lesion: int,
    large_lesion_grid_stride_ratio: float,
    max_large_lesion_grid_patches: int,
    rng: random.Random,
) -> list[tuple[float, float, str]]:
    center_x = float(bbox["center_x"])
    center_y = float(bbox["center_y"])
    bbox_w = float(bbox["bbox_w"])
    bbox_h = float(bbox["bbox_h"])

    centers = [(center_x, center_y, "positive_center")]
    for _ in range(max(0, positive_crops_per_lesion - 1)):
        dx = rng.uniform(-jitter_fraction * bbox_w, jitter_fraction * bbox_w)
        dy = rng.uniform(-jitter_fraction * bbox_h, jitter_fraction * bbox_h)
        centers.append((center_x + dx, center_y + dy, "positive_jitter"))

    if bbox["bbox_maxside"] > patch_size:
        step = max(1, int(round(patch_size * large_lesion_grid_stride_ratio)))
        x_start = int(bbox["bbox_x"] + patch_size / 2)
        x_end = int(bbox["bbox_x"] + bbox["bbox_w"] - patch_size / 2)
        y_start = int(bbox["bbox_y"] + patch_size / 2)
        y_end = int(bbox["bbox_y"] + bbox["bbox_h"] - patch_size / 2)
        xs = list(range(min(x_start, x_end), max(x_start, x_end) + 1, step)) or [center_x]
        ys = list(range(min(y_start, y_end), max(y_start, y_end) + 1, step)) or [center_y]
        grid_centers = [(float(x), float(y), "positive_large_lesion_grid") for y in ys for x in xs]
        if max_large_lesion_grid_patches > 0 and len(grid_centers) > max_large_lesion_grid_patches:
            selected_indices = np.linspace(
                0,
                len(grid_centers) - 1,
                num=max_large_lesion_grid_patches,
                dtype=int,
            )
            grid_centers = [grid_centers[index] for index in selected_indices]
        centers.extend(grid_centers)

    return centers


def random_negative_center(
    image_w: int,
    image_h: int,
    patch_size: int,
    rng: random.Random,
) -> tuple[float, float]:
    x = rng.uniform(patch_size / 2, max(patch_size / 2, image_w - patch_size / 2))
    y = rng.uniform(patch_size / 2, max(patch_size / 2, image_h - patch_size / 2))
    return x, y


def hard_negative_center(
    bbox: dict[str, int],
    image_w: int,
    image_h: int,
    patch_size: int,
    rng: random.Random,
) -> tuple[float, float]:
    radius_x = max(float(bbox["bbox_w"]), patch_size / 2)
    radius_y = max(float(bbox["bbox_h"]), patch_size / 2)
    angle = rng.uniform(0, 2 * math.pi)
    center_x = float(bbox["center_x"]) + math.cos(angle) * radius_x
    center_y = float(bbox["center_y"]) + math.sin(angle) * radius_y
    center_x = max(patch_size / 2, min(center_x, image_w - patch_size / 2))
    center_y = max(patch_size / 2, min(center_y, image_h - patch_size / 2))
    return center_x, center_y


def row_metadata(row: pd.Series) -> dict[str, Any]:
    output = {}
    for column in TEXT_COLUMNS:
        output[column] = "" if column not in row or pd.isna(row[column]) else row[column]
    return output


def save_patch(
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    output_dir: Path,
    patch_id: str,
) -> tuple[str, str]:
    image_rel = Path("images") / f"{patch_id}.jpg"
    mask_rel = Path("masks") / f"{patch_id}.png"
    image_path = output_dir / image_rel
    mask_path = output_dir / mask_rel
    cv2.imwrite(str(image_path), image_patch)
    cv2.imwrite(str(mask_path), (mask_patch > 0).astype(np.uint8) * 255)
    return str(image_rel), str(mask_rel)


def add_record(
    records: list[dict[str, Any]],
    row: pd.Series,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
    output_dir: Path,
    patch_id: str,
    patch_kind: str,
    crop_x: int,
    crop_y: int,
    patch_size: int,
    bbox: dict[str, int] | None,
) -> None:
    patch_path, mask_path = save_patch(image_patch, mask_patch, output_dir, patch_id)
    mask_area = int((mask_patch > 0).sum())
    proposed_patch_kind = patch_kind
    if patch_kind.startswith("positive") and mask_area == 0:
        patch_kind = "empty_positive_proposal"
    bbox_area = int(bbox["bbox_area"]) if bbox is not None else 0
    bbox_covered = bbox_intersection_area(crop_x, crop_y, patch_size, bbox) if bbox is not None else 0

    record = {
        "patch_id": patch_id,
        "patch_path": patch_path,
        "mask_path": mask_path,
        "image_id": str(row["image_id"]),
        "source_image_path": str(row["image_path"]),
        "source_mask_path": str(row["mask_path"]),
        "split": str(row.get("split", "")),
        "tumor": int(row["tumor"]),
        "is_positive": int(mask_area > 0),
        "patch_kind": patch_kind,
        "proposed_patch_kind": proposed_patch_kind,
        "patch_size": int(patch_size),
        "crop_x": int(crop_x),
        "crop_y": int(crop_y),
        "crop_w": int(patch_size),
        "crop_h": int(patch_size),
        "mask_area": mask_area,
        "mask_coverage": float(mask_area / float(patch_size * patch_size)),
        "bbox_x": int(bbox["bbox_x"]) if bbox is not None else "",
        "bbox_y": int(bbox["bbox_y"]) if bbox is not None else "",
        "bbox_w": int(bbox["bbox_w"]) if bbox is not None else "",
        "bbox_h": int(bbox["bbox_h"]) if bbox is not None else "",
        "bbox_area": bbox_area,
        "bbox_maxside": int(bbox["bbox_maxside"]) if bbox is not None else "",
        "bbox_coverage": float(bbox_covered / bbox_area) if bbox_area else 0.0,
        "patch_mean": float(image_patch.mean()),
        "patch_std": float(image_patch.std()),
    }
    record.update(row_metadata(row))
    records.append(record)


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()
    color = np.zeros_like(output)
    color[:, :, 2] = 255
    alpha = ((mask > 0).astype(np.float32) * 0.35)[:, :, None]
    output = (output * (1 - alpha) + color * alpha).astype(np.uint8)
    return output


def save_visual_checks(output_dir: Path, metadata_df: pd.DataFrame, preview_samples: int) -> list[str]:
    if metadata_df.empty or preview_samples <= 0:
        return []

    visual_dir = output_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)

    sampled_parts = []
    for patch_kind in metadata_df["patch_kind"].dropna().unique():
        part = metadata_df[metadata_df["patch_kind"] == patch_kind].head(max(1, preview_samples // 6))
        sampled_parts.append(part)
    sampled = pd.concat(sampled_parts, axis=0).head(preview_samples)
    if sampled.empty:
        sampled = metadata_df.head(preview_samples)

    cell = int(metadata_df["patch_size"].iloc[0])
    cols = 6
    rows = int(math.ceil(len(sampled) / cols))
    canvas = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)

    for idx, item in enumerate(sampled.itertuples(index=False)):
        image = cv2.imread(str(output_dir / item.patch_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(output_dir / item.mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        overlay = make_overlay(image, mask)
        r = idx // cols
        c = idx % cols
        canvas[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = overlay
        label = f"{item.patch_kind} | pos={item.is_positive}"
        cv2.putText(
            canvas,
            label[:45],
            (c * cell + 8, r * cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[:45],
            (c * cell + 8, r * cell + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    preview_path = visual_dir / "sample_patch_overlays.jpg"
    cv2.imwrite(str(preview_path), canvas)

    coverage_path = visual_dir / "mask_coverage_hist.png"
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(metadata_df["mask_coverage"].astype(float), bins=50, color="#2f6f8f", edgecolor="white")
        plt.xlabel("mask_coverage")
        plt.ylabel("patch_count")
        plt.title("Patch mask coverage")
        plt.tight_layout()
        plt.savefig(coverage_path, dpi=150)
        plt.close()
        return [str(preview_path), str(coverage_path)]
    except ImportError:
        return [str(preview_path)]


def generate_patches(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    rng = random.Random(args.seed)
    manifest_path = resolve_path(args.manifest, args.root_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if args.max_rows is not None:
        df = df.head(args.max_rows)

    output_dir = args.output_root / f"patches_{args.patch_size}"
    ensure_clean_output(output_dir, args.clean_output)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    counters = {
        "rows": int(len(df)),
        "tumor_rows": int((df["tumor"].astype(int) == 1).sum()),
        "normal_rows": int((df["tumor"].astype(int) == 0).sum()),
        "missing_images": 0,
        "missing_masks": 0,
        "tumor_rows_without_components": 0,
        "lesion_components": 0,
    }

    valid_rows: list[pd.Series] = []

    for _, row in df.iterrows():
        image_path = resolve_path(row["image_path"], args.root_dir)
        mask_path = resolve_path(row["mask_path"], args.root_dir)
        if not image_path.exists():
            counters["missing_images"] += 1
            continue
        if not mask_path.exists():
            counters["missing_masks"] += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        valid_rows.append(row)

        image_h, image_w = image.shape[:2]
        image_stem = Path(str(row["image_id"])).stem
        is_tumor = int(row["tumor"]) == 1
        components = component_bboxes(mask, args.min_component_area) if is_tumor else []
        counters["lesion_components"] += len(components)
        if is_tumor and not components:
            counters["tumor_rows_without_components"] += 1

        for component_index, bbox in enumerate(components, start=1):
            centers = jittered_centers(
                bbox,
                args.patch_size,
                args.jitter_fraction,
                args.positive_crops_per_lesion,
                args.large_lesion_grid_stride_ratio,
                args.max_large_lesion_grid_patches,
                rng,
            )
            for crop_index, (center_x, center_y, patch_kind) in enumerate(centers, start=1):
                image_patch, mask_patch, crop_x, crop_y = crop_from_center(
                    image,
                    mask,
                    center_x,
                    center_y,
                    args.patch_size,
                )
                patch_id = f"{image_stem}__lesion{component_index:02d}__{patch_kind}_{crop_index:02d}"
                add_record(
                    records,
                    row,
                    image_patch,
                    mask_patch,
                    output_dir,
                    patch_id,
                    patch_kind,
                    crop_x,
                    crop_y,
                    args.patch_size,
                    bbox,
                )

            for negative_index in range(args.hard_negatives_per_lesion):
                center_x, center_y = hard_negative_center(bbox, image_w, image_h, args.patch_size, rng)
                image_patch, mask_patch, crop_x, crop_y = crop_from_center(
                    image,
                    mask,
                    center_x,
                    center_y,
                    args.patch_size,
                )
                if (mask_patch > 0).any():
                    continue
                patch_id = f"{image_stem}__lesion{component_index:02d}__hard_negative_{negative_index + 1:02d}"
                add_record(
                    records,
                    row,
                    image_patch,
                    mask_patch,
                    output_dir,
                    patch_id,
                    "hard_negative",
                    crop_x,
                    crop_y,
                    args.patch_size,
                    bbox,
                )

        random_negative_count = int(math.floor(args.random_negatives_per_image))
        if rng.random() < args.random_negatives_per_image - random_negative_count:
            random_negative_count += 1
        for negative_index in range(random_negative_count):
            center_x, center_y = random_negative_center(image_w, image_h, args.patch_size, rng)
            image_patch, mask_patch, crop_x, crop_y = crop_from_center(
                image,
                mask,
                center_x,
                center_y,
                args.patch_size,
            )
            if (mask_patch > 0).any():
                continue
            patch_id = f"{image_stem}__random_negative_{negative_index + 1:02d}"
            add_record(
                records,
                row,
                image_patch,
                mask_patch,
                output_dir,
                patch_id,
                "random_negative",
                crop_x,
                crop_y,
                args.patch_size,
                None,
            )

    if 0 < args.target_positive_ratio < 1 and records:
        positive_count = sum(int(record["is_positive"]) for record in records)
        negative_count = len(records) - positive_count
        target_negative_count = int(math.ceil(positive_count * (1 - args.target_positive_ratio) / args.target_positive_ratio))
        extra_needed = max(0, target_negative_count - negative_count)
        max_attempts = max(1, extra_needed * args.max_balance_negative_attempts_multiplier)
        attempts = 0
        extra_added = 0

        while extra_added < extra_needed and attempts < max_attempts and valid_rows:
            attempts += 1
            row = rng.choice(valid_rows)
            image_path = resolve_path(row["image_path"], args.root_dir)
            mask_path = resolve_path(row["mask_path"], args.root_dir)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                continue
            image_h, image_w = image.shape[:2]
            center_x, center_y = random_negative_center(image_w, image_h, args.patch_size, rng)
            image_patch, mask_patch, crop_x, crop_y = crop_from_center(
                image,
                mask,
                center_x,
                center_y,
                args.patch_size,
            )
            if (mask_patch > 0).any():
                continue

            image_stem = Path(str(row["image_id"])).stem
            patch_id = f"{image_stem}__balance_negative_{extra_added + 1:05d}"
            add_record(
                records,
                row,
                image_patch,
                mask_patch,
                output_dir,
                patch_id,
                "balance_negative",
                crop_x,
                crop_y,
                args.patch_size,
                None,
            )
            extra_added += 1

    metadata_df = pd.DataFrame(records)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    visual_paths = save_visual_checks(output_dir, metadata_df, args.preview_samples)

    report = {
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "patch_size": int(args.patch_size),
        "jitter_fraction": float(args.jitter_fraction),
        "jitter_rule": "center_x += U(-jitter_fraction*bbox_w, +jitter_fraction*bbox_w); center_y += U(-jitter_fraction*bbox_h, +jitter_fraction*bbox_h)",
        "positive_crops_per_lesion": int(args.positive_crops_per_lesion),
        "hard_negatives_per_lesion": int(args.hard_negatives_per_lesion),
        "random_negatives_per_image": float(args.random_negatives_per_image),
        "large_lesion_grid_stride_ratio": float(args.large_lesion_grid_stride_ratio),
        "max_large_lesion_grid_patches": int(args.max_large_lesion_grid_patches),
        "target_positive_ratio": float(args.target_positive_ratio),
        "counters": counters,
        "total_patches": int(len(metadata_df)),
        "positive_patches": int(metadata_df["is_positive"].sum()) if not metadata_df.empty else 0,
        "negative_patches": int((metadata_df["is_positive"] == 0).sum()) if not metadata_df.empty else 0,
        "positive_ratio": float(metadata_df["is_positive"].mean()) if not metadata_df.empty else 0.0,
        "patch_kind_counts": metadata_df["patch_kind"].value_counts().to_dict() if not metadata_df.empty else {},
        "mask_coverage": {
            "min": float(metadata_df["mask_coverage"].min()) if not metadata_df.empty else 0.0,
            "mean": float(metadata_df["mask_coverage"].mean()) if not metadata_df.empty else 0.0,
            "median": float(metadata_df["mask_coverage"].median()) if not metadata_df.empty else 0.0,
            "max": float(metadata_df["mask_coverage"].max()) if not metadata_df.empty else 0.0,
        },
        "visual_checks": visual_paths,
    }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return metadata_df, report, visual_paths


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if args.jitter_fraction < 0:
        raise ValueError("--jitter-fraction must be non-negative")
    if args.large_lesion_grid_stride_ratio <= 0:
        raise ValueError("--large-lesion-grid-stride-ratio must be positive")
    if args.max_large_lesion_grid_patches < 0:
        raise ValueError("--max-large-lesion-grid-patches must be non-negative")
    if not 0 < args.target_positive_ratio < 1:
        raise ValueError("--target-positive-ratio must be in (0, 1)")

    metadata_df, report, visual_paths = generate_patches(args)
    print("=== BTXRD jitter patch dataset ===")
    print(f"Output:          {report['output_dir']}")
    print(f"Patch size:      {report['patch_size']}")
    print(f"Jitter fraction: {report['jitter_fraction']}")
    print(f"Total patches:   {report['total_patches']}")
    print(f"Positive patches:{report['positive_patches']}")
    print(f"Negative patches:{report['negative_patches']}")
    print(f"Positive ratio:  {report['positive_ratio']:.4f}")
    print("Patch kinds:")
    for key, value in report["patch_kind_counts"].items():
        print(f"- {key}: {value}")
    print("Output files:")
    print(f"- {Path(report['output_dir']) / 'metadata.csv'}")
    print(f"- {Path(report['output_dir']) / 'report.json'}")
    for path in visual_paths:
        print(f"- {path}")

    if metadata_df.empty:
        sys.exit("No patches were generated.")


if __name__ == "__main__":
    main()
