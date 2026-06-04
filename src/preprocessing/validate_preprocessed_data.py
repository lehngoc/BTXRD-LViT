from pathlib import Path
import argparse

import cv2
import pandas as pd


RAW_MASK_DIR = Path("data/processed/masks")
FINAL_IMAGE_DIR = Path("data/processed/images_preprocessed")
FINAL_MASK_DIR = Path("data/processed/masks_preprocessed")
REPORT_DIR = Path("data/processed/reports")

PIPELINE_REPORT_PATH = REPORT_DIR / "preprocess_pipeline_report.csv"
VALIDATION_REPORT_PATH = REPORT_DIR / "preprocess_validation_report.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Validate final preprocessed BTXRD data.")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix used by preprocess_btxrd_data.py.",
    )
    return parser.parse_args()


def tagged_path(path: Path, output_tag: str) -> Path:
    if not output_tag:
        return path
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return path.with_name(f"{path.name}_{safe_tag}")


def tagged_report_path(base_name: str, output_tag: str) -> Path:
    if not output_tag:
        return REPORT_DIR / f"{base_name}.csv"
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return REPORT_DIR / f"{base_name}_{safe_tag}.csv"


def main():
    args = parse_args()
    final_image_dir = tagged_path(FINAL_IMAGE_DIR, args.output_tag)
    final_mask_dir = tagged_path(FINAL_MASK_DIR, args.output_tag)
    pipeline_report_path = tagged_report_path("preprocess_pipeline_report", args.output_tag)
    validation_report_path = tagged_report_path(
        "preprocess_validation_report",
        args.output_tag,
    )

    print("=== Validate final preprocessed BTXRD data ===")

    if not pipeline_report_path.exists():
        raise FileNotFoundError(f"Missing pipeline report: {pipeline_report_path}")

    df = pd.read_csv(pipeline_report_path)
    records = []

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        stem = Path(image_id).stem

        raw_mask_path = RAW_MASK_DIR / f"{stem}.png"
        final_image_path = final_image_dir / f"{stem}.jpg"
        final_mask_path = final_mask_dir / f"{stem}.png"

        raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
        final_image = cv2.imread(str(final_image_path), cv2.IMREAD_GRAYSCALE)
        final_mask = cv2.imread(str(final_mask_path), cv2.IMREAD_GRAYSCALE)

        raw_mask_readable = raw_mask is not None
        final_image_readable = final_image is not None
        final_mask_readable = final_mask is not None

        raw_area = int((raw_mask > 0).sum()) if raw_mask_readable else -1
        final_area = int((final_mask > 0).sum()) if final_mask_readable else -1

        size_match = False
        final_w = -1
        final_h = -1
        if final_image_readable and final_mask_readable:
            size_match = final_image.shape[:2] == final_mask.shape[:2]
            final_h, final_w = final_image.shape[:2]

        area_diff = raw_area - final_area if raw_area >= 0 and final_area >= 0 else None
        if raw_area > 0 and final_area >= 0:
            area_keep_ratio = final_area / raw_area
        elif raw_area == 0 and final_area == 0:
            area_keep_ratio = 1.0
        else:
            area_keep_ratio = 0.0

        lost_tumor_pixels = raw_area > 0 and final_area >= 0 and final_area < raw_area
        normal_became_non_empty = raw_area == 0 and final_area > 0

        crop_ratio = float(row.get("crop_area_ratio", 1.0))
        marker_erase_area = int(row.get("marker_erase_area_pixels", 0))
        image_area = final_w * final_h if final_w > 0 and final_h > 0 else 0
        marker_erase_ratio = (
            marker_erase_area / image_area if image_area > 0 else 0.0
        )

        records.append(
            {
                "image_id": image_id,
                "raw_mask_readable": raw_mask_readable,
                "final_image_readable": final_image_readable,
                "final_mask_readable": final_mask_readable,
                "image_mask_size_match": size_match,
                "final_w": final_w,
                "final_h": final_h,
                "raw_mask_area": raw_area,
                "final_mask_area": final_area,
                "area_diff": area_diff,
                "area_keep_ratio": area_keep_ratio,
                "lost_tumor_pixels": lost_tumor_pixels,
                "normal_became_non_empty": normal_became_non_empty,
                "crop_area_ratio": crop_ratio,
                "crop_reason": row.get("crop_reason", ""),
                "mask_safety_fallback": row.get("mask_safety_fallback", False),
                "marker_removed": row.get("marker_removed", False),
                "num_markers": int(row.get("num_markers", 0)),
                "marker_erase_area_pixels": marker_erase_area,
                "marker_erase_ratio": marker_erase_ratio,
                "final_image_path": str(final_image_path),
                "final_mask_path": str(final_mask_path),
            }
        )

    report_df = pd.DataFrame(records)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(validation_report_path, index=False)

    unreadable_df = report_df[
        ~report_df["raw_mask_readable"]
        | ~report_df["final_image_readable"]
        | ~report_df["final_mask_readable"]
    ]
    size_mismatch_df = report_df[report_df["image_mask_size_match"] == False]
    tumor_lost_df = report_df[report_df["lost_tumor_pixels"] == True]
    normal_non_empty_df = report_df[report_df["normal_became_non_empty"] == True]
    aggressive_crop_df = report_df[report_df["crop_area_ratio"] < 0.50]
    large_marker_erase_df = report_df[report_df["marker_erase_ratio"] > 0.015]

    print("\n=== Validation Summary ===")
    print(f"Total samples:                  {len(report_df)}")
    print(f"Unreadable file rows:           {len(unreadable_df)}")
    print(f"Image/mask size mismatch rows:  {len(size_mismatch_df)}")
    print(f"Tumor masks with lost pixels:   {len(tumor_lost_df)}")
    print(f"Normal masks became non-empty:  {len(normal_non_empty_df)}")
    print(f"Aggressive crop ratio < 0.50:   {len(aggressive_crop_df)}")
    print(f"Marker erase ratio > 1.5%:      {len(large_marker_erase_df)}")

    if len(tumor_lost_df) == 0 and len(normal_non_empty_df) == 0:
        print("\nMask safety check passed.")
    else:
        print("\n[WARNING] Mask safety check failed. Inspect validation report.")

    if len(size_mismatch_df) > 0:
        print("\n[WARNING] Image/mask size mismatches:")
        print(size_mismatch_df.head(20).to_string(index=False))

    if len(tumor_lost_df) > 0:
        print("\n[WARNING] Tumor mask lost pixels:")
        print(tumor_lost_df.sort_values("area_keep_ratio").head(20).to_string(index=False))

    if len(large_marker_erase_df) > 0:
        print("\nLargest marker erase ratios:")
        print(
            large_marker_erase_df.sort_values("marker_erase_ratio", ascending=False)
            .head(20)[
                [
                    "image_id",
                    "num_markers",
                    "marker_erase_area_pixels",
                    "marker_erase_ratio",
                    "final_image_path",
                ]
            ]
            .to_string(index=False)
        )

    print(f"\nSaved validation report to: {validation_report_path}")


if __name__ == "__main__":
    main()
