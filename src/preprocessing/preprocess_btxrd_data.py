from pathlib import Path
import argparse
import json
import shutil

import cv2
import pandas as pd
from tqdm import tqdm

from crop_xray_border import detect_foreground_bbox
from remove_xray_markers import detect_marker_artifacts, remove_markers


RAW_DIR = Path("data/raw")
RAW_IMAGE_DIR = RAW_DIR / "images"
CSV_PATH = RAW_DIR / "dataset.csv"

RAW_MASK_DIR = Path("data/processed/masks")

FINAL_IMAGE_DIR = Path("data/processed/images_preprocessed")
FINAL_MASK_DIR = Path("data/processed/masks_preprocessed")
REPORT_DIR = Path("data/processed/reports")
PREVIEW_DIR = Path("data/processed/visual_checks/preprocess_pipeline")

PIPELINE_REPORT_PATH = REPORT_DIR / "preprocess_pipeline_report.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop X-ray borders and remove safe L/R marker artifacts."
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N samples from dataset.csv.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix for trial outputs, e.g. trial200.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing outputs for the selected output tag before running.",
    )
    return parser.parse_args()


def tagged_path(path: Path, output_tag: str) -> Path:
    if not output_tag:
        return path
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return path.with_name(f"{path.name}_{safe_tag}")


def tagged_report_path(output_tag: str) -> Path:
    if not output_tag:
        return PIPELINE_REPORT_PATH
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return REPORT_DIR / f"preprocess_pipeline_report_{safe_tag}.csv"


def resolve_outputs(output_tag: str) -> tuple[Path, Path, Path, Path]:
    return (
        tagged_path(FINAL_IMAGE_DIR, output_tag),
        tagged_path(FINAL_MASK_DIR, output_tag),
        tagged_report_path(output_tag),
        tagged_path(PREVIEW_DIR, output_tag),
    )


def clean_outputs(paths: list[Path]) -> None:
    workspace = Path.cwd().resolve()
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if workspace not in resolved.parents and resolved != workspace:
            raise ValueError(f"Refusing to delete outside workspace: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()


def find_image_file(image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
    candidates = [
        RAW_IMAGE_DIR / image_id,
        RAW_IMAGE_DIR / f"{stem}.jpg",
        RAW_IMAGE_DIR / f"{stem}.jpeg",
        RAW_IMAGE_DIR / f"{stem}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def empty_trim_meta() -> dict:
    return {
        "edge_trim_top": 0,
        "edge_trim_bottom": 0,
        "edge_trim_left": 0,
        "edge_trim_right": 0,
        "edge_trim_total": 0,
        "extra_vertical_frame_trim_left": 0,
        "extra_vertical_frame_trim_right": 0,
        "extra_vertical_frame_trim_total": 0,
    }


def marker_public_meta(markers: list[dict]) -> tuple[list[list[int]], list[list[int]]]:
    boxes = [marker["bbox"] for marker in markers]
    centers = [[marker["center_x"], marker["center_y"]] for marker in markers]
    return boxes, centers


def process_one(image_path: Path, mask_path: Path) -> tuple[dict, object, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image/mask size mismatch for {image_path.name}: "
            f"image={image.shape[:2]}, mask={mask.shape[:2]}"
        )

    h, w = image.shape[:2]
    raw_mask_area = int((mask > 0).sum())

    x1, y1, x2, y2, fallback, reason, trim_meta = detect_foreground_bbox(image)

    crop_image = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    crop_mask_area_before_safety = int((crop_mask > 0).sum())

    mask_safety_fallback = False
    if raw_mask_area > 0 and crop_mask_area_before_safety < raw_mask_area:
        x1, y1, x2, y2 = 0, 0, w, h
        crop_image = image
        crop_mask = mask
        fallback = True
        reason = "mask_safety_fallback"
        mask_safety_fallback = True
        trim_meta = empty_trim_meta()

    crop_mask_area_after_safety = int((crop_mask > 0).sum())

    markers = detect_marker_artifacts(crop_image, crop_mask)
    marker_clean_image, erase_mask = remove_markers(crop_image, markers)
    marker_boxes, marker_centers = marker_public_meta(markers)

    crop_w = x2 - x1
    crop_h = y2 - y1
    crop_area_ratio = (crop_w * crop_h) / (w * h)

    meta = {
        "orig_w": w,
        "orig_h": h,
        "crop_x1": x1,
        "crop_y1": y1,
        "crop_x2": x2,
        "crop_y2": y2,
        "final_w": crop_w,
        "final_h": crop_h,
        "crop_area_ratio": crop_area_ratio,
        "crop_fallback": bool(fallback),
        "crop_reason": reason,
        "mask_safety_fallback": bool(mask_safety_fallback),
        "raw_mask_area": raw_mask_area,
        "crop_mask_area_before_safety": crop_mask_area_before_safety,
        "final_mask_area": crop_mask_area_after_safety,
        **trim_meta,
        "marker_removed": bool(markers),
        "num_markers": len(markers),
        "marker_boxes": json.dumps(marker_boxes),
        "marker_centers": json.dumps(marker_centers),
        "marker_erase_area_pixels": int((erase_mask > 0).sum()),
    }

    return meta, marker_clean_image, crop_mask


def main():
    args = parse_args()
    output_image_dir, output_mask_dir, report_path, preview_dir = resolve_outputs(
        args.output_tag
    )

    if args.clean_output:
        clean_outputs([output_image_dir, output_mask_dir, report_path, preview_dir])

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=== BTXRD full preprocessing: crop borders + remove markers ===")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    if "image_id" not in df.columns:
        raise ValueError("dataset.csv must contain an 'image_id' column.")

    if args.max_images is not None:
        df = df.head(args.max_images)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        image_id = str(row["image_id"])
        stem = Path(image_id).stem

        image_path = find_image_file(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing image file for {image_id}")

        mask_path = RAW_MASK_DIR / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file for {image_id}: {mask_path}")

        meta, final_image, final_mask = process_one(image_path, mask_path)

        output_image_path = output_image_dir / f"{stem}.jpg"
        output_mask_path = output_mask_dir / f"{stem}.png"

        cv2.imwrite(str(output_image_path), final_image)
        cv2.imwrite(str(output_mask_path), final_mask)

        records.append(
            {
                "image_id": image_id,
                **meta,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "final_image_path": str(output_image_path),
                "final_mask_path": str(output_mask_path),
            }
        )

    report_df = pd.DataFrame(records)
    report_df.to_csv(report_path, index=False)

    print("\n=== Preprocessing Summary ===")
    print(f"Total images processed:      {len(report_df)}")
    print(f"Final images saved:          {len(list(output_image_dir.glob('*.jpg')))}")
    print(f"Final masks saved:           {len(list(output_mask_dir.glob('*.png')))}")
    print(f"Crop fallback count:         {int(report_df['crop_fallback'].sum())}")
    print(f"Mask-safety fallback count:  {int(report_df['mask_safety_fallback'].sum())}")
    print(f"Marker-removed image count:  {int(report_df['marker_removed'].sum())}")
    print(f"Total markers removed:       {int(report_df['num_markers'].sum())}")
    print(f"Mean crop area ratio:        {report_df['crop_area_ratio'].mean():.4f}")
    print(f"Min crop area ratio:         {report_df['crop_area_ratio'].min():.4f}")
    print(f"Report saved to:             {report_path}")
    print(f"Final images:                {output_image_dir}")
    print(f"Final masks:                 {output_mask_dir}")

    print("\nCrop reasons:")
    print(report_df["crop_reason"].value_counts())


if __name__ == "__main__":
    main()
