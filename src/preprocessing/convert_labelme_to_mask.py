from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
ANNOTATION_DIR = RAW_DIR / "Annotations"
CSV_PATH = RAW_DIR / "dataset.csv"

OUTPUT_MASK_DIR = Path("data/processed/masks")
OUTPUT_REPORT_DIR = Path("data/processed/reports")

OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_file(image_id: str) -> Path | None:
    """
    BTXRD paper mentions .jpeg, but Kaggle version may use .jpg.
    This function tries common image extensions.
    """
    image_id = str(image_id)
    stem = Path(image_id).stem

    candidates = [
        IMAGE_DIR / image_id,
        IMAGE_DIR / f"{stem}.jpg",
        IMAGE_DIR / f"{stem}.jpeg",
        IMAGE_DIR / f"{stem}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_annotation_file(image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
    path = ANNOTATION_DIR / f"{stem}.json"
    return path if path.exists() else None


def load_image_size(image_path: Path) -> tuple[int, int]:
    """
    Return width, height.
    """
    with Image.open(image_path) as img:
        return img.size


def polygon_to_mask(points: list, height: int, width: int) -> np.ndarray:
    """
    Convert one polygon to a binary mask.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(points) < 3:
        return mask

    polygon = np.array(points, dtype=np.float32)
    polygon = np.round(polygon).astype(np.int32)

    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)

    cv2.fillPoly(mask, [polygon], 255)
    return mask


def rectangle_to_mask(points: list, height: int, width: int) -> np.ndarray:
    """
    Fallback: convert LabelMe rectangle to mask.
    We only use this if no polygon exists in the JSON.
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    if len(points) != 2:
        return mask

    x1, y1 = points[0]
    x2, y2 = points[1]

    x1, x2 = sorted([int(round(x1)), int(round(x2))])
    y1, y2 = sorted([int(round(y1)), int(round(y2))])

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask


def create_mask_from_labelme(json_path: Path, height: int, width: int) -> tuple[np.ndarray, dict]:
    """
    Create binary segmentation mask from a LabelMe JSON file.

    Priority:
    1. Use polygon shapes for segmentation.
    2. Use rectangle only as fallback if no polygon exists.

    This avoids accidentally using bounding boxes as segmentation masks.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])

    final_mask = np.zeros((height, width), dtype=np.uint8)

    polygon_count = 0
    rectangle_count = 0
    used_rectangle_fallback = False

    polygon_masks = []

    for shape in shapes:
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if shape_type == "polygon":
            poly_mask = polygon_to_mask(points, height, width)
            if poly_mask.sum() > 0:
                polygon_masks.append(poly_mask)
                polygon_count += 1

        elif shape_type == "rectangle":
            rectangle_count += 1

    if len(polygon_masks) > 0:
        for poly_mask in polygon_masks:
            final_mask = np.maximum(final_mask, poly_mask)
    else:
        # Fallback: if the JSON contains no polygon, use rectangle.
        for shape in shapes:
            shape_type = shape.get("shape_type", "")
            points = shape.get("points", [])

            if shape_type == "rectangle":
                rect_mask = rectangle_to_mask(points, height, width)
                if rect_mask.sum() > 0:
                    final_mask = np.maximum(final_mask, rect_mask)
                    used_rectangle_fallback = True

    info = {
        "num_shapes": len(shapes),
        "polygon_count": polygon_count,
        "rectangle_count": rectangle_count,
        "used_rectangle_fallback": used_rectangle_fallback,
        "mask_area_pixels": int((final_mask > 0).sum()),
    }

    return final_mask, info


def main():
    print("=== Convert BTXRD LabelMe JSON to binary masks ===")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "image_id" not in df.columns:
        raise ValueError("dataset.csv must contain an 'image_id' column.")

    if "tumor" not in df.columns:
        raise ValueError("dataset.csv must contain a 'tumor' column.")

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting masks"):
        image_id = str(row["image_id"])
        stem = Path(image_id).stem
        tumor = int(row["tumor"])

        image_path = find_image_file(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing image file for {image_id}")

        width, height = load_image_size(image_path)

        output_mask_path = OUTPUT_MASK_DIR / f"{stem}.png"

        annotation_path = find_annotation_file(image_id)

        if tumor == 1:
            if annotation_path is None:
                raise FileNotFoundError(f"Tumor image missing annotation JSON: {image_id}")

            mask, info = create_mask_from_labelme(annotation_path, height, width)

            if info["mask_area_pixels"] == 0:
                print(f"[WARN] Empty mask generated for tumor image: {image_id}")

        else:
            # Normal image: expected mask is empty.
            mask = np.zeros((height, width), dtype=np.uint8)
            info = {
                "num_shapes": 0,
                "polygon_count": 0,
                "rectangle_count": 0,
                "used_rectangle_fallback": False,
                "mask_area_pixels": 0,
            }

        cv2.imwrite(str(output_mask_path), mask)

        records.append(
            {
                "image_id": image_id,
                "tumor": tumor,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path) if annotation_path else "",
                "mask_path": str(output_mask_path),
                "width": width,
                "height": height,
                **info,
            }
        )

    report_df = pd.DataFrame(records)
    report_path = OUTPUT_REPORT_DIR / "mask_conversion_report.csv"
    report_df.to_csv(report_path, index=False)

    total_masks = len(list(OUTPUT_MASK_DIR.glob("*.png")))
    tumor_mask_empty = int(((report_df["tumor"] == 1) & (report_df["mask_area_pixels"] == 0)).sum())
    normal_mask_non_empty = int(((report_df["tumor"] == 0) & (report_df["mask_area_pixels"] > 0)).sum())
    rectangle_fallback_count = int(report_df["used_rectangle_fallback"].sum())

    print("\n=== Mask Conversion Summary ===")
    print(f"Total masks saved: {total_masks}")
    print(f"Tumor masks empty: {tumor_mask_empty}")
    print(f"Normal masks non-empty: {normal_mask_non_empty}")
    print(f"Rectangle fallback count: {rectangle_fallback_count}")
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()