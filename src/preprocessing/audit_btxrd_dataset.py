from pathlib import Path
import json
import pandas as pd
from PIL import Image


RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
ANNOTATION_DIR = RAW_DIR / "Annotations"
CSV_PATH = RAW_DIR / "dataset.csv"

REPORT_DIR = Path("data/processed/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_file(image_id: str) -> Path | None:
    """
    BTXRD paper mentions .jpeg, but Kaggle version may use .jpg.
    This function tries common image extensions.
    """
    candidates = [
        IMAGE_DIR / image_id,
        IMAGE_DIR / image_id.replace(".jpeg", ".jpg"),
        IMAGE_DIR / image_id.replace(".jpg", ".jpeg"),
        IMAGE_DIR / f"{Path(image_id).stem}.jpg",
        IMAGE_DIR / f"{Path(image_id).stem}.jpeg",
        IMAGE_DIR / f"{Path(image_id).stem}.png",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def find_annotation_file(image_id: str) -> Path | None:
    stem = Path(image_id).stem
    path = ANNOTATION_DIR / f"{stem}.json"
    return path if path.exists() else None


def safe_sum(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def main():
    print("=== BTXRD Dataset Audit ===")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing image folder: {IMAGE_DIR}")

    if not ANNOTATION_DIR.exists():
        raise FileNotFoundError(f"Missing annotation folder: {ANNOTATION_DIR}")

    df = pd.read_csv(CSV_PATH)

    if "image_id" not in df.columns:
        raise ValueError("dataset.csv must contain an 'image_id' column.")

    image_files = sorted(
        list(IMAGE_DIR.glob("*.jpg"))
        + list(IMAGE_DIR.glob("*.jpeg"))
        + list(IMAGE_DIR.glob("*.png"))
    )
    annotation_files = sorted(ANNOTATION_DIR.glob("*.json"))

    print(f"Rows in dataset.csv: {len(df)}")
    print(f"Image files found:   {len(image_files)}")
    print(f"JSON files found:    {len(annotation_files)}")

    summary = {
        "csv_rows": int(len(df)),
        "image_files": int(len(image_files)),
        "annotation_json_files": int(len(annotation_files)),
        "normal_count": safe_sum(df, "tumor") and int((df["tumor"] == 0).sum()) if "tumor" in df.columns else None,
        "tumor_count": safe_sum(df, "tumor"),
        "benign_count": safe_sum(df, "benign"),
        "malignant_count": safe_sum(df, "malignant"),
    }

    print("\n=== Label Distribution ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    records = []
    missing_images = []
    missing_annotations_for_tumor = []
    annotations_for_normal = []

    widths = []
    heights = []

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        tumor = int(row["tumor"]) if "tumor" in df.columns and pd.notna(row["tumor"]) else 0

        image_path = find_image_file(image_id)
        annotation_path = find_annotation_file(image_id)

        has_image = image_path is not None
        has_annotation = annotation_path is not None

        width = None
        height = None

        if has_image:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)
            except Exception as e:
                print(f"[WARN] Cannot read image {image_path}: {e}")
        else:
            missing_images.append(image_id)

        if tumor == 1 and not has_annotation:
            missing_annotations_for_tumor.append(image_id)

        if tumor == 0 and has_annotation:
            annotations_for_normal.append(image_id)

        records.append(
            {
                "image_id": image_id,
                "tumor": tumor,
                "has_image": has_image,
                "image_path": str(image_path) if image_path else "",
                "has_annotation": has_annotation,
                "annotation_path": str(annotation_path) if annotation_path else "",
                "width": width,
                "height": height,
            }
        )

    audit_df = pd.DataFrame(records)
    audit_csv_path = REPORT_DIR / "btxrd_audit_details.csv"
    audit_df.to_csv(audit_csv_path, index=False)

    if widths and heights:
        summary.update(
            {
                "min_width": int(min(widths)),
                "max_width": int(max(widths)),
                "mean_width": float(sum(widths) / len(widths)),
                "min_height": int(min(heights)),
                "max_height": int(max(heights)),
                "mean_height": float(sum(heights) / len(heights)),
            }
        )

    summary.update(
        {
            "missing_images_count": len(missing_images),
            "missing_annotations_for_tumor_count": len(missing_annotations_for_tumor),
            "annotations_for_normal_count": len(annotations_for_normal),
            "missing_images_examples": missing_images[:20],
            "missing_annotations_for_tumor_examples": missing_annotations_for_tumor[:20],
            "annotations_for_normal_examples": annotations_for_normal[:20],
        }
    )

    summary_json_path = REPORT_DIR / "btxrd_audit_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Consistency Checks ===")
    print(f"Missing images: {len(missing_images)}")
    print(f"Tumor images missing JSON: {len(missing_annotations_for_tumor)}")
    print(f"Normal images with JSON: {len(annotations_for_normal)}")

    if widths and heights:
        print("\n=== Image Size Statistics ===")
        print(f"Width:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.2f}")
        print(f"Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.2f}")

    print("\nSaved reports:")
    print(f"- {audit_csv_path}")
    print(f"- {summary_json_path}")


if __name__ == "__main__":
    main()