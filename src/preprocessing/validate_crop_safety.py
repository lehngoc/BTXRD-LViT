from pathlib import Path

import cv2
import pandas as pd


REPORT_PATH = Path("data/processed/reports/crop_border_report.csv")
RAW_MASK_DIR = Path("data/processed/masks")
CLEAN_MASK_DIR = Path("data/processed/masks_clean")
OUTPUT_REPORT_PATH = Path("data/processed/reports/crop_safety_report.csv")


def main():
    print("=== Validate crop safety for tumor masks ===")

    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing crop report: {REPORT_PATH}")

    df = pd.read_csv(REPORT_PATH)

    records = []

    for _, row in df.iterrows():
        image_id = str(row["image_id"])
        stem = Path(image_id).stem

        raw_mask_path = RAW_MASK_DIR / f"{stem}.png"
        clean_mask_path = CLEAN_MASK_DIR / f"{stem}.png"

        raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)
        clean_mask = cv2.imread(str(clean_mask_path), cv2.IMREAD_GRAYSCALE)

        if raw_mask is None:
            raise ValueError(f"Cannot read raw mask: {raw_mask_path}")

        if clean_mask is None:
            raise ValueError(f"Cannot read clean mask: {clean_mask_path}")

        raw_area = int((raw_mask > 0).sum())
        clean_area = int((clean_mask > 0).sum())

        area_diff = raw_area - clean_area

        if raw_area > 0:
            area_keep_ratio = clean_area / raw_area
        else:
            area_keep_ratio = 1.0 if clean_area == 0 else 0.0

        lost_pixels = raw_area > 0 and clean_area < raw_area
        normal_became_non_empty = raw_area == 0 and clean_area > 0

        records.append(
            {
                "image_id": image_id,
                "raw_mask_area": raw_area,
                "clean_mask_area": clean_area,
                "area_diff": area_diff,
                "area_keep_ratio": area_keep_ratio,
                "lost_pixels": lost_pixels,
                "normal_became_non_empty": normal_became_non_empty,
                "crop_area_ratio": row.get("crop_area_ratio", None),
                "fallback": row.get("fallback", None),
                "reason": row.get("reason", None),
                "mask_safety_fallback": row.get("mask_safety_fallback", None),
            }
        )

    report_df = pd.DataFrame(records)
    report_df.to_csv(OUTPUT_REPORT_PATH, index=False)

    tumor_df = report_df[report_df["raw_mask_area"] > 0]
    normal_df = report_df[report_df["raw_mask_area"] == 0]

    tumor_lost_df = tumor_df[tumor_df["lost_pixels"] == True]
    normal_non_empty_df = normal_df[normal_df["normal_became_non_empty"] == True]

    print("\n=== Crop Safety Summary ===")
    print(f"Total samples: {len(report_df)}")
    print(f"Tumor samples: {len(tumor_df)}")
    print(f"Normal samples: {len(normal_df)}")
    print(f"Tumor masks with lost pixels after crop: {len(tumor_lost_df)}")
    print(f"Normal masks becoming non-empty after crop: {len(normal_non_empty_df)}")

    if len(tumor_lost_df) > 0:
        print("\n[WARNING] Some tumor masks lost pixels after crop.")
        print(
            tumor_lost_df.sort_values("area_keep_ratio")
            .head(20)
            .to_string(index=False)
        )
    else:
        print("\nAll tumor masks are preserved after crop.")

    if len(normal_non_empty_df) > 0:
        print("\n[WARNING] Some normal masks became non-empty after crop.")
        print(normal_non_empty_df.head(20).to_string(index=False))
    else:
        print("All normal masks remain empty after crop.")

    print(f"\nSaved report to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()