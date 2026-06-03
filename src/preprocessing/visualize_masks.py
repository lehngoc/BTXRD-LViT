from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd


IMAGE_DIR = Path("data/raw/images")
MASK_DIR = Path("data/processed/masks")
CSV_PATH = Path("data/raw/dataset.csv")
OUTPUT_DIR = Path("data/processed/visual_checks/mask_overlay")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_image_file(image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
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


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Create a simple red overlay for mask area.
    """
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()

    overlay = image_rgb.copy()
    overlay[mask > 0] = [0, 0, 255]

    blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
    return blended


def main():
    df = pd.read_csv(CSV_PATH)

    tumor_df = df[df["tumor"] == 1].copy()
    normal_df = df[df["tumor"] == 0].copy()

    tumor_samples = tumor_df.sample(n=min(20, len(tumor_df)), random_state=42)
    normal_samples = normal_df.sample(n=min(10, len(normal_df)), random_state=42)

    samples = pd.concat([tumor_samples, normal_samples], axis=0)

    for _, row in samples.iterrows():
        image_id = str(row["image_id"])
        stem = Path(image_id).stem
        tumor = int(row["tumor"])

        image_path = find_image_file(image_id)
        mask_path = MASK_DIR / f"{stem}.png"

        if image_path is None or not mask_path.exists():
            print(f"[WARN] Missing image or mask for {image_id}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"[WARN] Cannot read image or mask for {image_id}")
            continue

        overlay = make_overlay(image, mask)

        label = "tumor" if tumor == 1 else "normal"
        output_path = OUTPUT_DIR / f"{stem}_{label}_overlay.jpg"
        cv2.imwrite(str(output_path), overlay)

    print(f"Saved visual checks to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()