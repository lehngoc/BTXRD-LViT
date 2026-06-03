from pathlib import Path

import cv2
import numpy as np
import pandas as pd


REPORT_PATH = Path("data/processed/reports/crop_border_report.csv")
OUTPUT_DIR = Path("data/processed/visual_checks/crop_preview")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_bbox_on_image(image_gray, x1, y1, x2, y2):
    if len(image_gray.shape) == 2:
        image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image_gray.copy()

    cv2.rectangle(image_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 4)
    return image_bgr


def resize_keep_aspect(image, target_height=512):
    h, w = image.shape[:2]
    if h == 0:
        return image

    scale = target_height / h
    new_w = max(1, int(w * scale))
    return cv2.resize(image, (new_w, target_height))


def pad_to_height(img, target_h):
    h_img, _ = img.shape[:2]
    if h_img == target_h:
        return img

    pad = target_h - h_img
    return cv2.copyMakeBorder(
        img,
        0,
        pad,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def add_text_banner(image, text):
    """
    Add a small text banner on top of image.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    banner_h = 40
    banner = np.zeros((banner_h, image.shape[1], 3), dtype=np.uint8)

    cv2.putText(
        banner,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.concatenate([banner, image], axis=0)


def main():
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Missing crop report: {REPORT_PATH}")

    df = pd.read_csv(REPORT_PATH)

    samples = []

    # Random regular cases
    samples.append(df.sample(n=min(20, len(df)), random_state=42))

    # Most aggressive crops
    samples.append(df.sort_values("crop_area_ratio", ascending=True).head(30))

    # Fallback cases
    fallback_df = df[df["fallback"] == True]
    if len(fallback_df) > 0:
        samples.append(fallback_df.sample(n=min(30, len(fallback_df)), random_state=42))

    # Mask-safety fallback cases
    if "mask_safety_fallback" in df.columns:
        safety_df = df[df["mask_safety_fallback"] == True]
        if len(safety_df) > 0:
            samples.append(safety_df.head(30))

    sample_df = pd.concat(samples, axis=0).drop_duplicates("image_id")

    for _, row in sample_df.iterrows():
        image_id = row["image_id"]
        stem = Path(image_id).stem

        image_path = Path(row["image_path"])
        clean_image_path = Path(row["clean_image_path"])

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        clean_image = cv2.imread(str(clean_image_path), cv2.IMREAD_GRAYSCALE)

        if image is None or clean_image is None:
            print(f"[WARN] Cannot read image for {image_id}")
            continue

        x1 = int(row["crop_x1"])
        y1 = int(row["crop_y1"])
        x2 = int(row["crop_x2"])
        y2 = int(row["crop_y2"])

        boxed = draw_bbox_on_image(image, x1, y1, x2, y2)

        boxed_small = resize_keep_aspect(boxed, target_height=512)
        clean_small = resize_keep_aspect(
            cv2.cvtColor(clean_image, cv2.COLOR_GRAY2BGR),
            target_height=512,
        )

        boxed_small = add_text_banner(boxed_small, "Original + crop bbox")
        clean_small = add_text_banner(clean_small, "Clean crop")

        h1, _ = boxed_small.shape[:2]
        h2, _ = clean_small.shape[:2]
        h = max(h1, h2)

        boxed_small = pad_to_height(boxed_small, h)
        clean_small = pad_to_height(clean_small, h)

        preview = np.concatenate([boxed_small, clean_small], axis=1)

        reason = str(row.get("reason", "unknown"))
        fallback = str(row.get("fallback", "unknown"))
        mask_safety = str(row.get("mask_safety_fallback", "unknown"))

        output_name = (
            f"{stem}_ratio-{row['crop_area_ratio']:.3f}_"
            f"fallback-{fallback}_"
            f"maskSafety-{mask_safety}_"
            f"reason-{reason}.jpg"
        )

        # Avoid illegal filename chars
        output_name = output_name.replace("/", "-").replace("\\", "-").replace(":", "-")

        output_path = OUTPUT_DIR / output_name
        cv2.imwrite(str(output_path), preview)

    print(f"Saved crop previews to: {OUTPUT_DIR}")
    print(f"Total previews saved: {len(list(OUTPUT_DIR.glob('*.jpg')))}")


if __name__ == "__main__":
    main()