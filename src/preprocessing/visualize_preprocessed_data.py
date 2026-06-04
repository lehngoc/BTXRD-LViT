from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import pandas as pd


FINAL_IMAGE_DIR = Path("data/processed/images_preprocessed")
FINAL_MASK_DIR = Path("data/processed/masks_preprocessed")
REPORT_DIR = Path("data/processed/reports")
OUTPUT_DIR = Path("data/processed/visual_checks/preprocess_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize final BTXRD preprocessing.")
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix used by preprocess_btxrd_data.py.",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=160,
        help="Maximum number of preview images to write.",
    )
    return parser.parse_args()


def tagged_path(path: Path, output_tag: str) -> Path:
    if not output_tag:
        return path
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return path.with_name(f"{path.name}_{safe_tag}")


def tagged_report_path(output_tag: str) -> Path:
    if not output_tag:
        return REPORT_DIR / "preprocess_pipeline_report.csv"
    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return REPORT_DIR / f"preprocess_pipeline_report_{safe_tag}.csv"


def resize_keep_aspect(image: np.ndarray, target_height: int = 640) -> np.ndarray:
    h, w = image.shape[:2]
    if h == 0:
        return image
    scale = target_height / h
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(image, (new_w, target_height))


def pad_to_height(image: np.ndarray, target_h: int) -> np.ndarray:
    h, _ = image.shape[:2]
    if h == target_h:
        return image
    return cv2.copyMakeBorder(
        image,
        0,
        target_h - h,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


def add_banner(image: np.ndarray, text: str) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    banner_h = 44
    banner = np.zeros((banner_h, image.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        banner,
        text,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([banner, image], axis=0)


def overlay_mask(image_gray: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    mask = mask_gray > 0
    if mask.any():
        overlay = image_bgr.copy()
        overlay[mask] = (0, 0, 255)
        image_bgr = cv2.addWeighted(overlay, 0.35, image_bgr, 0.65, 0)
    return image_bgr


def draw_crop_box(image_gray: np.ndarray, row: pd.Series) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    x1 = int(row["crop_x1"])
    y1 = int(row["crop_y1"])
    x2 = int(row["crop_x2"])
    y2 = int(row["crop_y2"])
    cv2.rectangle(image_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 5)
    return image_bgr


def draw_marker_boxes(image_gray: np.ndarray, row: pd.Series) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    try:
        boxes = json.loads(row.get("marker_boxes", "[]"))
    except json.JSONDecodeError:
        boxes = []

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
    return image_bgr


def make_preview(row: pd.Series, output_path: Path) -> bool:
    raw_image = cv2.imread(str(row["image_path"]), cv2.IMREAD_GRAYSCALE)
    final_image = cv2.imread(str(row["final_image_path"]), cv2.IMREAD_GRAYSCALE)
    final_mask = cv2.imread(str(row["final_mask_path"]), cv2.IMREAD_GRAYSCALE)

    if raw_image is None or final_image is None or final_mask is None:
        print(f"[WARN] Cannot read preview inputs for {row['image_id']}")
        return False

    raw_boxed = draw_crop_box(raw_image, row)
    final_markers = draw_marker_boxes(final_image, row)
    final_overlay = overlay_mask(final_image, final_mask)

    ratio = float(row["crop_area_ratio"])
    marker_count = int(row["num_markers"])
    crop_reason = str(row["crop_reason"])
    mask_safety = str(row["mask_safety_fallback"])

    panels = [
        add_banner(
            resize_keep_aspect(raw_boxed),
            f"Raw + crop bbox | ratio={ratio:.3f} | {crop_reason}",
        ),
        add_banner(
            resize_keep_aspect(final_markers),
            f"Final image + marker boxes | markers={marker_count}",
        ),
        add_banner(
            resize_keep_aspect(final_overlay),
            f"Final mask overlay | maskSafety={mask_safety}",
        ),
    ]

    target_h = max(panel.shape[0] for panel in panels)
    panels = [pad_to_height(panel, target_h) for panel in panels]
    preview = np.concatenate(panels, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)
    return True


def select_samples(df: pd.DataFrame, max_previews: int) -> pd.DataFrame:
    samples = []

    samples.append(df.sample(n=min(30, len(df)), random_state=42))
    samples.append(df.sort_values("crop_area_ratio", ascending=True).head(35))
    samples.append(df.sort_values("edge_trim_total", ascending=False).head(25))
    samples.append(df[df["mask_safety_fallback"] == True].head(30))
    samples.append(df[df["marker_removed"] == True].head(35))
    samples.append(df.sort_values("marker_erase_area_pixels", ascending=False).head(30))

    sample_df = pd.concat(samples, axis=0).drop_duplicates("image_id")
    if len(sample_df) > max_previews:
        sample_df = sample_df.head(max_previews)
    return sample_df


def main():
    args = parse_args()
    report_path = tagged_report_path(args.output_tag)
    output_dir = tagged_path(OUTPUT_DIR, args.output_tag)

    if not report_path.exists():
        raise FileNotFoundError(f"Missing pipeline report: {report_path}")

    df = pd.read_csv(report_path)
    sample_df = select_samples(df, args.max_previews)

    saved = 0
    for _, row in sample_df.iterrows():
        stem = Path(str(row["image_id"])).stem
        output_name = (
            f"{stem}_ratio-{float(row['crop_area_ratio']):.3f}_"
            f"markers-{int(row['num_markers'])}_"
            f"safety-{row['mask_safety_fallback']}_"
            f"reason-{row['crop_reason']}.jpg"
        )
        output_name = output_name.replace("/", "-").replace("\\", "-").replace(":", "-")
        if make_preview(row, output_dir / output_name):
            saved += 1

    print(f"Saved preprocess previews to: {output_dir}")
    print(f"Total previews saved: {saved}")


if __name__ == "__main__":
    main()
