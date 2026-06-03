from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


RAW_DIR = Path("data/raw")
IMAGE_DIR = RAW_DIR / "images"
CSV_PATH = RAW_DIR / "dataset.csv"

MASK_DIR = Path("data/processed/masks")

CLEAN_IMAGE_DIR = Path("data/processed/images_clean")
CLEAN_MASK_DIR = Path("data/processed/masks_clean")
REPORT_DIR = Path("data/processed/reports")

CLEAN_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_MASK_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def is_low_information_edge(
    strip: np.ndarray,
    dark_threshold: int = 12,
    bright_threshold: int = 245,
    min_mid_ratio: float = 0.08,
    dark_ratio_threshold: float = 0.55,
    bright_ratio_threshold: float = 0.35,
) -> bool:
    """
    Detect whether an edge strip is likely to be irrelevant border.

    This handles both:
    - black border/background
    - white/saturated border/background

    The decision is image-only and does not use masks.
    """
    if strip.size == 0:
        return False

    strip = strip.astype(np.uint8)

    dark_ratio = float((strip <= dark_threshold).mean())
    bright_ratio = float((strip >= bright_threshold).mean())
    mid_ratio = float(((strip > dark_threshold) & (strip < bright_threshold)).mean())

    # Very dark edge: common black X-ray background
    if dark_ratio >= dark_ratio_threshold and mid_ratio < 0.35:
        return True

    # Very bright/saturated edge: white scanner border / page background
    if bright_ratio >= bright_ratio_threshold and mid_ratio < 0.45:
        return True

    # Edge mostly extreme values, very little useful radiograph content
    if (dark_ratio + bright_ratio) >= 0.80 and mid_ratio <= min_mid_ratio:
        return True

    return False


def is_outer_white_or_gray_frame(
    strip: np.ndarray,
    side: str,
    dark_threshold: int = 25,
    bright_threshold: int = 235,
) -> bool:
    """
    Detect soft white/gray outer frame regions that are not saturated enough
    for the extreme-edge detector.

    This is intentionally conservative because real anatomy can also be bright.
    """
    if strip.size == 0:
        return False

    strip = strip.astype(np.uint8)

    mean_val = float(strip.mean())
    std_val = float(strip.std())
    dark_ratio = float((strip <= dark_threshold).mean())
    bright_ratio = float((strip >= bright_threshold).mean())
    very_bright_ratio = float((strip >= 245).mean())

    # Strong white frame with little dark radiograph content.
    if bright_ratio > 0.45 and dark_ratio < 0.08:
        return True

    # Flat bright/gray frame, including scanner or page-like borders.
    if mean_val > 178 and dark_ratio < 0.03 and std_val < 50:
        return True

    # Horizontal frames can be gray rather than saturated white.
    if side in {"top", "bottom"}:
        if mean_val > 170 and dark_ratio < 0.02 and std_val < 55:
            return True

    # Vertical white frames are often narrow and may include anti-aliased edges.
    if side in {"left", "right"}:
        if mean_val > 185 and dark_ratio < 0.03 and std_val < 55:
            return True
        if very_bright_ratio > 0.25 and dark_ratio < 0.05 and std_val < 85:
            return True

    return False


def is_flat_vertical_frame_extension(
    strip: np.ndarray,
    dark_threshold: int = 25,
) -> bool:
    """
    Detect very flat left/right white-gray frame remnants after normal trimming
    already reached the conservative per-side limit.
    """
    if strip.size == 0:
        return False

    strip = strip.astype(np.uint8)

    mean_val = float(strip.mean())
    std_val = float(strip.std())
    dark_ratio = float((strip <= dark_threshold).mean())

    return mean_val > 160 and std_val < 25 and dark_ratio < 0.02


def trim_extreme_edges(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_trim_ratio: float = 0.20,
    max_vertical_frame_trim_ratio: float = 0.30,
    strip_ratio: float = 0.006,
) -> tuple[int, int, int, int, dict]:
    """
    Refine an image-only bbox by trimming low-information edge artifacts.

    This is designed to remove black/white/gray frame borders from X-ray images.
    It does NOT use mask information.

    Args:
        image_gray: grayscale image
        bbox: initial crop bbox, x1, y1, x2, y2, where x2/y2 are exclusive
        max_trim_ratio: maximum fraction of image size allowed to trim per side
        max_vertical_frame_trim_ratio: extra guarded limit for flat left/right frames
        strip_ratio: strip thickness used to test each edge

    Returns:
        refined bbox and trim metadata
    """
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox

    min_crop_w = int(0.25 * w)
    min_crop_h = int(0.25 * h)

    strip = max(4, int(min(h, w) * strip_ratio))

    max_trim_x = int(w * max_trim_ratio)
    max_trim_y = int(h * max_trim_ratio)
    max_vertical_frame_trim_x = int(w * max_vertical_frame_trim_ratio)

    trim_left = 0
    trim_right = 0
    trim_top = 0
    trim_bottom = 0
    extra_trim_left = 0
    extra_trim_right = 0

    # Trim top
    while (
        y1 + strip < y2
        and trim_top < max_trim_y
        and (y2 - (y1 + strip)) >= min_crop_h
    ):
        edge = image_gray[y1 : y1 + strip, x1:x2]
        if not (
            is_low_information_edge(edge)
            or is_outer_white_or_gray_frame(edge, side="top")
        ):
            break
        y1 += strip
        trim_top += strip

    # Trim bottom
    while (
        y2 - strip > y1
        and trim_bottom < max_trim_y
        and ((y2 - strip) - y1) >= min_crop_h
    ):
        edge = image_gray[y2 - strip : y2, x1:x2]
        if not (
            is_low_information_edge(edge)
            or is_outer_white_or_gray_frame(edge, side="bottom")
        ):
            break
        y2 -= strip
        trim_bottom += strip

    # Trim left
    while (
        x1 + strip < x2
        and trim_left < max_trim_x
        and (x2 - (x1 + strip)) >= min_crop_w
    ):
        edge = image_gray[y1:y2, x1 : x1 + strip]
        if not (
            is_low_information_edge(edge)
            or is_outer_white_or_gray_frame(edge, side="left")
        ):
            break
        x1 += strip
        trim_left += strip

    # Trim right
    while (
        x2 - strip > x1
        and trim_right < max_trim_x
        and ((x2 - strip) - x1) >= min_crop_w
    ):
        edge = image_gray[y1:y2, x2 - strip : x2]
        if not (
            is_low_information_edge(edge)
            or is_outer_white_or_gray_frame(edge, side="right")
        ):
            break
        x2 -= strip
        trim_right += strip

    # Extra guarded vertical trim:
    # Some BTXRD images keep a very wide white/gray scanner frame after the
    # regular 20% per-side cap. Only extend left/right trimming when the
    # normal trim already reached the cap and the remaining edge is extremely
    # flat, so real anatomy is unlikely to be removed.
    while (
        x1 + strip < x2
        and trim_left >= max_trim_x
        and trim_left < max_vertical_frame_trim_x
        and (x2 - (x1 + strip)) >= min_crop_w
    ):
        edge = image_gray[y1:y2, x1 : x1 + strip]
        if not is_flat_vertical_frame_extension(edge):
            break
        x1 += strip
        trim_left += strip
        extra_trim_left += strip

    while (
        x2 - strip > x1
        and trim_right >= max_trim_x
        and trim_right < max_vertical_frame_trim_x
        and ((x2 - strip) - x1) >= min_crop_w
    ):
        edge = image_gray[y1:y2, x2 - strip : x2]
        if not is_flat_vertical_frame_extension(edge):
            break
        x2 -= strip
        trim_right += strip
        extra_trim_right += strip

    meta = {
        "edge_trim_top": trim_top,
        "edge_trim_bottom": trim_bottom,
        "edge_trim_left": trim_left,
        "edge_trim_right": trim_right,
        "edge_trim_total": trim_top + trim_bottom + trim_left + trim_right,
        "extra_vertical_frame_trim_left": extra_trim_left,
        "extra_vertical_frame_trim_right": extra_trim_right,
        "extra_vertical_frame_trim_total": extra_trim_left + extra_trim_right,
    }

    return x1, y1, x2, y2, meta


def detect_foreground_bbox(
    image_gray: np.ndarray,
    margin_ratio: float = 0.03,
    min_area_ratio: float = 0.05,
) -> tuple[int, int, int, int, bool, str, dict]:
    """
    Detect the main radiograph foreground and return an image-only crop bbox.

    Important:
    - This function uses only image intensity.
    - It does not use mask information.
    - x2, y2 are exclusive coordinates.

    Returns:
        x1, y1, x2, y2, fallback, reason, edge_trim_meta
    """
    h, w = image_gray.shape[:2]

    empty_trim_meta = {
        "edge_trim_top": 0,
        "edge_trim_bottom": 0,
        "edge_trim_left": 0,
        "edge_trim_right": 0,
        "edge_trim_total": 0,
        "extra_vertical_frame_trim_left": 0,
        "extra_vertical_frame_trim_right": 0,
        "extra_vertical_frame_trim_total": 0,
    }

    if h == 0 or w == 0:
        return 0, 0, w, h, True, "invalid_image_size", empty_trim_meta

    # Light denoising
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)

    # Contrast normalization for more stable thresholding
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    # Otsu threshold. X-ray foreground is usually brighter than black borders.
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel_size = max(3, int(min(h, w) * 0.01))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    if num_labels <= 1:
        # Even when foreground detection fails, try edge trimming on the full image.
        x1, y1, x2, y2, trim_meta = trim_extreme_edges(image_gray, (0, 0, w, h))
        return x1, y1, x2, y2, True, "no_foreground_component", trim_meta

    image_area = h * w
    min_area = image_area * min_area_ratio

    selected_boxes = []

    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]

        if area < min_area:
            continue

        selected_boxes.append((x, y, x + bw, y + bh, area))

    if len(selected_boxes) == 0:
        # If no large component is found, use full image but still trim extreme edges.
        x1, y1, x2, y2, trim_meta = trim_extreme_edges(image_gray, (0, 0, w, h))
        return x1, y1, x2, y2, True, "no_component_large_enough", trim_meta

    x1 = min(box[0] for box in selected_boxes)
    y1 = min(box[1] for box in selected_boxes)
    x2 = max(box[2] for box in selected_boxes)
    y2 = max(box[3] for box in selected_boxes)

    margin_x = int(w * margin_ratio)
    margin_y = int(h * margin_ratio)

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)

    crop_w = x2 - x1
    crop_h = y2 - y1
    crop_area_ratio = (crop_w * crop_h) / image_area

    # Safety checks: avoid aggressive over-cropping
    if crop_w < 0.25 * w or crop_h < 0.25 * h:
        x1, y1, x2, y2, trim_meta = trim_extreme_edges(image_gray, (0, 0, w, h))
        return x1, y1, x2, y2, True, "crop_too_small", trim_meta

    # If crop is almost the full image, keep full image but still trim extreme edges.
    if crop_area_ratio > 0.98:
        x1, y1, x2, y2, trim_meta = trim_extreme_edges(image_gray, (0, 0, w, h))
        reason = "almost_full_image_edge_trimmed" if trim_meta["edge_trim_total"] > 0 else "almost_full_image"
        return x1, y1, x2, y2, True, reason, trim_meta

    # New refinement step:
    # Remove low-information black/white edge artifacts after initial foreground crop.
    x1_refined, y1_refined, x2_refined, y2_refined, trim_meta = trim_extreme_edges(
        image_gray,
        (x1, y1, x2, y2),
    )

    if trim_meta["edge_trim_total"] > 0:
        return x1_refined, y1_refined, x2_refined, y2_refined, False, "ok_edge_trimmed", trim_meta

    return x1, y1, x2, y2, False, "ok", trim_meta


def main():
    print("=== Crop BTXRD X-ray borders while preserving high resolution ===")

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "image_id" not in df.columns:
        raise ValueError("dataset.csv must contain an 'image_id' column.")

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cropping borders"):
        image_id = str(row["image_id"])
        stem = Path(image_id).stem

        image_path = find_image_file(image_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing image file for {image_id}")

        mask_path = MASK_DIR / f"{stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file for {image_id}: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        if mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {image_id}: "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        h, w = image.shape[:2]

        # Image-only foreground crop + image-only edge trimming
        x1, y1, x2, y2, fallback, reason, trim_meta = detect_foreground_bbox(image)

        image_crop = image[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        raw_mask_area = int((mask > 0).sum())
        crop_mask_area_before_safety = int((mask_crop > 0).sum())

        # Mask-safety fallback:
        # The mask is NOT used to localize or tighten the crop.
        # It is only used for offline QC to avoid corrupting ground-truth labels.
        # If image-only crop truncates any tumor pixel, keep the full image.
        mask_safety_fallback = False

        if raw_mask_area > 0 and crop_mask_area_before_safety < raw_mask_area:
            x1, y1, x2, y2 = 0, 0, w, h
            image_crop = image
            mask_crop = mask

            fallback = True
            reason = "mask_safety_fallback"
            mask_safety_fallback = True

            # Reset trim meta because final output is full image.
            trim_meta = {
                "edge_trim_top": 0,
                "edge_trim_bottom": 0,
                "edge_trim_left": 0,
                "edge_trim_right": 0,
                "edge_trim_total": 0,
                "extra_vertical_frame_trim_left": 0,
                "extra_vertical_frame_trim_right": 0,
                "extra_vertical_frame_trim_total": 0,
            }

        crop_mask_area_after_safety = int((mask_crop > 0).sum())

        clean_image_path = CLEAN_IMAGE_DIR / f"{stem}.jpg"
        clean_mask_path = CLEAN_MASK_DIR / f"{stem}.png"

        cv2.imwrite(str(clean_image_path), image_crop)
        cv2.imwrite(str(clean_mask_path), mask_crop)

        crop_w = x2 - x1
        crop_h = y2 - y1
        crop_area_ratio = (crop_w * crop_h) / (w * h)

        records.append(
            {
                "image_id": image_id,
                "orig_w": w,
                "orig_h": h,
                "crop_x1": x1,
                "crop_y1": y1,
                "crop_x2": x2,
                "crop_y2": y2,
                "clean_w": crop_w,
                "clean_h": crop_h,
                "crop_area_ratio": crop_area_ratio,
                "fallback": fallback,
                "reason": reason,
                "mask_safety_fallback": mask_safety_fallback,
                "raw_mask_area": raw_mask_area,
                "crop_mask_area_before_safety": crop_mask_area_before_safety,
                "crop_mask_area_after_safety": crop_mask_area_after_safety,
                **trim_meta,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "clean_image_path": str(clean_image_path),
                "clean_mask_path": str(clean_mask_path),
            }
        )

    report_df = pd.DataFrame(records)
    report_path = REPORT_DIR / "crop_border_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\n=== Crop Border Summary ===")
    print(f"Total images processed:      {len(report_df)}")
    print(f"Clean images saved:          {len(list(CLEAN_IMAGE_DIR.glob('*.jpg')))}")
    print(f"Clean masks saved:           {len(list(CLEAN_MASK_DIR.glob('*.png')))}")
    print(f"Fallback count:              {int(report_df['fallback'].sum())}")
    print(f"Mask-safety fallback count:  {int(report_df['mask_safety_fallback'].sum())}")
    print(
        f"Extra vertical trim count:   "
        f"{int((report_df['extra_vertical_frame_trim_total'] > 0).sum())}"
    )
    print(f"Mean crop area ratio:        {report_df['crop_area_ratio'].mean():.4f}")
    print(f"Min crop area ratio:         {report_df['crop_area_ratio'].min():.4f}")
    print(f"Max crop area ratio:         {report_df['crop_area_ratio'].max():.4f}")
    print(f"Report saved to:             {report_path}")

    print("\nCrop reasons:")
    print(report_df["reason"].value_counts())


if __name__ == "__main__":
    main()
