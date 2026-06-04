from pathlib import Path
import argparse
import json
import shutil

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


INPUT_IMAGE_DIR = Path("data/processed/images_clean")
INPUT_MASK_DIR = Path("data/processed/masks_clean")

OUTPUT_IMAGE_DIR = Path("data/processed/images_no_markers")
OUTPUT_MASK_DIR = Path("data/processed/masks_no_markers")

REPORT_DIR = Path("data/processed/reports")
PREVIEW_DIR = Path("data/processed/visual_checks/marker_removal")

MARKER_STROKE_THRESHOLD = 145
MAX_MARKERS_PER_IMAGE = 6
MIN_MARKER_SCORE = 1.15
BRIGHT_BLOB_THRESHOLD = 175


def list_clean_images() -> list[Path]:
    image_files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(INPUT_IMAGE_DIR.glob(pattern))
    return sorted(image_files)


def find_mask_file(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        path = INPUT_MASK_DIR / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Remove safe X-ray L/R marker artifacts.")
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Process only the first N clean images.",
    )
    parser.add_argument(
        "--output-tag",
        type=str,
        default="",
        help="Optional suffix for trial outputs, e.g. trial200.",
    )
    return parser.parse_args()


def resolve_outputs(output_tag: str) -> tuple[Path, Path, Path, Path]:
    if not output_tag:
        return (
            OUTPUT_IMAGE_DIR,
            OUTPUT_MASK_DIR,
            REPORT_DIR / "xray_marker_removal_report.csv",
            PREVIEW_DIR,
        )

    safe_tag = output_tag.replace("/", "_").replace("\\", "_").replace(":", "_")
    return (
        Path(f"data/processed/images_no_markers_{safe_tag}"),
        Path(f"data/processed/masks_no_markers_{safe_tag}"),
        REPORT_DIR / f"xray_marker_removal_report_{safe_tag}.csv",
        Path(f"data/processed/visual_checks/marker_removal_{safe_tag}"),
    )


def is_peripheral_marker_region(cx: float, cy: float, width: int, height: int) -> bool:
    near_side = cx < 0.40 * width or cx > 0.60 * width
    bottom_band = cy > 0.68 * height
    not_top_artifact = cy > 0.18 * height
    top_side_marker = is_top_side_marker_region(cx, cy, width, height)
    return bottom_band or (near_side and not_top_artifact) or top_side_marker


def is_top_side_marker_region(cx: float, cy: float, width: int, height: int) -> bool:
    return cy < 0.18 * height and (cx < 0.25 * width or cx > 0.75 * width)


def is_bottom_marker_region(cy: float, height: int) -> bool:
    return cy > 0.68 * height


def is_strict_middle_side_marker_region(
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> bool:
    return (
        0.18 * height <= cy <= 0.68 * height
        and (cx < 0.18 * width or cx > 0.82 * width)
    )


def is_plausible_top_text_marker(
    bbox: tuple[int, int, int, int],
    area: int,
    width: int,
    height: int,
) -> bool:
    min_dim = min(width, height)
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    min_w = max(22, int(min_dim * 0.010))
    min_h = max(35, int(min_dim * 0.015))
    max_box = max(min_h + 1, int(min_dim * 0.12))
    min_area = max(450, int(min_dim * 0.35))

    return (
        min_w <= box_w <= max_box
        and min_h <= box_h <= max_box
        and area >= min_area
    )


def is_marker_erase_area_reasonable(
    erase_mask: np.ndarray,
    safe_circled_marker: bool,
) -> bool:
    erase_area_ratio = float((erase_mask > 0).sum()) / float(erase_mask.size)
    max_ratio = 0.006
    return erase_area_ratio <= max_ratio


def is_plausible_text_stroke_shape(
    bbox: tuple[int, int, int, int],
    area: int,
    width: int,
    height: int,
) -> bool:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    min_dim = min(width, height)
    aspect = box_w / float(box_h)
    density = area / float(box_w * box_h)
    max_text_box = max(36, int(min_dim * 0.10))

    return max(box_w, box_h) <= max_text_box and aspect <= 0.92 and density <= 0.58


def has_dark_isolation_shell(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """
    Bare L/R strokes should be isolated by dark film immediately around them.
    Bright bone segments can pass broad dark-background checks because there is
    black space between fingers, but their immediate shell usually contains
    more connected anatomy.
    """
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(6, int(round(max(box_w, box_h) * 0.35)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    if patch.size == 0:
        return False

    ix1 = x1 - px1
    iy1 = y1 - py1
    ix2 = x2 - px1
    iy2 = y2 - py1
    shell_mask = np.ones(patch.shape[:2], dtype=bool)
    shell_mask[max(0, iy1):iy2, max(0, ix1):ix2] = False
    shell = patch[shell_mask]

    if shell.size < 20:
        return False

    dark_ratio = float((shell <= 90).mean())
    bright_ratio = float((shell >= MARKER_STROKE_THRESHOLD).mean())
    mean_val = float(shell.mean())

    return dark_ratio >= 0.52 and bright_ratio <= 0.16 and mean_val <= 118


def padded_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    pad: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad),
        min(height, y2 + pad),
    )


def is_safe_dark_background(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """
    Only erase candidate strokes when they sit mostly on dark film/background.
    If the region looks like tissue/bone, keep it unchanged.
    """
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(10, int(round(max(box_w, box_h) * 0.85)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    if patch.size == 0:
        return False

    dark_ratio = float((patch <= 75).mean())
    bright_ratio = float((patch >= 165).mean())
    mean_val = float(patch.mean())
    median_val = float(np.median(patch))

    return (
        dark_ratio >= 0.62
        and 0.002 <= bright_ratio <= 0.28
        and mean_val <= 112
        and median_val <= 90
    )


def is_safe_dark_blob_background(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> bool:
    """
    Larger white marker dots can occupy more of the local patch than letters, so
    this uses a separate dark-background check with a wider bright-ratio window.
    """
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(14, int(round(max(box_w, box_h) * 0.65)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    if patch.size == 0:
        return False

    dark_ratio = float((patch <= 80).mean())
    bright_ratio = float((patch >= 170).mean())
    mean_val = float(patch.mean())
    median_val = float(np.median(patch))

    return (
        dark_ratio >= 0.45
        and 0.01 <= bright_ratio <= 0.55
        and mean_val <= 130
        and median_val <= 105
    )


def marker_overlaps_tumor(
    marker_mask: np.ndarray,
    tumor_mask: np.ndarray | None,
) -> bool:
    if tumor_mask is None:
        return False
    if tumor_mask.shape != marker_mask.shape:
        return True
    return bool(np.logical_and(marker_mask > 0, tumor_mask > 0).any())


def is_likely_circled_marker(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    area: int,
) -> bool:
    h, w = image_gray.shape[:2]
    min_dim = min(h, w)
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    aspect = box_w / float(box_h)
    density = area / float(box_w * box_h)
    box_ratio = max(box_w, box_h) / max(1.0, float(min_dim))

    if not 0.72 <= aspect <= 1.38:
        return False
    if not 0.055 <= density <= 0.24:
        return False
    if not 0.028 <= box_ratio <= 0.12:
        return False

    pad = max(4, int(round(max(box_w, box_h) * 0.10)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    if patch.size == 0:
        return False

    dark_ratio = float((patch <= 85).mean())
    bright_ratio = float((patch >= MARKER_STROKE_THRESHOLD).mean())
    core_patch = image_gray[y1:y2, x1:x2]
    bright = core_patch >= MARKER_STROKE_THRESHOLD
    ring_band = max(2, int(round(min(core_patch.shape[:2]) * 0.18)))
    ring_side_ratios = [
        float(bright[:ring_band, :].mean()),
        float(bright[-ring_band:, :].mean()),
        float(bright[:, :ring_band].mean()),
        float(bright[:, -ring_band:].mean()),
    ]

    ix1 = px1 + int(round((px2 - px1) * 0.25))
    ix2 = px1 + int(round((px2 - px1) * 0.75))
    iy1 = py1 + int(round((py2 - py1) * 0.25))
    iy2 = py1 + int(round((py2 - py1) * 0.75))
    inner = image_gray[iy1:iy2, ix1:ix2]
    inner_dark_ratio = float((inner <= 95).mean()) if inner.size else 0.0

    return (
        dark_ratio >= 0.16
        and bright_ratio >= 0.035
        and min(ring_side_ratios) >= 0.10
        and inner_dark_ratio >= 0.18
    )


def build_marker_stroke_mask(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    Build an erase mask from bright marker strokes only. This supports bare
    L/R letters and circled L/R markers without inpainting the whole marker area.
    """
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(4, int(round(max(box_w, box_h) * 0.18)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    if patch.size == 0:
        return np.zeros(image_gray.shape[:2], dtype=np.uint8)

    bright_mask = (patch >= MARKER_STROKE_THRESHOLD).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright_mask,
        connectivity=8,
    )

    filtered = np.zeros(image_gray.shape[:2], dtype=np.uint8)
    min_area = max(5, int((box_w * box_h) * 0.004))
    max_area = max(min_area + 1, int((box_w * box_h) * 1.25))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            component = labels == label_id
            filtered[py1:py2, px1:px2][component] = 255

    if filtered.any():
        kernel_size = max(3, int(round(max(box_w, box_h) * 0.075)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        filtered = cv2.dilate(filtered, kernel, iterations=1)

    # Circled markers often leave a dark disk after the white strokes are removed,
    # especially when they sit over gray tissue. Remove only local high-contrast
    # connected components inside the same marker bbox.
    if filtered.any():
        patch_median = float(np.median(patch))
        dark_limit = int(min(85, max(35, patch_median - 38)))
        bright_limit = int(max(MARKER_STROKE_THRESHOLD, min(195, patch_median + 58)))
        contrast_artifact = np.logical_or(
            patch <= dark_limit,
            patch >= bright_limit,
        ).astype(np.uint8) * 255
        num_contrast, contrast_labels, contrast_stats, _ = (
            cv2.connectedComponentsWithStats(
                contrast_artifact,
                connectivity=8,
            )
        )
        contrast_filtered = np.zeros_like(filtered)
        dark_min_area = max(8, int((box_w * box_h) * 0.01))
        dark_max_area = max(dark_min_area + 1, int((box_w * box_h) * 2.5))

        for label_id in range(1, num_contrast):
            area = int(contrast_stats[label_id, cv2.CC_STAT_AREA])
            if dark_min_area <= area <= dark_max_area:
                component = contrast_labels == label_id
                contrast_filtered[py1:py2, px1:px2][component] = 255

        if contrast_filtered.any():
            filtered = np.maximum(filtered, contrast_filtered)

    return filtered


def build_bright_blob_mask(
    image_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(3, int(round(max(box_w, box_h) * 0.08)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)
    patch = image_gray[py1:py2, px1:px2]

    mask = np.zeros(image_gray.shape[:2], dtype=np.uint8)
    if patch.size == 0:
        return mask

    local = (patch >= BRIGHT_BLOB_THRESHOLD).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(local, connectivity=8)
    min_area = max(16, int((box_w * box_h) * 0.20))
    max_area = max(min_area + 1, int((box_w * box_h) * 1.40))

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            component = labels == label_id
            mask[py1:py2, px1:px2][component] = 255
            cx = px1 + int(stats[label_id, cv2.CC_STAT_LEFT] + stats[label_id, cv2.CC_STAT_WIDTH] / 2)
            cy = py1 + int(stats[label_id, cv2.CC_STAT_TOP] + stats[label_id, cv2.CC_STAT_HEIGHT] / 2)
            axes = (
                max(4, int(round(stats[label_id, cv2.CC_STAT_WIDTH] * 0.62))),
                max(4, int(round(stats[label_id, cv2.CC_STAT_HEIGHT] * 0.62))),
            )
            cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, thickness=-1)

    if mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def marker_score(
    bbox: tuple[int, int, int, int],
    area: int,
    width: int,
    height: int,
) -> float:
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    edge_distance_ratio = min(cx, width - cx) / max(1.0, float(width))
    edge_score = 1.0 - min(edge_distance_ratio / 0.40, 1.0)
    lower_score = min(cy / max(1.0, float(height)), 1.0)
    density = area / float(box_w * box_h)
    density_score = 1.0 - min(abs(density - 0.28) / 0.28, 1.0)

    return 2.0 * edge_score + lower_score + density_score


def detect_marker_strokes(
    image_gray: np.ndarray,
    tumor_mask: np.ndarray | None,
) -> list[dict]:
    h, w = image_gray.shape[:2]
    min_dim = min(h, w)
    bright_mask = (image_gray >= MARKER_STROKE_THRESHOLD).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask,
        connectivity=8,
    )

    min_area = max(8, int(h * w * 0.000002))
    max_area = max(min_area + 1, int(h * w * 0.0060))
    min_h = max(8, int(min_dim * 0.008))
    max_box = max(min_h + 1, int(min_dim * 0.20))

    markers = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]

        if area < min_area or area > max_area:
            continue
        if bw < 3 or bh < min_h or bw > max_box or bh > max_box:
            continue
        if not is_peripheral_marker_region(float(cx), float(cy), w, h):
            continue

        bbox = (x, y, x + bw, y + bh)
        safe_background = is_safe_dark_background(image_gray, bbox)
        safe_circled_marker = is_likely_circled_marker(image_gray, bbox, area)
        if not safe_background and not safe_circled_marker:
            continue
        if not safe_circled_marker and not is_plausible_text_stroke_shape(bbox, area, w, h):
            continue
        if not safe_circled_marker and not has_dark_isolation_shell(image_gray, bbox):
            continue
        if is_top_side_marker_region(float(cx), float(cy), w, h):
            if not safe_background or not is_plausible_top_text_marker(bbox, area, w, h):
                continue
        elif not is_bottom_marker_region(float(cy), h):
            if not safe_circled_marker and not is_strict_middle_side_marker_region(float(cx), float(cy), w, h):
                continue

        erase_mask = build_marker_stroke_mask(image_gray, bbox)
        if not erase_mask.any() or marker_overlaps_tumor(erase_mask, tumor_mask):
            continue
        if not is_marker_erase_area_reasonable(erase_mask, safe_circled_marker):
            continue

        ys, xs = np.where(erase_mask > 0)
        if len(xs) == 0:
            continue

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max() + 1)
        y2 = int(ys.max() + 1)

        markers.append(
            {
                "center_x": int(round(cx)),
                "center_y": int(round(cy)),
                "bbox": [x1, y1, x2, y2],
                "area_pixels": int((erase_mask > 0).sum()),
                "erase_mask": erase_mask,
                "score": marker_score(bbox, area, w, h),
            }
        )

    markers = [
        marker
        for marker in sorted(markers, key=lambda marker: marker["score"], reverse=True)
        if marker["score"] >= MIN_MARKER_SCORE
    ]

    return markers[:MAX_MARKERS_PER_IMAGE]


def detect_bright_blob_artifacts(
    image_gray: np.ndarray,
    tumor_mask: np.ndarray | None,
    existing_markers: list[dict],
) -> list[dict]:
    h, w = image_gray.shape[:2]
    min_dim = min(h, w)
    bright_mask = (image_gray >= BRIGHT_BLOB_THRESHOLD).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright_mask,
        connectivity=8,
    )

    min_area = max(30, int(h * w * 0.000025))
    max_area = max(min_area + 1, int(h * w * 0.012))
    min_box = max(10, int(min_dim * 0.018))
    max_box = max(min_box + 1, int(min_dim * 0.18))

    existing_mask = build_erase_mask(image_gray.shape[:2], existing_markers)
    blobs = []

    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        bw = int(stats[label_id, cv2.CC_STAT_WIDTH])
        bh = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = centroids[label_id]

        if area < min_area or area > max_area:
            continue
        touches_side = x <= 2 or x + bw >= w - 2
        touches_vertical_edge = y <= 2 or y + bh >= h - 2
        if touches_vertical_edge:
            continue
        side_max_box = max_box if not touches_side else max(max_box, int(min_dim * 0.30))

        if bw < min_box or bh < min_box or bw > side_max_box or bh > side_max_box:
            continue
        aspect = bw / max(1.0, float(bh))
        density = area / max(1.0, float(bw * bh))
        min_aspect = 0.22 if touches_side else 0.55
        max_aspect = 2.25 if touches_side else 1.85
        if not min_aspect <= aspect <= max_aspect or density < 0.35:
            continue
        if not is_peripheral_marker_region(float(cx), float(cy), w, h):
            continue

        bbox = (x, y, x + bw, y + bh)
        if not is_safe_dark_blob_background(image_gray, bbox):
            continue

        erase_mask = build_bright_blob_mask(image_gray, bbox)
        if not erase_mask.any():
            continue
        if existing_mask.any() and np.logical_and(erase_mask > 0, existing_mask > 0).any():
            continue
        if marker_overlaps_tumor(erase_mask, tumor_mask):
            continue

        blobs.append(
            {
                "center_x": int(round(cx)),
                "center_y": int(round(cy)),
                "bbox": list(bbox),
                "area_pixels": int((erase_mask > 0).sum()),
                "erase_mask": erase_mask,
                "score": marker_score(bbox, area, w, h) + 0.15,
                "kind": "bright_blob",
            }
        )

    return sorted(blobs, key=lambda marker: marker["score"], reverse=True)


def detect_marker_artifacts(
    image_gray: np.ndarray,
    tumor_mask: np.ndarray | None,
) -> list[dict]:
    h, w = image_gray.shape[:2]
    markers = detect_marker_strokes(image_gray, tumor_mask)
    blobs = detect_bright_blob_artifacts(image_gray, tumor_mask, markers)
    combined = sorted(
        [*markers, *blobs],
        key=lambda marker: marker["score"],
        reverse=True,
    )
    if w / max(1.0, float(h)) > 1.05 and len(combined) >= 3:
        combined = [
            marker
            for marker in combined
            if marker.get("kind") == "bright_blob"
            or marker["bbox"][2] <= 0.08 * w
            or marker["bbox"][0] >= 0.92 * w
        ]
    return combined[:MAX_MARKERS_PER_IMAGE]


def build_erase_mask(shape: tuple[int, int], markers: list[dict]) -> np.ndarray:
    erase_mask = np.zeros(shape, dtype=np.uint8)
    for marker in markers:
        marker_mask = marker.get("erase_mask")
        if marker_mask is not None and marker_mask.shape == shape:
            erase_mask = np.maximum(erase_mask, marker_mask)
    return erase_mask


def local_background_value(
    image_gray: np.ndarray,
    erase_mask: np.ndarray,
    bbox: list[int],
    prefer_dark: bool = False,
) -> int:
    h, w = image_gray.shape[:2]
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    pad = max(10, int(round(max(box_w, box_h) * 0.75)))
    px1, py1, px2, py2 = padded_bbox(x1, y1, x2, y2, w, h, pad)

    patch = image_gray[py1:py2, px1:px2]
    patch_mask = erase_mask[py1:py2, px1:px2] > 0

    if prefer_dark:
        dark_background = patch[np.logical_and(~patch_mask, patch <= 90)]
        if dark_background.size >= 20:
            return int(min(np.percentile(dark_background, 35), 45))

    background = patch[np.logical_and(~patch_mask, patch <= 150)]

    if background.size < 20:
        background = patch[~patch_mask]
    if background.size == 0:
        return int(np.median(image_gray))

    return int(np.median(background))


def remove_markers(image_gray: np.ndarray, markers: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    erase_mask = build_erase_mask(image_gray.shape[:2], markers)

    if not erase_mask.any():
        return image_gray.copy(), erase_mask

    cleaned = image_gray.copy()
    inpaint_mask = erase_mask.copy()

    for marker in markers:
        if marker.get("kind") != "bright_blob":
            continue
        marker_mask = marker.get("erase_mask")
        if marker_mask is None or marker_mask.shape != image_gray.shape[:2]:
            continue
        fill_value = local_background_value(
            image_gray,
            marker_mask,
            marker["bbox"],
            prefer_dark=True,
        )
        cleaned[marker_mask > 0] = fill_value
        inpaint_mask[marker_mask > 0] = 0

    if inpaint_mask.any():
        cleaned = cv2.inpaint(cleaned, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return cleaned, erase_mask


def draw_marker_preview(
    original: np.ndarray,
    cleaned: np.ndarray,
    markers: list[dict],
    output_path: Path,
) -> None:
    original_bgr = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    for marker in markers:
        x1, y1, x2, y2 = marker["bbox"]
        cv2.rectangle(original_bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)

    target_height = 768
    scale = target_height / max(1, original_bgr.shape[0])
    new_w = max(1, int(original_bgr.shape[1] * scale))

    original_small = cv2.resize(original_bgr, (new_w, target_height))
    cleaned_small = cv2.resize(cleaned_bgr, (new_w, target_height))

    preview = np.concatenate([original_small, cleaned_small], axis=1)
    cv2.imwrite(str(output_path), preview)


def main():
    args = parse_args()
    output_image_dir, output_mask_dir, report_path, preview_dir = resolve_outputs(
        args.output_tag
    )

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=== Remove X-ray L/R marker artifacts ===")

    image_files = list_clean_images()
    if not image_files:
        raise FileNotFoundError(f"No clean images found in: {INPUT_IMAGE_DIR}")

    if args.max_images is not None:
        image_files = image_files[: args.max_images]

    records = []

    for image_path in tqdm(image_files, desc="Removing markers"):
        stem = image_path.stem
        mask_path = find_mask_file(stem)

        if mask_path is None:
            raise FileNotFoundError(f"Missing clean mask for {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        tumor_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        if tumor_mask is None:
            raise ValueError(f"Cannot read mask: {mask_path}")
        if image.shape[:2] != tumor_mask.shape[:2]:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, mask={tumor_mask.shape[:2]}"
            )

        markers = detect_marker_artifacts(image, tumor_mask)
        cleaned, erase_mask = remove_markers(image, markers)

        output_image_path = output_image_dir / f"{stem}.jpg"
        output_mask_path = output_mask_dir / f"{stem}.png"

        if markers:
            cv2.imwrite(str(output_image_path), cleaned)
            preview_path = preview_dir / f"{stem}_markers-{len(markers)}.jpg"
            draw_marker_preview(image, cleaned, markers, preview_path)
        else:
            shutil.copyfile(image_path, output_image_path)
            preview_path = None

        shutil.copyfile(mask_path, output_mask_path)

        records.append(
            {
                "image_id": image_path.name,
                "marker_removed": bool(markers),
                "num_markers": len(markers),
                "marker_boxes": json.dumps([m["bbox"] for m in markers]),
                "marker_centers": json.dumps(
                    [[m["center_x"], m["center_y"]] for m in markers]
                ),
                "erase_area_pixels": int((erase_mask > 0).sum()),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "output_image_path": str(output_image_path),
                "output_mask_path": str(output_mask_path),
                "preview_path": str(preview_path) if preview_path else "",
            }
        )

    report_df = pd.DataFrame(records)
    report_df.to_csv(report_path, index=False)

    removed_df = report_df[report_df["marker_removed"] == True]

    print("\n=== Marker Removal Summary ===")
    print(f"Total images processed: {len(report_df)}")
    print(f"Images with markers removed: {len(removed_df)}")
    print(f"Total markers removed: {int(report_df['num_markers'].sum())}")
    print(f"Output images saved to: {output_image_dir}")
    print(f"Output masks saved to:  {output_mask_dir}")
    print(f"Report saved to:        {report_path}")
    print(f"Previews saved to:      {preview_dir}")


if __name__ == "__main__":
    main()
