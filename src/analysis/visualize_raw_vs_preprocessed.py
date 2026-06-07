from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and preprocessed BTXRD images/masks.")
    parser.add_argument("--manifest", default="data/exports/btxrd_preprocessed/val.csv")
    parser.add_argument("--raw-image-dir", default="data/raw/images")
    parser.add_argument("--raw-annotation-dir", default="data/raw/Annotations")
    parser.add_argument("--output-dir", default="data/processed/visual_checks/raw_vs_preprocessed")
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path_value: str | Path, root_dir: Path = Path(".")) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    if path.is_absolute():
        return path
    return root_dir / path


def find_raw_image(image_id: str, raw_image_dir: Path) -> Path:
    stem = Path(image_id).stem
    candidates = [
        raw_image_dir / image_id,
        raw_image_dir / f"{stem}.jpeg",
        raw_image_dir / f"{stem}.jpg",
        raw_image_dir / f"{stem}.png",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing raw image for {image_id}")


def polygon_to_mask(points: list, height: int, width: int) -> np.ndarray:
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


def load_labelme_mask(json_path: Path, height: int, width: int) -> np.ndarray:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    final_mask = np.zeros((height, width), dtype=np.uint8)
    polygon_masks = []

    for shape in shapes:
        if shape.get("shape_type", "") == "polygon":
            mask = polygon_to_mask(shape.get("points", []), height, width)
            if mask.sum() > 0:
                polygon_masks.append(mask)

    if polygon_masks:
        for mask in polygon_masks:
            final_mask = np.maximum(final_mask, mask)
        return final_mask

    for shape in shapes:
        if shape.get("shape_type", "") == "rectangle":
            mask = rectangle_to_mask(shape.get("points", []), height, width)
            if mask.sum() > 0:
                final_mask = np.maximum(final_mask, mask)

    return final_mask


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    return np.asarray(pil_image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    pil_mask = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    return (np.asarray(pil_mask.resize((size, size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[float, float, float]) -> np.ndarray:
    if mask.shape[:2] != image.shape[:2]:
        mask = resize_mask(mask, image.shape[0])

    out = image.copy()
    color_arr = np.array(color, dtype=np.float32)
    active = mask > 0
    out[active] = 0.55 * out[active] + 0.45 * color_arr
    return out.clip(0, 1)


def mask_area_ratio(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def save_case_figure(row: pd.Series, raw_image_dir: Path, raw_annotation_dir: Path, output_dir: Path, image_size: int) -> dict:
    image_id = str(row["image_id"])
    stem = Path(image_id).stem
    raw_image_path = find_raw_image(image_id, raw_image_dir)
    raw_annotation_path = raw_annotation_dir / f"{stem}.json"
    pre_image_path = resolve_path(row["image_path"])
    pre_mask_path = resolve_path(row["mask_path"])

    raw_image = load_rgb(raw_image_path)
    raw_h, raw_w = raw_image.shape[:2]
    raw_mask = load_labelme_mask(raw_annotation_path, raw_h, raw_w)
    pre_image = load_rgb(pre_image_path)
    pre_mask = load_mask(pre_mask_path)

    raw_image_224 = resize_rgb(raw_image, image_size)
    raw_mask_224 = resize_mask(raw_mask, image_size)
    pre_image_224 = resize_rgb(pre_image, image_size)
    pre_mask_224 = resize_mask(pre_mask, image_size)

    panels = [
        ("raw + annotation", overlay_mask(raw_image_224, raw_mask_224, (0.0, 1.0, 0.0))),
        ("preprocessed + mask", overlay_mask(pre_image_224, pre_mask_224, (0.0, 1.0, 0.0))),
        ("raw 224", raw_image_224),
        ("preprocessed 224", pre_image_224),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"{image_id} | raw mask={mask_area_ratio(raw_mask_224):.4f} "
        f"pre mask={mask_area_ratio(pre_mask_224):.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_raw_vs_preprocessed.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return {
        "image_id": image_id,
        "raw_image_path": str(raw_image_path),
        "raw_annotation_path": str(raw_annotation_path),
        "preprocessed_image_path": str(pre_image_path),
        "preprocessed_mask_path": str(pre_mask_path),
        "raw_shape": f"{raw_h}x{raw_w}",
        "preprocessed_shape": f"{pre_image.shape[0]}x{pre_image.shape[1]}",
        "raw_mask_area_ratio_224": mask_area_ratio(raw_mask_224),
        "preprocessed_mask_area_ratio_224": mask_area_ratio(pre_mask_224),
        "abs_mask_area_ratio_diff_224": abs(mask_area_ratio(raw_mask_224) - mask_area_ratio(pre_mask_224)),
        "output_path": str(output_path),
    }


def save_index_grid(records: list[dict], output_dir: Path) -> None:
    if not records:
        return

    image_paths = [Path(record["output_path"]) for record in records]
    ncols = 2
    nrows = math.ceil(len(image_paths) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_arr = np.array(axes).reshape(-1)

    for ax, image_path, record in zip(axes_arr, image_paths, records):
        ax.imshow(load_rgb(image_path))
        ax.set_title(
            f"{record['image_id']} diff={record['abs_mask_area_ratio_diff_224']:.4f}",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes_arr[len(image_paths):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "raw_vs_preprocessed_index.png", dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    raw_image_dir = resolve_path(args.raw_image_dir)
    raw_annotation_dir = resolve_path(args.raw_annotation_dir)
    output_dir = resolve_path(args.output_dir)

    df = pd.read_csv(manifest_path)
    tumor_df = df[df["tumor"].astype(int) == 1].copy()
    if args.split_name:
        tumor_df = tumor_df[tumor_df["split"].astype(str) == args.split_name]

    sample_df = tumor_df.sample(
        n=min(args.max_samples, len(tumor_df)),
        random_state=args.seed,
    ).reset_index(drop=True)

    records = []
    for _, row in sample_df.iterrows():
        records.append(save_case_figure(row, raw_image_dir, raw_annotation_dir, output_dir, args.image_size))

    report_path = output_dir / "raw_vs_preprocessed_report.csv"
    pd.DataFrame(records).to_csv(report_path, index=False)
    save_index_grid(records, output_dir)
    print(f"Saved {len(records)} raw-vs-preprocessed visual checks to {output_dir}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
