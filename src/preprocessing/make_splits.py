from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RAW_CSV_PATH = Path("data/raw/dataset.csv")
TEXT_CSV_PATH = Path("data/processed/text_annotations.csv")

FINAL_IMAGE_DIR = Path("data/processed/images_preprocessed")
FINAL_MASK_DIR = Path("data/processed/masks_preprocessed")

FALLBACK_IMAGE_DIRS = [
    Path("data/processed/images_preprocessed"),
    Path("data/processed/images_marker_clean"),
    Path("data/processed/images_clean"),
    Path("data/raw/images"),
]

FALLBACK_MASK_DIRS = [
    Path("data/processed/masks_preprocessed"),
    Path("data/processed/masks_marker_clean"),
    Path("data/processed/masks_clean"),
    Path("data/processed/masks"),
]

OUTPUT_SPLIT_DIR = Path("data/processed/splits")
OUTPUT_REPORT_DIR = Path("data/processed/reports")
VERSIONED_SPLIT_DIR = Path("configs/splits")

OUTPUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
VERSIONED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_ALL_CSV = OUTPUT_SPLIT_DIR / "split.csv"
TRAIN_CSV = OUTPUT_SPLIT_DIR / "train.csv"
VAL_CSV = OUTPUT_SPLIT_DIR / "val.csv"
TEST_CSV = OUTPUT_SPLIT_DIR / "test.csv"

VERSIONED_SPLIT_CSV = VERSIONED_SPLIT_DIR / f"btxrd_split_seed{SEED}.csv"

SUMMARY_JSON = OUTPUT_REPORT_DIR / "split_summary.json"
SUMMARY_CSV = OUTPUT_REPORT_DIR / "split_summary.csv"
DISTRIBUTION_CSV = OUTPUT_REPORT_DIR / "split_distribution_long.csv"


def value_is_one(value) -> bool:
    if pd.isna(value):
        return False

    try:
        return int(value) == 1
    except Exception:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_diagnosis_group(row: pd.Series) -> str:
    tumor = value_is_one(row.get("tumor", 0))
    benign = value_is_one(row.get("benign", 0))
    malignant = value_is_one(row.get("malignant", 0))

    if not tumor:
        return "normal"

    if benign:
        return "benign"

    if malignant:
        return "malignant"

    return "tumor_unknown"


def find_file_by_stem(stem: str, directories: list[Path], extensions: list[str]) -> Path | None:
    for directory in directories:
        for ext in extensions:
            path = directory / f"{stem}{ext}"
            if path.exists():
                return path

    return None


def build_paths(image_id: str) -> tuple[str, str, bool, bool]:
    stem = Path(str(image_id)).stem

    image_path = find_file_by_stem(
        stem=stem,
        directories=FALLBACK_IMAGE_DIRS,
        extensions=[".jpg", ".jpeg", ".png"],
    )

    mask_path = find_file_by_stem(
        stem=stem,
        directories=FALLBACK_MASK_DIRS,
        extensions=[".png", ".jpg", ".jpeg"],
    )

    has_image = image_path is not None
    has_mask = mask_path is not None

    return (
        str(image_path) if image_path else "",
        str(mask_path) if mask_path else "",
        has_image,
        has_mask,
    )


def normalize_label(value, fallback: str) -> str:
    if pd.isna(value):
        return fallback

    value_str = str(value).strip()

    if value_str == "":
        return fallback

    return value_str


def build_base_dataframe() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing raw CSV: {RAW_CSV_PATH}")

    if not TEXT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing text annotation CSV: {TEXT_CSV_PATH}. "
            f"Run generate_text_annotations.py first."
        )

    raw_df = pd.read_csv(RAW_CSV_PATH)
    text_df = pd.read_csv(TEXT_CSV_PATH)

    required_raw_cols = ["image_id", "tumor", "benign", "malignant"]
    missing_raw = [col for col in required_raw_cols if col not in raw_df.columns]
    if missing_raw:
        raise ValueError(f"Missing required columns in dataset.csv: {missing_raw}")

    required_text_cols = [
        "image_id",
        "text_lvit_prompt",
        "text_short",
        "text_diagnosis_only",
        "text_anatomy_aware",
        "segmentation_target_text",
        "region_text",
        "view_text",
    ]
    missing_text = [col for col in required_text_cols if col not in text_df.columns]
    if missing_text:
        raise ValueError(f"Missing required columns in text_annotations.csv: {missing_text}")

    # Use text_df as the main table because it already contains generated text fields.
    keep_raw_cols = [
        col for col in ["image_id", "center", "age", "gender", "tumor", "benign", "malignant"]
        if col in raw_df.columns
    ]

    merged_df = text_df.merge(
        raw_df[keep_raw_cols],
        on="image_id",
        how="left",
        suffixes=("", "_raw"),
    )

    # Prefer original labels from raw CSV if duplicate columns appear.
    for col in ["age", "gender", "tumor", "benign", "malignant"]:
        raw_col = f"{col}_raw"
        if raw_col in merged_df.columns:
            merged_df[col] = merged_df[raw_col]
            merged_df = merged_df.drop(columns=[raw_col])

    merged_df["diagnosis_group"] = merged_df.apply(get_diagnosis_group, axis=1)

    merged_df["region_for_split"] = merged_df["region_text"].apply(
        lambda x: normalize_label(x, "unknown_region")
    )
    merged_df["view_for_split"] = merged_df["view_text"].apply(
        lambda x: normalize_label(x, "unknown_view")
    )

    image_paths = []
    mask_paths = []
    has_images = []
    has_masks = []

    for image_id in merged_df["image_id"]:
        image_path, mask_path, has_image, has_mask = build_paths(image_id)
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        has_images.append(has_image)
        has_masks.append(has_mask)

    merged_df["image_path"] = image_paths
    merged_df["mask_path"] = mask_paths
    merged_df["has_image"] = has_images
    merged_df["has_mask"] = has_masks

    return merged_df


def build_hierarchical_stratify_key(df: pd.DataFrame, min_count: int = 10) -> pd.Series:
    """
    Build a robust stratification key.

    Priority:
    1. diagnosis + region + view
    2. diagnosis + region
    3. diagnosis

    Rare keys are automatically backed off to a more general key.
    """
    level1 = (
        df["diagnosis_group"].astype(str)
        + "__"
        + df["region_for_split"].astype(str)
        + "__"
        + df["view_for_split"].astype(str)
    )

    level2 = (
        df["diagnosis_group"].astype(str)
        + "__"
        + df["region_for_split"].astype(str)
    )

    level3 = df["diagnosis_group"].astype(str)

    level1_counts = level1.value_counts()
    level2_counts = level2.value_counts()

    final_keys = []

    for idx in df.index:
        key1 = level1.loc[idx]
        key2 = level2.loc[idx]
        key3 = level3.loc[idx]

        if level1_counts[key1] >= min_count:
            final_keys.append(key1)
        elif level2_counts[key2] >= min_count:
            final_keys.append(key2)
        else:
            final_keys.append(key3)

    return pd.Series(final_keys, index=df.index)


def split_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")

    df = df.copy()
    df["stratify_key"] = build_hierarchical_stratify_key(df)

    if (df["has_image"] == False).any() or (df["has_mask"] == False).any():
        missing = df[(df["has_image"] == False) | (df["has_mask"] == False)]
        raise FileNotFoundError(
            "Some image/mask files are missing. Examples:\n"
            + missing[["image_id", "has_image", "has_mask", "image_path", "mask_path"]]
            .head(20)
            .to_string(index=False)
        )

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=SEED,
        shuffle=True,
        stratify=df["stratify_key"],
    )

    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    # For the second split, use the same key if possible.
    # If any key appears fewer than 2 times in temp, fall back to diagnosis only.
    temp_key_counts = temp_df["stratify_key"].value_counts()
    if temp_key_counts.min() >= 2:
        temp_stratify = temp_df["stratify_key"]
    else:
        temp_stratify = temp_df["diagnosis_group"]

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction_of_temp,
        random_state=SEED,
        shuffle=True,
        stratify=temp_stratify,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], axis=0)
    split_df = split_df.sort_values("image_id").reset_index(drop=True)

    return split_df


def summarize_split(split_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    summary_rows = []

    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name]

        row = {
            "split": split_name,
            "count": int(len(part)),
            "normal": int((part["diagnosis_group"] == "normal").sum()),
            "benign": int((part["diagnosis_group"] == "benign").sum()),
            "malignant": int((part["diagnosis_group"] == "malignant").sum()),
            "tumor_total": int((part["diagnosis_group"] != "normal").sum()),
            "empty_mask": int((part["segmentation_target_text"] == "empty mask").sum()),
            "segment_tumor": int((part["segmentation_target_text"] == "segment tumor region only").sum()),
        }

        row["normal_ratio"] = row["normal"] / row["count"] if row["count"] else 0
        row["tumor_ratio"] = row["tumor_total"] / row["count"] if row["count"] else 0

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    distribution_rows = []

    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name]

        for col, category_type in [
            ("diagnosis_group", "diagnosis"),
            ("region_for_split", "region"),
            ("view_for_split", "view"),
        ]:
            counts = part[col].value_counts().sort_index()
            for label, count in counts.items():
                distribution_rows.append(
                    {
                        "split": split_name,
                        "category_type": category_type,
                        "label": label,
                        "count": int(count),
                        "ratio_within_split": float(count / len(part)) if len(part) else 0.0,
                    }
                )

    distribution_df = pd.DataFrame(distribution_rows)

    summary = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_rows": int(len(split_df)),
        "train_rows": int((split_df["split"] == "train").sum()),
        "val_rows": int((split_df["split"] == "val").sum()),
        "test_rows": int((split_df["split"] == "test").sum()),
        "has_overlap": False,
    }

    split_sets = {
        "train": set(split_df[split_df["split"] == "train"]["image_id"]),
        "val": set(split_df[split_df["split"] == "val"]["image_id"]),
        "test": set(split_df[split_df["split"] == "test"]["image_id"]),
    }

    if split_sets["train"] & split_sets["val"]:
        summary["has_overlap"] = True
    if split_sets["train"] & split_sets["test"]:
        summary["has_overlap"] = True
    if split_sets["val"] & split_sets["test"]:
        summary["has_overlap"] = True

    return summary, summary_df, distribution_df


def save_outputs(split_df: pd.DataFrame) -> None:
    split_df.to_csv(SPLIT_ALL_CSV, index=False)

    split_df[split_df["split"] == "train"].to_csv(TRAIN_CSV, index=False)
    split_df[split_df["split"] == "val"].to_csv(VAL_CSV, index=False)
    split_df[split_df["split"] == "test"].to_csv(TEST_CSV, index=False)

    # Versioned split file should be committed to Git for reproducibility.
    versioned_cols = ["image_id", "split", "diagnosis_group", "region_for_split", "view_for_split"]
    split_df[versioned_cols].to_csv(VERSIONED_SPLIT_CSV, index=False)

    summary, summary_df, distribution_df = summarize_split(split_df)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    distribution_df.to_csv(DISTRIBUTION_CSV, index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    print("=== Make BTXRD train/val/test split ===")
    print(f"Seed: {SEED}")
    print(f"Ratios: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")

    df = build_base_dataframe()
    split_df = split_dataframe(df)

    save_outputs(split_df)

    summary, summary_df, distribution_df = summarize_split(split_df)

    print("\n=== Split Summary ===")
    print(summary_df.to_string(index=False))

    print("\n=== Diagnosis distribution ===")
    diagnosis_dist = distribution_df[distribution_df["category_type"] == "diagnosis"]
    print(diagnosis_dist.to_string(index=False))

    print("\nOutput files:")
    print(f"- {SPLIT_ALL_CSV}")
    print(f"- {TRAIN_CSV}")
    print(f"- {VAL_CSV}")
    print(f"- {TEST_CSV}")
    print(f"- {VERSIONED_SPLIT_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")
    print(f"- {DISTRIBUTION_CSV}")

    if summary["has_overlap"]:
        raise RuntimeError("Split overlap detected.")

    print("\nSplit safety check passed: no overlap across train/val/test.")


if __name__ == "__main__":
    main()