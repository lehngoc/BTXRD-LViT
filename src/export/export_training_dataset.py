from pathlib import Path
import json
import shutil

import pandas as pd


TEXT_CSV_PATH = Path("data/processed/text_annotations.csv")
SPLIT_CSV_PATH = Path("configs/splits/btxrd_split_seed42.csv")

IMAGE_DIR = Path("data/processed/images_preprocessed")
MASK_DIR = Path("data/processed/masks_preprocessed")

EXPORT_DIR = Path("data/exports/btxrd_preprocessed")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ALL_EXPORT_CSV = EXPORT_DIR / "all.csv"
TRAIN_EXPORT_CSV = EXPORT_DIR / "train.csv"
VAL_EXPORT_CSV = EXPORT_DIR / "val.csv"
TEST_EXPORT_CSV = EXPORT_DIR / "test.csv"
SUMMARY_JSON = EXPORT_DIR / "export_summary.json"
SUMMARY_CSV = EXPORT_DIR / "export_summary.csv"


REQUIRED_TEXT_COLUMNS = [
    "image_id",
    "age",
    "gender",
    "age_group",
    "tumor",
    "benign",
    "malignant",
    "diagnosis_text",
    "tumor_type_text",
    "tumor_subtype_text",
    "anatomy_text",
    "has_specific_anatomy",
    "bone_text",
    "joint_text",
    "region_text",
    "view_text",
    "text_diagnosis_only",
    "text_anatomy_aware",
    "text_short",
    "text_lvit_prompt",
    "segmentation_target_text",
    "negative_prompt",
]

REQUIRED_SPLIT_COLUMNS = [
    "image_id",
    "split",
    "diagnosis_group",
    "region_for_split",
    "view_for_split",
]


def find_file_by_stem(stem: str, directory: Path, extensions: list[str]) -> Path | None:
    for ext in extensions:
        path = directory / f"{stem}{ext}"
        if path.exists():
            return path

    return None


def validate_input_files() -> None:
    if not TEXT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing text annotation CSV: {TEXT_CSV_PATH}. "
            f"Run generate_text_annotations.py first."
        )

    if not SPLIT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing split CSV: {SPLIT_CSV_PATH}. "
            f"Run make_splits.py first."
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing preprocessed image directory: {IMAGE_DIR}")

    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Missing preprocessed mask directory: {MASK_DIR}")


def load_and_merge() -> pd.DataFrame:
    text_df = pd.read_csv(TEXT_CSV_PATH)
    split_df = pd.read_csv(SPLIT_CSV_PATH)

    missing_text_cols = [col for col in REQUIRED_TEXT_COLUMNS if col not in text_df.columns]
    if missing_text_cols:
        raise ValueError(f"Missing columns in text_annotations.csv: {missing_text_cols}")

    missing_split_cols = [col for col in REQUIRED_SPLIT_COLUMNS if col not in split_df.columns]
    if missing_split_cols:
        raise ValueError(f"Missing columns in split CSV: {missing_split_cols}")

    df = text_df.merge(
        split_df,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if df["split"].isna().any():
        missing = df[df["split"].isna()]["image_id"].head(20).tolist()
        raise ValueError(f"Some images do not have split assignment. Examples: {missing}")

    return df


def attach_image_mask_paths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    image_paths = []
    mask_paths = []
    has_image = []
    has_mask = []

    for image_id in df["image_id"]:
        stem = Path(str(image_id)).stem

        image_path = find_file_by_stem(
            stem=stem,
            directory=IMAGE_DIR,
            extensions=[".jpg", ".jpeg", ".png"],
        )

        mask_path = find_file_by_stem(
            stem=stem,
            directory=MASK_DIR,
            extensions=[".png", ".jpg", ".jpeg"],
        )

        image_paths.append(str(image_path) if image_path else "")
        mask_paths.append(str(mask_path) if mask_path else "")
        has_image.append(image_path is not None)
        has_mask.append(mask_path is not None)

    df["image_path"] = image_paths
    df["mask_path"] = mask_paths
    df["has_image"] = has_image
    df["has_mask"] = has_mask

    return df


def validate_export_df(df: pd.DataFrame) -> None:
    if len(df) != 3746:
        raise ValueError(f"Expected 3746 rows, got {len(df)}")

    if df["image_id"].duplicated().any():
        dup = df[df["image_id"].duplicated()]["image_id"].head(20).tolist()
        raise ValueError(f"Duplicated image_id found: {dup}")

    if not set(df["split"].unique()).issubset({"train", "val", "test"}):
        raise ValueError(f"Invalid split values: {df['split'].unique()}")

    missing_images = df[df["has_image"] == False]
    if len(missing_images) > 0:
        raise FileNotFoundError(
            "Missing preprocessed images. Examples:\n"
            + missing_images[["image_id", "image_path"]].head(20).to_string(index=False)
        )

    missing_masks = df[df["has_mask"] == False]
    if len(missing_masks) > 0:
        raise FileNotFoundError(
            "Missing preprocessed masks. Examples:\n"
            + missing_masks[["image_id", "mask_path"]].head(20).to_string(index=False)
        )

    train_ids = set(df[df["split"] == "train"]["image_id"])
    val_ids = set(df[df["split"] == "val"]["image_id"])
    test_ids = set(df[df["split"] == "test"]["image_id"])

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Split overlap detected.")

    normal_df = df[df["tumor"] == 0]
    tumor_df = df[df["tumor"] == 1]

    if not (normal_df["segmentation_target_text"] == "empty mask").all():
        raise ValueError("Some normal rows do not have empty mask target text.")

    if not (tumor_df["segmentation_target_text"] == "segment tumor region only").all():
        raise ValueError("Some tumor rows do not have segment tumor target text.")

    if not (normal_df["negative_prompt"].fillna("").str.len() > 0).all():
        raise ValueError("Some normal rows do not have negative prompts.")


def build_summary(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []

    for split_name in ["train", "val", "test"]:
        part = df[df["split"] == split_name]

        row = {
            "split": split_name,
            "count": int(len(part)),
            "normal": int((part["tumor"] == 0).sum()),
            "tumor": int((part["tumor"] == 1).sum()),
            "benign": int(part["benign"].sum()),
            "malignant": int(part["malignant"].sum()),
            "empty_mask": int((part["segmentation_target_text"] == "empty mask").sum()),
            "segment_tumor": int((part["segmentation_target_text"] == "segment tumor region only").sum()),
            "has_image": int(part["has_image"].sum()),
            "has_mask": int(part["has_mask"].sum()),
        }

        row["normal_ratio"] = row["normal"] / row["count"] if row["count"] else 0
        row["tumor_ratio"] = row["tumor"] / row["count"] if row["count"] else 0

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    summary = {
        "total_rows": int(len(df)),
        "train_rows": int((df["split"] == "train").sum()),
        "val_rows": int((df["split"] == "val").sum()),
        "test_rows": int((df["split"] == "test").sum()),
        "normal_rows": int((df["tumor"] == 0).sum()),
        "tumor_rows": int((df["tumor"] == 1).sum()),
        "benign_rows": int(df["benign"].sum()),
        "malignant_rows": int(df["malignant"].sum()),
        "missing_images": int((df["has_image"] == False).sum()),
        "missing_masks": int((df["has_mask"] == False).sum()),
        "export_dir": str(EXPORT_DIR),
    }

    return summary, summary_df


def save_exports(df: pd.DataFrame) -> None:
    export_cols = [
        "image_id",
        "split",
        "image_path",
        "mask_path",
        "tumor",
        "benign",
        "malignant",
        "diagnosis_group",
        "diagnosis_text",
        "tumor_type_text",
        "tumor_subtype_text",
        "age",
        "gender",
        "age_group",
        "region_text",
        "view_text",
        "anatomy_text",
        "has_specific_anatomy",
        "bone_text",
        "joint_text",
        "segmentation_target_text",
        "text_diagnosis_only",
        "text_anatomy_aware",
        "text_short",
        "text_lvit_prompt",
        "negative_prompt",
    ]

    missing_cols = [col for col in export_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing export columns: {missing_cols}")

    export_df = df[export_cols].copy()

    export_df.to_csv(ALL_EXPORT_CSV, index=False)
    export_df[export_df["split"] == "train"].to_csv(TRAIN_EXPORT_CSV, index=False)
    export_df[export_df["split"] == "val"].to_csv(VAL_EXPORT_CSV, index=False)
    export_df[export_df["split"] == "test"].to_csv(TEST_EXPORT_CSV, index=False)

    summary, summary_df = build_summary(df)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def print_preview(df: pd.DataFrame) -> None:
    print("\n=== Export Summary ===")
    _, summary_df = build_summary(df)
    print(summary_df.to_string(index=False))

    print("\n=== Export Preview ===")
    preview_cols = [
        "image_id",
        "split",
        "image_path",
        "mask_path",
        "tumor",
        "diagnosis_group",
        "text_lvit_prompt",
    ]
    print(df[preview_cols].head(5).to_string(index=False))

    print("\nOutput files:")
    print(f"- {ALL_EXPORT_CSV}")
    print(f"- {TRAIN_EXPORT_CSV}")
    print(f"- {VAL_EXPORT_CSV}")
    print(f"- {TEST_EXPORT_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")


def main():
    print("=== Export BTXRD preprocessed training dataset ===")

    validate_input_files()

    df = load_and_merge()
    df = attach_image_mask_paths(df)

    validate_export_df(df)

    save_exports(df)

    print_preview(df)

    print("\nExport safety check passed.")


if __name__ == "__main__":
    main()