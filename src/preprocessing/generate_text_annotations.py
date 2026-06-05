from pathlib import Path
import json

import pandas as pd


RAW_CSV_PATH = Path("data/raw/dataset.csv")

OUTPUT_TEXT_CSV = Path("data/processed/text_annotations.csv")
REPORT_DIR = Path("data/processed/reports")
SUMMARY_CSV = REPORT_DIR / "text_annotation_summary.csv"
EXAMPLES_CSV = REPORT_DIR / "text_annotation_examples.csv"
SUMMARY_JSON = REPORT_DIR / "text_annotation_summary.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TEXT_CSV.parent.mkdir(parents=True, exist_ok=True)


BONE_COLUMNS = [
    "hand",
    "ulna",
    "radius",
    "humerus",
    "foot",
    "tibia",
    "fibula",
    "femur",
    "hip bone",
]

JOINT_COLUMNS = [
    "ankle-joint",
    "knee-joint",
    "hip-joint",
    "wrist-joint",
    "elbow-joint",
    "shoulder-joint",
]

REGION_COLUMNS = [
    "upper limb",
    "lower limb",
    "pelvis",
]

VIEW_COLUMNS = [
    "frontal",
    "lateral",
    "oblique",
]

BENIGN_TUMOR_COLUMNS = [
    "osteochondroma",
    "multiple osteochondromas",
    "simple bone cyst",
    "giant cell tumor",
    "osteofibroma",
    "synovial osteochondroma",
    "other bt",
]

MALIGNANT_TUMOR_COLUMNS = [
    "osteosarcoma",
    "other mt",
]

TUMOR_TYPE_COLUMNS = BENIGN_TUMOR_COLUMNS + MALIGNANT_TUMOR_COLUMNS


COLUMN_DISPLAY_NAME = {
    "hand": "hand",
    "ulna": "ulna",
    "radius": "radius",
    "humerus": "humerus",
    "foot": "foot",
    "tibia": "tibia",
    "fibula": "fibula",
    "femur": "femur",
    "hip bone": "hip bone",
    "ankle-joint": "ankle joint",
    "knee-joint": "knee joint",
    "hip-joint": "hip joint",
    "wrist-joint": "wrist joint",
    "elbow-joint": "elbow joint",
    "shoulder-joint": "shoulder joint",
    "upper limb": "upper limb",
    "lower limb": "lower limb",
    "pelvis": "pelvis",
    "frontal": "frontal",
    "lateral": "lateral",
    "oblique": "oblique",
    "osteochondroma": "osteochondroma",
    "multiple osteochondromas": "multiple osteochondromas",
    "simple bone cyst": "simple bone cyst",
    "giant cell tumor": "giant cell tumor",
    "osteofibroma": "osteofibroma",
    "synovial osteochondroma": "synovial osteochondroma",
    "other bt": "other benign bone tumor",
    "osteosarcoma": "osteosarcoma",
    "other mt": "other malignant bone tumor",
}


def value_is_one(value) -> bool:
    """
    Robustly interpret a CSV value as binary 1.
    """
    if pd.isna(value):
        return False

    try:
        return int(value) == 1
    except Exception:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_active_columns(row: pd.Series, columns: list[str]) -> list[str]:
    active = []

    for col in columns:
        if col in row.index and value_is_one(row[col]):
            active.append(col)

    return active


def format_list(items: list[str]) -> str:
    """
    Convert ['a', 'b', 'c'] to 'a, b, and c'.
    """
    if len(items) == 0:
        return ""

    if len(items) == 1:
        return items[0]

    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    return ", ".join(items[:-1]) + f", and {items[-1]}"


def get_age_group(age) -> str:
    if pd.isna(age):
        return "unknown age group"

    try:
        age_int = int(age)
    except Exception:
        return "unknown age group"

    if age_int < 18:
        return "pediatric"
    if age_int < 40:
        return "young adult"
    if age_int < 65:
        return "adult"

    return "older adult"


def get_gender_text(gender) -> str:
    if pd.isna(gender):
        return "patient"

    gender_str = str(gender).strip().upper()

    if gender_str == "M":
        return "male patient"

    if gender_str == "F":
        return "female patient"

    return "patient"


def get_age_gender_patient_text(age, age_group: str, gender_text: str) -> str:
    """
    Build a grammatically correct patient description.
    """
    if age_group == "unknown age group":
        if gender_text == "patient":
            return "This is a patient."
        return f"This is a {gender_text}."

    article = "an" if age_group in {"adult", "older adult"} else "a"

    if pd.isna(age):
        return f"This is {article} {age_group} {gender_text}."

    try:
        age_int = int(age)
        return f"This is {article} {age_group} {gender_text}, age {age_int}."
    except Exception:
        return f"This is {article} {age_group} {gender_text}."


def get_view_text(row: pd.Series) -> tuple[str, str]:
    active_views = get_active_columns(row, VIEW_COLUMNS)
    view_names = [COLUMN_DISPLAY_NAME[col] for col in active_views]

    if len(view_names) == 0:
        return "X-ray", "unknown view"

    view_phrase = format_list(view_names)

    if len(view_names) == 1:
        return f"{view_phrase} X-ray", view_phrase

    return f"{view_phrase} X-ray", view_phrase


def get_region_text(row: pd.Series) -> tuple[str, str]:
    active_regions = get_active_columns(row, REGION_COLUMNS)
    region_names = [COLUMN_DISPLAY_NAME[col] for col in active_regions]

    if len(region_names) == 0:
        return "unspecified body region", "unknown region"

    return format_list(region_names), format_list(region_names)


def get_anatomy_text(row: pd.Series) -> tuple[str, str, str]:
    active_bones = get_active_columns(row, BONE_COLUMNS)
    active_joints = get_active_columns(row, JOINT_COLUMNS)

    bone_names = [COLUMN_DISPLAY_NAME[col] for col in active_bones]
    joint_names = [COLUMN_DISPLAY_NAME[col] for col in active_joints]

    if len(bone_names) == 0 and len(joint_names) == 0:
        return "unspecified anatomical site", "unknown bone site", "unknown joint site"

    anatomy_parts = []

    if len(bone_names) > 0:
        anatomy_parts.append(format_list(bone_names))

    if len(joint_names) > 0:
        anatomy_parts.append(format_list(joint_names))

    anatomy_text = format_list(anatomy_parts)

    bone_text = format_list(bone_names) if len(bone_names) > 0 else "unknown bone site"
    joint_text = format_list(joint_names) if len(joint_names) > 0 else "unknown joint site"

    return anatomy_text, bone_text, joint_text


def get_tumor_type_text(row: pd.Series) -> tuple[str, str]:
    active_tumor_types = get_active_columns(row, TUMOR_TYPE_COLUMNS)
    tumor_names = [COLUMN_DISPLAY_NAME[col] for col in active_tumor_types]

    if len(tumor_names) == 0:
        return "unspecified bone tumor", "unknown tumor subtype"

    tumor_type_text = format_list(tumor_names)

    return tumor_type_text, tumor_type_text


def get_diagnosis_text(row: pd.Series) -> str:
    tumor = value_is_one(row.get("tumor", 0))
    benign = value_is_one(row.get("benign", 0))
    malignant = value_is_one(row.get("malignant", 0))

    if not tumor:
        return "no bone tumor"

    if benign and malignant:
        return "bone tumor with inconsistent benign and malignant labels"

    if benign:
        return "benign bone tumor"

    if malignant:
        return "malignant bone tumor"

    return "bone tumor"


def build_location_phrases(
    view_modality_text: str,
    region_text: str,
    anatomy_text: str,
) -> tuple[str, str, str, bool]:
    """
    Build location phrases while avoiding awkward prompts like:
    'focused on the unspecified anatomical site'.
    """
    has_specific_anatomy = anatomy_text != "unspecified anatomical site"

    if has_specific_anatomy:
        anatomy_sentence = (
            f"{view_modality_text} of the {region_text}, focused on the {anatomy_text}."
        )
        text_location_phrase = f"{view_modality_text} of the {region_text}, {anatomy_text}"
        anatomy_aware_location = (
            f"{view_modality_text} of the {region_text}, focused on the {anatomy_text}"
        )
    else:
        anatomy_sentence = f"{view_modality_text} of the {region_text}."
        text_location_phrase = f"{view_modality_text} of the {region_text}"
        anatomy_aware_location = f"{view_modality_text} of the {region_text}"

    return anatomy_sentence, text_location_phrase, anatomy_aware_location, has_specific_anatomy


def build_tumor_disease_sentence(
    tumor_subtype_text: str,
    benign: bool,
    malignant: bool,
) -> str:
    if benign:
        tumor_category_text = "a benign bone tumor"
    elif malignant:
        tumor_category_text = "a malignant bone tumor"
    else:
        tumor_category_text = "a bone tumor"

    if tumor_subtype_text == "unknown tumor subtype":
        return f"The image contains {tumor_category_text}."

    if tumor_subtype_text.startswith("other "):
        return (
            f"The image contains {tumor_category_text} classified as "
            f"{tumor_subtype_text}."
        )

    return f"The image contains {tumor_subtype_text}, {tumor_category_text}."


def build_texts(row: pd.Series) -> dict:
    image_id = str(row["image_id"])

    age = row.get("age", None)
    age_group = get_age_group(age)
    gender_text = get_gender_text(row.get("gender", None))
    patient_text = get_age_gender_patient_text(age, age_group, gender_text)

    view_modality_text, view_text = get_view_text(row)
    region_text, region_label_text = get_region_text(row)
    anatomy_text, bone_text, joint_text = get_anatomy_text(row)
    tumor_type_text, tumor_subtype_text = get_tumor_type_text(row)
    diagnosis_text = get_diagnosis_text(row)

    (
        anatomy_sentence,
        text_location_phrase,
        anatomy_aware_location,
        has_specific_anatomy,
    ) = build_location_phrases(
        view_modality_text=view_modality_text,
        region_text=region_text,
        anatomy_text=anatomy_text,
    )

    tumor = value_is_one(row.get("tumor", 0))
    benign = value_is_one(row.get("benign", 0))
    malignant = value_is_one(row.get("malignant", 0))

    if tumor:
        disease_sentence = build_tumor_disease_sentence(
            tumor_subtype_text=tumor_subtype_text,
            benign=benign,
            malignant=malignant,
        )

        text_short = (
            f"{text_location_phrase}. "
            f"{diagnosis_text}."
        )

        # Text input for LViT training.
        # Keep it as metadata-style description, without patient age/gender
        # and without explicit segmentation instructions.
        text_lvit_prompt = (
            f"{anatomy_sentence} "
            f"{disease_sentence}"
        )

        segmentation_target_text = "segment tumor region only"
        negative_prompt = ""

    else:
        disease_sentence = (
            "No benign or malignant bone tumor is annotated; "
            "the expected segmentation mask is empty."
        )

        text_short = (
            f"{text_location_phrase}. "
            f"No bone tumor."
        )

        # Text input for LViT training.
        # Keep negative information, but remove patient age/gender
        # and avoid direct segmentation instruction wording.
        text_lvit_prompt = (
            f"{anatomy_sentence} "
            f"{disease_sentence}"
        )

        segmentation_target_text = "empty mask"

        negative_prompt = (
            "No benign or malignant bone tumor is present; "
            "the expected segmentation mask is empty."
        )

    # More compact variants for later ablation.
    text_diagnosis_only = (
        "The image contains a bone tumor."
        if tumor
        else "The image contains no bone tumor."
    )

    if tumor:
        text_anatomy_aware = (
            f"{anatomy_aware_location}. "
            f"The image contains {diagnosis_text}."
        )
    else:
        text_anatomy_aware = (
            f"{anatomy_aware_location}. "
            f"No bone tumor is present."
        )

    return {
        "image_id": image_id,
        "age": age,
        "gender": row.get("gender", ""),
        "age_group": age_group,
        "tumor": int(tumor),
        "benign": int(benign),
        "malignant": int(malignant),
        "diagnosis_text": diagnosis_text,
        "tumor_type_text": tumor_type_text,
        "tumor_subtype_text": tumor_subtype_text,
        "anatomy_text": anatomy_text,
        "has_specific_anatomy": int(has_specific_anatomy),
        "bone_text": bone_text,
        "joint_text": joint_text,
        "region_text": region_label_text,
        "view_text": view_text,
        "text_diagnosis_only": text_diagnosis_only,
        "text_anatomy_aware": text_anatomy_aware,
        "text_short": text_short,
        "text_lvit_prompt": text_lvit_prompt,
        "segmentation_target_text": segmentation_target_text,
        "negative_prompt": negative_prompt,
    }


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "image_id",
        "age",
        "gender",
        "tumor",
        "benign",
        "malignant",
    ]

    missing_required = [col for col in required_columns if col not in df.columns]

    if len(missing_required) > 0:
        raise ValueError(f"Missing required columns in dataset.csv: {missing_required}")

    metadata_columns = (
        BONE_COLUMNS
        + JOINT_COLUMNS
        + REGION_COLUMNS
        + VIEW_COLUMNS
        + TUMOR_TYPE_COLUMNS
    )

    missing_metadata = [col for col in metadata_columns if col not in df.columns]

    if len(missing_metadata) > 0:
        raise ValueError(f"Missing metadata columns in dataset.csv: {missing_metadata}")


def generate_summary(out_df: pd.DataFrame) -> dict:
    summary = {
        "total_rows": int(len(out_df)),
        "tumor_rows": int(out_df["tumor"].sum()),
        "normal_rows": int((out_df["tumor"] == 0).sum()),
        "benign_rows": int(out_df["benign"].sum()),
        "malignant_rows": int(out_df["malignant"].sum()),
        "empty_mask_text_rows": int((out_df["segmentation_target_text"] == "empty mask").sum()),
        "segment_tumor_text_rows": int((out_df["segmentation_target_text"] == "segment tumor region only").sum()),
        "unknown_anatomy_rows": int((out_df["anatomy_text"] == "unspecified anatomical site").sum()),
        "specific_anatomy_rows": int(out_df["has_specific_anatomy"].sum()),
        "unknown_region_rows": int((out_df["region_text"] == "unknown region").sum()),
        "unknown_view_rows": int((out_df["view_text"] == "unknown view").sum()),
        "unknown_tumor_subtype_rows": int(
            ((out_df["tumor"] == 1) & (out_df["tumor_subtype_text"] == "unknown tumor subtype")).sum()
        ),
        "normal_negative_prompt_rows": int(
            ((out_df["tumor"] == 0) & (out_df["negative_prompt"].str.len() > 0)).sum()
        ),
    }

    return summary


def save_examples(out_df: pd.DataFrame) -> None:
    examples = []

    groups = {
        "normal": out_df[out_df["tumor"] == 0],
        "benign": out_df[(out_df["tumor"] == 1) & (out_df["benign"] == 1)],
        "malignant": out_df[(out_df["tumor"] == 1) & (out_df["malignant"] == 1)],
    }

    for group_name, group_df in groups.items():
        if len(group_df) == 0:
            continue

        n = min(10, len(group_df))
        sample_df = group_df.sample(n=n, random_state=42)

        for _, row in sample_df.iterrows():
            examples.append(
                {
                    "group": group_name,
                    "image_id": row["image_id"],
                    "text_short": row["text_short"],
                    "text_lvit_prompt": row["text_lvit_prompt"],
                    "segmentation_target_text": row["segmentation_target_text"],
                }
            )

    pd.DataFrame(examples).to_csv(EXAMPLES_CSV, index=False)


def main():
    print("=== Generate structured BTXRD text annotations ===")

    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing dataset CSV: {RAW_CSV_PATH}")

    df = pd.read_csv(RAW_CSV_PATH)

    validate_required_columns(df)

    records = []

    for _, row in df.iterrows():
        records.append(build_texts(row))

    out_df = pd.DataFrame(records)

    out_df.to_csv(OUTPUT_TEXT_CSV, index=False)

    summary = generate_summary(out_df)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(SUMMARY_CSV, index=False)

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_examples(out_df)

    print("\n=== Text Annotation Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nOutput files:")
    print(f"- {OUTPUT_TEXT_CSV}")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")
    print(f"- {EXAMPLES_CSV}")

    print("\nExample prompts:")
    preview_cols = ["image_id", "tumor", "benign", "malignant", "text_lvit_prompt"]
    print(out_df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()