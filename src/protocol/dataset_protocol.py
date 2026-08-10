from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


CANONICAL_SPLIT_COUNTS = {"train": 2622, "val": 562, "test": 562}
SSL_GROUPS = ("normal", "benign", "malignant")
PHASE0_SEED = 42
PHASE0_LABELED_FRACTION = 0.5


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit_hash(repo_root: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_canonical_split(
    split_path: str | Path,
    *,
    enforce_btxrd_counts: bool = True,
) -> pd.DataFrame:
    """Load and validate the committed, source-image-level BTXRD split."""
    path = Path(split_path)
    frame = pd.read_csv(path)
    required = {"image_id", "split", "diagnosis_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical split is missing columns: {sorted(missing)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Canonical split contains duplicated image_id values.")
    if set(frame["split"]) - set(CANONICAL_SPLIT_COUNTS):
        raise ValueError("Canonical split contains an unknown split name.")
    if set(frame["diagnosis_group"]) - set(SSL_GROUPS):
        raise ValueError("Canonical split contains an unknown diagnosis_group.")
    if enforce_btxrd_counts:
        counts = frame["split"].value_counts().to_dict()
        if counts != CANONICAL_SPLIT_COUNTS:
            raise ValueError(
                f"Canonical split counts must be {CANONICAL_SPLIT_COUNTS}, got {counts}."
            )
    return frame.sort_values("image_id").reset_index(drop=True)


def stratified_labeled_ids(
    train_frame: pd.DataFrame,
    *,
    labeled_fraction: float = 0.5,
    seed: int = 42,
) -> set[str]:
    """Choose an exact global labeled quota while preserving diagnosis strata.

    Per-stratum quotas start at floor(size * fraction); remaining seats are
    assigned by largest fractional remainder, then diagnosis name for a stable
    tie break. Sampling within each stratum is deterministic with ``seed``.
    """
    if not 0 < labeled_fraction < 1:
        raise ValueError("labeled_fraction must be strictly between zero and one.")
    if set(train_frame["diagnosis_group"]) - set(SSL_GROUPS):
        raise ValueError("Training frame has an unsupported diagnosis_group.")

    groups = {
        name: train_frame[train_frame["diagnosis_group"] == name].sort_values("image_id")
        for name in SSL_GROUPS
    }
    raw_quotas = {name: len(group) * labeled_fraction for name, group in groups.items()}
    quotas = {name: int(value) for name, value in raw_quotas.items()}
    target = round(len(train_frame) * labeled_fraction)
    seats_left = target - sum(quotas.values())
    order = sorted(
        SSL_GROUPS,
        key=lambda name: (-(raw_quotas[name] - quotas[name]), name),
    )
    for name in order[:seats_left]:
        quotas[name] += 1

    rng = random.Random(seed)
    selected: set[str] = set()
    for name in SSL_GROUPS:
        ids = groups[name]["image_id"].astype(str).tolist()
        selected.update(rng.sample(ids, quotas[name]))
    return selected


def build_ssl_manifests(
    *,
    canonical_split_path: str | Path,
    export_manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = PHASE0_SEED,
    labeled_fraction: float = PHASE0_LABELED_FRACTION,
    repo_root: str | Path = ".",
    protocol_config_path: str | Path | None = None,
    enforce_btxrd_counts: bool = True,
) -> dict[str, Any]:
    """Create source-image-level D_L/D_U manifests from frozen inputs.

    D_U deliberately excludes ``mask_path``. Downstream unlabeled datasets
    reject manifests that reintroduce it, making mask access impossible through
    the intended training path.
    """
    if seed != PHASE0_SEED or labeled_fraction != PHASE0_LABELED_FRACTION:
        raise ValueError("Phase 0 is frozen to seed=42 and labeled_fraction=0.5.")
    split_path = Path(canonical_split_path)
    export_path = Path(export_manifest_path)
    canonical = verify_canonical_split(split_path, enforce_btxrd_counts=enforce_btxrd_counts)
    exported = pd.read_csv(export_path)
    required_export = {"image_id", "image_path", "mask_path", "text_lvit_prompt"}
    missing_export = required_export - set(exported.columns)
    if missing_export:
        raise ValueError(f"Export manifest is missing columns: {sorted(missing_export)}")
    if exported["image_id"].duplicated().any():
        raise ValueError("Export manifest contains duplicated image_id values.")

    canonical_for_merge = canonical.rename(
        columns={"split": "canonical_split", "diagnosis_group": "canonical_diagnosis_group"}
    )
    merged = canonical_for_merge.merge(exported, on="image_id", how="left")
    if merged["image_path"].isna().any() or merged["mask_path"].isna().any():
        raise ValueError("Every canonical sample must have an exported image and mask path.")
    if "split" in exported.columns and not (merged["canonical_split"] == merged["split"]).all():
        raise ValueError("Export manifest split assignments differ from the canonical split.")
    if "diagnosis_group" in exported.columns and not (
        merged["canonical_diagnosis_group"] == merged["diagnosis_group"]
    ).all():
        raise ValueError("Export manifest diagnosis groups differ from the canonical split.")

    merged["split"] = merged["canonical_split"]
    merged["diagnosis_group"] = merged["canonical_diagnosis_group"]
    merged = merged.drop(columns=["canonical_split", "canonical_diagnosis_group"])
    train = merged[merged["split"] == "train"].copy()
    labeled_ids = stratified_labeled_ids(train, labeled_fraction=labeled_fraction, seed=seed)
    labeled = train[train["image_id"].astype(str).isin(labeled_ids)].copy()
    unlabeled = train[~train["image_id"].astype(str).isin(labeled_ids)].copy()
    labeled["ssl_partition"] = "D_L"
    unlabeled["ssl_partition"] = "D_U"
    unlabeled = unlabeled.drop(columns=[column for column in unlabeled.columns if column == "mask_path"])

    if set(labeled["image_id"]) & set(unlabeled["image_id"]):
        raise RuntimeError("D_L and D_U overlap.")
    if len(labeled) + len(unlabeled) != len(train):
        raise RuntimeError("D_L and D_U do not cover the canonical training set.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    labeled_path = destination / "btxrd_train_dl_seed42.csv"
    unlabeled_path = destination / "btxrd_train_du_seed42.csv"
    labeled.sort_values("image_id").to_csv(labeled_path, index=False)
    unlabeled.sort_values("image_id").to_csv(unlabeled_path, index=False)

    provenance = {
        "protocol": "phase0_ssl_split_v1",
        "seed": seed,
        "labeled_fraction": labeled_fraction,
        "canonical_split": str(split_path),
        "canonical_split_sha256": file_sha256(split_path),
        "export_manifest": str(export_path),
        "export_manifest_sha256": file_sha256(export_path),
        "git_commit": git_commit_hash(repo_root),
        "dl_manifest": str(labeled_path),
        "du_manifest": str(unlabeled_path),
        "counts": {
            "train": int(len(train)),
            "D_L": int(len(labeled)),
            "D_U": int(len(unlabeled)),
            "D_L_by_diagnosis": labeled["diagnosis_group"].value_counts().sort_index().to_dict(),
            "D_U_by_diagnosis": unlabeled["diagnosis_group"].value_counts().sort_index().to_dict(),
        },
    }
    if protocol_config_path is not None:
        config_path = Path(protocol_config_path)
        provenance["protocol_config"] = str(config_path)
        provenance["protocol_config_sha256"] = file_sha256(config_path)
    provenance_path = destination / "phase0_ssl_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
