from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.protocol.dataset_protocol import build_ssl_manifests, stratified_labeled_ids, verify_canonical_split


class Phase0ProtocolTests(unittest.TestCase):
    def _train_frame(self) -> pd.DataFrame:
        rows = []
        for diagnosis, count in (("normal", 5), ("benign", 5), ("malignant", 4)):
            rows.extend(
                {"image_id": f"{diagnosis}_{index:02d}.jpg", "split": "train", "diagnosis_group": diagnosis}
                for index in range(count)
            )
        return pd.DataFrame(rows)

    def test_stratified_selection_is_deterministic_disjoint_and_near_half(self) -> None:
        train = self._train_frame()
        first = stratified_labeled_ids(train, seed=42)
        second = stratified_labeled_ids(train, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), round(len(train) * 0.5))
        for diagnosis, group in train.groupby("diagnosis_group"):
            selected = group["image_id"].isin(first).sum()
            self.assertLessEqual(abs(selected - len(group) * 0.5), 1)

    def test_build_manifests_removes_du_mask_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._train_frame()
            canonical = pd.concat([
                train,
                pd.DataFrame([
                    {"image_id": "val.jpg", "split": "val", "diagnosis_group": "normal"},
                    {"image_id": "test.jpg", "split": "test", "diagnosis_group": "normal"},
                ]),
            ],
                ignore_index=True,
            )
            canonical_path = root / "split.csv"
            canonical.to_csv(canonical_path, index=False)
            exported = canonical.assign(
                image_path=lambda frame: frame["image_id"],
                mask_path=lambda frame: frame["image_id"].str.replace(".jpg", ".png", regex=False),
                text_lvit_prompt="text",
            )
            export_path = root / "all.csv"
            exported.to_csv(export_path, index=False)
            config_path = root / "config.yaml"
            config_path.write_text("seed: 42\n", encoding="utf-8")

            result = build_ssl_manifests(
                canonical_split_path=canonical_path,
                export_manifest_path=export_path,
                output_dir=root / "output",
                protocol_config_path=config_path,
                enforce_btxrd_counts=False,
            )
            dl = pd.read_csv(root / "output" / "btxrd_train_dl_seed42.csv")
            du = pd.read_csv(root / "output" / "btxrd_train_du_seed42.csv")
            self.assertNotIn("mask_path", du.columns)
            self.assertIn("mask_path", dl.columns)
            self.assertFalse(set(dl.image_id) & set(du.image_id))
            self.assertEqual(len(dl) + len(du), len(train))
            self.assertEqual(result["counts"]["D_L"], 7)
            provenance = json.loads((root / "output" / "phase0_ssl_provenance.json").read_text())
            self.assertEqual(provenance["seed"], 42)

    def test_split_validation_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            pd.DataFrame([
                {"image_id": "same.jpg", "split": "train", "diagnosis_group": "normal"},
                {"image_id": "same.jpg", "split": "val", "diagnosis_group": "normal"},
            ]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                verify_canonical_split(path, enforce_btxrd_counts=False)

    def test_phase0_builder_rejects_non_frozen_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "seed=42"):
                build_ssl_manifests(
                    canonical_split_path=root / "missing.csv",
                    export_manifest_path=root / "missing_export.csv",
                    output_dir=root / "output",
                    seed=7,
                    enforce_btxrd_counts=False,
                )

    def test_export_without_split_or_diagnosis_uses_canonical_source_of_truth(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = self._train_frame()
            canonical = pd.concat([
                train,
                pd.DataFrame([
                    {"image_id": "val.jpg", "split": "val", "diagnosis_group": "normal"},
                    {"image_id": "test.jpg", "split": "test", "diagnosis_group": "normal"},
                ]),
            ], ignore_index=True)
            canonical_path = root / "split.csv"
            canonical.to_csv(canonical_path, index=False)
            exported = canonical.assign(
                image_path=lambda frame: frame["image_id"],
                mask_path=lambda frame: frame["image_id"].str.replace(".jpg", ".png", regex=False),
                text_lvit_prompt="text",
            ).drop(columns=["split", "diagnosis_group"])
            export_path = root / "all.csv"
            exported.to_csv(export_path, index=False)
            result = build_ssl_manifests(
                canonical_split_path=canonical_path,
                export_manifest_path=export_path,
                output_dir=root / "output",
                enforce_btxrd_counts=False,
            )
            self.assertEqual(result["counts"]["D_L"] + result["counts"]["D_U"], len(train))


if __name__ == "__main__":
    unittest.main()
