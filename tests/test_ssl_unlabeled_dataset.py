from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.ssl_dataset import BTXRDUnlabeledDataset


class UnlabeledDatasetTests(unittest.TestCase):
    def test_du_reader_returns_image_and_text_without_mask_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (11, 8)).save(image_path)
            manifest = root / "du.csv"
            pd.DataFrame([{
                "image_id": "image.jpg", "image_path": str(image_path),
                "text_lvit_prompt": "normal X-ray", "ssl_partition": "D_U",
            }]).to_csv(manifest, index=False)
            sample = BTXRDUnlabeledDataset(manifest)[0]
            self.assertEqual(set(sample), {"image", "image_id", "text"})
            self.assertEqual(tuple(sample["image"].shape), (3, 8, 11))

    def test_du_reader_resizes_only_when_legacy_size_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (11, 8)).save(image_path)
            manifest = root / "du.csv"
            pd.DataFrame([{
                "image_id": "image.jpg", "image_path": str(image_path),
                "text_lvit_prompt": "normal X-ray", "ssl_partition": "D_U",
            }]).to_csv(manifest, index=False)
            sample = BTXRDUnlabeledDataset(manifest, image_size=224)[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 224, 224))

    def test_du_reader_rejects_mask_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "du.csv"
            pd.DataFrame([{
                "image_id": "image.jpg", "image_path": "image.jpg", "mask_path": "mask.png",
                "text_lvit_prompt": "text", "ssl_partition": "D_U",
            }]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "mask_path"):
                BTXRDUnlabeledDataset(path)


if __name__ == "__main__":
    unittest.main()
