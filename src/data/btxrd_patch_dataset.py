from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = [
    "patch_id",
    "patch_path",
    "mask_path",
    "image_id",
    "tumor",
    "is_positive",
    "patch_kind",
]


class BTXRDPatchSegmentationDataset(Dataset):
    """BTXRD segmentation patches backed by extract_patches.py metadata."""

    def __init__(
        self,
        csv_path: str | Path,
        expected_size: int | tuple[int, int] = 384,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        text_column: str = "text_lvit_prompt",
        include_text: bool = True,
        max_samples: int | None = None,
        root_dir: str | Path = ".",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.csv_path = self._resolve_csv_path(csv_path)
        self.patch_root = self.csv_path.parent
        self.expected_size = self._normalize_size(expected_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.text_column = text_column
        self.include_text = include_text

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing patch metadata CSV: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()
        if max_samples is not None:
            self.df = self.df.head(max_samples)
        self.df = self.df.reset_index(drop=True)

    @staticmethod
    def _normalize_size(size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(size, int):
            return size, size
        if len(size) != 2:
            raise ValueError(f"expected_size must be int or (height, width), got {size}")
        return int(size[0]), int(size[1])

    def _validate_columns(self) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing}")
        if self.include_text and self.text_column not in self.df.columns:
            raise ValueError(f"Missing text column in {self.csv_path}: {self.text_column}")

    def _resolve_csv_path(self, path_value: str | Path) -> Path:
        path = Path(str(path_value).replace("\\", "/"))
        if path.is_absolute() or path.exists():
            return path
        rooted_path = self.root_dir / path
        return rooted_path if rooted_path.exists() else path

    def _resolve_patch_path(self, path_value: str | Path) -> Path:
        path = Path(str(path_value).replace("\\", "/"))
        if path.is_absolute():
            return path

        metadata_relative = self.patch_root / path
        if metadata_relative.exists():
            return metadata_relative

        rooted_path = self.root_dir / path
        if rooted_path.exists():
            return rooted_path

        return metadata_relative

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image = self._load_image(row["patch_path"])
        mask = self._load_mask(row["mask_path"])
        is_positive = int(row["is_positive"])

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "patch_id": str(row["patch_id"]),
            "image_id": str(row["image_id"]),
            "patch_kind": str(row["patch_kind"]),
            "is_positive": torch.tensor(is_positive, dtype=torch.long),
            "tumor": torch.tensor(is_positive, dtype=torch.long),
            "source_tumor": torch.tensor(int(row["tumor"]), dtype=torch.long),
            "diagnosis_group": str(row.get("diagnosis_group", "")),
        }

        for column in ["crop_x", "crop_y", "crop_w", "crop_h"]:
            if column in row and not pd.isna(row[column]):
                sample[column] = torch.tensor(int(row[column]), dtype=torch.long)

        if self.include_text:
            text_value = row.get(self.text_column, "")
            sample["text"] = "" if pd.isna(text_value) else str(text_value)

        return sample

    def _load_image(self, path_value: str | Path) -> torch.Tensor:
        path = self._resolve_patch_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing patch image: {path}")

        image = Image.open(path).convert("RGB")
        width, height = image.size
        expected_h, expected_w = self.expected_size
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                f"Unexpected patch image size for {path}: {(height, width)}, "
                f"expected {(expected_h, expected_w)}"
            )

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return (image_tensor - self.image_mean) / self.image_std

    def _load_mask(self, path_value: str | Path) -> torch.Tensor:
        path = self._resolve_patch_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing patch mask: {path}")

        mask = Image.open(path).convert("L")
        width, height = mask.size
        expected_h, expected_w = self.expected_size
        if (height, width) != (expected_h, expected_w):
            raise ValueError(
                f"Unexpected patch mask size for {path}: {(height, width)}, "
                f"expected {(expected_h, expected_w)}"
            )

        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(mask_np).unsqueeze(0)
