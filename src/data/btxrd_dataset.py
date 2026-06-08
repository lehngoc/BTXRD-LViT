from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = [
    "image_id",
    "image_path",
    "mask_path",
    "tumor",
    "diagnosis_group",
]


class BTXRDSegmentationDataset(Dataset):
    """BTXRD segmentation dataset backed by exported train/val/test CSV files.

    Exported CSVs may contain Windows-style paths. Paths are normalized at load
    time so the same manifests can be used on Linux/Kaggle.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_size: int | tuple[int, int] = 224,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        text_column: str = "text_lvit_prompt",
        include_text: bool = True,
        max_samples: int | None = None,
        label_fraction: float = 1.0,
        label_seed: int = 42,
        tumor_only: bool = False,
        root_dir: str | Path = ".",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.csv_path = self._resolve_input_csv_path(csv_path)
        self.image_size = self._normalize_size(image_size)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
        self.text_column = text_column
        self.include_text = include_text

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Missing dataset CSV: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate_columns()
        self.df = self._apply_tumor_filter(self.df, tumor_only)
        self.df = self._apply_label_fraction(self.df, label_fraction, label_seed)

        if max_samples is not None:
            self.df = self.df.head(max_samples)

        self.df = self.df.reset_index(drop=True)

    @staticmethod
    def _normalize_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
        if isinstance(image_size, int):
            return image_size, image_size

        if len(image_size) != 2:
            raise ValueError(f"image_size must be int or (height, width), got {image_size}")

        return int(image_size[0]), int(image_size[1])

    def _validate_columns(self) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {self.csv_path}: {missing}")

        if self.include_text and self.text_column not in self.df.columns:
            raise ValueError(f"Missing text column in {self.csv_path}: {self.text_column}")

    @staticmethod
    def _apply_tumor_filter(df: pd.DataFrame, tumor_only: bool) -> pd.DataFrame:
        if not tumor_only:
            return df

        return df[df["tumor"].astype(int) == 1].copy()

    def _resolve_input_csv_path(self, path_value: str | Path) -> Path:
        path = Path(str(path_value).replace("\\", "/"))

        if path.is_absolute() or path.exists():
            return path

        rooted_path = self.root_dir / path
        if rooted_path.exists():
            return rooted_path

        return path

    @staticmethod
    def _apply_label_fraction(df: pd.DataFrame, label_fraction: float, seed: int) -> pd.DataFrame:
        if not 0 < label_fraction <= 1:
            raise ValueError(f"label_fraction must be in (0, 1], got {label_fraction}")

        if label_fraction == 1:
            return df

        tumor_df = df[df["tumor"].astype(int) == 1]
        normal_df = df[df["tumor"].astype(int) == 0]
        sampled_tumor = tumor_df.sample(frac=label_fraction, random_state=seed)

        return pd.concat([normal_df, sampled_tumor], axis=0).sort_index()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        image = self._load_image(row["image_path"])
        mask = self._load_mask(row["mask_path"])

        sample: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "image_id": str(row["image_id"]),
            "tumor": torch.tensor(int(row["tumor"]), dtype=torch.long),
            "diagnosis_group": str(row["diagnosis_group"]),
        }

        if self.include_text:
            text_value = row.get(self.text_column, "")
            sample["text"] = "" if pd.isna(text_value) else str(text_value)

        return sample

    def _resolve_path(self, path_value: str | Path) -> Path:
        path = Path(str(path_value).replace("\\", "/"))
        if path.is_absolute():
            return path

        return self.root_dir / path

    def _load_image(self, path_value: str | Path) -> torch.Tensor:
        path = self._resolve_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing image file: {path}")

        height, width = self.image_size
        image = Image.open(path).convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)

        return (image_tensor - self.image_mean) / self.image_std

    def _load_mask(self, path_value: str | Path) -> torch.Tensor:
        path = self._resolve_path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Missing mask file: {path}")

        height, width = self.image_size
        mask = Image.open(path).convert("L")
        mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
        mask_np = np.asarray(mask, dtype=np.uint8)
        mask_np = (mask_np > 0).astype(np.float32)

        return torch.from_numpy(mask_np).unsqueeze(0)
