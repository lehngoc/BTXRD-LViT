from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BTXRDUnlabeledDataset(Dataset):
    """Image/text-only D_U reader; it rejects any manifest exposing GT masks."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        image_size: int | tuple[int, int] | None = None,
        text_column: str = "text_lvit_prompt",
        root_dir: str | Path = ".",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.csv_path = self._resolve_path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        required = {"image_id", "image_path", "ssl_partition", text_column}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"D_U manifest is missing columns: {sorted(missing)}")
        if "mask_path" in self.frame.columns:
            raise ValueError("D_U manifests must not expose mask_path.")
        if set(self.frame["ssl_partition"].astype(str)) != {"D_U"}:
            raise ValueError("BTXRDUnlabeledDataset only accepts D_U manifests.")
        self.text_column = text_column
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(str(value).replace("\\", "/"))
        return path if path.is_absolute() else self.root_dir / path

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = self._resolve_path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height), resample=Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return {
            "image": image_tensor,
            "image_id": str(row["image_id"]),
            "text": "" if pd.isna(row[self.text_column]) else str(row[self.text_column]),
        }
