from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.inference.sliding_window import predict_sliding_window
from src.training.patch_pipeline import build_patch_dataset
from src.training.train_lvit_t import build_model
from src.training.train_lvit_tw import build_criterion
from src.training.utils import load_config


class CountingTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_preparations = 0

    def prepare_text_features(self, text: list[str], images: torch.Tensor) -> torch.Tensor:
        self.text_preparations += 1
        return images.new_zeros((1, 10, 8))

    def forward(self, images: torch.Tensor, text: torch.Tensor | None = None) -> torch.Tensor:
        assert text is not None
        return images.new_zeros((images.shape[0], 1, images.shape[2], images.shape[3]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test text-conditioned LViT-T Patch384.")
    parser.add_argument("--config", default="configs/train_lvit_t_patch384.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["model"]["text_encoder_provider"] = "hash"

    dataset = build_patch_dataset(
        smoke_cfg["data"],
        smoke_cfg["training"],
        "train",
        max_samples=1,
        include_text=True,
    )
    sample = dataset[0]
    assert sample["text"]
    assert tuple(sample["image"].shape) == (3, 384, 384)
    assert tuple(sample["mask"].shape) == (1, 384, 384)
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0

    model = build_model(smoke_cfg).eval()
    with torch.no_grad():
        logits = model(sample["image"].unsqueeze(0), text=[sample["text"]])
    assert logits.shape == sample["mask"].unsqueeze(0).shape
    assert torch.isfinite(logits).all()

    criterion = build_criterion(smoke_cfg["training"])
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = criterion(
            logits,
            sample["mask"].unsqueeze(0),
            tumor=sample["is_positive"].unsqueeze(0),
        )
    assert torch.isfinite(loss)

    image = torch.zeros(384, 384, 3, dtype=torch.uint8).numpy()
    probability, window_count = predict_sliding_window(
        model=model,
        image=image,
        device=torch.device("cpu"),
        patch_size=384,
        stride=192,
        batch_size=1,
        text_prompt=sample["text"],
        image_mean=tuple(smoke_cfg["training"]["image_mean"]),
        image_std=tuple(smoke_cfg["training"]["image_std"]),
    )
    assert probability.shape == (384, 384)
    assert window_count == 1

    counting_model = CountingTextModel()
    _, window_count = predict_sliding_window(
        model=counting_model,
        image=torch.zeros(500, 500, 3, dtype=torch.uint8).numpy(),
        device=torch.device("cpu"),
        patch_size=384,
        stride=192,
        batch_size=1,
        text_prompt=sample["text"],
    )
    assert window_count == 4
    assert counting_model.text_preparations == 1
    print("LViT-T Patch384 text pipeline smoke test passed.")


if __name__ == "__main__":
    main()
