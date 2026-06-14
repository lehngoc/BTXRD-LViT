from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models import LViTT
from src.training.train_lvit_tw import train_from_config
from src.training.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E3 LViT-T with text_lvit_prompt on BTXRD.")
    parser.add_argument("--config", default="configs/train_lvit_t_preprocessed_224.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    return LViTT(
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
        base_channels=model_cfg.get("base_channels", 48),
        transformer_depth=model_cfg.get("transformer_depth", 2),
        transformer_heads=model_cfg.get("transformer_heads", 4),
        transformer_dropout=model_cfg.get("transformer_dropout", 0.0),
        text_vocab_size=model_cfg.get("text_vocab_size", 4096),
        text_embed_dim=model_cfg.get("text_embed_dim", 128),
        text_max_tokens=model_cfg.get("text_max_tokens", 48),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_from_config(cfg, args, build_model(cfg), use_text=True)


if __name__ == "__main__":
    main()
