from __future__ import annotations

import hashlib

import torch
from torch import nn

from src.models.lvit_tw import LViTTW


class HashTextEncoder(nn.Module):
    """Small deterministic text encoder for controlled BTXRD prompt ablations."""

    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 128,
        output_dim: int = 384,
        max_tokens: int = 48,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
        return int.from_bytes(digest, byteorder="little") % (self.vocab_size - 1) + 1

    def tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor:
        rows: list[list[int]] = []
        for text in texts:
            tokens = str(text).lower().replace(".", " ").replace(",", " ").replace(";", " ").split()
            token_ids = [self._token_id(token) for token in tokens[: self.max_tokens]]
            token_ids.extend([0] * (self.max_tokens - len(token_ids)))
            rows.append(token_ids)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        token_ids = self.tokenize(texts, device=device)
        mask = token_ids.ne(0).unsqueeze(-1)
        embeddings = self.embedding(token_ids)
        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1)
        pooled = summed / counts
        return self.proj(pooled)


class LViTT(LViTTW):
    """Text-conditioned LViT-style model using BTXRD text_lvit_prompt."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 48,
        transformer_depth: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        text_vocab_size: int = 4096,
        text_embed_dim: int = 128,
        text_max_tokens: int = 48,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        bottleneck_channels = base_channels * 8
        self.text_encoder = HashTextEncoder(
            vocab_size=text_vocab_size,
            embed_dim=text_embed_dim,
            output_dim=bottleneck_channels,
            max_tokens=text_max_tokens,
        )
        self.text_scale = nn.Linear(bottleneck_channels, bottleneck_channels)
        self.text_shift = nn.Linear(bottleneck_channels, bottleneck_channels)

    def forward(self, images: torch.Tensor, text: list[str] | None = None) -> torch.Tensor:
        if text is None:
            raise ValueError("LViTT requires text prompts. Use LViTTW for no-text experiments.")

        x1, x2, x3, x4 = self.encode(images)
        text_features = self.text_encoder(list(text), device=images.device)
        scale = torch.tanh(self.text_scale(text_features)).view(images.shape[0], -1, 1, 1)
        shift = self.text_shift(text_features).view(images.shape[0], -1, 1, 1)
        x4 = x4 * (1.0 + scale) + shift
        return self.decode(x1, x2, x3, x4)
