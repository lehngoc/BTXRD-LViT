from __future__ import annotations

import hashlib

import torch
from torch import nn

from src.models.lvit_tw import LViTTW


class HashTextEmbedding(nn.Module):
    """Deterministic fallback for smoke tests without HuggingFace dependencies."""

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 768,
        max_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)

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
        return self.embedding(self.tokenize(texts, device=device))


class HuggingFaceBertEmbedding(nn.Module):
    """Maintained BERT replacement that preserves the original LViT [B, 10, 768] text interface."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 10,
        hidden_size: int = 768,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFace text encoding requires the 'transformers' package. "
                "Install requirements.txt or set model.text_encoder_provider: hash for smoke tests."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.max_tokens = max_tokens
        self.freeze = freeze
        bert_hidden_size = int(self.bert.config.hidden_size)
        self.proj = nn.Identity() if bert_hidden_size == hidden_size else nn.Linear(bert_hidden_size, hidden_size)

        if freeze:
            self.bert.eval()
            for parameter in self.bert.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        if self.freeze:
            with torch.no_grad():
                hidden = self.bert(**encoded).last_hidden_state
        else:
            hidden = self.bert(**encoded).last_hidden_state

        return self.proj(hidden[:, : self.max_tokens, :])


class LViTT(LViTTW):
    """Text-conditioned LViT-T with the original Double-U image/text fusion path."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        transformer_depth: int = 1,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        image_size: int = 224,
        text_encoder_provider: str = "huggingface",
        text_encoder_model_name: str = "bert-base-uncased",
        text_vocab_size: int = 8192,
        text_embed_dim: int = 768,
        text_max_tokens: int = 10,
        text_encoder_freeze: bool = True,
        text_encoder_local_files_only: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            transformer_depth=transformer_depth,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            image_size=image_size,
            text_tokens=text_max_tokens,
            text_dim=text_embed_dim,
        )
        provider = text_encoder_provider.lower()
        if provider == "huggingface":
            self.text_encoder = HuggingFaceBertEmbedding(
                model_name=text_encoder_model_name,
                max_tokens=text_max_tokens,
                hidden_size=text_embed_dim,
                freeze=text_encoder_freeze,
                local_files_only=text_encoder_local_files_only,
            )
        elif provider == "hash":
            self.text_encoder = HashTextEmbedding(
                vocab_size=text_vocab_size,
                hidden_size=text_embed_dim,
                max_tokens=text_max_tokens,
            )
        else:
            raise ValueError(f"Unsupported text_encoder_provider: {text_encoder_provider}")

    def prepare_text_features(self, text: torch.Tensor | list[str] | None, images: torch.Tensor) -> torch.Tensor:
        if text is None:
            raise ValueError("LViTT requires text prompts or precomputed [B, tokens, 768] text features.")

        if torch.is_tensor(text):
            if text.ndim != 3:
                raise ValueError(f"Expected text tensor [B, tokens, dim], got shape {tuple(text.shape)}")
            return text.to(device=images.device, dtype=images.dtype)

        return self.text_encoder(list(text), device=images.device).to(dtype=images.dtype)
