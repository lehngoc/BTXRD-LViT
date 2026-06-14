from __future__ import annotations

import math

import torch
from torch import nn


def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class ConvBatchNorm(nn.Module):
    """Original LViT CNN unit: Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


def _make_nconv(in_channels: int, out_channels: int, nb_conv: int) -> nn.Sequential:
    layers: list[nn.Module] = [ConvBatchNorm(in_channels, out_channels)]
    layers.extend(ConvBatchNorm(out_channels, out_channels) for _ in range(nb_conv - 1))
    return nn.Sequential(*layers)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nb_conv: int = 2) -> None:
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nconvs = _make_nconv(in_channels, out_channels, nb_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nconvs(self.maxpool(x))


class PixLevelModule(nn.Module):
    """Pixel-Level Attention Module (PLAM) from the official LViT code."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        middle_layer_size_ratio = 2
        self.conv_avg = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu_avg = nn.ReLU(inplace=True)
        self.conv_max = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.relu_max = nn.ReLU(inplace=True)
        self.bottleneck = nn.Sequential(
            nn.Linear(3, 3 * middle_layer_size_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(3 * middle_layer_size_ratio, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_avg = self.relu_avg(self.conv_avg(x)).mean(dim=1, keepdim=True)
        x_max = self.relu_max(self.conv_max(x)).max(dim=1, keepdim=True).values
        x_out = x_avg + x_max
        attention = torch.cat((x_avg, x_max, x_out), dim=1)
        attention = attention.transpose(1, 3)
        attention = self.bottleneck(attention)
        attention = attention.transpose(1, 3)
        return attention * x


class UpBlockAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, nb_conv: int = 2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.pix_module = PixLevelModule(in_channels // 2)
        self.nconvs = _make_nconv(in_channels, out_channels, nb_conv)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        skip_x_att = self.pix_module(skip_x)
        return self.nconvs(torch.cat([skip_x_att, up], dim=1))


class Reconstruct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, scale_factor: tuple[int, int]) -> None:
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_patch, hidden = x.size()
        height = width = int(math.sqrt(n_patch))
        if height * width != n_patch:
            raise ValueError(f"Token count must be square for reconstruction, got {n_patch}")

        x = x.permute(0, 2, 1).contiguous().view(batch_size, hidden, height, width)
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.activation(self.norm(self.conv(x)))


class Embeddings(nn.Module):
    def __init__(self, patch_size: int, img_size: int, in_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) * (img_size // patch_size)
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        return self.dropout(x + self.position_embeddings)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act_layer = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.act_layer(self.fc1(x)))
        x = self.dropout(self.act_layer(self.fc2(x)))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, n_tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_dim=dim, hidden_dim=int(dim * mlp_ratio), out_dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvTransBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class VisionTransformer(nn.Module):
    """Official LViT-style transformer branch block."""

    def __init__(
        self,
        img_size: int,
        channel_num: int,
        patch_size: int,
        embed_dim: int,
        depth: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        text_tokens: int = 10,
    ) -> None:
        super().__init__()
        self.embeddings = Embeddings(
            patch_size=patch_size,
            img_size=img_size,
            in_channels=channel_num,
            dropout=0.1,
        )
        self.dim = embed_dim
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.encoder_blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.ctbn = ConvTransBN(in_channels=embed_dim, out_channels=embed_dim // 2)
        self.ctbn2 = ConvTransBN(in_channels=embed_dim * 2, out_channels=embed_dim)
        self.text_to_patches = ConvTransBN(in_channels=text_tokens, out_channels=(img_size // patch_size) ** 2)

    def forward(
        self,
        x: torch.Tensor,
        skip_x: torch.Tensor,
        text: torch.Tensor,
        reconstruct: bool = False,
    ) -> torch.Tensor:
        if not reconstruct:
            x = self.embeddings(x)
            if self.dim == 64:
                x = x + self.text_to_patches(text)
            x = self.encoder_blocks(x)
        else:
            x = self.encoder_blocks(x)

        if (self.dim == 64 and not reconstruct) or (self.dim == 512 and reconstruct):
            return x

        if not reconstruct:
            x = self.ctbn(x.transpose(1, 2)).transpose(1, 2)
            return torch.cat([x, skip_x], dim=2)

        skip_x = self.ctbn2(skip_x.transpose(1, 2)).transpose(1, 2)
        return x + skip_x


class LViTTW(nn.Module):
    """No-text LViT-TW using the official LViT Double-U backbone without text signal."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 64,
        transformer_depth: int = 1,
        transformer_heads: int = 8,
        transformer_dropout: float = 0.0,
        image_size: int = 224,
        text_tokens: int = 10,
        text_dim: int = 768,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.image_size = image_size
        self.text_tokens = text_tokens
        self.text_dim = text_dim

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.inc = ConvBatchNorm(in_channels, c1)
        self.down_vit = VisionTransformer(image_size, c1, 16, c1, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.down_vit1 = VisionTransformer(image_size // 2, c2, 8, c2, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.down_vit2 = VisionTransformer(image_size // 4, c3, 4, c3, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.down_vit3 = VisionTransformer(image_size // 8, c4, 2, c4, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.up_vit = VisionTransformer(image_size, c1, 16, c1, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.up_vit1 = VisionTransformer(image_size // 2, c2, 8, c2, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.up_vit2 = VisionTransformer(image_size // 4, c3, 4, c3, transformer_depth, transformer_heads, drop_rate=transformer_dropout)
        self.up_vit3 = VisionTransformer(image_size // 8, c4, 2, c4, transformer_depth, transformer_heads, drop_rate=transformer_dropout)

        self.down1 = DownBlock(c1, c2)
        self.down2 = DownBlock(c2, c3)
        self.down3 = DownBlock(c3, c4)
        self.down4 = DownBlock(c4, c4)

        self.up4 = UpBlockAttention(c4 * 2, c3)
        self.up3 = UpBlockAttention(c3 * 2, c2)
        self.up2 = UpBlockAttention(c2 * 2, c1)
        self.up1 = UpBlockAttention(c1 * 2, c1)
        self.outc = nn.Conv2d(c1, out_channels, kernel_size=1, stride=1)

        self.reconstruct1 = Reconstruct(c1, c1, kernel_size=1, scale_factor=(16, 16))
        self.reconstruct2 = Reconstruct(c2, c2, kernel_size=1, scale_factor=(8, 8))
        self.reconstruct3 = Reconstruct(c3, c3, kernel_size=1, scale_factor=(4, 4))
        self.reconstruct4 = Reconstruct(c4, c4, kernel_size=1, scale_factor=(2, 2))

        self.text_module4 = nn.Conv1d(in_channels=text_dim, out_channels=c4, kernel_size=3, padding=1)
        self.text_module3 = nn.Conv1d(in_channels=c4, out_channels=c3, kernel_size=3, padding=1)
        self.text_module2 = nn.Conv1d(in_channels=c3, out_channels=c2, kernel_size=3, padding=1)
        self.text_module1 = nn.Conv1d(in_channels=c2, out_channels=c1, kernel_size=3, padding=1)

    def prepare_text_features(self, text: torch.Tensor | list[str] | None, images: torch.Tensor) -> torch.Tensor:
        del text
        return images.new_zeros((images.shape[0], self.text_tokens, self.text_dim))

    def _project_text(self, text_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text4 = self.text_module4(text_features.transpose(1, 2)).transpose(1, 2)
        text3 = self.text_module3(text4.transpose(1, 2)).transpose(1, 2)
        text2 = self.text_module2(text3.transpose(1, 2)).transpose(1, 2)
        text1 = self.text_module1(text2.transpose(1, 2)).transpose(1, 2)
        return text1, text2, text3, text4

    def forward(self, images: torch.Tensor, text: torch.Tensor | list[str] | None = None) -> torch.Tensor:
        images = images.float()
        text1, text2, text3, text4 = self._project_text(self.prepare_text_features(text, images))

        x1 = self.inc(images)
        y1 = self.down_vit(x1, x1, text1)

        x2 = self.down1(x1)
        y2 = self.down_vit1(x2, y1, text2)

        x3 = self.down2(x2)
        y3 = self.down_vit2(x3, y2, text3)

        x4 = self.down3(x3)
        y4 = self.down_vit3(x4, y3, text4)

        x5 = self.down4(x4)
        y4 = self.up_vit3(y4, y4, text4, reconstruct=True)
        y3 = self.up_vit2(y3, y4, text3, reconstruct=True)
        y2 = self.up_vit1(y2, y3, text2, reconstruct=True)
        y1 = self.up_vit(y1, y2, text1, reconstruct=True)

        x1 = self.reconstruct1(y1) + x1
        x2 = self.reconstruct2(y2) + x2
        x3 = self.reconstruct3(y3) + x3
        x4 = self.reconstruct4(y4) + x4

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        return self.outc(x)
