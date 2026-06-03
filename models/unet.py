"""A compact class-conditioned U-Net for pixel-space DDPM experiments.

The architecture is intentionally explicit: every encoder and decoder stage is
implemented with ordinary PyTorch modules, and every residual block receives a
combined timestep and class-label context vector.  A learned null class token
supports classifier-free guidance (CFG).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _group_count(num_channels: int, preferred_groups: int = 32) -> int:
    """Return the largest valid GroupNorm group count up to ``preferred_groups``."""

    for num_groups in reversed(range(1, min(preferred_groups, num_channels) + 1)):
        if num_channels % num_groups == 0:
            return num_groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Map integer diffusion steps ``[B]`` to sinusoidal vectors ``[B, D]``."""

    def __init__(self, embedding_dim: int, *, max_period: int = 10_000) -> None:
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if max_period < 1:
            raise ValueError("max_period must be positive")

        self.embedding_dim = embedding_dim
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(half_dim) / max(half_dim - 1, 1)
        self.register_buffer("frequencies", torch.exp(exponent), persistent=False)

    def forward(self, timesteps: Tensor) -> Tensor:
        """Return a deterministic embedding with shape ``[B, embedding_dim]``."""

        if timesteps.ndim != 1:
            raise ValueError("timesteps must have shape [B]")

        angles = timesteps.float().unsqueeze(1) * self.frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if embedding.shape[1] < self.embedding_dim:
            embedding = F.pad(embedding, (0, self.embedding_dim - embedding.shape[1]))
        return embedding


class ClassConditioningEmbedding(nn.Module):
    """Map optional class labels ``[B]`` to learned context vectors ``[B, D]``.

    Label values ``0`` through ``num_classes - 1`` select class tokens.
    ``None`` selects the learned null token for the full batch.  Individual
    label values of ``-1`` select that token during CFG training dropout.
    """

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")

        self.num_classes = num_classes
        self.null_class_index = num_classes
        self.embedding = nn.Embedding(num_classes + 1, embedding_dim)

    def forward(
        self,
        class_labels: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Return class or null embeddings with shape ``[B, embedding_dim]``."""

        if class_labels is None:
            embedding_indices = torch.full(
                (batch_size,),
                self.null_class_index,
                dtype=torch.long,
                device=device,
            )
        else:
            if class_labels.ndim != 1 or class_labels.shape[0] != batch_size:
                raise ValueError(f"class_labels must have shape [{batch_size}]")
            if class_labels.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("class_labels must contain integer indices")
            if torch.any(class_labels < -1) or torch.any(class_labels >= self.num_classes):
                raise ValueError(
                    f"class_labels must lie in [-1, {self.num_classes - 1}]"
                )

            class_labels = class_labels.to(device=device, dtype=torch.long)
            embedding_indices = torch.where(
                class_labels == -1,
                self.null_class_index,
                class_labels,
            )
        return self.embedding(embedding_indices)


class ConditionedResidualBlock(nn.Module):
    """Residual convolutional block with additive context injection.

    ``x`` has shape ``[B, C_in, H, W]`` and ``conditioning`` has shape
    ``[B, D]``.  The context projection broadcasts to ``[B, C_out, 1, 1]``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1 or conditioning_dim < 1:
            raise ValueError("channel counts and conditioning_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.input_norm = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conditioning_projection = nn.Linear(conditioning_dim, out_channels)
        self.output_norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.output_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """Return a conditioned feature map with shape ``[B, C_out, H, W]``."""

        if conditioning.ndim != 2 or conditioning.shape[0] != x.shape[0]:
            raise ValueError("conditioning must have shape [B, conditioning_dim]")

        hidden = self.input_conv(F.silu(self.input_norm(x)))
        context = self.conditioning_projection(conditioning).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + context
        hidden = self.output_conv(self.dropout(F.silu(self.output_norm(hidden))))
        return self.skip(x) + hidden


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over flattened image positions."""

    def __init__(self, channels: int, *, num_heads: int = 4) -> None:
        super().__init__()
        if channels < 1 or num_heads < 1 or channels % num_heads != 0:
            raise ValueError("channels must be positive and divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        """Return an attended feature map with the same shape as ``x``."""

        batch_size, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(batch_size, 3, self.num_heads, self.head_dim, height * width)
        query, key, value = qkv.unbind(dim=1)
        attended = F.scaled_dot_product_attention(
            query.transpose(-1, -2),
            key.transpose(-1, -2),
            value.transpose(-1, -2),
        )
        attended = attended.transpose(-1, -2).reshape(batch_size, channels, height, width)
        return x + self.projection(attended)


class Downsample(nn.Module):
    """Halve spatial resolution with a learned strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    """Double spatial resolution and refine with a learned convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class _DownStage(nn.Module):
    def __init__(
        self,
        blocks: Sequence[ConditionedResidualBlock],
        attentions: Sequence[nn.Module],
        downsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)
        self.downsample = downsample

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        skip_features: list[Tensor],
    ) -> Tensor:
        for block, attention in zip(self.blocks, self.attentions):
            x = attention(block(x, conditioning))
            skip_features.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
            skip_features.append(x)
        return x


class _UpStage(nn.Module):
    def __init__(
        self,
        blocks: Sequence[ConditionedResidualBlock],
        attentions: Sequence[nn.Module],
        upsample: nn.Module | None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.attentions = nn.ModuleList(attentions)
        self.upsample = upsample

    def forward(
        self,
        x: Tensor,
        conditioning: Tensor,
        skip_features: list[Tensor],
    ) -> Tensor:
        for block, attention in zip(self.blocks, self.attentions):
            x = torch.cat((x, skip_features.pop()), dim=1)
            x = attention(block(x, conditioning))
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ConditionalUNet(nn.Module):
    """Class-conditioned epsilon-predicting U-Net for ``[B, C, H, W]`` images.

    CIFAR-10 defaults use feature resolutions ``32 -> 16 -> 8``.  Attention is
    enabled only at the deepest level by default, keeping desktop and Colab
    memory use modest.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 64,
        conditioning_dim: int = 256,
        channel_multipliers: Sequence[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_levels: Sequence[int] = (2,),
        num_attention_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels < 1 or out_channels < 1:
            raise ValueError("in_channels and out_channels must be positive")
        if base_channels < 2 or conditioning_dim < 1:
            raise ValueError("base_channels must be >= 2 and conditioning_dim positive")
        if not channel_multipliers or any(multiplier < 1 for multiplier in channel_multipliers):
            raise ValueError("channel_multipliers must contain positive integers")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        if any(level < 0 or level >= len(channel_multipliers) for level in attention_levels):
            raise ValueError("attention_levels contains an invalid stage index")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.conditioning_dim = conditioning_dim
        self.channel_multipliers = tuple(channel_multipliers)
        self.num_res_blocks = num_res_blocks
        self.attention_levels = tuple(attention_levels)
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.spatial_divisor = 2 ** (len(channel_multipliers) - 1)

        self.time_embedding = SinusoidalTimeEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, conditioning_dim),
            nn.SiLU(),
            nn.Linear(conditioning_dim, conditioning_dim),
        )
        self.class_embedding = ClassConditioningEmbedding(num_classes, conditioning_dim)
        self.input_projection = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        current_channels = base_channels
        skip_channels = [current_channels]
        down_stages = []
        for level, multiplier in enumerate(channel_multipliers):
            output_channels = base_channels * multiplier
            blocks = []
            attentions = []
            for _ in range(num_res_blocks):
                blocks.append(
                    ConditionedResidualBlock(
                        current_channels,
                        output_channels,
                        conditioning_dim,
                        dropout=dropout,
                    )
                )
                attentions.append(
                    self._attention_or_identity(
                        output_channels,
                        enabled=level in attention_levels,
                        num_heads=num_attention_heads,
                    )
                )
                current_channels = output_channels
                skip_channels.append(current_channels)
            downsample = Downsample(current_channels) if level < len(channel_multipliers) - 1 else None
            if downsample is not None:
                skip_channels.append(current_channels)
            down_stages.append(_DownStage(blocks, attentions, downsample))
        self.down_stages = nn.ModuleList(down_stages)

        self.middle_block1 = ConditionedResidualBlock(
            current_channels,
            current_channels,
            conditioning_dim,
            dropout=dropout,
        )
        self.middle_attention = SpatialSelfAttention(
            current_channels,
            num_heads=num_attention_heads,
        )
        self.middle_block2 = ConditionedResidualBlock(
            current_channels,
            current_channels,
            conditioning_dim,
            dropout=dropout,
        )

        up_stages = []
        for level, multiplier in reversed(tuple(enumerate(channel_multipliers))):
            output_channels = base_channels * multiplier
            blocks = []
            attentions = []
            for _ in range(num_res_blocks + 1):
                blocks.append(
                    ConditionedResidualBlock(
                        current_channels + skip_channels.pop(),
                        output_channels,
                        conditioning_dim,
                        dropout=dropout,
                    )
                )
                attentions.append(
                    self._attention_or_identity(
                        output_channels,
                        enabled=level in attention_levels,
                        num_heads=num_attention_heads,
                    )
                )
                current_channels = output_channels
            upsample = Upsample(current_channels) if level > 0 else None
            up_stages.append(_UpStage(blocks, attentions, upsample))
        if skip_channels:
            raise RuntimeError("internal U-Net skip-channel accounting error")
        self.up_stages = nn.ModuleList(up_stages)

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_projection = nn.Conv2d(
            current_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    @staticmethod
    def _attention_or_identity(
        channels: int,
        *,
        enabled: bool,
        num_heads: int,
    ) -> nn.Module:
        return SpatialSelfAttention(channels, num_heads=num_heads) if enabled else nn.Identity()

    def config_dict(self) -> dict[str, Any]:
        """Return constructor values suitable for a checkpoint."""

        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "num_classes": self.num_classes,
            "base_channels": self.base_channels,
            "conditioning_dim": self.conditioning_dim,
            "channel_multipliers": list(self.channel_multipliers),
            "num_res_blocks": self.num_res_blocks,
            "attention_levels": list(self.attention_levels),
            "num_attention_heads": self.num_attention_heads,
            "dropout": self.dropout,
        }

    def build_conditioning(
        self,
        timesteps: Tensor,
        class_labels: Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        """Combine timestep and class vectors into shape ``[B, conditioning_dim]``."""

        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(f"timesteps must have shape [{batch_size}]")
        time_context = self.time_mlp(self.time_embedding(timesteps.to(device=device)))
        class_context = self.class_embedding(
            class_labels,
            batch_size=batch_size,
            device=device,
        )
        return time_context + class_context

    def forward_features(self, x: Tensor, conditioning: Tensor) -> Tensor:
        """Run the encoder, bottleneck, and decoder feature path."""

        skip_features = [x]
        for stage in self.down_stages:
            x = stage(x, conditioning, skip_features)

        x = self.middle_block1(x, conditioning)
        x = self.middle_attention(x)
        x = self.middle_block2(x, conditioning)

        for stage in self.up_stages:
            x = stage(x, conditioning, skip_features)
        if skip_features:
            raise RuntimeError("internal U-Net skip-feature accounting error")
        return x

    def forward(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        class_labels: Tensor | None = None,
    ) -> Tensor:
        """Predict epsilon with the same spatial shape as ``x_t``."""

        if x_t.ndim != 4:
            raise ValueError("x_t must have shape [B, C, H, W]")
        if x_t.shape[1] != self.in_channels:
            raise ValueError(f"x_t must have {self.in_channels} channels")
        if x_t.shape[2] % self.spatial_divisor or x_t.shape[3] % self.spatial_divisor:
            raise ValueError(
                f"height and width must be divisible by {self.spatial_divisor}"
            )

        conditioning = self.build_conditioning(
            timesteps,
            class_labels,
            batch_size=x_t.shape[0],
            device=x_t.device,
        )
        features = self.forward_features(self.input_projection(x_t), conditioning)
        return self.output_projection(F.silu(self.output_norm(features)))

