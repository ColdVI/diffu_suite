"""Conversions and grids for normalized diffusion image tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torchvision.utils import make_grid


def pil_to_tensor(image: Image.Image, *, image_size: int | None = None) -> Tensor:
    """Convert a PIL image to an RGB ``[1, 3, H, W]`` tensor in ``[-1, 1]``."""

    image = ImageOps.exif_transpose(image).convert("RGB")
    if image_size is not None:
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        )
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def mask_to_tensor(mask: Image.Image, *, image_size: int) -> Tensor:
    """Convert a mask to ``[1, 1, H, W]`` where white pixels mean regenerate."""

    mask = ImageOps.exif_transpose(mask).convert("L")
    mask = ImageOps.fit(
        mask,
        (image_size, image_size),
        method=Image.Resampling.NEAREST,
    )
    array = np.asarray(mask, dtype=np.float32).copy()
    return (torch.from_numpy(array).unsqueeze(0).unsqueeze(0) >= 127.5).float()


def tensor_to_pil(image: Tensor) -> Image.Image:
    """Convert one normalized ``[C, H, W]`` or ``[1, C, H, W]`` tensor to PIL."""

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("a batched tensor must contain exactly one image")
        image = image[0]
    if image.ndim != 3 or image.shape[0] not in (1, 3):
        raise ValueError("image must have shape [1|3, H, W] or [1, 1|3, H, W]")

    array = (
        image.detach()
        .float()
        .cpu()
        .clamp(-1.0, 1.0)
        .add(1.0)
        .mul(127.5)
        .round()
        .byte()
    )
    if array.shape[0] == 1:
        return Image.fromarray(array.squeeze(0).numpy(), mode="L")
    return Image.fromarray(array.permute(1, 2, 0).numpy(), mode="RGB")


def tensor_grid_to_pil(
    images: Tensor,
    *,
    nrow: int = 8,
    padding: int = 2,
) -> Image.Image:
    """Convert normalized ``[B, C, H, W]`` images into a PIL contact sheet."""

    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    grid = make_grid(images.detach().float().cpu(), nrow=nrow, padding=padding, normalize=False)
    return tensor_to_pil(grid)


def save_tensor_grid(
    images: Tensor,
    path: str | Path,
    *,
    nrow: int = 8,
    padding: int = 2,
) -> Path:
    """Save a normalized tensor batch as a contact-sheet image."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_grid_to_pil(images, nrow=nrow, padding=padding).save(path)
    return path

