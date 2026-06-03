"""Checkpoint serialization shared by training, inference, and the web app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.ddpm import DDPM
from models.unet import ConditionalUNet
from utils.ema import ExponentialMovingAverage


@dataclass(frozen=True)
class LoadedCustomCheckpoint:
    """A restored custom model, diffusion process, and training metadata."""

    model: ConditionalUNet
    diffusion: DDPM
    metadata: dict[str, Any]
    raw: dict[str, Any]


def save_custom_checkpoint(
    path: str | Path,
    *,
    model: ConditionalUNet,
    diffusion: DDPM,
    optimizer: torch.optim.Optimizer | None = None,
    ema: ExponentialMovingAverage | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist the complete reproducible state of a custom DDPM experiment."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "format_version": 1,
        "model_config": model.config_dict(),
        "diffusion_config": diffusion.config_dict(),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if ema is not None:
        checkpoint["ema_state"] = ema.state_dict()
    torch.save(checkpoint, path)
    return path


def load_custom_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    use_ema: bool = True,
) -> LoadedCustomCheckpoint:
    """Restore a custom model and schedule from a trusted local checkpoint."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ConditionalUNet(**checkpoint["model_config"]).to(device)
    diffusion = DDPM(**checkpoint["diffusion_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])

    if use_ema and "ema_state" in checkpoint:
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(checkpoint["ema_state"])
        ema.copy_to(model)
    model.eval()
    return LoadedCustomCheckpoint(
        model=model,
        diffusion=diffusion,
        metadata=dict(checkpoint.get("metadata", {})),
        raw=checkpoint,
    )


def restore_optimizer(
    optimizer: torch.optim.Optimizer,
    checkpoint: dict[str, Any],
) -> None:
    """Restore optimizer state when resuming a training run."""

    if "optimizer_state" not in checkpoint:
        raise ValueError("checkpoint does not contain optimizer state")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
