"""Exponential moving average weights for diffusion training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ExponentialMovingAverage:
    """Track a smoothed copy of trainable model parameters."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie in (0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend current trainable parameters into the shadow copy."""

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the smoothed parameters into ``model`` in place."""

        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.decay = float(state["decay"])
        shadow = state["shadow"]
        if not isinstance(shadow, Mapping):
            raise TypeError("EMA shadow state must be a mapping")
        self.shadow = {str(name): tensor.clone() for name, tensor in shadow.items()}
