"""Custom DDPM restoration trajectories for inpainting and super-resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from models.ddpm import DDPM

ProgressCallback = Callable[[int, Tensor], None]


@dataclass(frozen=True)
class RestorationOutput:
    """Final restored image and optional reverse-process trajectory."""

    image: Tensor
    trajectory: Tensor | None = None


class DiffusionRestorationPipeline:
    """Apply a trained pixel-space DDPM as a restoration prior.

    These methods do not require a separately trained restoration U-Net.
    Inpainting repeatedly locks known pixels to a forward-noised source path.
    Super-resolution repeatedly replaces generated low-frequency content with
    the low-frequency content of an upsampled reference.  The latter is an
    ILVR-style exploratory baseline rather than a dedicated SR model.
    """

    def __init__(self, model: nn.Module, diffusion: DDPM) -> None:
        self.model = model
        self.diffusion = diffusion

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4 or not images.is_floating_point():
            raise ValueError("images must be floating-point tensors with shape [B, C, H, W]")

    @staticmethod
    def _generator(
        device: torch.device,
        *,
        seed: int | None,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if generator is not None and seed is not None:
            raise ValueError("pass either seed or generator, not both")
        if seed is None:
            return generator
        return torch.Generator(device=device).manual_seed(seed)

    @staticmethod
    def _trajectory_or_none(states: list[Tensor] | None) -> Tensor | None:
        return torch.stack(states, dim=1) if states is not None else None

    @torch.no_grad()
    def inpaint(
        self,
        source: Tensor,
        regenerate_mask: Tensor,
        *,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        clip_denoised: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        return_all_timesteps: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> RestorationOutput:
        """Regenerate white-mask regions while preserving known source pixels.

        ``source`` uses ``[B, C, H, W]`` values in ``[-1, 1]``.
        ``regenerate_mask`` uses ``[B, 1|C, H, W]`` values in ``[0, 1]``:
        ``1`` means regenerate and ``0`` means retain the source observation.
        """

        self._validate_images(source)
        if regenerate_mask.ndim != 4 or regenerate_mask.shape[0] != source.shape[0]:
            raise ValueError("regenerate_mask must have shape [B, 1|C, H, W]")
        if regenerate_mask.shape[1] not in (1, source.shape[1]):
            raise ValueError("regenerate_mask must have one channel or match source channels")
        if regenerate_mask.shape[2:] != source.shape[2:]:
            raise ValueError("regenerate_mask spatial dimensions must match source")
        if torch.any(regenerate_mask < 0.0) or torch.any(regenerate_mask > 1.0):
            raise ValueError("regenerate_mask values must lie in [0, 1]")

        device = source.device
        generator = self._generator(device, seed=seed, generator=generator)
        regenerate_mask = regenerate_mask.to(device=device, dtype=source.dtype)
        if class_labels is not None:
            class_labels = class_labels.to(device)

        x_t = torch.randn(
            source.shape,
            dtype=source.dtype,
            device=device,
            generator=generator,
        )
        known_noise = torch.randn(
            source.shape,
            dtype=source.dtype,
            device=device,
            generator=generator,
        )
        states = [x_t.detach().clone()] if return_all_timesteps else None

        for step in reversed(range(self.diffusion.num_timesteps)):
            timesteps = torch.full((source.shape[0],), step, device=device, dtype=torch.long)
            x_t = self.diffusion.p_sample(
                self.model,
                x_t,
                timesteps,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
                generator=generator,
                model_kwargs=model_kwargs,
            )
            if step == 0:
                known_state = source
            else:
                previous_timesteps = torch.full(
                    (source.shape[0],),
                    step - 1,
                    device=device,
                    dtype=torch.long,
                )
                known_state = self.diffusion.q_sample(
                    source,
                    previous_timesteps,
                    noise=known_noise,
                )
            x_t = regenerate_mask * x_t + (1.0 - regenerate_mask) * known_state
            if states is not None:
                states.append(x_t.detach().clone())
            if progress_callback is not None:
                progress_callback(step, x_t)

        return RestorationOutput(image=x_t, trajectory=self._trajectory_or_none(states))

    @staticmethod
    def prepare_low_resolution_reference(source: Tensor, *, downsample_factor: int) -> Tensor:
        """Create a nearest-neighbor upsampled low-resolution observation."""

        if downsample_factor < 2:
            raise ValueError("downsample_factor must be at least 2")
        height, width = source.shape[-2:]
        if height % downsample_factor or width % downsample_factor:
            raise ValueError("source dimensions must be divisible by downsample_factor")
        low_resolution = F.avg_pool2d(source, kernel_size=downsample_factor)
        return F.interpolate(low_resolution, size=(height, width), mode="nearest")

    @staticmethod
    def _low_frequency(images: Tensor, factor: int) -> Tensor:
        low_resolution = F.avg_pool2d(images, kernel_size=factor)
        return F.interpolate(low_resolution, size=images.shape[-2:], mode="nearest")

    @torch.no_grad()
    def super_resolve(
        self,
        upsampled_reference: Tensor,
        *,
        downsample_factor: int = 4,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        clip_denoised: bool = True,
        seed: int | None = None,
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        return_all_timesteps: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> RestorationOutput:
        """Generate details while preserving an upsampled low-resolution source."""

        self._validate_images(upsampled_reference)
        if downsample_factor < 2:
            raise ValueError("downsample_factor must be at least 2")
        height, width = upsampled_reference.shape[-2:]
        if height % downsample_factor or width % downsample_factor:
            raise ValueError("reference dimensions must be divisible by downsample_factor")

        device = upsampled_reference.device
        generator = self._generator(device, seed=seed, generator=generator)
        if class_labels is not None:
            class_labels = class_labels.to(device)
        x_t = torch.randn(
            upsampled_reference.shape,
            dtype=upsampled_reference.dtype,
            device=device,
            generator=generator,
        )
        known_noise = torch.randn(
            upsampled_reference.shape,
            dtype=upsampled_reference.dtype,
            device=device,
            generator=generator,
        )
        states = [x_t.detach().clone()] if return_all_timesteps else None

        for step in reversed(range(self.diffusion.num_timesteps)):
            timesteps = torch.full(
                (upsampled_reference.shape[0],),
                step,
                device=device,
                dtype=torch.long,
            )
            x_t = self.diffusion.p_sample(
                self.model,
                x_t,
                timesteps,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
                generator=generator,
                model_kwargs=model_kwargs,
            )
            if step == 0:
                reference_state = upsampled_reference
            else:
                previous_timesteps = torch.full(
                    (upsampled_reference.shape[0],),
                    step - 1,
                    device=device,
                    dtype=torch.long,
                )
                reference_state = self.diffusion.q_sample(
                    upsampled_reference,
                    previous_timesteps,
                    noise=known_noise,
                )
            x_t = x_t + self._low_frequency(
                reference_state,
                downsample_factor,
            ) - self._low_frequency(x_t, downsample_factor)
            if states is not None:
                states.append(x_t.detach().clone())
            if progress_callback is not None:
                progress_callback(step, x_t)

        return RestorationOutput(image=x_t, trajectory=self._trajectory_or_none(states))

