"""Lazy Stable Diffusion ControlNet inference conditioned by Canny edges."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from utils.device import resolve_device

DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_MODEL = "lllyasviel/sd-controlnet-canny"


def _require_diffusers() -> tuple[Any, Any, Any]:
    try:
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ControlNet requires the optional advanced dependencies. "
            "Install them with: pip install -r requirements-advanced.txt"
        ) from exc
    return ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler


def _inference_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


class CannyControlNetStudio:
    """Generate images that follow a user image's Canny edge structure."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        controlnet_model: str = DEFAULT_CONTROLNET_MODEL,
        device: str = "auto",
        cpu_offload: bool = False,
    ) -> None:
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.device = resolve_device(device)
        self.cpu_offload = cpu_offload
        self.pipeline: Any | None = None

    def load(self) -> None:
        """Download and initialize ControlNet and its Stable Diffusion base once."""

        if self.pipeline is not None:
            return
        controlnet_class, pipeline_class, scheduler_class = _require_diffusers()
        dtype = _inference_dtype(self.device)
        controlnet = controlnet_class.from_pretrained(
            self.controlnet_model,
            torch_dtype=dtype,
        )
        self.pipeline = pipeline_class.from_pretrained(
            self.base_model,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
        self.pipeline.scheduler = scheduler_class.from_config(self.pipeline.scheduler.config)
        if self.cpu_offload and self.device.type == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    @staticmethod
    def canny_image(
        image: Image.Image,
        *,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """Extract a three-channel Canny edge image suitable for ControlNet."""

        if not 0 <= low_threshold <= high_threshold <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low <= high <= 255")
        image = ImageOps.exif_transpose(image).convert("RGB")
        edges = cv2.Canny(np.asarray(image), low_threshold, high_threshold)
        edges = np.repeat(edges[:, :, None], 3, axis=2)
        return Image.fromarray(edges)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        source_image: Image.Image,
        *,
        negative_prompt: str | None = None,
        low_threshold: int = 100,
        high_threshold: int = 200,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: int = 7,
    ) -> tuple[Image.Image, Image.Image]:
        """Return the generated image and the edge condition used to produce it."""

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        self.load()
        edges = self.canny_image(
            source_image,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=edges,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )
        return result.images[0], edges

