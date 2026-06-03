"""Lazy Stable Diffusion text-to-image inference with optional LoRA adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from utils.device import resolve_device

DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _require_diffusers() -> Any:
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as exc:
        raise RuntimeError(
            "LoRA inference requires the optional advanced dependencies. "
            "Install them with: pip install -r requirements-advanced.txt"
        ) from exc
    return AutoPipelineForText2Image


def _inference_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type in ("cuda", "mps") else torch.float32


class LoraTextToImageStudio:
    """Load a Stable Diffusion base model and attach or unload LoRA weights."""

    def __init__(
        self,
        *,
        base_model: str = DEFAULT_BASE_MODEL,
        device: str = "auto",
        cpu_offload: bool = False,
    ) -> None:
        self.base_model = base_model
        self.device = resolve_device(device)
        self.cpu_offload = cpu_offload
        self.pipeline: Any | None = None
        self.adapter_name: str | None = None

    def load_base(self) -> None:
        """Download and initialize the configured base pipeline once."""

        if self.pipeline is not None:
            return
        auto_pipeline = _require_diffusers()
        self.pipeline = auto_pipeline.from_pretrained(
            self.base_model,
            torch_dtype=_inference_dtype(self.device),
        )
        if self.cpu_offload and self.device.type == "cuda":
            self.pipeline.enable_model_cpu_offload()
        else:
            self.pipeline.to(self.device)

    def load_lora(
        self,
        source: str | Path,
        *,
        weight_name: str | None = None,
        adapter_name: str = "portfolio",
    ) -> None:
        """Load local or Hub-hosted LoRA weights into the base model."""

        self.load_base()
        kwargs = {"adapter_name": adapter_name}
        if weight_name:
            kwargs["weight_name"] = weight_name
        self.pipeline.load_lora_weights(str(source), **kwargs)
        self.adapter_name = adapter_name

    def unload_lora(self) -> None:
        """Remove all active LoRA adapters while retaining the base weights."""

        if self.pipeline is not None:
            self.pipeline.unload_lora_weights()
        self.adapter_name = None

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        lora_scale: float = 1.0,
        seed: int = 7,
    ) -> Image.Image:
        """Generate one image from a text prompt and the currently loaded adapter."""

        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        self.load_base()
        if self.adapter_name is not None:
            self.pipeline.set_adapters(self.adapter_name, adapter_weights=lora_scale)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]

