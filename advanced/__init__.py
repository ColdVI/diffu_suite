"""Optional Hugging Face Diffusers integrations for production-scale demos."""

from .controlnet import CannyControlNetStudio
from .lora_inference import LoraTextToImageStudio

__all__ = ["CannyControlNetStudio", "LoraTextToImageStudio"]

