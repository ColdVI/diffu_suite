#!/usr/bin/env python3
"""Launch the two-tab DiffuSuite Gradio dashboard."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from advanced.controlnet import CannyControlNetStudio
from advanced.lora_inference import LoraTextToImageStudio
from pipelines.restoration import DiffusionRestorationPipeline
from training.dataset import CIFAR10_CLASSES
from utils.checkpoints import LoadedCustomCheckpoint, load_custom_checkpoint
from utils.device import resolve_device
from utils.diagnostics import (
    collect_schedule_diagnostics,
    plot_diagnostics,
    snapshot_indices,
)
from utils.images import mask_to_tensor, pil_to_tensor, tensor_to_pil

CUSTOM_CACHE: dict[tuple[str, str], LoadedCustomCheckpoint] = {}
LORA_CACHE: dict[tuple[str, str], LoraTextToImageStudio] = {}
CONTROLNET_CACHE: dict[tuple[str, str], CannyControlNetStudio] = {}


def _error_message(exc: Exception) -> None:
    raise gr.Error(str(exc)) from exc


def _custom_checkpoint(path: str, device_name: str) -> LoadedCustomCheckpoint:
    path = str(Path(path).expanduser())
    key = (path, device_name)
    if key not in CUSTOM_CACHE:
        CUSTOM_CACHE[key] = load_custom_checkpoint(
            path,
            device=resolve_device(device_name),
        )
    return CUSTOM_CACHE[key]


def custom_generate(
    checkpoint_path: str,
    class_name: str,
    guidance_scale: float,
    seed: int,
    device_name: str,
) -> Image.Image:
    """Generate one CIFAR-10 image with the custom checkpoint."""

    try:
        loaded = _custom_checkpoint(checkpoint_path, device_name)
        device = next(loaded.model.parameters()).device
        class_labels = torch.tensor([CIFAR10_CLASSES.index(class_name)], device=device)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        image = loaded.diffusion.sample_loop(
            loaded.model,
            (1, 3, 32, 32),
            class_labels=class_labels,
            guidance_scale=float(guidance_scale),
            clip_denoised=True,
            device=device,
            generator=generator,
        )
        return tensor_to_pil(image).resize((256, 256), Image.Resampling.NEAREST)
    except Exception as exc:
        _error_message(exc)


def custom_inpaint(
    checkpoint_path: str,
    source_image: Image.Image,
    regenerate_mask: Image.Image,
    class_name: str,
    guidance_scale: float,
    seed: int,
    device_name: str,
) -> Image.Image:
    """Restore the white-mask area of an uploaded image."""

    try:
        if source_image is None or regenerate_mask is None:
            raise ValueError("upload both a source image and a white regenerate-mask image")
        loaded = _custom_checkpoint(checkpoint_path, device_name)
        device = next(loaded.model.parameters()).device
        source = pil_to_tensor(source_image, image_size=32).to(device)
        mask = mask_to_tensor(regenerate_mask, image_size=32).to(device)
        class_labels = torch.tensor([CIFAR10_CLASSES.index(class_name)], device=device)
        output = DiffusionRestorationPipeline(loaded.model, loaded.diffusion).inpaint(
            source,
            mask,
            class_labels=class_labels,
            guidance_scale=float(guidance_scale),
            seed=int(seed),
        )
        return tensor_to_pil(output.image).resize((256, 256), Image.Resampling.NEAREST)
    except Exception as exc:
        _error_message(exc)


def custom_super_resolution(
    checkpoint_path: str,
    source_image: Image.Image,
    class_name: str,
    guidance_scale: float,
    downsample_factor: int,
    seed: int,
    device_name: str,
) -> tuple[Image.Image, Image.Image]:
    """Restore details from a deliberately degraded low-resolution observation."""

    try:
        if source_image is None:
            raise ValueError("upload a source image for super-resolution")
        loaded = _custom_checkpoint(checkpoint_path, device_name)
        device = next(loaded.model.parameters()).device
        source = pil_to_tensor(source_image, image_size=32).to(device)
        class_labels = torch.tensor([CIFAR10_CLASSES.index(class_name)], device=device)
        pipeline = DiffusionRestorationPipeline(loaded.model, loaded.diffusion)
        reference = pipeline.prepare_low_resolution_reference(
            source,
            downsample_factor=int(downsample_factor),
        )
        output = pipeline.super_resolve(
            reference,
            downsample_factor=int(downsample_factor),
            class_labels=class_labels,
            guidance_scale=float(guidance_scale),
            seed=int(seed),
        )
        def resize(image: torch.Tensor) -> Image.Image:
            return tensor_to_pil(image).resize(
                (256, 256),
                Image.Resampling.NEAREST,
            )

        return resize(reference), resize(output.image)
    except Exception as exc:
        _error_message(exc)


def schedule_diagnostic(source_image: Image.Image, timesteps: int, seed: int) -> Image.Image:
    """Render a matched-noise linear-versus-cosine degradation report."""

    try:
        if source_image is None:
            raise ValueError("upload an image for schedule diagnostics")
        x_start = pil_to_tensor(source_image, image_size=128)
        generator = torch.Generator().manual_seed(int(seed))
        noise = torch.randn(x_start.shape, generator=generator)
        indices = snapshot_indices(int(timesteps), 8)
        linear = collect_schedule_diagnostics(
            "linear",
            num_timesteps=int(timesteps),
            x_start=x_start,
            noise=noise,
            indices=indices,
        )
        cosine = collect_schedule_diagnostics(
            "cosine",
            num_timesteps=int(timesteps),
            x_start=x_start,
            noise=noise,
            indices=indices,
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as output:
            plot_diagnostics(
                x_start,
                indices,
                linear,
                cosine,
                source_name="uploaded image",
                output_path=Path(output.name),
            )
            with Image.open(output.name) as image:
                return image.copy()
    except Exception as exc:
        _error_message(exc)


def lora_generate(
    base_model: str,
    lora_source: str,
    prompt: str,
    negative_prompt: str,
    lora_scale: float,
    guidance_scale: float,
    steps: int,
    seed: int,
    device_name: str,
) -> Image.Image:
    """Run optional Stable Diffusion LoRA text-to-image inference."""

    try:
        key = (base_model.strip(), device_name)
        studio = LORA_CACHE.setdefault(
            key,
            LoraTextToImageStudio(base_model=key[0], device=device_name),
        )
        if lora_source.strip():
            studio.unload_lora()
            studio.load_lora(lora_source.strip())
        return studio.generate(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            lora_scale=float(lora_scale),
            seed=int(seed),
        )
    except Exception as exc:
        _error_message(exc)


def controlnet_generate(
    base_model: str,
    prompt: str,
    source_image: Image.Image,
    low_threshold: int,
    high_threshold: int,
    guidance_scale: float,
    control_scale: float,
    steps: int,
    seed: int,
    device_name: str,
) -> tuple[Image.Image, Image.Image]:
    """Run optional Stable Diffusion Canny-ControlNet inference."""

    try:
        if source_image is None:
            raise ValueError("upload a source image for ControlNet")
        key = (base_model.strip(), device_name)
        studio = CONTROLNET_CACHE.setdefault(
            key,
            CannyControlNetStudio(base_model=key[0], device=device_name),
        )
        generated, edges = studio.generate(
            prompt,
            source_image,
            low_threshold=int(low_threshold),
            high_threshold=int(high_threshold),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
            controlnet_conditioning_scale=float(control_scale),
            seed=int(seed),
        )
        return edges, generated
    except Exception as exc:
        _error_message(exc)


def create_demo() -> gr.Blocks:
    """Construct the complete dashboard without launching a server."""

    with gr.Blocks(title="DiffuSuite") as demo:
        gr.Markdown(
            "# DiffuSuite\n"
            "From-scratch DDPM mathematics and optional production-scale diffusion tools."
        )
        with gr.Tab("Custom Mathematical Core"):
            gr.Markdown(
                "Checkpoint generation uses the schedule stored during training. "
                "The diagnostics panel compares schedules independently with matched noise."
            )
            checkpoint = gr.Textbox(
                label="Custom checkpoint",
                value="runs/cifar10_cosine/checkpoints/latest.pt",
            )
            with gr.Row():
                class_name = gr.Dropdown(
                    choices=list(CIFAR10_CLASSES),
                    value="airplane",
                    label="CIFAR-10 class",
                )
                cfg = gr.Slider(0.0, 4.0, value=1.0, step=0.1, label="CFG scale w")
                seed = gr.Number(value=7, precision=0, label="Seed")
                custom_device = gr.Dropdown(
                    choices=("auto", "cuda", "mps", "cpu"),
                    value="auto",
                    label="Device",
                )
            generate_button = gr.Button("Generate class sample", variant="primary")
            generated_image = gr.Image(label="Generated CIFAR-10 image")
            generate_button.click(
                custom_generate,
                inputs=(checkpoint, class_name, cfg, seed, custom_device),
                outputs=generated_image,
            )

            gr.Markdown("## Inpainting")
            with gr.Row():
                inpaint_source = gr.Image(type="pil", label="Source image")
                inpaint_mask = gr.Image(type="pil", label="White area to regenerate")
                inpaint_output = gr.Image(label="Restored image")
            inpaint_button = gr.Button("Restore masked region")
            inpaint_button.click(
                custom_inpaint,
                inputs=(
                    checkpoint,
                    inpaint_source,
                    inpaint_mask,
                    class_name,
                    cfg,
                    seed,
                    custom_device,
                ),
                outputs=inpaint_output,
            )

            gr.Markdown("## Exploratory Super-Resolution")
            with gr.Row():
                super_res_source = gr.Image(type="pil", label="Reference image")
                super_res_observation = gr.Image(label="Low-resolution observation")
                super_res_output = gr.Image(label="Restored detail sample")
            super_res_factor = gr.Dropdown(
                choices=(2, 4, 8),
                value=4,
                label="Downsample factor",
            )
            super_res_button = gr.Button("Run super-resolution prior")
            super_res_button.click(
                custom_super_resolution,
                inputs=(
                    checkpoint,
                    super_res_source,
                    class_name,
                    cfg,
                    super_res_factor,
                    seed,
                    custom_device,
                ),
                outputs=(super_res_observation, super_res_output),
            )

            gr.Markdown("## Linear vs. Cosine Forward Degradation")
            with gr.Row():
                diagnostic_source = gr.Image(type="pil", label="Diagnostic source")
                with gr.Column():
                    diagnostic_timesteps = gr.Slider(
                        20,
                        1_000,
                        value=1_000,
                        step=10,
                        label="Forward-process timesteps T",
                    )
                    diagnostic_button = gr.Button("Compare schedules")
                diagnostic_output = gr.Image(label="Schedule report")
            diagnostic_button.click(
                schedule_diagnostic,
                inputs=(diagnostic_source, diagnostic_timesteps, seed),
                outputs=diagnostic_output,
            )

        with gr.Tab("Production-Grade Studio"):
            gr.Markdown(
                "These tools lazily load Hugging Face Diffusers models. "
                "Install `requirements-advanced.txt` and prefer a Colab GPU."
            )
            advanced_device = gr.Dropdown(
                choices=("auto", "cuda", "mps", "cpu"),
                value="auto",
                label="Device",
            )
            base_model = gr.Textbox(
                value="stable-diffusion-v1-5/stable-diffusion-v1-5",
                label="Stable Diffusion base model",
            )
            with gr.Tab("LoRA Text-to-Image"):
                lora_source = gr.Textbox(
                    label="Optional local or Hub LoRA source",
                    placeholder="runs/lora or a Hugging Face repository id",
                )
                lora_prompt = gr.Textbox(label="Prompt", value="a ceramic teapot on a table")
                lora_negative = gr.Textbox(label="Negative prompt")
                with gr.Row():
                    lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="LoRA scale")
                    lora_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
                    lora_steps = gr.Slider(10, 60, value=30, step=1, label="Steps")
                    lora_seed = gr.Number(value=7, precision=0, label="Seed")
                lora_button = gr.Button("Generate with LoRA", variant="primary")
                lora_output = gr.Image(label="Generated image")
                lora_button.click(
                    lora_generate,
                    inputs=(
                        base_model,
                        lora_source,
                        lora_prompt,
                        lora_negative,
                        lora_scale,
                        lora_cfg,
                        lora_steps,
                        lora_seed,
                        advanced_device,
                    ),
                    outputs=lora_output,
                )
            with gr.Tab("Canny ControlNet"):
                with gr.Row():
                    control_source = gr.Image(type="pil", label="Layout source")
                    control_edges = gr.Image(label="Canny condition")
                    control_output = gr.Image(label="Generated image")
                control_prompt = gr.Textbox(
                    label="Prompt",
                    value="a cinematic photograph of a modern building",
                )
                with gr.Row():
                    canny_low = gr.Slider(0, 255, value=100, step=1, label="Canny low")
                    canny_high = gr.Slider(0, 255, value=200, step=1, label="Canny high")
                    control_cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
                    control_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Control")
                    control_steps = gr.Slider(10, 60, value=30, step=1, label="Steps")
                    control_seed = gr.Number(value=7, precision=0, label="Seed")
                control_button = gr.Button("Generate with ControlNet", variant="primary")
                control_button.click(
                    controlnet_generate,
                    inputs=(
                        base_model,
                        control_prompt,
                        control_source,
                        canny_low,
                        canny_high,
                        control_cfg,
                        control_scale,
                        control_steps,
                        control_seed,
                        advanced_device,
                    ),
                    outputs=(control_edges, control_output),
                )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_demo().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
