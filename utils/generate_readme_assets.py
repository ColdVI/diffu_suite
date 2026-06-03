#!/usr/bin/env python3
"""Generate reproducible CIFAR-10 figures and forward-process videos for README."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ddpm import DDPM
from training.dataset import CIFAR10_CLASSES
from utils.diagnostics import (
    collect_schedule_diagnostics,
    plot_diagnostics,
    snapshot_indices,
)
from utils.images import pil_to_tensor, tensor_to_pil
from utils.trajectory_video import save_gif, save_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/examples"))
    parser.add_argument("--timesteps", type=int, default=1_000)
    parser.add_argument("--video-frames", type=int, default=41)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def sample_path(data_root: Path, class_id: int, *, split: str = "test") -> Path:
    """Select a deterministic example path for a class."""

    paths = sorted((data_root / split / str(class_id)).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG files found for {split} class {class_id}")
    return paths[0]


def captioned_panel(
    image: Image.Image,
    caption: str,
    *,
    panel_size: int = 160,
) -> Image.Image:
    """Upscale a tiny CIFAR image and add a caption below it."""

    canvas = Image.new("RGB", (panel_size, panel_size + 24), "white")
    resized = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.NEAREST)
    canvas.paste(resized, (0, 0))
    ImageDraw.Draw(canvas).text((5, panel_size + 5), caption, fill=(0, 0, 0))
    return canvas


def save_contact_sheet(data_root: Path, output_dir: Path) -> Path:
    """Write one labeled CIFAR-10 input example per class."""

    panels = []
    for class_id, class_name in enumerate(CIFAR10_CLASSES):
        with Image.open(sample_path(data_root, class_id)) as image:
            panels.append(captioned_panel(image, f"{class_id}: {class_name}"))
    sheet = Image.new("RGB", (5 * 160, 2 * 184), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % 5) * 160, (index // 5) * 184))
    path = output_dir / "cifar10_inputs.png"
    sheet.save(path)
    return path


def save_mask_example(source: Image.Image, output_dir: Path) -> Path:
    """Write a three-panel inpainting input explanation."""

    source = source.convert("RGB").resize((256, 256), Image.Resampling.NEAREST)
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((88, 72, 184, 176), fill=255)
    masked = source.copy()
    masked.paste((0, 0, 0), mask=mask)

    canvas = Image.new("RGB", (3 * 256, 292), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, caption) in enumerate(
        (
            (source, "Input source"),
            (mask.convert("RGB"), "White = regenerate"),
            (masked, "Masked model input"),
        )
    ):
        canvas.paste(image, (index * 256, 0))
        draw.text((index * 256 + 8, 266), caption, fill=(0, 0, 0))
    path = output_dir / "inpainting_input_mask.png"
    canvas.save(path)
    return path


def save_super_resolution_example(source: Image.Image, output_dir: Path) -> Path:
    """Write a source-versus-low-resolution observation figure."""

    source = source.convert("RGB")
    original = source.resize((256, 256), Image.Resampling.NEAREST)
    low_resolution = source.resize((8, 8), Image.Resampling.BOX)
    observed = low_resolution.resize((256, 256), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (512, 292), "white")
    canvas.paste(original, (0, 0))
    canvas.paste(observed, (256, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 266), "Reference 32x32", fill=(0, 0, 0))
    draw.text((264, 266), "Observed 8x8 upsampled input", fill=(0, 0, 0))
    path = output_dir / "super_resolution_input.png"
    canvas.save(path)
    return path


def forward_comparison_frames(
    source_tensor: torch.Tensor,
    noise: torch.Tensor,
    *,
    num_timesteps: int,
    num_frames: int,
) -> list[np.ndarray]:
    """Render synchronized linear-versus-cosine forward-process frames."""

    if num_frames < 2:
        raise ValueError("video_frames must be at least 2")
    linear = DDPM(num_timesteps=num_timesteps, schedule="linear")
    cosine = DDPM(num_timesteps=num_timesteps, schedule="cosine")
    indices = torch.linspace(0, num_timesteps - 1, num_frames - 1).round().long()
    frames = []
    for frame_number in range(num_frames):
        if frame_number == 0:
            literature_timestep = 0
            linear_image = source_tensor
            cosine_image = source_tensor
            linear_alpha = cosine_alpha = 1.0
        else:
            index = indices[frame_number - 1]
            timestep = index.reshape(1)
            literature_timestep = index.item() + 1
            linear_image = linear.q_sample(source_tensor, timestep, noise=noise)
            cosine_image = cosine.q_sample(source_tensor, timestep, noise=noise)
            linear_alpha = linear.alphas_cumprod[index].item()
            cosine_alpha = cosine.alphas_cumprod[index].item()

        canvas = Image.new("RGB", (512, 294), "white")
        canvas.paste(
            tensor_to_pil(linear_image).resize((256, 256), Image.Resampling.NEAREST),
            (0, 38),
        )
        canvas.paste(
            tensor_to_pil(cosine_image).resize((256, 256), Image.Resampling.NEAREST),
            (256, 38),
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 512, 38), fill=(20, 20, 20))
        draw.text((8, 7), f"Forward degradation at t={literature_timestep}", fill="white")
        draw.text((8, 22), f"Linear alpha_bar={linear_alpha:.4f}", fill=(255, 170, 160))
        draw.text((264, 22), f"Cosine alpha_bar={cosine_alpha:.4f}", fill=(160, 210, 255))
        frames.append(np.asarray(canvas))
    return frames


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet = save_contact_sheet(args.data_root, args.output_dir)

    source_path = sample_path(args.data_root, 8)
    with Image.open(source_path) as image:
        source_image = image.convert("RGB")
    source_asset = args.output_dir / "source_ship.png"
    source_image.resize((256, 256), Image.Resampling.NEAREST).save(source_asset)
    mask_asset = save_mask_example(source_image, args.output_dir)
    super_resolution_asset = save_super_resolution_example(source_image, args.output_dir)

    source_tensor = pil_to_tensor(source_image, image_size=128)
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(source_tensor.shape, generator=generator)
    indices = snapshot_indices(args.timesteps, 8)
    linear = collect_schedule_diagnostics(
        "linear",
        num_timesteps=args.timesteps,
        x_start=source_tensor,
        noise=noise,
        indices=indices,
    )
    cosine = collect_schedule_diagnostics(
        "cosine",
        num_timesteps=args.timesteps,
        x_start=source_tensor,
        noise=noise,
        indices=indices,
    )
    diagnostics_asset = args.output_dir / "forward_degradation.png"
    plot_diagnostics(
        source_tensor,
        indices,
        linear,
        cosine,
        source_name=source_path.name,
        output_path=diagnostics_asset,
    )

    video_source = pil_to_tensor(source_image)
    video_noise = torch.randn(video_source.shape, generator=generator)
    frames = forward_comparison_frames(
        video_source,
        video_noise,
        num_timesteps=args.timesteps,
        num_frames=args.video_frames,
    )
    gif_asset = save_gif(frames, args.output_dir / "forward_linear_vs_cosine.gif", fps=8)
    mp4_asset = save_mp4(frames, args.output_dir / "forward_linear_vs_cosine.mp4", fps=8)

    print("Generated README assets:")
    for path in (
        contact_sheet,
        source_asset,
        mask_asset,
        super_resolution_asset,
        diagnostics_asset,
        gif_asset,
        mp4_asset,
    ):
        print(f"- {path.resolve()}")


if __name__ == "__main__":
    main()

