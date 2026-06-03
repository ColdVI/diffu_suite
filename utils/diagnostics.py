#!/usr/bin/env python3
"""Visualize forward DDPM information destruction for two variance schedules.

Run from the repository root:

    python3 utils/diagnostics.py path/to/source.png

The generated figure compares matched closed-form samples from linear and
cosine schedules.  Both rows use the same source image and the same Gaussian
noise tensor, isolating the effect of the schedule itself.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ddpm import DDPM, ScheduleName


@dataclass(frozen=True)
class ScheduleDiagnostics:
    """Forward-process samples and scalar traces for one schedule."""

    name: str
    process: DDPM
    snapshots: list[Tensor]
    mse_curve: Tensor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare forward image degradation under linear and cosine DDPM schedules."
        )
    )
    parser.add_argument("image", type=Path, help="Source image to degrade.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/forward_degradation.png"),
        help="Destination PNG path. Default: artifacts/forward_degradation.png",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000,
        help="Number of diffusion steps T. Default: 1000",
    )
    parser.add_argument(
        "--num-snapshots",
        type=int,
        default=8,
        help="Number of noisy snapshots per schedule, excluding clean x_0. Default: 8",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=192,
        help="Center-cropped square image size used for the grid. Default: 192",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the shared Gaussian noise tensor. Default: 7",
    )
    return parser.parse_args()


def load_image(path: Path, image_size: int) -> Tensor:
    """Load and normalize one RGB image to a ``[1, 3, H, W]`` tensor in ``[-1, 1]``."""

    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(f"source image does not exist: {path}")

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
        )
        image_array = np.asarray(image, dtype=np.float32).copy()

    return torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0


def image_for_plot(image: Tensor) -> np.ndarray:
    """Convert a normalized image tensor to an ``[H, W, 3]`` NumPy array."""

    return (
        image.detach()
        .squeeze(0)
        .clamp(-1.0, 1.0)
        .add(1.0)
        .div(2.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def snapshot_indices(num_timesteps: int, num_snapshots: int) -> Tensor:
    """Choose evenly spaced zero-based buffer indices, including the final step."""

    if num_timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    if not 2 <= num_snapshots <= num_timesteps:
        raise ValueError("num_snapshots must lie between 2 and timesteps")
    return torch.linspace(0, num_timesteps - 1, num_snapshots).round().long()


def fixed_noise_mse_curve(x_start: Tensor, noise: Tensor, alphas_cumprod: Tensor) -> Tensor:
    """Compute exact per-pixel MSE traces for fixed-noise closed-form samples.

    Expanding ``T`` full-resolution image tensors is unnecessary.  The MSE can
    be reduced analytically from the source, noise, and cross-term moments.
    """

    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    source_moment = x_start.square().mean()
    noise_moment = noise.square().mean()
    cross_moment = (x_start * noise).mean()

    signal_delta = sqrt_alphas_cumprod - 1.0
    return (
        signal_delta.square() * source_moment
        + sqrt_one_minus_alphas_cumprod.square() * noise_moment
        + 2.0 * signal_delta * sqrt_one_minus_alphas_cumprod * cross_moment
    )


def collect_schedule_diagnostics(
    schedule_name: ScheduleName,
    *,
    num_timesteps: int,
    x_start: Tensor,
    noise: Tensor,
    indices: Tensor,
) -> ScheduleDiagnostics:
    """Generate matched forward-process snapshots and MSE values."""

    process = DDPM(num_timesteps=num_timesteps, schedule=schedule_name)
    snapshots = []
    for index in indices:
        timestep = index.reshape(1)
        snapshots.append(process.q_sample(x_start, timestep, noise=noise))

    return ScheduleDiagnostics(
        name=schedule_name,
        process=process,
        snapshots=snapshots,
        mse_curve=fixed_noise_mse_curve(x_start, noise, process.alphas_cumprod),
    )


def add_snapshot_row(
    figure: plt.Figure,
    grid: matplotlib.gridspec.GridSpec,
    row: int,
    diagnostics: ScheduleDiagnostics,
    *,
    clean_image: Tensor,
    indices: Tensor,
    color: str,
) -> None:
    """Render one schedule's clean source and degradation snapshots."""

    axes = [figure.add_subplot(grid[row, column]) for column in range(len(indices) + 1)]
    axes[0].imshow(image_for_plot(clean_image))
    axes[0].set_title("$x_0$ clean", fontsize=9)

    for axis, index, snapshot in zip(axes[1:], indices, diagnostics.snapshots):
        alpha_bar = diagnostics.process.alphas_cumprod[index].item()
        axis.imshow(image_for_plot(snapshot))
        axis.set_title(
            f"$x_{{{index.item() + 1}}}$\n"
            f"$\\bar{{\\alpha}}={alpha_bar:.3f}$",
            fontsize=8,
        )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.4)

    axes[0].set_ylabel(
        diagnostics.name.capitalize(),
        color=color,
        fontsize=11,
        fontweight="bold",
    )


def plot_diagnostics(
    x_start: Tensor,
    indices: Tensor,
    linear: ScheduleDiagnostics,
    cosine: ScheduleDiagnostics,
    *,
    source_name: str,
    output_path: Path,
) -> None:
    """Create and save the complete schedule-comparison figure."""

    num_columns = len(indices) + 1
    figure = plt.figure(figsize=(max(16, num_columns * 2.0), 8.8), layout="constrained")
    grid = figure.add_gridspec(3, num_columns, height_ratios=(1.0, 1.0, 1.2))

    colors = {"linear": "#c0392b", "cosine": "#2471a3"}
    add_snapshot_row(
        figure,
        grid,
        0,
        linear,
        clean_image=x_start,
        indices=indices,
        color=colors["linear"],
    )
    add_snapshot_row(
        figure,
        grid,
        1,
        cosine,
        clean_image=x_start,
        indices=indices,
        color=colors["cosine"],
    )

    chart_grid = grid[2, :].subgridspec(1, 2, wspace=0.16)
    variance_axis = figure.add_subplot(chart_grid[0, 0])
    mse_axis = figure.add_subplot(chart_grid[0, 1])
    literature_steps = np.arange(1, linear.process.num_timesteps + 1)

    for diagnostics in (linear, cosine):
        color = colors[diagnostics.name]
        alphas_cumprod = diagnostics.process.alphas_cumprod.cpu().numpy()
        variance_axis.plot(
            literature_steps,
            alphas_cumprod,
            color=color,
            linewidth=2.0,
            label=f"{diagnostics.name}: signal $\\bar{{\\alpha}}_t$",
        )
        variance_axis.plot(
            literature_steps,
            1.0 - alphas_cumprod,
            color=color,
            linestyle="--",
            linewidth=1.7,
            label=f"{diagnostics.name}: variance $1-\\bar{{\\alpha}}_t$",
        )
        mse_axis.plot(
            literature_steps,
            diagnostics.mse_curve.cpu().numpy(),
            color=color,
            linewidth=2.0,
            label=diagnostics.name.capitalize(),
        )

    variance_axis.set_title("Closed-Form Signal Retention and Forward Variance")
    variance_axis.set_xlabel("Literature timestep $t$")
    variance_axis.set_ylabel("Coefficient value")
    variance_axis.set_ylim(-0.03, 1.03)
    variance_axis.grid(alpha=0.25)
    variance_axis.legend(fontsize=8)

    mse_axis.set_title("Measured Source-to-$x_t$ MSE with Shared Noise")
    mse_axis.set_xlabel("Literature timestep $t$")
    mse_axis.set_ylabel("Mean squared error")
    mse_axis.grid(alpha=0.25)
    mse_axis.legend(fontsize=8)

    figure.suptitle(
        "DDPM Forward Degradation: Linear vs. Cosine Schedule\n"
        f"source={source_name!r}, shared Gaussian noise seed comparison",
        fontsize=14,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def log_summary(
    output_path: Path,
    indices: Tensor,
    linear: ScheduleDiagnostics,
    cosine: ScheduleDiagnostics,
) -> None:
    """Print a compact textual summary beside the visual artifact."""

    print(f"Saved diagnostic figure: {output_path.resolve()}")
    print("Selected literature timesteps:", ", ".join(str(index.item() + 1) for index in indices))
    for diagnostics in (linear, cosine):
        process = diagnostics.process
        print(
            f"{diagnostics.name:>6} | "
            f"alpha_bar_T={process.alphas_cumprod[-1].item():.8f} | "
            f"variance_T={(1.0 - process.alphas_cumprod[-1]).item():.8f} | "
            f"fixed-noise mse_T={diagnostics.mse_curve[-1].item():.8f}"
        )


def main() -> None:
    """Run the command-line diagnostic."""

    args = parse_args()
    indices = snapshot_indices(args.timesteps, args.num_snapshots)
    x_start = load_image(args.image, args.image_size)

    generator = torch.Generator(device=x_start.device).manual_seed(args.seed)
    noise = torch.randn(
        x_start.shape,
        dtype=x_start.dtype,
        device=x_start.device,
        generator=generator,
    )

    linear = collect_schedule_diagnostics(
        "linear",
        num_timesteps=args.timesteps,
        x_start=x_start,
        noise=noise,
        indices=indices,
    )
    cosine = collect_schedule_diagnostics(
        "cosine",
        num_timesteps=args.timesteps,
        x_start=x_start,
        noise=noise,
        indices=indices,
    )
    plot_diagnostics(
        x_start,
        indices,
        linear,
        cosine,
        source_name=args.image.name,
        output_path=args.output,
    )
    log_summary(args.output, indices, linear, cosine)


if __name__ == "__main__":
    main()
