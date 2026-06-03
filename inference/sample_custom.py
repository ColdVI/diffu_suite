#!/usr/bin/env python3
"""Generate CIFAR-10 class samples from a trained custom DDPM checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.dataset import CIFAR10_CLASSES
from utils.checkpoints import load_custom_checkpoint
from utils.device import resolve_device, seed_everything
from utils.images import save_tensor_grid
from utils.trajectory_video import save_trajectory_media


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/generated/classes.png"))
    parser.add_argument("--classes", type=_csv_ints, default=tuple(range(10)))
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="CFG literature w.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--trajectory-stem",
        type=Path,
        default=None,
        help="Optional path stem for the first sample's GIF and MP4 reverse trajectory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive")
    if any(class_id < 0 or class_id >= len(CIFAR10_CLASSES) for class_id in args.classes):
        raise ValueError("classes must contain CIFAR-10 ids from 0 to 9")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    loaded = load_custom_checkpoint(args.checkpoint, device=device, use_ema=not args.no_ema)
    labels = torch.tensor(
        [class_id for class_id in args.classes for _ in range(args.samples_per_class)],
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = loaded.diffusion.sample_loop(
        loaded.model,
        (len(labels), 3, 32, 32),
        class_labels=labels,
        guidance_scale=args.guidance_scale,
        clip_denoised=True,
        device=device,
        generator=generator,
        return_all_timesteps=args.trajectory_stem is not None,
    )
    if args.trajectory_stem is None:
        samples = result
    else:
        samples, trajectory = result
        gif_path, mp4_path = save_trajectory_media(trajectory, args.trajectory_stem)
        print(f"saved reverse trajectory to {gif_path.resolve()} and {mp4_path.resolve()}")
    save_tensor_grid(samples, args.output, nrow=args.samples_per_class)
    print(f"saved {len(labels)} samples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
