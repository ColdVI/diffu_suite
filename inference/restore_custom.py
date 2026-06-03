#!/usr/bin/env python3
"""Run custom DDPM inpainting or exploratory super-resolution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.restoration import DiffusionRestorationPipeline
from utils.checkpoints import load_custom_checkpoint
from utils.device import resolve_device, seed_everything
from utils.images import mask_to_tensor, pil_to_tensor, tensor_to_pil
from utils.trajectory_video import save_trajectory_media


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--task", choices=("inpaint", "super-res"), default="inpaint")
    parser.add_argument("--mask", type=Path, help="Required for inpainting; white means regenerate.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/restored/result.png"))
    parser.add_argument("--trajectory-stem", type=Path, default=None)
    parser.add_argument("--class-id", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    loaded = load_custom_checkpoint(args.checkpoint, device=device)
    pipeline = DiffusionRestorationPipeline(loaded.model, loaded.diffusion)
    with Image.open(args.source) as image:
        source = pil_to_tensor(image, image_size=32).to(device)
    class_labels = (
        torch.tensor([args.class_id], device=device) if args.class_id is not None else None
    )

    if args.task == "inpaint":
        if args.mask is None:
            raise ValueError("--mask is required for inpainting")
        with Image.open(args.mask) as image:
            mask = mask_to_tensor(image, image_size=32).to(device)
        result = pipeline.inpaint(
            source,
            mask,
            class_labels=class_labels,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_all_timesteps=args.trajectory_stem is not None,
        )
    else:
        reference = pipeline.prepare_low_resolution_reference(
            source,
            downsample_factor=args.downsample_factor,
        )
        result = pipeline.super_resolve(
            reference,
            downsample_factor=args.downsample_factor,
            class_labels=class_labels,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            return_all_timesteps=args.trajectory_stem is not None,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(result.image).save(args.output)
    print(f"saved restored image: {args.output.resolve()}")
    if args.trajectory_stem is not None and result.trajectory is not None:
        gif_path, mp4_path = save_trajectory_media(result.trajectory, args.trajectory_stem)
        print(f"saved trajectory: {gif_path.resolve()} and {mp4_path.resolve()}")


if __name__ == "__main__":
    main()

