#!/usr/bin/env python3
"""Launch Hugging Face Diffusers' maintained DreamBooth-LoRA trainer.

This wrapper validates a small concept-image directory and builds a reproducible
``accelerate launch`` command for the official Diffusers training example.  It
keeps upstream trainer behavior upstream while giving DiffuSuite a stable,
portfolio-friendly entrypoint.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

DEFAULT_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_data_dir", type=Path)
    parser.add_argument("--instance-prompt", required=True, help='Example: "a photo of sks ceramic".')
    parser.add_argument("--output-dir", type=Path, default=Path("runs/lora"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--trainer-script",
        type=Path,
        default=Path("third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"),
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-train-steps", type=int, default=800)
    parser.add_argument("--checkpointing-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def concept_images(root: Path) -> list[Path]:
    """List supported concept images immediately below ``root``."""

    if not root.is_dir():
        raise FileNotFoundError(f"concept-image directory does not exist: {root}")
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the official DreamBooth-LoRA trainer invocation."""

    images = concept_images(args.instance_data_dir)
    if not images:
        raise ValueError("instance_data_dir must contain JPG, PNG, or WEBP images")
    if not args.trainer_script.is_file() and not args.dry_run:
        raise FileNotFoundError(
            f"official trainer script not found: {args.trainer_script}\n"
            "Clone it first with: git clone --depth 1 "
            "https://github.com/huggingface/diffusers third_party/diffusers"
        )

    return [
        "accelerate",
        "launch",
        str(args.trainer_script),
        "--pretrained_model_name_or_path",
        args.base_model,
        "--instance_data_dir",
        str(args.instance_data_dir),
        "--output_dir",
        str(args.output_dir),
        "--instance_prompt",
        args.instance_prompt,
        "--resolution",
        str(args.resolution),
        "--train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--learning_rate",
        str(args.learning_rate),
        "--max_train_steps",
        str(args.max_train_steps),
        "--checkpointing_steps",
        str(args.checkpointing_steps),
        "--rank",
        str(args.rank),
        "--mixed_precision",
        args.mixed_precision,
        "--gradient_checkpointing",
        "--seed",
        str(args.seed),
    ]


def main() -> None:
    args = parse_args()
    images = concept_images(args.instance_data_dir)
    command = build_command(args)
    print(f"validated {len(images)} concept images")
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

