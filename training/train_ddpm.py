#!/usr/bin/env python3
"""Train the custom class-conditioned DDPM on extracted CIFAR-10 PNG files."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ddpm import DDPM
from models.unet import ConditionalUNet
from training.dataset import CIFAR10_CLASSES, build_dataloader
from utils.checkpoints import (
    load_custom_checkpoint,
    restore_optimizer,
    save_custom_checkpoint,
)
from utils.device import resolve_device, seed_everything
from utils.ema import ExponentialMovingAverage
from utils.images import save_tensor_grid


def _csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/cifar10_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cifar10_cosine"))
    parser.add_argument("--schedule", choices=("linear", "cosine"), default="cosine")
    parser.add_argument("--timesteps", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--condition-dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--conditioning-dim", type=int, default=256)
    parser.add_argument("--channel-multipliers", type=_csv_ints, default=(1, 2, 4))
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--attention-levels", type=_csv_ints, default=(2,))
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--sample-every", type=int, default=2_000)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset subset.")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA float16 automatic mixed precision.",
    )
    return parser.parse_args()


def build_new_experiment(args: argparse.Namespace, device: torch.device) -> tuple[
    ConditionalUNet,
    DDPM,
    torch.optim.Optimizer,
    ExponentialMovingAverage,
    int,
    int,
]:
    """Create or resume model state and optimizer state."""

    if args.resume is not None:
        loaded = load_custom_checkpoint(args.resume, device=device, use_ema=False)
        model = loaded.model
        diffusion = loaded.diffusion
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        restore_optimizer(optimizer, loaded.raw)
        ema = ExponentialMovingAverage(model, decay=args.ema_decay)
        if "ema_state" in loaded.raw:
            ema.load_state_dict(loaded.raw["ema_state"])
        return (
            model,
            diffusion,
            optimizer,
            ema,
            int(loaded.raw.get("epoch", 0)),
            int(loaded.raw.get("global_step", 0)),
        )

    model = ConditionalUNet(
        num_classes=len(CIFAR10_CLASSES),
        base_channels=args.base_channels,
        conditioning_dim=args.conditioning_dim,
        channel_multipliers=args.channel_multipliers,
        num_res_blocks=args.num_res_blocks,
        attention_levels=args.attention_levels,
        num_attention_heads=args.attention_heads,
        dropout=args.dropout,
    ).to(device)
    diffusion = DDPM(
        num_timesteps=args.timesteps,
        schedule=args.schedule,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    return model, diffusion, optimizer, ema, 0, 0


@torch.no_grad()
def save_preview(
    model: ConditionalUNet,
    diffusion: DDPM,
    ema: ExponentialMovingAverage,
    *,
    path: Path,
    sample_count: int,
    device: torch.device,
    seed: int,
) -> None:
    """Sample a small EMA preview grid without mutating the training model."""

    preview_model = ConditionalUNet(**model.config_dict()).to(device)
    preview_model.load_state_dict(model.state_dict())
    ema.copy_to(preview_model)
    preview_model.eval()
    class_labels = torch.arange(sample_count, device=device) % len(CIFAR10_CLASSES)
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = diffusion.sample_loop(
        preview_model,
        (sample_count, 3, 32, 32),
        class_labels=class_labels,
        guidance_scale=1.0,
        clip_denoised=True,
        device=device,
        generator=generator,
    )
    save_tensor_grid(samples, path, nrow=min(5, sample_count))


def save_training_state(
    args: argparse.Namespace,
    *,
    model: ConditionalUNet,
    diffusion: DDPM,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    epoch: int,
    global_step: int,
) -> Path:
    """Write both a numbered checkpoint and a convenient latest checkpoint."""

    metadata = {
        "dataset": "CIFAR-10 extracted ImageFolder",
        "class_names": list(CIFAR10_CLASSES),
        "command": " ".join(sys.argv),
    }
    numbered_path = args.output_dir / "checkpoints" / f"step_{global_step:07d}.pt"
    save_custom_checkpoint(
        numbered_path,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        ema=ema,
        epoch=epoch,
        global_step=global_step,
        metadata=metadata,
    )
    latest_path = args.output_dir / "checkpoints" / "latest.pt"
    save_custom_checkpoint(
        latest_path,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        ema=ema,
        epoch=epoch,
        global_step=global_step,
        metadata=metadata,
    )
    return latest_path


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(
        args.data_root,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.workers,
        limit=args.limit,
    )
    model, diffusion, optimizer, ema, start_epoch, global_step = build_new_experiment(
        args,
        device,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} amp={use_amp} samples={len(loader.dataset)} "
        f"parameters={parameter_count:,} schedule={diffusion.schedule} "
        f"T={diffusion.num_timesteps}"
    )

    model.train()
    started_at = time.perf_counter()
    stop_training = False
    latest_checkpoint: Path | None = None
    for epoch in range(start_epoch, args.epochs):
        for images, class_labels in loader:
            images = images.to(device)
            class_labels = class_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = diffusion.training_loss(
                    model,
                    images,
                    class_labels=class_labels,
                    condition_dropout_prob=args.condition_dropout,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"epoch={epoch + 1:03d} step={global_step:07d} "
                    f"loss={loss.item():.6f} elapsed={elapsed:.1f}s"
                )

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                preview_path = args.output_dir / "samples" / f"step_{global_step:07d}.png"
                save_preview(
                    model,
                    diffusion,
                    ema,
                    path=preview_path,
                    sample_count=args.sample_count,
                    device=device,
                    seed=args.seed + global_step,
                )
                print(f"saved preview: {preview_path}")

            if args.save_every > 0 and global_step % args.save_every == 0:
                latest_checkpoint = save_training_state(
                    args,
                    model=model,
                    diffusion=diffusion,
                    optimizer=optimizer,
                    ema=ema,
                    epoch=epoch,
                    global_step=global_step,
                )
                print(f"saved checkpoint: {latest_checkpoint}")

            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break

    latest_checkpoint = save_training_state(
        args,
        model=model,
        diffusion=diffusion,
        optimizer=optimizer,
        ema=ema,
        epoch=epoch if "epoch" in locals() else start_epoch,
        global_step=global_step,
    )
    print(f"training complete: step={global_step} checkpoint={latest_checkpoint}")


if __name__ == "__main__":
    main()

