"""Encode forward or reverse diffusion trajectories as GIF and MP4 files."""

from __future__ import annotations

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from torch import Tensor

from utils.images import tensor_to_pil


def trajectory_frames(
    trajectory: Tensor,
    *,
    sample_index: int = 0,
    upscale: int = 8,
    labels: list[str] | None = None,
) -> list[np.ndarray]:
    """Convert ``[B, S, C, H, W]`` or ``[S, C, H, W]`` into RGB frames."""

    if trajectory.ndim == 5:
        trajectory = trajectory[sample_index]
    if trajectory.ndim != 4:
        raise ValueError("trajectory must have shape [B, S, C, H, W] or [S, C, H, W]")
    if upscale < 1:
        raise ValueError("upscale must be positive")
    if labels is not None and len(labels) != trajectory.shape[0]:
        raise ValueError("labels must contain one string per trajectory frame")

    frames = []
    for frame_index, frame in enumerate(trajectory):
        image = tensor_to_pil(frame).convert("RGB")
        image = image.resize(
            (image.width * upscale, image.height * upscale),
            Image.Resampling.NEAREST,
        )
        if labels is not None:
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
            draw.text((4, 3), labels[frame_index], fill=(255, 255, 255))
        frames.append(np.asarray(image))
    return frames


def save_gif(
    frames: list[np.ndarray],
    path: str | Path,
    *,
    fps: float = 8.0,
) -> Path:
    """Write RGB frames as a looping GIF."""

    if not frames:
        raise ValueError("frames cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    return path


def save_mp4(
    frames: list[np.ndarray],
    path: str | Path,
    *,
    fps: float = 8.0,
) -> Path:
    """Write RGB frames as an MP4 using OpenCV's bundled video backend."""

    if not frames:
        raise ValueError("frames cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize an MP4 writer")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("all frames must have the same dimensions")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def save_trajectory_media(
    trajectory: Tensor,
    output_stem: str | Path,
    *,
    sample_index: int = 0,
    upscale: int = 8,
    fps: float = 8.0,
    labels: list[str] | None = None,
) -> tuple[Path, Path]:
    """Write sibling GIF and MP4 files for a tensor trajectory."""

    output_stem = Path(output_stem)
    frames = trajectory_frames(
        trajectory,
        sample_index=sample_index,
        upscale=upscale,
        labels=labels,
    )
    return (
        save_gif(frames, output_stem.with_suffix(".gif"), fps=fps),
        save_mp4(frames, output_stem.with_suffix(".mp4"), fps=fps),
    )

