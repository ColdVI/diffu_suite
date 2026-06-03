"""Fast structural tests for the custom mathematical core."""

from __future__ import annotations

import torch
from torch import nn

from advanced.controlnet import CannyControlNetStudio
from models.ddpm import DDPM
from models.unet import ConditionalUNet
from pipelines.restoration import DiffusionRestorationPipeline
from training.dataset import Cifar10ImageFolder
from utils.checkpoints import load_custom_checkpoint, save_custom_checkpoint
from utils.ema import ExponentialMovingAverage
from utils.images import tensor_to_pil


class ZeroDenoiser(nn.Module):
    def forward(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(x_t)


def tiny_unet() -> ConditionalUNet:
    return ConditionalUNet(
        base_channels=8,
        conditioning_dim=32,
        channel_multipliers=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
        num_attention_heads=1,
        dropout=0.0,
    )


def test_schedules_and_closed_form_reconstruction() -> None:
    x_start = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(x_start)
    timesteps = torch.tensor([0, 11])
    for schedule in ("linear", "cosine"):
        diffusion = DDPM(num_timesteps=20, schedule=schedule)
        assert len(diffusion.state_dict()) == 18
        assert all(value.shape == (20,) for value in diffusion.state_dict().values())
        assert torch.all(diffusion.betas > 0.0)
        assert torch.all(diffusion.betas < 1.0)
        x_t = diffusion.q_sample(x_start, timesteps, noise=noise)
        recovered = diffusion.predict_start_from_noise(x_t, timesteps, noise)
        assert torch.allclose(recovered, x_start, atol=2e-5)


def test_unet_predicts_same_shape() -> None:
    model = tiny_unet()
    output = model(
        torch.randn(2, 3, 32, 32),
        torch.tensor([0, 7]),
        class_labels=torch.tensor([3, -1]),
    )
    assert output.shape == (2, 3, 32, 32)


def test_checkpoint_roundtrip(tmp_path) -> None:
    model = tiny_unet()
    diffusion = DDPM(num_timesteps=5, schedule="cosine")
    ema = ExponentialMovingAverage(model)
    path = save_custom_checkpoint(
        tmp_path / "checkpoint.pt",
        model=model,
        diffusion=diffusion,
        ema=ema,
        global_step=12,
    )
    loaded = load_custom_checkpoint(path)
    assert loaded.raw["global_step"] == 12
    assert loaded.model.config_dict() == model.config_dict()
    assert loaded.diffusion.config_dict() == diffusion.config_dict()


def test_inpaint_locks_known_pixels() -> None:
    source = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, 8:24, 8:24] = 1.0
    pipeline = DiffusionRestorationPipeline(
        ZeroDenoiser(),
        DDPM(num_timesteps=5, schedule="cosine"),
    )
    output = pipeline.inpaint(source, mask, seed=7, return_all_timesteps=True)
    known = (1.0 - mask).expand_as(source).bool()
    assert torch.equal(output.image[known], source[known])
    assert output.trajectory is not None
    assert output.trajectory.shape == (1, 6, 3, 32, 32)


def test_super_resolution_locks_low_frequencies() -> None:
    source = torch.randn(1, 3, 32, 32).clamp(-1.0, 1.0)
    pipeline = DiffusionRestorationPipeline(
        ZeroDenoiser(),
        DDPM(num_timesteps=5, schedule="cosine"),
    )
    reference = pipeline.prepare_low_resolution_reference(source, downsample_factor=4)
    output = pipeline.super_resolve(reference, downsample_factor=4, seed=7)
    assert torch.allclose(
        pipeline._low_frequency(output.image, 4),
        pipeline._low_frequency(reference, 4),
        atol=1e-6,
    )


def test_local_cifar10_folder_if_present() -> None:
    try:
        dataset = Cifar10ImageFolder("data/cifar10_dataset", split="train", limit=1)
    except FileNotFoundError:
        return
    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert 0 <= label.item() <= 9


def test_canny_preprocessing() -> None:
    image = tensor_to_pil(torch.randn(3, 32, 32))
    edges = CannyControlNetStudio.canny_image(image)
    assert edges.size == (32, 32)
    assert edges.mode == "RGB"

