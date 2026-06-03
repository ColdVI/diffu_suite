"""Denoising Diffusion Probabilistic Model process utilities.

This module implements the mathematics around an epsilon-predicting denoiser.
It deliberately does not own a U-Net: pass any compatible ``nn.Module`` to
``training_loss`` or ``sample_loop``.  Keeping the process separate makes the
same schedule reusable for unconditional generation, class-conditioned
generation, diagnostics, inpainting, and super-resolution.

Tensor conventions
------------------
``T`` is ``num_timesteps`` and ``B`` is the batch size.

Persistent schedule buffers all have shape ``[T]``:

* ``diffusion_steps``: literature indices ``[1, ..., T]``.
* ``betas``: :math:`\\beta_t`.
* ``alphas``: :math:`\\alpha_t = 1 - \\beta_t`.
* ``alphas_cumprod``: :math:`\\bar{\\alpha}_t`.
* ``alphas_cumprod_prev``: :math:`\\bar{\\alpha}_{t-1}`, prepended with 1.
* ``sqrt_betas``: :math:`\\sqrt{\\beta_t}`.
* ``sqrt_alphas``: :math:`\\sqrt{\\alpha_t}`.
* ``sqrt_recip_alphas``: :math:`1 / \\sqrt{\\alpha_t}`.
* ``sqrt_alphas_cumprod``: :math:`\\sqrt{\\bar{\\alpha}_t}`.
* ``sqrt_one_minus_alphas_cumprod``:
  :math:`\\sqrt{1 - \\bar{\\alpha}_t}`.
* ``log_one_minus_alphas_cumprod``:
  :math:`\\log(1 - \\bar{\\alpha}_t)`.
* ``sqrt_recip_alphas_cumprod``:
  :math:`1 / \\sqrt{\\bar{\\alpha}_t}`.
* ``sqrt_recipm1_alphas_cumprod``:
  :math:`\\sqrt{1 / \\bar{\\alpha}_t - 1}`.
* ``posterior_variance``: :math:`\\tilde{\\beta}_t`.
* ``posterior_log_variance_clipped``:
  :math:`\\log(\\max(\\tilde{\\beta}_t, 10^{-20}))`.
* ``posterior_std``: :math:`\\sigma_t = \\sqrt{\\tilde{\\beta}_t}`.
* ``posterior_mean_coef1``: coefficient on :math:`x_0` in
  :math:`q(x_{t-1} | x_t, x_0)`.
* ``posterior_mean_coef2``: coefficient on :math:`x_t` in
  :math:`q(x_{t-1} | x_t, x_0)`.

Public methods accept zero-based timestep tensors with shape ``[B]`` because
they index the PyTorch buffers directly.  A code timestep of ``0`` therefore
corresponds to the literature timestep :math:`t = 1`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ScheduleName = Literal["linear", "cosine"]
Reduction = Literal["mean", "sum", "none"]
ProgressCallback = Callable[[int, Tensor], None]


@dataclass(frozen=True)
class ReverseProcessOutput:
    """Statistics for one learned reverse-process transition.

    All tensors have the same shape as the current noisy batch ``x_t``:
    ``[B, C, H, W]`` for images.
    """

    mean: Tensor
    variance: Tensor
    log_variance: Tensor
    predicted_noise: Tensor
    predicted_start: Tensor


def make_beta_schedule(
    schedule: ScheduleName,
    num_timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_offset: float = 0.008,
    cosine_max_beta: float = 0.999,
) -> Tensor:
    """Build a variance schedule as a float64 tensor with shape ``[T]``.

    Float64 construction prevents avoidable cumulative-product drift.  The
    ``DDPM`` constructor stores the resulting schedule buffers as float32.
    """

    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")

    if schedule == "linear":
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("linear schedule requires 0 < beta_start < beta_end < 1")
        return torch.linspace(
            beta_start,
            beta_end,
            num_timesteps,
            dtype=torch.float64,
        )

    if schedule == "cosine":
        if cosine_offset < 0.0:
            raise ValueError("cosine_offset must be non-negative")
        if not 0.0 < cosine_max_beta < 1.0:
            raise ValueError("cosine_max_beta must lie in (0, 1)")

        # Nichol & Dhariwal: alpha_bar(t) = cos^2(((t / T + s) / (1 + s)) pi/2).
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos(
            ((steps / num_timesteps + cosine_offset) / (1.0 + cosine_offset))
            * torch.pi
            / 2.0
        ).square()
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=cosine_max_beta)

    raise ValueError(f"unsupported schedule: {schedule!r}")


class DDPM(nn.Module):
    """Schedule buffers and sampling equations for an epsilon-predicting DDPM.

    A compatible denoiser must implement::

        predicted_noise = model(x_t, timesteps, class_labels=class_labels, **kwargs)

    where ``predicted_noise`` has the same shape as ``x_t``.  For
    classifier-free guidance training, a class label of ``-1`` is the null
    token for individual examples.  During CFG sampling, ``class_labels=None``
    asks the model for an unconditional batch.
    """

    def __init__(
        self,
        *,
        num_timesteps: int = 1_000,
        schedule: ScheduleName = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        cosine_offset: float = 0.008,
        cosine_max_beta: float = 0.999,
    ) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.cosine_offset = cosine_offset
        self.cosine_max_beta = cosine_max_beta

        betas = make_beta_schedule(
            schedule,
            num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_offset=cosine_offset,
            cosine_max_beta=cosine_max_beta,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        buffers = {
            "diffusion_steps": torch.arange(1, num_timesteps + 1, dtype=torch.long),
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_betas": torch.sqrt(betas),
            "sqrt_alphas": torch.sqrt(alphas),
            "sqrt_recip_alphas": torch.rsqrt(alphas),
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "log_one_minus_alphas_cumprod": torch.log(1.0 - alphas_cumprod),
            "sqrt_recip_alphas_cumprod": torch.rsqrt(alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1.0),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(
                posterior_variance.clamp(min=1e-20)
            ),
            "posterior_std": torch.sqrt(posterior_variance),
            "posterior_mean_coef1": (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            ),
            "posterior_mean_coef2": (
                (1.0 - alphas_cumprod_prev)
                * torch.sqrt(alphas)
                / (1.0 - alphas_cumprod)
            ),
        }

        for name, value in buffers.items():
            stored_value = value if value.dtype == torch.long else value.float()
            self.register_buffer(name, stored_value, persistent=True)

    def extra_repr(self) -> str:
        """Include the schedule configuration in ``print(ddpm)`` output."""

        return f"num_timesteps={self.num_timesteps}, schedule={self.schedule!r}"

    def config_dict(self) -> dict[str, Any]:
        """Return constructor values suitable for a checkpoint."""

        return {
            "num_timesteps": self.num_timesteps,
            "schedule": self.schedule,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "cosine_offset": self.cosine_offset,
            "cosine_max_beta": self.cosine_max_beta,
        }

    def _validate_timesteps(
        self,
        timesteps: Tensor,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(f"timesteps must have shape [{batch_size}]")
        if timesteps.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("timesteps must contain integer indices")
        if torch.any(timesteps < 0) or torch.any(timesteps >= self.num_timesteps):
            raise ValueError(
                f"timesteps must lie in [0, {self.num_timesteps - 1}]"
            )
        if device is not None and timesteps.device != device:
            raise ValueError(f"timesteps must be on device {device}")

    def _validate_class_labels(
        self,
        class_labels: Tensor | None,
        batch_size: int,
    ) -> None:
        if class_labels is None:
            return
        if class_labels.ndim != 1 or class_labels.shape[0] != batch_size:
            raise ValueError(f"class_labels must have shape [{batch_size}]")
        if class_labels.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("class_labels must contain integer indices")

    @staticmethod
    def _validate_matching_tensor(value: Tensor, reference: Tensor, name: str) -> None:
        if value.shape != reference.shape:
            raise ValueError(f"{name} must have the same shape as the reference tensor")
        if value.device != reference.device:
            raise ValueError(f"{name} must be on device {reference.device}")
        if value.dtype != reference.dtype:
            raise TypeError(f"{name} must have dtype {reference.dtype}")

    def _extract(self, values: Tensor, timesteps: Tensor, x: Tensor) -> Tensor:
        """Gather ``[T]`` values and reshape them to ``[B, 1, ..., 1]``.

        Extracted coefficients adopt ``x``'s device and dtype.  This keeps the
        process compatible with mixed-precision denoisers while the persistent
        schedule buffers themselves remain float32.
        """

        if not x.is_floating_point():
            raise TypeError("diffusion states must use a floating-point dtype")
        gathered = values.gather(0, timesteps.to(device=values.device, dtype=torch.long))
        return gathered.to(device=x.device, dtype=x.dtype).reshape(
            timesteps.shape[0],
            *((1,) * (x.ndim - 1)),
        )

    def sample_timesteps(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Draw uniformly distributed zero-based timesteps with shape ``[B]``."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=device,
            generator=generator,
        )

    def q_mean_variance(self, x_start: Tensor, timesteps: Tensor) -> tuple[Tensor, Tensor]:
        """Return mean and variance of ``q(x_t | x_0)``.

        For image inputs, both returned tensors broadcast to ``[B, C, H, W]``.
        """

        self._validate_timesteps(timesteps, x_start.shape[0], device=x_start.device)
        mean = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
        variance = self._extract(
            1.0 - self.alphas_cumprod,
            timesteps,
            x_start,
        )
        return mean, variance

    def q_sample(
        self,
        x_start: Tensor,
        timesteps: Tensor,
        *,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample ``q(x_t | x_0)`` in one step with the reparameterization trick.

        .. math::

            x_t = \\sqrt{\\bar{\\alpha}_t} x_0
                + \\sqrt{1 - \\bar{\\alpha}_t} \\epsilon
        """

        self._validate_timesteps(timesteps, x_start.shape[0], device=x_start.device)
        if noise is None:
            noise = torch.randn(
                x_start.shape,
                dtype=x_start.dtype,
                device=x_start.device,
                generator=generator,
            )
        self._validate_matching_tensor(noise, x_start, "noise")

        return (
            self._extract(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                x_start,
            )
            * noise
        )

    def predict_start_from_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Tensor,
    ) -> Tensor:
        """Recover an estimate of ``x_0`` from ``x_t`` and predicted noise."""

        self._validate_timesteps(timesteps, x_t.shape[0], device=x_t.device)
        self._validate_matching_tensor(predicted_noise, x_t, "predicted_noise")
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t)
            * predicted_noise
        )

    def predict_noise_from_start(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_start: Tensor,
    ) -> Tensor:
        """Recover predicted noise from ``x_t`` and an estimate of ``x_0``."""

        self._validate_timesteps(timesteps, x_t.shape[0], device=x_t.device)
        self._validate_matching_tensor(predicted_start, x_t, "predicted_start")
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t) * x_t
            - predicted_start
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t)

    def q_posterior_mean_variance(
        self,
        x_start: Tensor,
        x_t: Tensor,
        timesteps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return statistics of ``q(x_{t-1} | x_t, x_0)``."""

        self._validate_matching_tensor(x_start, x_t, "x_start")
        self._validate_timesteps(timesteps, x_t.shape[0], device=x_t.device)

        mean = (
            self._extract(self.posterior_mean_coef1, timesteps, x_t) * x_start
            + self._extract(self.posterior_mean_coef2, timesteps, x_t) * x_t
        )
        variance = self._extract(self.posterior_variance, timesteps, x_t)
        log_variance = self._extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t,
        )
        return mean, variance, log_variance

    def reverse_mean_from_noise(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        predicted_noise: Tensor,
    ) -> Tensor:
        """Compute the epsilon-parameterized ancestral reverse mean exactly.

        .. math::

            \\mu_\\theta(x_t, t) =
            \\frac{1}{\\sqrt{\\alpha_t}}
            \\left(
                x_t - \\frac{\\beta_t}{\\sqrt{1 - \\bar{\\alpha}_t}}
                \\epsilon_\\theta(x_t, t)
            \\right)
        """

        self._validate_timesteps(timesteps, x_t.shape[0], device=x_t.device)
        self._validate_matching_tensor(predicted_noise, x_t, "predicted_noise")

        return self._extract(self.sqrt_recip_alphas, timesteps, x_t) * (
            x_t
            - self._extract(self.betas, timesteps, x_t)
            * predicted_noise
            / self._extract(
                self.sqrt_one_minus_alphas_cumprod,
                timesteps,
                x_t,
            )
        )

    def guided_noise_prediction(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Predict noise, optionally using classifier-free guidance.

        ``guidance_scale`` is the literature value ``w``:

        .. math::

            \\tilde{\\epsilon}_\\theta(x_t, c) =
            (1 + w) \\epsilon_\\theta(x_t, c)
            - w \\epsilon_\\theta(x_t, \\emptyset)

        ``guidance_scale=0`` performs one model evaluation.  A positive scale
        with class labels performs conditional and unconditional evaluations.
        """

        self._validate_timesteps(timesteps, x_t.shape[0], device=x_t.device)
        self._validate_class_labels(class_labels, x_t.shape[0])
        if guidance_scale < 0.0:
            raise ValueError("guidance_scale must be non-negative")

        if class_labels is not None:
            class_labels = class_labels.to(x_t.device)
        kwargs = dict(model_kwargs or {})
        conditional_noise = model(
            x_t,
            timesteps,
            class_labels=class_labels,
            **kwargs,
        )
        if conditional_noise.shape != x_t.shape:
            raise ValueError("model must predict a noise tensor with the same shape as x_t")

        if class_labels is None or guidance_scale == 0.0:
            return conditional_noise

        unconditional_noise = model(
            x_t,
            timesteps,
            class_labels=None,
            **kwargs,
        )
        if unconditional_noise.shape != x_t.shape:
            raise ValueError("model must predict a noise tensor with the same shape as x_t")

        return (
            (1.0 + guidance_scale) * conditional_noise
            - guidance_scale * unconditional_noise
        )

    def p_mean_variance(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        clip_denoised: bool = False,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> ReverseProcessOutput:
        """Return learned statistics of ``p_theta(x_{t-1} | x_t)``.

        By default, ``mean`` is the exact epsilon-parameterized reverse mean.
        If ``clip_denoised`` is enabled, the method clips the recovered
        ``x_0`` estimate to ``[-1, 1]`` and recomputes the posterior mean.  The
        latter is often useful for image sampling but intentionally opt-in so
        the default remains a transparent implementation of the equation.
        """

        predicted_noise = self.guided_noise_prediction(
            model,
            x_t,
            timesteps,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            model_kwargs=model_kwargs,
        )
        predicted_start = self.predict_start_from_noise(
            x_t,
            timesteps,
            predicted_noise,
        )

        if clip_denoised:
            predicted_start = predicted_start.clamp(-1.0, 1.0)
            mean, variance, log_variance = self.q_posterior_mean_variance(
                predicted_start,
                x_t,
                timesteps,
            )
        else:
            mean = self.reverse_mean_from_noise(x_t, timesteps, predicted_noise)
            variance = self._extract(self.posterior_variance, timesteps, x_t)
            log_variance = self._extract(
                self.posterior_log_variance_clipped,
                timesteps,
                x_t,
            )

        return ReverseProcessOutput(
            mean=mean,
            variance=variance,
            log_variance=log_variance,
            predicted_noise=predicted_noise,
            predicted_start=predicted_start,
        )

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        clip_denoised: bool = False,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Sample one ancestral transition ``x_t -> x_{t-1}``.

        ``posterior_std`` is :math:`\\sigma_t`.  The random term is disabled at
        code timestep ``0`` because the final transition returns ``x_0``.
        """

        reverse = self.p_mean_variance(
            model,
            x_t,
            timesteps,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
        )
        if noise is None:
            noise = torch.randn(
                x_t.shape,
                dtype=x_t.dtype,
                device=x_t.device,
                generator=generator,
            )
        self._validate_matching_tensor(noise, x_t, "noise")

        nonzero_mask = (timesteps != 0).to(dtype=x_t.dtype).reshape(
            x_t.shape[0],
            *((1,) * (x_t.ndim - 1)),
        )
        sigma_t = self._extract(self.posterior_std, timesteps, x_t)
        return reverse.mean + nonzero_mask * sigma_t * noise

    @torch.no_grad()
    def sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        *,
        class_labels: Tensor | None = None,
        guidance_scale: float = 0.0,
        clip_denoised: bool = False,
        initial_noise: Tensor | None = None,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        return_all_timesteps: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the complete ancestral reverse process.

        Args:
            model: Epsilon-predicting denoiser.
            shape: Sample shape ``[B, C, H, W]`` for image generation.
            return_all_timesteps: If true, also return a trajectory tensor with
                shape ``[B, T + 1, C, H, W]``.  It starts with ``x_T`` and
                stores every subsequent reverse state through ``x_0``.
            progress_callback: Optional ``callback(code_timestep, x_t)``.
        """

        if len(shape) < 2 or shape[0] < 1:
            raise ValueError("shape must contain a positive batch dimension")

        if initial_noise is not None:
            if initial_noise.shape != shape:
                raise ValueError("initial_noise must match shape")
            x_t = initial_noise
            sample_device = x_t.device
        else:
            if device is None:
                try:
                    sample_device = next(model.parameters()).device
                except StopIteration:
                    sample_device = self.betas.device
            else:
                sample_device = torch.device(device)
            x_t = torch.randn(
                shape,
                device=sample_device,
                generator=generator,
            )

        if class_labels is not None:
            class_labels = class_labels.to(sample_device)
        self._validate_class_labels(class_labels, shape[0])

        trajectory = [x_t.detach().clone()] if return_all_timesteps else None
        for step in reversed(range(self.num_timesteps)):
            timesteps = torch.full(
                (shape[0],),
                step,
                device=sample_device,
                dtype=torch.long,
            )
            x_t = self.p_sample(
                model,
                x_t,
                timesteps,
                class_labels=class_labels,
                guidance_scale=guidance_scale,
                clip_denoised=clip_denoised,
                generator=generator,
                model_kwargs=model_kwargs,
            )
            if trajectory is not None:
                trajectory.append(x_t.detach().clone())
            if progress_callback is not None:
                progress_callback(step, x_t)

        if trajectory is None:
            return x_t
        return x_t, torch.stack(trajectory, dim=1)

    def training_loss(
        self,
        model: nn.Module,
        x_start: Tensor,
        *,
        class_labels: Tensor | None = None,
        timesteps: Tensor | None = None,
        noise: Tensor | None = None,
        condition_dropout_prob: float = 0.0,
        reduction: Reduction = "mean",
        generator: torch.Generator | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> Tensor:
        """Compute the standard simple epsilon-prediction loss.

        .. math::

            L_{simple}(\\theta) =
            \\mathbb{E}_{t, x_0, \\epsilon}
            \\left[\\left\\|\\epsilon - \\epsilon_\\theta(x_t, t, c)\\right\\|^2\\right]

        ``condition_dropout_prob`` replaces selected labels with ``-1``.  The
        conditional U-Net scaffold maps that value to its learned null token,
        providing the unconditional examples required for CFG sampling.
        """

        if not 0.0 <= condition_dropout_prob <= 1.0:
            raise ValueError("condition_dropout_prob must lie in [0, 1]")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

        batch_size = x_start.shape[0]
        self._validate_class_labels(class_labels, batch_size)
        if timesteps is None:
            timesteps = self.sample_timesteps(
                batch_size,
                device=x_start.device,
                generator=generator,
            )
        self._validate_timesteps(timesteps, batch_size, device=x_start.device)

        if noise is None:
            noise = torch.randn(
                x_start.shape,
                dtype=x_start.dtype,
                device=x_start.device,
                generator=generator,
            )
        self._validate_matching_tensor(noise, x_start, "noise")

        effective_labels = (
            class_labels.to(device=x_start.device) if class_labels is not None else None
        )
        if class_labels is not None and condition_dropout_prob > 0.0:
            drop_condition = (
                torch.rand(
                    (batch_size,),
                    device=x_start.device,
                    generator=generator,
                )
                < condition_dropout_prob
            )
            effective_labels = effective_labels.clone()
            effective_labels[drop_condition] = -1

        x_t = self.q_sample(x_start, timesteps, noise=noise)
        predicted_noise = model(
            x_t,
            timesteps,
            class_labels=effective_labels,
            **dict(model_kwargs or {}),
        )
        if predicted_noise.shape != noise.shape:
            raise ValueError("model must predict a noise tensor with the same shape as x_t")
        return F.mse_loss(predicted_noise, noise, reduction=reduction)
