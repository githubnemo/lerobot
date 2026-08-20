"""Backbone seam, official Cosmos rollout, temporal alignment, and metrics.

The sampler below is the pinned mimic-video Rectified-Flow AB2 sampler at
commit e3355dbc93132b576c02f920a59b4fc18a4f5906.  The model-specific backend
reuses ``CosmosPredict2Extractor``'s preprocessing and latent-prefix helpers;
this module does not alter the frozen extractor contract.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional

UPSTREAM_REPOSITORY = "https://github.com/mimic-video/mimic-video"
UPSTREAM_COMMIT = "e3355dbc93132b576c02f920a59b4fc18a4f5906"
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VideoBackboneSpec:
    """Geometry and artifacts needed by one video2world backend."""

    name: str
    model_size: str
    checkpoint_path: Path
    tokenizer_path: Path
    device: str
    dtype: str | torch.dtype
    input_frames: int
    input_height: int
    input_width: int
    fps: int
    latent_channels: int
    latent_conditional_frames: int
    state_t: int
    latent_height: int
    latent_width: int
    token_height: int
    token_width: int
    feature_dim: int
    temporal_compression_factor: int
    pixel_chunk_duration: int
    feature_layer: int
    prompt_tokens: int
    prompt_dim: int
    sigma_data: float
    sigma_conditional: float
    guidance: float
    sampling_steps: int
    sigma_min: float
    sigma_max: float
    schedule_order: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path).expanduser())
        object.__setattr__(self, "tokenizer_path", Path(self.tokenizer_path).expanduser())
        for name in (
            "input_frames",
            "input_height",
            "input_width",
            "fps",
            "latent_channels",
            "latent_conditional_frames",
            "state_t",
            "latent_height",
            "latent_width",
            "token_height",
            "token_width",
            "feature_dim",
            "temporal_compression_factor",
            "pixel_chunk_duration",
            "prompt_tokens",
            "prompt_dim",
            "sampling_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.latent_conditional_frames > self.state_t:
            raise ValueError("latent_conditional_frames cannot exceed state_t")
        if self.sigma_data <= 0 or self.sigma_conditional <= 0:
            raise ValueError("sigma_data and sigma_conditional must be positive")
        if not 0 < self.sigma_min < self.sigma_max:
            raise ValueError("sigma_min must be smaller than sigma_max")
        if self.schedule_order <= 0:
            raise ValueError("schedule_order must be positive")

    @property
    def pixel_frames(self) -> int:
        """Number of decoded frames represented by ``state_t`` latents."""
        return (self.state_t - 1) * self.temporal_compression_factor + 1

    @property
    def conditional_pixel_frames(self) -> int:
        """Number of RGB frames represented by the clean latent prefix."""
        return (self.latent_conditional_frames - 1) * self.temporal_compression_factor + 1

    @property
    def latent_chunk_duration(self) -> int:
        """VAE latent duration corresponding to its configured pixel chunk."""
        return 1 + (self.pixel_chunk_duration - 1) // self.temporal_compression_factor

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe backbone provenance."""
        return {
            "name": self.name,
            "model_size": self.model_size,
            "checkpoint_path": str(self.checkpoint_path),
            "tokenizer_path": str(self.tokenizer_path),
            "device": self.device,
            "dtype": str(self.dtype),
            "input_frames": self.input_frames,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "fps": self.fps,
            "latent_channels": self.latent_channels,
            "latent_conditional_frames": self.latent_conditional_frames,
            "state_t": self.state_t,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "token_height": self.token_height,
            "token_width": self.token_width,
            "feature_dim": self.feature_dim,
            "temporal_compression_factor": self.temporal_compression_factor,
            "pixel_chunk_duration": self.pixel_chunk_duration,
            "latent_chunk_duration": self.latent_chunk_duration,
            "pixel_frames": self.pixel_frames,
            "conditional_pixel_frames": self.conditional_pixel_frames,
            "feature_layer": self.feature_layer,
            "prompt_tokens": self.prompt_tokens,
            "prompt_dim": self.prompt_dim,
        }


def cosmos_2b_backbone_spec(
    checkpoint_path: Path,
    tokenizer_path: Path,
    *,
    device: str = "cuda",
    dtype: str | torch.dtype = "bfloat16",
    guidance: float = 7.0,
    sampling_steps: int = 35,
) -> VideoBackboneSpec:
    """Create the pinned 2B/480p/10fps spec behind the generic seam."""
    return VideoBackboneSpec(
        name="cosmos_predict2_video2world",
        model_size="2B",
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
        dtype=dtype,
        input_frames=5,
        input_height=480,
        input_width=640,
        fps=10,
        latent_channels=16,
        latent_conditional_frames=2,
        state_t=16,
        latent_height=60,
        latent_width=80,
        token_height=30,
        token_width=40,
        feature_dim=2048,
        temporal_compression_factor=4,
        pixel_chunk_duration=81,
        feature_layer=20,
        prompt_tokens=512,
        prompt_dim=1024,
        sigma_data=1.0,
        sigma_conditional=0.0001,
        guidance=guidance,
        sampling_steps=sampling_steps,
        sigma_min=0.002,
        sigma_max=80.0,
        schedule_order=7.0,
    )


@dataclass(frozen=True, slots=True)
class TemporalAlignment:
    """Absolute frame mapping for one decoded video window."""

    conditioning_indices: tuple[int, ...]
    output_indices: tuple[int, ...]
    predicted_future_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditioning_indices": list(self.conditioning_indices),
            "output_indices": list(self.output_indices),
            "predicted_future_indices": list(self.predicted_future_indices),
            "conditioning_output_count": len(self.conditioning_indices),
            "predicted_future_count": len(self.predicted_future_indices),
        }


def temporal_alignment(
    *,
    current_frame_index: int,
    input_frames: int,
    state_t: int,
    latent_conditional_frames: int,
    temporal_compression_factor: int,
) -> TemporalAlignment:
    """Map decoded pixel frames to dataset indices for a causal prefix.

    A causal VAE's first latent frame is pixel frame zero and each later latent
    frame advances by the temporal compression factor.  The five RGB inputs
    therefore become latent frames at offsets 0 and 4; decoded output frame 5
    is the first genuinely predicted frame.
    """
    if input_frames != 1 + (latent_conditional_frames - 1) * temporal_compression_factor:
        raise ValueError(
            "input_frames must equal the RGB duration of the latent prefix: "
            f"got input_frames={input_frames}, latent_conditional_frames={latent_conditional_frames}, "
            f"temporal_compression_factor={temporal_compression_factor}"
        )
    output_count = (state_t - 1) * temporal_compression_factor + 1
    origin = current_frame_index - input_frames + 1
    output_indices = tuple(origin + offset for offset in range(output_count))
    conditioning_indices = output_indices[:input_frames]
    predicted_future_indices = output_indices[input_frames:]
    return TemporalAlignment(conditioning_indices, output_indices, predicted_future_indices)


class OfficialRectifiedFlowAB2Scheduler:
    """Pinned upstream Rectified-Flow Adams-Bashforth-2 scheduler.

    This is the minimal dependency-free copy of the official scheduler's
    numerical operations.  Its schedule and step equations match
    ``model/cosmos_predict2/schedulers/rectified_flow_scheduler.py`` at the
    pinned upstream commit above.
    """

    def __init__(self, sigma_min: float, sigma_max: float, order: float, steps: int) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.order = order
        self.steps = steps
        self.sigmas: Tensor | None = None
        self.timesteps: Tensor | None = None

    def set_timesteps(self, device: torch.device) -> Tensor:
        n_sigma = self.steps + 1
        i = torch.arange(n_sigma, device=device, dtype=torch.float64)
        ramp = self.sigma_max ** (1 / self.order) + i / (n_sigma - 1) * (
            self.sigma_min ** (1 / self.order) - self.sigma_max ** (1 / self.order)
        )
        self.sigmas = ramp**self.order
        self.timesteps = torch.arange(self.steps, device=device, dtype=torch.long)
        return self.timesteps

    @staticmethod
    def _phi1(t: Tensor) -> Tensor:
        input_dtype = t.dtype
        t64 = t.to(torch.float64)
        return (torch.expm1(t64) / t64).to(input_dtype)

    @classmethod
    def _phi2(cls, t: Tensor) -> Tensor:
        input_dtype = t.dtype
        t64 = t.to(torch.float64)
        return ((cls._phi1(t64) - 1.0) / t64).to(input_dtype)

    @staticmethod
    def _batch_mul(coefficient: Tensor, value: Tensor) -> Tensor:
        return coefficient.reshape((coefficient.shape[0],) + (1,) * (value.ndim - 1)) * value

    @classmethod
    def _res_x0_rk2_step(
        cls,
        x_s: Tensor,
        t: Tensor,
        s: Tensor,
        x0_s: Tensor,
        s1: Tensor,
        x0_s1: Tensor,
    ) -> Tensor:
        s = -torch.log(s)
        t = -torch.log(t)
        m = -torch.log(s1)
        dt = t - s
        if torch.any(torch.isclose(dt, torch.zeros_like(dt), atol=1e-6)):
            raise ValueError("Rectified-Flow AB2 step size is too small")
        if torch.any(torch.isclose(m - s, torch.zeros_like(dt), atol=1e-6)):
            raise ValueError("Rectified-Flow AB2 intermediate step size is too small")
        c2 = (m - s) / dt
        phi1_val = cls._phi1(-dt)
        phi2_val = cls._phi2(-dt)
        b1 = torch.nan_to_num(phi1_val - 1.0 / c2 * phi2_val, nan=0.0)
        b2 = torch.nan_to_num(1.0 / c2 * phi2_val, nan=0.0)
        return cls._batch_mul(torch.exp(-dt), x_s) + cls._batch_mul(
            dt, cls._batch_mul(b1, x0_s) + cls._batch_mul(b2, x0_s1)
        )

    def step(
        self, x0_pred: Tensor, index: int, sample: Tensor, x0_prev: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Advance one official scheduler step."""
        if self.sigmas is None:
            raise RuntimeError("set_timesteps must be called before step")
        dtype_target = sample.dtype
        dtype_work = torch.float64
        x_t = sample.to(dtype_work)
        x0_t = x0_pred.to(dtype_work)
        sigma_t = self.sigmas[index]
        sigma_s = self.sigmas[index + 1]
        ones = torch.ones(x_t.shape[0], device=x_t.device, dtype=dtype_work)
        if x0_prev is None:
            x_next = self._batch_mul((sigma_t - sigma_s) / sigma_t * ones, x0_t) + self._batch_mul(
                sigma_s / sigma_t * ones, x_t
            )
        else:
            x_next = self._res_x0_rk2_step(
                x_t,
                sigma_s * ones,
                sigma_t * ones,
                x0_t,
                self.sigmas[index - 1] * ones,
                x0_prev,
            )
        return x_next.to(dtype_target), x0_t.to(dtype_target)


@dataclass(frozen=True, slots=True)
class PreparedConditioning:
    conditional_latent: Tensor
    prompt_embedding: Tensor
    latent_conditional_frames: int


class CosmosVideo2WorldBackend:
    """Generic preview backend with the current Cosmos 2B implementation."""

    def __init__(self, spec: VideoBackboneSpec, *, seed: int = 0) -> None:
        if spec.model_size != "2B":
            raise NotImplementedError(f"Only model_size='2B' is implemented; received {spec.model_size!r}")
        from .cosmos_predict2_extractor import CosmosPredict2Extractor, CosmosPredict2ExtractorConfig

        self.spec = spec
        self.device = torch.device(spec.device)
        config = CosmosPredict2ExtractorConfig(
            checkpoint_path=spec.checkpoint_path,
            tokenizer_path=spec.tokenizer_path,
            device=spec.device,
            dtype=spec.dtype,
            hidden_layer=spec.feature_layer,
            seed=seed,
            input_frames=spec.input_frames,
            input_height=spec.input_height,
            input_width=spec.input_width,
            latent_channels=spec.latent_channels,
            latent_conditional_frames=spec.latent_conditional_frames,
            state_t=spec.state_t,
            latent_height=spec.latent_height,
            latent_width=spec.latent_width,
            token_height=spec.token_height,
            token_width=spec.token_width,
            hidden_dim=spec.feature_dim,
            sigma_data=spec.sigma_data,
            sigma_conditional=spec.sigma_conditional,
        )
        self.extractor = CosmosPredict2Extractor(config)
        self.dtype = self.extractor.dtype
        self._scaling = None

    def prepare_conditioning(
        self,
        rgb_history: Tensor,
        prompt_embedding: Tensor,
        *,
        num_latent_conditional_frames: int | None = None,
    ) -> PreparedConditioning:
        if rgb_history.ndim != 5 or rgb_history.shape[1] != 3 or rgb_history.shape[3:] != (480, 640):
            raise ValueError(
                f"rgb_history must have shape [B, 3, T, 480, 640], got {tuple(rgb_history.shape)}"
            )
        input_frames = rgb_history.shape[2]
        if input_frames not in (1, 5):
            raise ValueError(f"rgb_history must contain 1 or 5 frames, got {input_frames}")
        if rgb_history.dtype == torch.uint8:
            images = rgb_history.to(device=self.device, dtype=self.dtype) / 127.5 - 1.0
        elif torch.is_floating_point(rgb_history):
            images = rgb_history.to(device=self.device, dtype=self.dtype)
            min_value, max_value = images.amin().item(), images.amax().item()
            if min_value >= 0.0 and max_value <= 1.0:
                images = images * 2.0 - 1.0
            elif not (min_value >= -1.0 and max_value <= 1.0):
                raise ValueError("floating-point images must be in [0, 1] or [-1, 1]")
        else:
            raise TypeError("rgb_history must be uint8 or floating point")
        padded_video = torch.zeros(
            images.shape[0], 3, self.spec.pixel_frames, 480, 640, device=self.device, dtype=self.dtype
        )
        padded_video[:, :, :input_frames] = images
        conditional_latent = self.extractor.tokenizer.encode(padded_video)
        expected_latent = (
            images.shape[0],
            self.spec.latent_channels,
            self.spec.state_t,
            self.spec.latent_height,
            self.spec.latent_width,
        )
        if tuple(conditional_latent.shape) != expected_latent:
            raise ValueError(
                f"tokenizer output must have shape {expected_latent}, got {tuple(conditional_latent.shape)}"
            )
        conditional_latent = conditional_latent.to(device=self.device, dtype=self.dtype)
        expected_prompt = (rgb_history.shape[0], self.spec.prompt_tokens, self.spec.prompt_dim)
        if tuple(prompt_embedding.shape) != expected_prompt:
            raise ValueError(
                f"prompt_embedding must have shape {expected_prompt}, got {tuple(prompt_embedding.shape)}"
            )
        if not torch.is_floating_point(prompt_embedding) or not torch.isfinite(prompt_embedding).all():
            raise ValueError("prompt_embedding must be finite floating point")
        conditional_frames = num_latent_conditional_frames or (1 if input_frames == 1 else 2)
        return PreparedConditioning(
            conditional_latent, prompt_embedding.to(self.device, self.dtype), conditional_frames
        )

    @torch.no_grad()
    def decode_latent(self, latent: Tensor) -> Tensor:
        """Decode a normalized Cosmos latent to ``[-1, 1]`` pixels."""
        decoded = self.extractor.tokenizer.decode(latent / self.spec.sigma_data)
        return decoded.clamp(-1.0, 1.0)

    def _denoise(self, sample: Tensor, sigma: Tensor, condition: PreparedConditioning) -> Tensor:
        from ._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
        from .cosmos_predict2_extractor import _backbone_autocast_context

        if self._scaling is None:
            self._scaling = RectifiedFlowScaling(sigma_data=self.spec.sigma_data, t_scaling_factor=1.0)
        batch_size = sample.shape[0]
        state_t = sample.shape[2]
        sigma_b_1_t_1_1 = sigma.reshape(batch_size, 1, 1, 1, 1).expand(batch_size, 1, state_t, 1, 1)
        c_skip, c_out, c_in, c_noise = self._scaling(sigma_b_1_t_1_1)
        network_input = sample * c_in.to(sample.dtype)
        condition_mask = torch.zeros(
            batch_size,
            1,
            state_t,
            self.spec.latent_height,
            self.spec.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask[:, :, : condition.latent_conditional_frames] = 1
        condition_state = condition.conditional_latent / self.spec.sigma_data
        network_input = condition_state.to(sample.dtype) * condition_mask.to(sample.dtype) + network_input * (
            1.0 - condition_mask.to(sample.dtype)
        )
        sigma_conditional = torch.full_like(sigma_b_1_t_1_1, self.spec.sigma_conditional)
        _, _, _, c_noise_conditional = self._scaling(sigma_conditional)
        condition_mask_b_1_t_1_1 = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
        c_noise = c_noise_conditional * condition_mask_b_1_t_1_1 + c_noise * (1.0 - condition_mask_b_1_t_1_1)
        padding_mask = torch.zeros(
            batch_size,
            1,
            self.spec.latent_height,
            self.spec.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        with _backbone_autocast_context(self.device, self.dtype):
            net_output = self.extractor.backbone(
                x_B_C_T_H_W=network_input.to(self.dtype),
                timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]),
                crossattn_emb=condition.prompt_embedding,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=torch.full(
                    (batch_size, 1), float(self.spec.fps), device=self.device, dtype=torch.float32
                ),
                padding_mask=padding_mask,
                data_type=self.extractor._data_type(),
                use_cuda_graphs=False,
            )
        if not isinstance(net_output, Tensor):
            raise TypeError(f"Cosmos backbone must return a tensor during rollout, got {type(net_output)}")
        x0_pred = c_skip.to(sample.dtype) * sample + c_out.to(sample.dtype) * net_output.float()
        return condition_state.to(sample.dtype) * condition_mask.to(sample.dtype) + x0_pred * (
            1.0 - condition_mask.to(sample.dtype)
        )

    @torch.no_grad()
    def rollout(self, condition: PreparedConditioning, *, seed: int) -> Tensor:
        """Run the complete official 35-step rollout and return decoded pixels."""
        scheduler = OfficialRectifiedFlowAB2Scheduler(
            sigma_min=self.spec.sigma_min,
            sigma_max=self.spec.sigma_max,
            order=self.spec.schedule_order,
            steps=self.spec.sampling_steps,
        )
        scheduler.set_timesteps(self.device)
        shape = (
            condition.conditional_latent.shape[0],
            self.spec.latent_channels,
            self.spec.state_t,
            self.spec.latent_height,
            self.spec.latent_width,
        )
        from .cosmos_predict2_extractor import arch_invariant_rand

        sample = arch_invariant_rand(shape, seed).to(self.device, torch.float32) * scheduler.sigmas[0]
        uncondition = PreparedConditioning(
            conditional_latent=condition.conditional_latent,
            prompt_embedding=torch.zeros_like(condition.prompt_embedding),
            latent_conditional_frames=condition.latent_conditional_frames,
        )
        x0_prev: Tensor | None = None
        assert scheduler.timesteps is not None
        for index in range(self.spec.sampling_steps):
            sigma = scheduler.sigmas[index].to(self.device, torch.float32).repeat(shape[0])
            conditional_x0 = self._denoise(sample, sigma, condition)
            if self.spec.guidance != 0.0:
                unconditioned_x0 = self._denoise(sample, sigma, uncondition)
                x0_pred = conditional_x0 + self.spec.guidance * (conditional_x0 - unconditioned_x0)
            else:
                x0_pred = conditional_x0
            sample, x0_prev = scheduler.step(x0_pred, index, sample, x0_prev)
        sigma_min = scheduler.sigmas[-1].to(self.device, torch.float32).repeat(shape[0])
        conditional_x0 = self._denoise(sample, sigma_min, condition)
        if self.spec.guidance != 0.0:
            unconditioned_x0 = self._denoise(sample, sigma_min, uncondition)
            samples = conditional_x0 + self.spec.guidance * (conditional_x0 - unconditioned_x0)
        else:
            samples = conditional_x0
        return self.decode_latent(samples)


def set_deterministic_seed(seed: int) -> None:
    """Set host and CUDA seeds used by the preview process."""
    if type(seed) is not int or seed < 0 or seed >= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 2]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> dict[str, Any]:
    """Hash a checkpoint in bounded memory for report provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _as_bchw_float(video: Tensor) -> Tensor:
    if video.ndim != 4:
        raise ValueError(f"video must have shape [T, C, H, W], got {tuple(video.shape)}")
    if video.shape[1] not in (1, 3, 4):
        raise ValueError(f"video channel dimension must be 1, 3, or 4, got {video.shape[1]}")
    video = video.float()
    if video.amax().item() > 1.0:
        video = video / 255.0
    if video.amin().item() < 0.0 or video.amax().item() > 1.0:
        raise ValueError("metric videos must be in [0, 1] or uint8")
    return video


def _ssim_frame(predicted: Tensor, target: Tensor) -> float:
    channels = predicted.shape[0]
    kernel_1d = torch.arange(11, dtype=torch.float32, device=predicted.device) - 5
    kernel_1d = torch.exp(-(kernel_1d**2) / (2 * 1.5**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).expand(channels, 1, 11, 11)
    pred = functional.pad(predicted.unsqueeze(0), (5, 5, 5, 5), mode="reflect")
    truth = functional.pad(target.unsqueeze(0), (5, 5, 5, 5), mode="reflect")
    mu_pred = functional.conv2d(pred, kernel, groups=channels)
    mu_truth = functional.conv2d(truth, kernel, groups=channels)
    mu_pred_sq = mu_pred.square()
    mu_truth_sq = mu_truth.square()
    sigma_pred = functional.conv2d(pred.square(), kernel, groups=channels) - mu_pred_sq
    sigma_truth = functional.conv2d(truth.square(), kernel, groups=channels) - mu_truth_sq
    sigma_cross = functional.conv2d(pred * truth, kernel, groups=channels) - mu_pred * mu_truth
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_pred * mu_truth + c1) * (2 * sigma_cross + c2)) / (
        (mu_pred_sq + mu_truth_sq + c1) * (sigma_pred + sigma_truth + c2)
    )
    return float(score.mean().clamp(-1.0, 1.0).item())


def compute_frame_metrics(
    predicted: Tensor,
    target: Tensor,
    *,
    frame_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compute deterministic per-frame PSNR and Gaussian-window SSIM."""
    predicted = _as_bchw_float(predicted)
    target = _as_bchw_float(target)
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted and target shapes differ: {tuple(predicted.shape)} vs {tuple(target.shape)}"
        )
    if frame_indices is None:
        frame_indices = tuple(range(predicted.shape[0]))
    if len(frame_indices) != predicted.shape[0]:
        raise ValueError("frame_indices length must match video length")
    rows: list[dict[str, Any]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    for position, frame_index in enumerate(frame_indices):
        mse = float((predicted[position] - target[position]).square().mean().item())
        psnr = None if mse == 0.0 else -10.0 * math.log10(mse)
        ssim = _ssim_frame(predicted[position], target[position])
        rows.append({"position": position, "frame_index": int(frame_index), "psnr_db": psnr, "ssim": ssim})
        if psnr is not None:
            psnr_values.append(psnr)
        ssim_values.append(ssim)
    return {
        "per_frame": rows,
        "mean_psnr_db": None if not psnr_values else float(sum(psnr_values) / len(psnr_values)),
        "mean_ssim": None if not ssim_values else float(sum(ssim_values) / len(ssim_values)),
    }


def compute_baseline_metrics(
    conditioning: Tensor,
    target: Tensor,
    *,
    frame_indices: Sequence[int],
) -> dict[str, Any]:
    """Score static, conditioning-mean, and ground-truth-copy baselines."""
    conditioning = _as_bchw_float(conditioning)
    target = _as_bchw_float(target)
    if conditioning.shape[1:] != target.shape[1:]:
        raise ValueError("conditioning and target frame shapes must match")
    static = conditioning[-1:].expand(target.shape[0], -1, -1, -1)
    mean_frame = conditioning.mean(dim=0, keepdim=True).expand(target.shape[0], -1, -1, -1)
    return {
        "static": compute_frame_metrics(static, target, frame_indices=frame_indices),
        "conditioning_mean": compute_frame_metrics(mean_frame, target, frame_indices=frame_indices),
        "ground_truth_copy": compute_frame_metrics(target, target, frame_indices=frame_indices),
    }


def compute_optional_lpips(predicted: Tensor, target: Tensor) -> dict[str, Any]:
    """Use an already-installed LPIPS package, without installing or downloading anything."""
    if importlib.util.find_spec("lpips") is None:
        return {"status": "skipped", "reason": "lpips is not installed in the existing environment"}
    try:
        import lpips  # type: ignore[import-not-found]

        model = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().to(predicted.device)
        with torch.no_grad():
            values = model(predicted.float() * 2 - 1, target.float() * 2 - 1).flatten().tolist()
        values = [float(value) for value in values]
        return {"status": "computed", "per_frame": values, "mean": float(sum(values) / len(values))}
    except Exception as exc:  # noqa: BLE001
        return {"status": "skipped", "reason": f"installed LPIPS could not initialize: {exc}"}


REPORT_KEYS = frozenset(
    {
        "schema_version",
        "upstream",
        "seed",
        "backbone",
        "checkpoint",
        "sampler",
        "resolution",
        "fps",
        "window",
        "temporal_alignment",
        "metrics",
        "runtime",
        "artifacts",
    }
)


def validate_report_schema(report: Mapping[str, Any]) -> None:
    """Reject malformed or non-strict report payloads before JSON serialization."""
    if set(report) != REPORT_KEYS:
        raise ValueError(f"report keys must be exactly {sorted(REPORT_KEYS)}, got {sorted(report)}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported report schema version")
    if not isinstance(report["seed"], int) or report["seed"] < 0:
        raise ValueError("report seed must be a non-negative integer")
    window = report["window"]
    if not isinstance(window, Mapping) or set(window) != {
        "episode_index",
        "current_frame_index",
        "conditioning_frame_indices",
        "ground_truth_future_frame_indices",
    }:
        raise ValueError("report window does not match the strict schema")
    alignment = report["temporal_alignment"]
    if not isinstance(alignment, Mapping) or not {
        "conditioning_indices",
        "output_indices",
        "predicted_future_indices",
    }.issubset(alignment):
        raise ValueError("report temporal_alignment is incomplete")
    metrics = report["metrics"]
    if not isinstance(metrics, Mapping) or not {"per_frame", "mean_psnr_db", "mean_ssim", "lpips"}.issubset(
        metrics
    ):
        raise ValueError("report metrics are incomplete")
    json.dumps(report, allow_nan=False)


def build_prediction_report(
    *,
    spec: VideoBackboneSpec,
    seed: int,
    checkpoint: Mapping[str, Any],
    episode_index: int,
    current_frame_index: int,
    alignment: TemporalAlignment,
    sampler: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    alignment_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate the stable report object."""
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        "seed": seed,
        "backbone": spec.to_dict(),
        "checkpoint": dict(checkpoint),
        "sampler": dict(sampler),
        "resolution": {"height": spec.input_height, "width": spec.input_width},
        "fps": spec.fps,
        "window": {
            "episode_index": episode_index,
            "current_frame_index": current_frame_index,
            "conditioning_frame_indices": list(alignment.conditioning_indices),
            "ground_truth_future_frame_indices": list(alignment.predicted_future_indices),
        },
        "temporal_alignment": {**alignment.to_dict(), "probe": dict(alignment_probe)},
        "metrics": dict(metrics),
        "runtime": dict(runtime),
        "artifacts": dict(artifacts),
    }
    validate_report_schema(report)
    return report


def write_strict_json(report: Mapping[str, Any], path: Path) -> None:
    """Validate and write a strict JSON report."""
    validate_report_schema(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
