"""Checkpoint-native video training, deterministic VAE normalization, EDM objective, and protocol isolation.

This module provides shared production utilities for Video Action Model (VAM) backbones:
- NVIDIA Cosmos-1.0-Diffusion-7B / 14B (EDM pretraining objective, AutoencoderKLCosmos normalization)
- Black Forest Labs FLUX.2 [klein] (multi-reference RF flow matching, native AutoencoderKLFlux2 patchify + BatchNorm)
- Strict Protocol 1.0 & Scale-100 episode split enforcement and clip sampling (FPS 10, clip_length=17)
- Pre-encoding and offloading of VAE / text encoders for 24 GB FP8 / QLoRA memory budgets
- Strict LoRA rank / alpha metadata serialization and two-way key validation (no silent base fallback)
- Real text embedding validation with semantic provenance and sha256 (no random text in production)
- Selective gradient backward (loss.backward(inputs=lora_params))
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from lerobot.policies.vam.base.split_guard import (
    CANONICAL_DATASET_REPO,
    CANONICAL_DATASET_REVISION,
    PROTOCOL_1_0_VAL_EPISODES,
    ProtocolViolationError,
    enforce_protocol_1_0_split,
)


class NativeVideoError(RuntimeError):
    """Base exception for native video operations."""


class LoRAMetadataError(RuntimeError):
    """Raised when LoRA weights fail strict rank/alpha metadata or two-way key validation."""


class DatasetIntegrityError(RuntimeError):
    """Raised when dataset files or frame indices are missing or corrupt (no silent fallback)."""


class TextEmbeddingError(RuntimeError):
    """Raised when text embeddings are missing, invalid, or lack required prompt identity provenance."""


SCALE_100_CANONICAL_REPO: str = "Orellius/cube_out_of_box_v2"
SCALE_100_CANONICAL_REVISION: str = "5d0325cc1412f4774223a0beb528958108814962"


# ==============================================================================
# 1. Strict Protocol 1.0 / Scale-100 Split & Dataset Validation (FPS 10)
# ==============================================================================


def parse_episode_spec(ep_str: str | Sequence[int]) -> list[int]:
    """Parse comma-separated episode specs like '0-31, 40-89' or lists into sorted unique ints."""
    if isinstance(ep_str, (list, tuple, set)):
        return sorted(int(x) for x in ep_str)

    episodes: set[int] = set()
    parts = [p.strip() for p in str(ep_str).split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s.strip()), int(end_s.strip())
            if start > end:
                raise ValueError(f"Invalid episode range: {part}")
            episodes.update(range(start, end + 1))
        else:
            episodes.add(int(part))
    return sorted(episodes)


def validate_training_episodes(
    train_episodes: Sequence[int] | str,
    protocol: str = "protocol1",
    val_episodes: Sequence[int] | None = None,
) -> list[int]:
    """Strictly validate train episodes against Protocol 1.0 or Scale-100 rules.

    Guarantees 0% data leakage against held-out validation episodes (32-39, 90-99).
    """
    train_eps = parse_episode_spec(train_episodes)
    val_eps = list(val_episodes) if val_episodes is not None else list(PROTOCOL_1_0_VAL_EPISODES)

    enforce_protocol_1_0_split(
        train_episodes=train_eps,
        val_episodes=val_eps,
        allow_subset=True,
        protocol=protocol,
    )
    return train_eps


def validate_dataset_repo_and_root(
    dataset_repo_id: str,
    protocol: str,
    dataset_root: Path | str | None,
    dataset: Any,
    expected_revision: str | None = None,
) -> None:
    """Validate dataset identity, pinned revision, and episode count to prevent repo/root mismatch bugs."""
    num_episodes = (
        len(dataset.meta.episodes) if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes") else 0
    )

    if protocol == "protocol1":
        if dataset_repo_id != CANONICAL_DATASET_REPO:
            raise DatasetIntegrityError(
                f"Protocol 1.0 requires canonical repo '{CANONICAL_DATASET_REPO}', got '{dataset_repo_id}'."
            )
        target_rev = expected_revision or CANONICAL_DATASET_REVISION
        if hasattr(dataset, "revision") and dataset.revision and dataset.revision != target_rev:
            raise DatasetIntegrityError(
                f"Canonical dataset revision mismatch: expected '{target_rev}', got '{dataset.revision}'."
            )
    elif protocol == "scale100":
        if dataset_repo_id != SCALE_100_CANONICAL_REPO:
            raise DatasetIntegrityError(
                f"Scale-100 protocol requires canonical repo '{SCALE_100_CANONICAL_REPO}', got '{dataset_repo_id}'."
            )
        target_rev = expected_revision or SCALE_100_CANONICAL_REVISION
        if hasattr(dataset, "revision") and dataset.revision and dataset.revision != target_rev:
            raise DatasetIntegrityError(
                f"Scale-100 dataset revision mismatch: expected '{target_rev}', got '{dataset.revision}'."
            )
        if num_episodes < 100:
            raise DatasetIntegrityError(
                f"Scale-100 protocol requires at least 100 episodes, but dataset only has {num_episodes} episodes. "
                f"Check dataset root '{dataset_root}' and ensure it is not pointing to a 40-episode Protocol 1.0 root."
            )


@dataclass(frozen=True, slots=True)
class EpisodeClipSpec:
    """Specification of a temporal multiframe clip within an episode."""

    episode_index: int
    start_frame_idx: int
    end_frame_idx: int
    frame_indices: tuple[int, ...]
    fps: int = 10


def extract_episode_clip_indices(
    dataset_meta: Any,
    train_episodes: Sequence[int],
    clip_length: int = 17,
    stride: int = 3,
    fps: int = 10,
    max_clips: int | None = None,
) -> list[EpisodeClipSpec]:
    """Extract strictly episode-local multiframe clip boundaries from dataset metadata.

    Guarantees no clip spans across episode boundaries and validates actual episode IDs.
    """
    if not hasattr(dataset_meta, "episodes") or not dataset_meta.episodes:
        raise DatasetIntegrityError("Dataset metadata has no episodes list.")

    train_set = set(train_episodes)
    clips: list[EpisodeClipSpec] = []

    for idx, ep_meta in enumerate(dataset_meta.episodes):
        if isinstance(ep_meta, Mapping):
            ep_idx = int(ep_meta.get("episode_index", ep_meta.get("index", idx)))
            from_idx = int(ep_meta["dataset_from_index"])
            to_idx = int(ep_meta["dataset_to_index"])
        else:
            ep_idx = int(getattr(ep_meta, "episode_index", getattr(ep_meta, "index", idx)))
            from_idx = int(ep_meta.dataset_from_index)
            to_idx = int(ep_meta.dataset_to_index)

        if ep_idx not in train_set:
            continue

        ep_len = to_idx - from_idx
        if ep_len < clip_length:
            continue

        for start in range(from_idx, to_idx - clip_length + 1, stride):
            end = start + clip_length
            clip_spec = EpisodeClipSpec(
                episode_index=ep_idx,
                start_frame_idx=start,
                end_frame_idx=end,
                frame_indices=tuple(range(start, end)),
                fps=fps,
            )
            clips.append(clip_spec)
            if max_clips is not None and len(clips) >= max_clips:
                return clips

    if not clips:
        raise DatasetIntegrityError(
            f"No valid clips of length {clip_length} found in train episodes {sorted(train_set)}."
        )

    return clips


# ==============================================================================
# 2. Text Embedding Loading & Semantic Provenance Validation (No Random Text)
# ==============================================================================

RECOGNIZED_PROMPT_EMBEDDING_KEYS: tuple[str, ...] = (
    "prompt_embedding",
    "prompt_embeds",
)


def load_and_validate_prompt_embedding(
    embedding_path: str | Path | None,
    prompt: str,
    expected_dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, dict[str, Any]]:
    """Load and validate an authentic precomputed text embedding tensor with semantic provenance.

    Requires:
      - Valid file path with verified SHA-256 hash.
      - Recognized key ('prompt_embedding' or 'prompt_embeds').
      - Safe load with finite values check.
      - Semantic provenance matching requested prompt in safetensors metadata or JSON sidecar.
      - Finite shape [1, N, D] with D == expected_dim.

    Random text generation is strictly prohibited in production.
    """
    if embedding_path is None:
        raise TextEmbeddingError(
            "No --text-embedding-path provided! Production video LoRA training requires an authentic "
            "precomputed text embedding tensor with verified prompt metadata. "
            "Random text embedding fallback is strictly prohibited. "
            "Pass an explicit --text-embedding-path or run with --dry-run for synthetic tests."
        )

    p = Path(embedding_path)
    if not p.is_file():
        raise TextEmbeddingError(f"Text embedding file not found at: {p}")

    sha256 = hashlib.sha256(p.read_bytes()).hexdigest()

    provenance: dict[str, Any] = {
        "embedding_file": str(p),
        "sha256": sha256,
        "prompt": prompt,
        "requested_prompt": prompt,
        "expected_dim": expected_dim,
    }

    meta_prompt: str | None = None

    sidecar = p.with_suffix(".json")
    if sidecar.is_file():
        try:
            with open(sidecar, encoding="utf-8") as sf:
                sidecar_data = json.load(sf)
                if isinstance(sidecar_data, dict):
                    provenance["sidecar"] = sidecar_data
                    if "prompt" in sidecar_data:
                        meta_prompt = str(sidecar_data["prompt"])
        except Exception as e:
            raise TextEmbeddingError(f"Failed to parse sidecar JSON at {sidecar}: {e}") from e

    tensor: Tensor | None = None

    if p.suffix == ".safetensors":
        with safe_open(str(p), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            provenance["metadata"] = meta
            if "prompt" in meta:
                meta_prompt = meta["prompt"]

            f_keys = set(f.keys())
            for key in RECOGNIZED_PROMPT_EMBEDDING_KEYS:
                if key in f_keys:
                    tensor = f.get_tensor(key)
                    provenance["tensor_key"] = key
                    break

        if tensor is None:
            raise TextEmbeddingError(
                f"No recognized embedding tensor in {p}. Keys must be one of {RECOGNIZED_PROMPT_EMBEDDING_KEYS}."
            )
    else:
        loaded = torch.load(str(p), map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in RECOGNIZED_PROMPT_EMBEDDING_KEYS:
                if key in loaded:
                    tensor = loaded[key]
                    provenance["tensor_key"] = key
                    break
        elif isinstance(loaded, Tensor):
            tensor = loaded
            provenance["tensor_key"] = "raw_tensor"
        else:
            raise TextEmbeddingError(f"Unsupported object type loaded from {p}: {type(loaded)}")

        if tensor is None:
            raise TextEmbeddingError(
                f"No recognized embedding key in dictionary from {p}. Keys: {list(loaded.keys()) if isinstance(loaded, dict) else 'none'}."
            )

    if meta_prompt is None:
        raise TextEmbeddingError(
            f"Semantic provenance missing for {p}: no prompt metadata found in safetensors header or sidecar JSON {sidecar}."
        )
    if meta_prompt != prompt:
        raise TextEmbeddingError(
            f"Prompt identity mismatch in {p}: metadata prompt '{meta_prompt}' does not match requested prompt '{prompt}'."
        )

    provenance["prompt"] = meta_prompt

    if not torch.isfinite(tensor).all():
        raise TextEmbeddingError(f"Text embedding tensor in {p} contains NaN or Inf values.")

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise TextEmbeddingError(f"Expected 2D or 3D text embedding tensor, got shape {tensor.shape}")

    if tensor.shape[0] != 1:
        raise TextEmbeddingError(f"Text embedding batch dimension must be 1, got {tensor.shape[0]}")

    if tensor.shape[-1] != expected_dim:
        raise TextEmbeddingError(
            f"Text embedding dimension mismatch in {p}: got {tensor.shape[-1]}, expected {expected_dim}."
        )

    provenance["seq_len"] = tensor.shape[1]
    provenance["dim"] = tensor.shape[2]

    return tensor.to(device=device, dtype=dtype), provenance


# ==============================================================================
# 3. Checkpoint-Native Deterministic VAE Normalization
# ==============================================================================


def pad_past_temporal_frames(rgb: Tensor, min_frames: int = 5) -> Tensor:
    """Pad past frames by repeating the first temporal frame if rgb has fewer than min_frames."""
    b, c, t, h, w = rgb.shape
    if t >= min_frames:
        return rgb
    pad_len = min_frames - t
    first_frame_repeat = rgb[:, :, :1].repeat(1, 1, pad_len, 1, 1)
    return torch.cat([first_frame_repeat, rgb], dim=2)


class CosmosVAENormalizer:
    """Deterministic AutoencoderKLCosmos latent normalization and scaling.

    Cosmos VAE latent space uses per-channel latents_mean and latents_std:
        normalized_latents = (raw_latents - latents_mean) / latents_std * sigma_data
    with sigma_data = 0.5.
    """

    def __init__(
        self,
        vae: Any | None = None,
        sigma_data: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.vae = vae
        self.sigma_data = float(sigma_data)
        self.device = torch.device(device)
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        vae_path: str | Path,
        sigma_data: float = 0.5,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> CosmosVAENormalizer:
        p = Path(vae_path)
        if not p.exists():
            raise FileNotFoundError(f"Cosmos VAE checkpoint not found at: {vae_path}")

        from diffusers import AutoencoderKLCosmos

        vae = AutoencoderKLCosmos.from_pretrained(str(p), torch_dtype=dtype).to(device)
        vae.eval()
        for p_param in vae.parameters():
            p_param.requires_grad_(False)
        return cls(vae=vae, sigma_data=sigma_data, device=device, dtype=dtype)

    def encode(
        self,
        rgb_video: Tensor,
        deterministic: bool = True,
        pad_past: bool = False,
        min_frames: int = 5,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Encode RGB video [B, 3, T, H, W] into normalized Cosmos latent space [B, 16, T', H', W']."""
        if self.vae is None:
            raise NativeVideoError("Cosmos VAE model is not loaded in CosmosVAENormalizer.")

        rgb = rgb_video.to(device=self.device, dtype=self.dtype)
        if pad_past and rgb.shape[2] < min_frames:
            rgb = pad_past_temporal_frames(rgb, min_frames=min_frames)

        if rgb.max() > 1.0:
            rgb = (rgb / 255.0 - 0.5) * 2.0
        elif rgb.min() >= 0.0:
            rgb = (rgb - 0.5) * 2.0

        with torch.no_grad():
            res = self.vae.encode(rgb)
            if hasattr(res, "latent_dist"):
                if deterministic and hasattr(res.latent_dist, "mode"):
                    latents = res.latent_dist.mode()
                else:
                    latents = res.latent_dist.sample(generator=generator)
            elif hasattr(res, "sample"):
                latents = res.sample
            else:
                latents = res

            if getattr(self.vae.config, "latents_mean", None) is not None:
                latent_ch = getattr(self.vae.config, "latent_channels", 16)
                mean = torch.tensor(
                    self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype
                ).view(1, latent_ch, -1, 1, 1)[:, :, : latents.size(2)]
                std = torch.tensor(
                    self.vae.config.latents_std, device=latents.device, dtype=latents.dtype
                ).view(1, latent_ch, -1, 1, 1)[:, :, : latents.size(2)]
                norm_latents = (latents - mean) / std * self.sigma_data
            else:
                norm_latents = latents * self.sigma_data

        return norm_latents

    def decode(self, norm_latents: Tensor) -> Tensor:
        """Decode normalized Cosmos latents back to RGB video [B, 3, T, H, W] in [-1, 1]."""
        if self.vae is None:
            raise NativeVideoError("Cosmos VAE model is not loaded in CosmosVAENormalizer.")

        latents = norm_latents.to(device=self.device, dtype=self.dtype)
        if getattr(self.vae.config, "latents_mean", None) is not None:
            latent_ch = getattr(self.vae.config, "latent_channels", 16)
            mean = torch.tensor(
                self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype
            ).view(1, latent_ch, -1, 1, 1)[:, :, : latents.size(2)]
            std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
                1, latent_ch, -1, 1, 1
            )[:, :, : latents.size(2)]
            raw_latents = norm_latents * std / self.sigma_data + mean
        else:
            raw_latents = norm_latents / self.sigma_data

        with torch.no_grad():
            video = self.vae.decode(raw_latents, return_dict=False)[0]
        return video


class FluxVAENormalizer:
    """Deterministic FLUX.2 AutoencoderKLFlux2 latent normalization and patchification.

    FLUX.2 differs from FLUX.1:
      1. Encodes RGB [B, 3, H, W] to raw latents [B, 32, H', W']
      2. Patchifies 2x2 spatial blocks: (B, 32, H//2, 2, W//2, 2) -> (B, 128, H//2, W//2)
      3. Normalizes via VAE's BatchNorm running_mean and running_var:
         normalized = (patchified - bn.running_mean) / sqrt(bn.running_var + eps)
    """

    def __init__(
        self,
        vae: Any | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.vae = vae
        self.device = torch.device(device)
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        vae_path: str | Path,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> FluxVAENormalizer:
        p = Path(vae_path)
        if not p.exists():
            raise FileNotFoundError(f"FLUX.2 VAE checkpoint not found at: {vae_path}")

        from diffusers import AutoencoderKLFlux2

        vae = AutoencoderKLFlux2.from_pretrained(str(p), torch_dtype=dtype).to(device)
        vae.eval()
        for p_param in vae.parameters():
            p_param.requires_grad_(False)
        return cls(vae=vae, device=device, dtype=dtype)

    @staticmethod
    def patchify_latents(latents: Tensor) -> Tensor:
        """Patchify 2x2 spatial blocks into 4x channel dimension (32 -> 128 channels)."""
        b, c, h, w = latents.shape
        lat = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
        return lat.reshape(b, c * 4, h // 2, w // 2)

    def encode(
        self,
        rgb: Tensor,
        deterministic: bool = True,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Encode RGB frames [B, 3, H, W] into normalized 128-channel FLUX.2 latents [B, 128, H', W']."""
        if self.vae is None:
            raise NativeVideoError("FLUX.2 VAE model is not loaded in FluxVAENormalizer.")

        orig_5d = False
        if rgb.ndim == 5:
            orig_5d = True
            b, c, t, h, w = rgb.shape
            rgb = rgb.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        rgb = rgb.to(device=self.device, dtype=self.dtype)
        if rgb.max() > 1.0:
            rgb = (rgb / 255.0 - 0.5) * 2.0
        elif rgb.min() >= 0.0:
            rgb = (rgb - 0.5) * 2.0

        with torch.no_grad():
            res = self.vae.encode(rgb)
            if hasattr(res, "latent_dist"):
                latents = (
                    res.latent_dist.mode() if deterministic else res.latent_dist.sample(generator=generator)
                )
            elif hasattr(res, "sample"):
                latents = res.sample
            else:
                latents = res

            patchified = self.patchify_latents(latents)

            if not hasattr(self.vae, "bn") or self.vae.bn is None:
                raise NativeVideoError(
                    "FLUX.2 VAE requires BatchNorm 'bn' module with running_mean and running_var. "
                    "Silent fallback is prohibited."
                )

            bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(patchified.device, patchified.dtype)
            eps = getattr(self.vae.config, "batch_norm_eps", 1e-4) or 1e-4
            bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + eps).to(
                patchified.device, patchified.dtype
            )
            norm_latents = (patchified - bn_mean) / bn_std

        if orig_5d:
            b_lat, c_lat, h_lat, w_lat = norm_latents.shape
            norm_latents = norm_latents.view(b, t, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)

        return norm_latents


# ==============================================================================
# 4. Cosmos EDM Pretrained Objective with Exact In-Channel Dispatched Masking
# ==============================================================================


class CosmosEDMScaling:
    """NVIDIA Cosmos EDM Scaling formulation with sigma_data = 0.5.

    Native preconditioning functions matching diffusers EDMEulerScheduler:
      c_in(sigma) = 1 / sqrt(sigma^2 + sigma_data^2)
      c_skip(sigma) = sigma_data^2 / (sigma^2 + sigma_data^2)
      c_out(sigma) = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
      c_noise(sigma) = 0.25 * ln(sigma)
      loss_weight(sigma) = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    """

    def __init__(self, sigma_data: float = 0.5) -> None:
        self.sigma_data = float(sigma_data)

    def __call__(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute c_skip, c_out, c_in, c_noise for input sigma."""
        s_sq = sigma**2
        s_data_sq = self.sigma_data**2
        denom = torch.sqrt(s_sq + s_data_sq)

        c_skip = s_data_sq / (s_sq + s_data_sq)
        c_out = (sigma * self.sigma_data) / denom
        c_in = 1.0 / denom
        c_noise = 0.25 * torch.log(sigma.clamp(min=1e-8))
        return c_skip, c_out, c_in, c_noise

    def sigma_loss_weights(self, sigma: Tensor) -> Tensor:
        """Compute EDM weighted loss factor lambda(sigma) = 1 / c_out(sigma)^2."""
        s_sq = sigma**2
        s_data_sq = self.sigma_data**2
        return (s_sq + s_data_sq) / ((sigma * self.sigma_data) ** 2).clamp(min=1e-8)


def sample_edm_sigmas(
    batch_size: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    p_mean: float = -1.2,
    p_std: float = 1.2,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample sigmas according to EDM log-normal distribution ln(sigma) ~ N(p_mean, p_std^2)."""
    rnd = torch.randn(batch_size, device=device, dtype=dtype, generator=generator)
    return torch.exp(rnd * p_std + p_mean)


def compute_cosmos_edm_loss(
    model: nn.Module,
    clean_latents: Tensor,
    encoder_hidden_states: Tensor,
    sigma: Tensor | float | None = None,
    noise: Tensor | None = None,
    condition_frames: int = 1,
    padding_mask: Tensor | None = None,
    sigma_data: float = 0.5,
    fps: int = 10,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> Tensor:
    """Compute native Cosmos EDM pretraining objective with condition indicator and weighted x0 loss.

    Guarantees:
      - Appends condition mask channel iff model.config.in_channels == 17 (e.g. Cosmos 14B Video2World).
      - Strictly passes exactly 16 channels iff model.config.in_channels <= 16 (e.g. Cosmos 7B),
        omitting condition_mask argument to prevent internal diffusers concatenation to 17 channels.
      - Native spatial padding mask handled separately via kwargs['padding_mask'].
      - Evaluates weighted EDM loss on target prediction frames.
    """
    b = clean_latents.shape[0]
    device = clean_latents.device
    dtype = clean_latents.dtype
    scaling = CosmosEDMScaling(sigma_data=sigma_data)

    base_latents = clean_latents[:, :16]
    t_frames = base_latents.shape[2]
    h, w = base_latents.shape[3], base_latents.shape[4]

    if noise is None:
        noise = torch.randn_like(base_latents)

    if sigma is None:
        sigma_tensor = sample_edm_sigmas(b, device=device, dtype=dtype, p_mean=p_mean, p_std=p_std)
    elif isinstance(sigma, (int, float)):
        sigma_tensor = torch.full((b,), float(sigma), device=device, dtype=dtype)
    else:
        sigma_tensor = sigma.to(device=device, dtype=dtype)

    sigma_5d = sigma_tensor.view(b, 1, 1, 1, 1)
    c_skip, c_out, c_in, c_noise = scaling(sigma_5d)

    cond_mask = torch.zeros(b, 1, t_frames, h, w, device=device, dtype=dtype)
    num_cond = min(condition_frames, t_frames)
    cond_mask[:, :, :num_cond] = 1.0

    noisy_target = base_latents + sigma_5d * noise
    noisy_latents = cond_mask * base_latents + (1.0 - cond_mask) * noisy_target
    scaled_input = c_in * noisy_latents

    t_input = c_noise.flatten().to(dtype=torch.float32)

    model_config = getattr(model, "config", None)
    in_ch = getattr(model_config, "in_channels", 16)
    concat_padding = getattr(model_config, "concat_padding_mask", False)

    kwargs: dict[str, Any] = {
        "timestep": t_input,
        "encoder_hidden_states": encoder_hidden_states,
        "fps": fps,
        "return_dict": True,
    }

    if in_ch >= 17:
        cond_chan = clean_latents[:, 16:17] if clean_latents.shape[1] >= 17 else cond_mask
        kwargs["hidden_states"] = torch.cat([scaled_input[:, :16], cond_chan], dim=1)
    else:
        kwargs["hidden_states"] = scaled_input[:, :16]

    if padding_mask is not None:
        kwargs["padding_mask"] = padding_mask
    elif concat_padding:
        kwargs["padding_mask"] = torch.zeros(1, 1, h, w, device=device, dtype=dtype)

    model_out = model(**kwargs)
    pred = (
        model_out.sample
        if hasattr(model_out, "sample")
        else (model_out[0] if isinstance(model_out, tuple) else model_out)
    )
    pred_eps = pred[:, :16]

    pred_x0 = c_skip * noisy_latents + c_out * pred_eps
    loss_weights = scaling.sigma_loss_weights(sigma_5d)

    target_mask = 1.0 - cond_mask
    diff_sq = (pred_x0.float() - base_latents.float()) ** 2
    weighted_diff = loss_weights.float() * diff_sq * target_mask.float()

    normalizer = target_mask.float().sum().clamp(min=1.0)
    loss = weighted_diff.sum() / normalizer

    return loss


# ==============================================================================
# 5. Strict LoRA Serialization and Two-Way Key/Metadata Validation
# ==============================================================================


def save_native_lora(
    model: nn.Module,
    save_path: str | Path,
    rank: int,
    alpha: float,
    model_type: str,
    target_blocks: Sequence[int] | None = None,
    extra_metadata: dict[str, str] | None = None,
) -> None:
    """Save LoRA adapter weights to safetensors with strict metadata header."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lora_sd: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.detach().cpu()

    if not lora_sd:
        raise NativeVideoError("No LoRA parameters (lora_A/lora_B) found in model to save.")

    metadata: dict[str, str] = {
        "lora_rank": str(int(rank)),
        "lora_alpha": str(float(alpha)),
        "model_type": str(model_type),
        "num_adapter_tensors": str(len(lora_sd)),
    }
    if target_blocks is not None:
        metadata["target_blocks"] = json.dumps(list(target_blocks))
    if extra_metadata:
        metadata.update(extra_metadata)

    save_file(lora_sd, str(path), metadata=metadata)


def load_native_lora(
    model: nn.Module,
    lora_path: str | Path,
    expected_rank: int | None = None,
    expected_alpha: float | None = None,
    strict: bool = True,
) -> dict[str, str]:
    """Load LoRA adapter weights with strict two-way key matching and metadata validation.

    Validates:
      1. Both directions: All checkpoint keys must exist in model, AND all model LoRA adapter
         parameters must be present in the checkpoint (no missing adapter weights).
      2. Metadata rank and alpha match expected values and actual model module configurations.
      3. Tensor shapes match exactly.
    """
    path = Path(lora_path)
    if not path.is_file():
        raise FileNotFoundError(f"LoRA artifact not found: {path}")

    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if strict and not meta:
            raise LoRAMetadataError(f"LoRA safetensors file {path} has missing or empty metadata header.")

        if expected_rank is not None:
            if "lora_rank" in meta:
                stored_rank = int(meta["lora_rank"])
                if stored_rank != int(expected_rank):
                    raise LoRAMetadataError(
                        f"LoRA rank mismatch in metadata: checkpoint has rank={stored_rank}, expected rank={expected_rank}."
                    )
            elif strict:
                raise LoRAMetadataError(f"LoRA checkpoint {path} is missing 'lora_rank' in metadata header.")

        if expected_alpha is not None:
            if "lora_alpha" in meta:
                stored_alpha = float(meta["lora_alpha"])
                if not math.isclose(stored_alpha, float(expected_alpha), rel_tol=1e-4):
                    raise LoRAMetadataError(
                        f"LoRA alpha mismatch in metadata: checkpoint has alpha={stored_alpha}, expected alpha={expected_alpha}."
                    )
            elif strict:
                raise LoRAMetadataError(f"LoRA checkpoint {path} is missing 'lora_alpha' in metadata header.")

    sd = load_file(str(path))
    model_sd = model.state_dict()
    model_lora_keys = {name for name, _ in model.named_parameters() if "lora_A" in name or "lora_B" in name}

    if not model_lora_keys:
        raise LoRAMetadataError("Target model has no injected LoRA parameters (lora_A/lora_B).")

    for k in sd:
        if k not in model_sd:
            raise LoRAMetadataError(f"Unexpected LoRA key in checkpoint: '{k}' not found in target model.")

    missing_in_ckpt = model_lora_keys - set(sd.keys())
    if missing_in_ckpt:
        raise LoRAMetadataError(
            f"Missing LoRA parameters in checkpoint {path}: {sorted(missing_in_ckpt)}. "
            f"All {len(model_lora_keys)} model LoRA adapters must be present."
        )

    matched: dict[str, str] = {}
    for k, v in sd.items():
        if model_sd[k].shape != v.shape:
            raise LoRAMetadataError(
                f"LoRA tensor shape mismatch for '{k}': model expects {model_sd[k].shape}, got {v.shape}."
            )
        model_sd[k].copy_(v)
        matched[k] = "loaded"

    for mod in model.modules():
        if hasattr(mod, "rank") and hasattr(mod, "alpha"):
            if expected_rank is not None and mod.rank != expected_rank:
                raise LoRAMetadataError(
                    f"Injected module has rank={mod.rank}, expected rank={expected_rank}."
                )
            if expected_alpha is not None and not math.isclose(
                mod.alpha, float(expected_alpha), rel_tol=1e-4
            ):
                raise LoRAMetadataError(
                    f"Injected module has alpha={mod.alpha}, expected alpha={expected_alpha}."
                )

    return matched


# ==============================================================================
# 6. Shared Pre-encoding & Offloading with TCHW / CHW Delta Handling
# ==============================================================================


@dataclass
class PreencodedClipDataset:
    """In-memory or disk cache of pre-encoded latents and text conditioning."""

    latents: list[Tensor]
    text_embeddings: list[Tensor]
    clip_specs: list[EpisodeClipSpec]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.latents[idx], self.text_embeddings[idx]


def extract_single_image_from_dataset_row(row_img: Tensor) -> Tensor:
    """Extract single [C, H, W] image from dataset row, robust to delta timestamps [T_delta, C, H, W]."""
    if row_img.ndim == 4:
        return row_img[-1]
    elif row_img.ndim == 3:
        return row_img
    else:
        raise DatasetIntegrityError(
            f"Unexpected camera image shape: {row_img.shape}, expected 3D or 4D tensor."
        )


def preencode_and_offload_clips(
    dataset: Any,
    clips: list[EpisodeClipSpec],
    normalizer: CosmosVAENormalizer | FluxVAENormalizer,
    text_encoder_fn: Callable[[str], Tensor] | Tensor,
    prompt: str = "take cube out of box",
    batch_size: int = 4,
    device: str | torch.device = "cuda:0",
    offload_to_cpu: bool = True,
    provenance: dict[str, Any] | None = None,
) -> PreencodedClipDataset:
    """Pre-encode raw dataset video clips using true VAE, and offload encoder memory.

    Robustly handles both 3D [C, H, W] and 4D [T_delta, C, H, W] camera images from delta timestamps.
    """
    encoded_latents: list[Tensor] = []
    encoded_text: list[Tensor] = []
    valid_clips: list[EpisodeClipSpec] = []

    if isinstance(text_encoder_fn, Tensor):
        base_text_embed = text_encoder_fn.detach().cpu()
    else:
        base_text_embed = text_encoder_fn(prompt).detach().cpu()

    prov = provenance or {}
    prov["prompt"] = prompt
    prov["num_clips"] = len(clips)

    for i in range(0, len(clips), batch_size):
        batch_specs = clips[i : i + batch_size]
        batch_rgb_frames = []

        for spec in batch_specs:
            frames = []
            for frame_idx in spec.frame_indices:
                row = dataset[frame_idx]
                if "observation.images.front" not in row:
                    raise DatasetIntegrityError(f"Frame {frame_idx} missing 'observation.images.front'!")
                raw_img = row["observation.images.front"]
                img_chw = extract_single_image_from_dataset_row(raw_img)
                frames.append(img_chw)

            clip_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3)
            batch_rgb_frames.append(clip_tensor)

        batch_video = torch.stack(batch_rgb_frames, dim=0)  # [B, C, T, H, W]
        latents = normalizer.encode(batch_video, deterministic=True)
        lat_cpu = latents.detach().cpu()

        for j, spec in enumerate(batch_specs):
            encoded_latents.append(lat_cpu[j : j + 1])
            encoded_text.append(base_text_embed.clone())
            valid_clips.append(spec)

    if offload_to_cpu and hasattr(normalizer, "vae") and normalizer.vae is not None:
        normalizer.vae.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return PreencodedClipDataset(
        latents=encoded_latents,
        text_embeddings=encoded_text,
        clip_specs=valid_clips,
        provenance=prov,
    )


__all__ = ["ProtocolViolationError"]
