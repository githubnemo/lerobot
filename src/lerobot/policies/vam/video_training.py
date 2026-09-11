"""Shared native video training utilities, EDM loss, and LoRA injection helpers.

Exposes unified interfaces for video diffusion LoRA training across:
- Cosmos 7B / Cosmos 14B (EDM objective)
- FLUX.2 [klein] 9B (Rectified Flow)
- Episode-local multi-frame extraction and strict Protocol 1.0 / Scale-100 splits.
"""

from __future__ import annotations

import math

import torch

from lerobot.policies.vam.base.native_video import (
    CosmosEDMScaling,
    CosmosVAENormalizer,
    DatasetIntegrityError,
    EpisodeClipSpec,
    FluxVAENormalizer,
    LoRAMetadataError,
    NativeVideoError,
    PreencodedClipDataset,
    compute_cosmos_edm_loss,
    extract_episode_clip_indices,
    load_native_lora,
    parse_episode_spec,
    preencode_and_offload_clips,
    sample_edm_sigmas,
    save_native_lora,
    validate_training_episodes,
)


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a learning rate scheduler with linear warmup and cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-3, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / base_lr if base_lr > min_lr else 0.01
        return max(min_factor, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


__all__ = [
    "CosmosEDMScaling",
    "CosmosVAENormalizer",
    "DatasetIntegrityError",
    "EpisodeClipSpec",
    "FluxVAENormalizer",
    "LoRAMetadataError",
    "NativeVideoError",
    "PreencodedClipDataset",
    "build_cosine_schedule_with_warmup",
    "compute_cosmos_edm_loss",
    "extract_episode_clip_indices",
    "load_native_lora",
    "parse_episode_spec",
    "preencode_and_offload_clips",
    "sample_edm_sigmas",
    "save_native_lora",
    "validate_training_episodes",
]
