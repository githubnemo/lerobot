"""Reusable Cosmos layer-20 context-grid transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

CONTEXT_GRID_T, CONTEXT_GRID_H, CONTEXT_GRID_W = 16, 30, 40
CONTEXT_CONDITIONING_FRAMES = 2
CONTEXT_TRANSFORMS = (
    "none",
    "cond_frames",
    "gen_frames_pool2",
    "pool2",
    "pool4",
    "frame_mean",
    "global_mean",
)
CONTEXT_GRID_FLATTEN_ORDER = "T,H,W"
RAW_CONTEXT_TOKENS = CONTEXT_GRID_T * CONTEXT_GRID_H * CONTEXT_GRID_W


@dataclass(frozen=True, slots=True)
class ContextTransformSpec:
    """The input and output grid semantics of one context transform."""

    name: str
    output_grid: tuple[int, int, int]

    @property
    def output_tokens(self) -> int:
        temporal, height, width = self.output_grid
        return temporal * height * width


def context_transform_spec(transform: str) -> ContextTransformSpec:
    """Return the validated output grid for ``transform``."""
    if transform not in CONTEXT_TRANSFORMS:
        raise ValueError(f"unknown context transform {transform!r}")
    if transform == "cond_frames":
        grid = (CONTEXT_CONDITIONING_FRAMES, CONTEXT_GRID_H, CONTEXT_GRID_W)
    elif transform == "gen_frames_pool2":
        grid = (CONTEXT_GRID_T - CONTEXT_CONDITIONING_FRAMES, 15, 20)
    elif transform == "pool2":
        grid = (CONTEXT_GRID_T, 15, 20)
    elif transform == "pool4":
        grid = (CONTEXT_GRID_T, 8, 10)
    elif transform == "frame_mean":
        grid = (CONTEXT_GRID_T, 1, 1)
    elif transform == "global_mean":
        grid = (1, 1, 1)
    else:
        grid = (CONTEXT_GRID_T, CONTEXT_GRID_H, CONTEXT_GRID_W)
    return ContextTransformSpec(transform, grid)


def context_transform_metadata(transform: str) -> dict[str, Any]:
    """Return JSON-safe provenance for the stored context representation."""
    spec = context_transform_spec(transform)
    return {
        "name": spec.name,
        "input_grid": {
            "temporal": CONTEXT_GRID_T,
            "height": CONTEXT_GRID_H,
            "width": CONTEXT_GRID_W,
            "flatten_order": CONTEXT_GRID_FLATTEN_ORDER,
        },
        "output_grid": {
            "temporal": spec.output_grid[0],
            "height": spec.output_grid[1],
            "width": spec.output_grid[2],
            "flatten_order": CONTEXT_GRID_FLATTEN_ORDER,
        },
        "input_tokens": RAW_CONTEXT_TOKENS,
        "output_tokens": spec.output_tokens,
    }


def apply_context_transform(context: torch.Tensor, transform: str) -> torch.Tensor:
    """Reduce a flattened ``[B, 19200, C]`` context along its latent grid."""
    if transform == "none":
        return context
    batch, tokens, channels = context.shape
    if tokens != RAW_CONTEXT_TOKENS:
        raise ValueError(f"context transform {transform!r} expects {RAW_CONTEXT_TOKENS} tokens, got {tokens}")
    grid = context.view(batch, CONTEXT_GRID_T, CONTEXT_GRID_H, CONTEXT_GRID_W, channels)
    if transform == "cond_frames":
        reduced = grid[:, :CONTEXT_CONDITIONING_FRAMES]
    elif transform == "frame_mean":
        return grid.mean(dim=(2, 3))
    elif transform == "global_mean":
        return grid.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    elif transform in ("pool2", "pool4", "gen_frames_pool2"):
        source = grid[:, CONTEXT_CONDITIONING_FRAMES:] if transform == "gen_frames_pool2" else grid
        target_hw = (8, 10) if transform == "pool4" else (15, 20)
        frames = source.shape[1]
        flat = source.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, CONTEXT_GRID_H, CONTEXT_GRID_W)
        pooled = functional.adaptive_avg_pool2d(flat, target_hw)
        reduced = pooled.reshape(batch, frames, channels, *target_hw).permute(0, 1, 3, 4, 2)
    else:
        raise ValueError(f"unknown context transform {transform!r}")
    return reduced.reshape(batch, -1, channels)
