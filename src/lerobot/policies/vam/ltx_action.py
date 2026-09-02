"""Minimal LTX feature adapter for the existing World2Action decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn

LTX_GRID = (8, 15, 20)
LTX_STATE_T = (2, 8)
LTX_CHANNELS = 4096
WORLD2ACTION_CHANNELS = 2048
LTX_CONTEXT_TRANSFORMS = ("pool2", "pool4")
LTX_CONTEXT_SPATIAL_GRIDS = {
    "pool2": (8, 10),
    "pool4": (4, 5),
}
LTX_CONTEXT_GRIDS = {transform: (8, *spatial) for transform, spatial in LTX_CONTEXT_SPATIAL_GRIDS.items()}


def _ltx_grid_shape(hidden_grid: torch.Tensor) -> tuple[int, int, int, int, int]:
    if hidden_grid.ndim != 5 or hidden_grid.shape[1] not in LTX_STATE_T:
        raise ValueError(
            f"LTX hidden grid must have shape [B, T, 15, 20, 4096] with T in {LTX_STATE_T}, "
            f"got {tuple(hidden_grid.shape)}"
        )
    if tuple(hidden_grid.shape[2:]) != (15, 20, LTX_CHANNELS):
        raise ValueError(
            f"LTX hidden grid must have shape [B, T, 15, 20, 4096], got {tuple(hidden_grid.shape)}"
        )
    if not torch.is_floating_point(hidden_grid) or not torch.isfinite(hidden_grid).all().item():
        raise ValueError("LTX hidden grid must be finite floating point")
    return tuple(hidden_grid.shape)  # type: ignore[return-value]


def frame_mean_ltx_context(hidden_grid: torch.Tensor) -> torch.Tensor:
    """Spatially average `[B,8,15,20,4096]` into eight causal video tokens."""
    _ltx_grid_shape(hidden_grid)
    return hidden_grid.mean(dim=(2, 3)).contiguous()


def ltx_context_grid(context_transform: str, temporal_frames: int = 8) -> tuple[int, int, int]:
    """Return the output T,H,W grid for a supported LTX context transform."""
    if temporal_frames not in LTX_STATE_T:
        raise ValueError(f"temporal_frames must be one of {LTX_STATE_T}")
    try:
        return (temporal_frames, *LTX_CONTEXT_SPATIAL_GRIDS[context_transform])
    except KeyError as exc:
        raise ValueError(f"unknown LTX context transform {context_transform!r}") from exc


def apply_ltx_context_transform(hidden_grid: torch.Tensor, context_transform: str) -> torch.Tensor:
    """Apply the shared LTX spatial reduction used by cache builders and policies."""
    _ltx_grid_shape(hidden_grid)
    output_grid = ltx_context_grid(context_transform, temporal_frames=hidden_grid.shape[1])
    batch, frames, height, width, channels = hidden_grid.shape
    flat = hidden_grid.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
    pooled = functional.adaptive_avg_pool2d(flat, output_grid[1:])
    return (
        pooled.reshape(batch, frames, channels, *output_grid[1:])
        .permute(0, 1, 3, 4, 2)
        .reshape(batch, math.prod(output_grid), channels)
        .contiguous()
    )


def pool2_ltx_context(hidden_grid: torch.Tensor) -> torch.Tensor:
    """Pool each 15x20 latent frame to 8x10, mirroring the Cosmos pool2 arm."""
    return apply_ltx_context_transform(hidden_grid, "pool2")


def pool4_ltx_context(hidden_grid: torch.Tensor) -> torch.Tensor:
    """Pool each 15x20 latent frame to 4x5 for compact probe caches."""
    return apply_ltx_context_transform(hidden_grid, "pool4")


class LTXContextAdapter(nn.Module):
    """Normalize and project frame-level LTX 4096-d tokens to World2Action 2048-d."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(LTX_CHANNELS, elementwise_affine=False)
        self.projection = nn.Linear(LTX_CHANNELS, WORLD2ACTION_CHANNELS, bias=False)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3 or context.shape[-1] != LTX_CHANNELS:
            raise ValueError(
                f"LTX frame context must have shape [B, N, {LTX_CHANNELS}], got {tuple(context.shape)}"
            )
        if not torch.is_floating_point(context) or not torch.isfinite(context).all().item():
            raise ValueError("LTX frame context must be finite floating point")
        return self.projection(self.norm(context))
