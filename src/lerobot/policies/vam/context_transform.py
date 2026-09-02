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
    "cosmos3_edge_none",
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


def _validate_temporal_frames(temporal_frames: int) -> int:
    if temporal_frames not in (2, 3, 4, 6, 8, 12, 16):
        raise ValueError(f"temporal_frames must be in (2, 3, 4, 6, 8, 12, 16), got {temporal_frames}")
    return temporal_frames


def _input_grid_from_tokens(tokens: int) -> tuple[int, int, int]:
    for t in (2, 3, 4, 6, 8, 12, 16):
        if tokens == t * 30 * 40:
            return t, 30, 40
        if tokens == t * 15 * 20:
            return t, 15, 20
    raise ValueError(
        f"Cosmos context must contain T*1200 or T*300 tokens for T in (2, 3, 4, 6, 8, 12, 16), got {tokens}"
    )


def context_transform_spec(transform: str, temporal_frames: int = 16) -> ContextTransformSpec:
    """Return the validated output grid for ``transform`` and input frame count."""
    if transform not in CONTEXT_TRANSFORMS:
        raise ValueError(f"unknown context transform {transform!r}")
    temporal_frames = _validate_temporal_frames(temporal_frames)
    if transform == "cond_frames":
        grid = (min(CONTEXT_CONDITIONING_FRAMES, temporal_frames), CONTEXT_GRID_H, CONTEXT_GRID_W)
    elif transform == "gen_frames_pool2":
        if temporal_frames <= CONTEXT_CONDITIONING_FRAMES:
            raise ValueError("gen_frames_pool2 requires future frames beyond the two conditional frames")
        grid = (temporal_frames - CONTEXT_CONDITIONING_FRAMES, 15, 20)
    elif transform == "pool2":
        grid = (temporal_frames, 15, 20)
    elif transform == "pool4":
        grid = (temporal_frames, 8, 10)
    elif transform == "frame_mean":
        grid = (temporal_frames, 1, 1)
    elif transform == "global_mean":
        grid = (1, 1, 1)
    elif transform == "cosmos3_edge_none":
        grid = (temporal_frames, 15, 20)
    else:
        grid = (temporal_frames, CONTEXT_GRID_H, CONTEXT_GRID_W)
    return ContextTransformSpec(transform, grid)


def context_transform_metadata(transform: str, temporal_frames: int = 16) -> dict[str, Any]:
    """Return JSON-safe provenance for the stored context representation."""
    temporal_frames = _validate_temporal_frames(temporal_frames)
    spec = context_transform_spec(transform, temporal_frames=temporal_frames)
    return {
        "name": spec.name,
        "input_grid": {
            "temporal": temporal_frames,
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
        "input_tokens": temporal_frames * CONTEXT_GRID_H * CONTEXT_GRID_W,
        "output_tokens": spec.output_tokens,
    }


def apply_context_transform(context: torch.Tensor, transform: str) -> torch.Tensor:
    """Reduce a flattened Cosmos context along its inferred latent grid."""
    if transform not in CONTEXT_TRANSFORMS:
        raise ValueError(f"unknown context transform {transform!r}")
    if not isinstance(context, torch.Tensor) or context.ndim != 3:
        raise ValueError(f"context must have shape [B, N, C], got {getattr(context, 'shape', type(context))}")
    batch, tokens, channels = context.shape
    temporal_frames, height, width = _input_grid_from_tokens(tokens)
    spec = context_transform_spec(transform, temporal_frames=temporal_frames)
    if transform == "none":
        return context
    grid = context.view(batch, temporal_frames, height, width, channels)
    if transform == "cond_frames":
        reduced = grid[:, : spec.output_grid[0]]
    elif transform == "frame_mean":
        return grid.mean(dim=(2, 3))
    elif transform == "global_mean":
        return grid.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    elif transform in ("pool2", "pool4", "gen_frames_pool2"):
        source = grid[:, CONTEXT_CONDITIONING_FRAMES:] if transform == "gen_frames_pool2" else grid
        target_hw = (8, 10) if transform == "pool4" else (15, 20)
        frames = source.shape[1]
        flat = source.permute(0, 1, 4, 2, 3).reshape(batch * frames, channels, height, width)
        pooled = functional.adaptive_avg_pool2d(flat, target_hw)
        reduced = pooled.reshape(batch, frames, channels, *target_hw).permute(0, 1, 3, 4, 2)
    else:
        raise ValueError(f"unknown context transform {transform!r}")
    return reduced.reshape(batch, -1, channels)
