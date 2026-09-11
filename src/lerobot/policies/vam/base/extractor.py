"""Standardized Abstract Base Extractor for Video Action Models (VAM).

Defines the contract for physical AI feature extractors coupling video backbones
(e.g., Cosmos, LTX, FLUX) with downstream action decoders (e.g., SmolExpert).
Enforces true VAE latent contracts, standardized 3D spatiotemporal grid formatting,
and spatial pooling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


class VAEContractViolationError(RuntimeError):
    """Raised when VAE latent encoding contract is bypassed or violated."""


class ExtractorConfigError(ValueError):
    """Raised when extractor configuration parameters are invalid."""


@dataclass(frozen=True, slots=True)
class BaseExtractorConfig:
    """Standard configuration contract for Video Action Model extractors."""

    backbone_name: str = "base"
    checkpoint_path: str | Path | None = None
    vae_path: str | Path | None = None
    device: str = "cpu"
    dtype: str = "float32"
    hidden_layers: tuple[int, ...] = (20,)
    pool_spatial: int | None = 2
    concat_layers: bool = True
    state_t: int = 2
    latent_channels: int = 16
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backbone_name or not isinstance(self.backbone_name, str):
            raise ExtractorConfigError("backbone_name must be a non-empty string")
        if not self.hidden_layers or any(
            not isinstance(layer, int) or layer < 0 for layer in self.hidden_layers
        ):
            raise ExtractorConfigError("hidden_layers must be a non-empty tuple of non-negative integers")
        if self.pool_spatial is not None and (
            not isinstance(self.pool_spatial, int) or self.pool_spatial <= 0
        ):
            raise ExtractorConfigError("pool_spatial must be None or a positive integer")
        if not isinstance(self.latent_channels, int) or self.latent_channels <= 0:
            raise ExtractorConfigError("latent_channels must be a positive integer")
        if self.dtype not in ("float32", "bfloat16", "float16"):
            raise ExtractorConfigError(
                f"dtype must be 'float32', 'bfloat16', or 'float16', got {self.dtype!r}"
            )


@dataclass(frozen=True, slots=True)
class VAMExtractionOutput:
    """Standardized output container for extracted VAM intermediate representations."""

    features: Tensor  # [B, S_pooled, C_total] - primary representation for policy heads
    features_by_layer: dict[int, Tensor]  # layer_idx -> [B, S_pooled, C]
    grid_coords: Tensor  # [B, S_pooled, 3] (t, h, w) coordinate tuples
    grid_shape: tuple[int, int, int]  # (T', H', W') spatiotemporal grid dimensions
    provenance: dict[str, Any]  # Metadata (SHA, shapes, config, timestamps)
    grid_features_by_layer: dict[int, Tensor] = field(default_factory=dict)  # layer_idx -> [B, T', H', W', C]

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]


class BaseVAMExtractor(nn.Module, ABC):
    """Abstract base class establishing the contract for all physical AI feature extractors."""

    def __init__(self, config: BaseExtractorConfig) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        if config.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif config.dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    @abstractmethod
    def encode_latents(self, rgb_frames: Tensor) -> Tensor:
        """Encode raw RGB video [B, 3, T, H, W] into exact VAE latent space [B, C, T', H', W'].

        Subclasses must execute an authorized causal/spatial VAE encoder.
        Heuristic bilinear downsampling or mock zero-padding is strictly forbidden.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
    ) -> dict[int, Tensor]:
        """Execute transformer forward pass and tap specified hidden layers."""
        raise NotImplementedError

    @abstractmethod
    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        """Compute unpooled spatiotemporal grid shape (T', H', W') from input latents."""
        raise NotImplementedError

    @abstractmethod
    def get_provenance(self) -> dict[str, Any]:
        """Return provenance metadata dictionary."""
        raise NotImplementedError

    def validate_vae_contract(self, latents: Tensor) -> None:
        """Enforce strict VAE output shape and channel invariants."""
        if not isinstance(latents, Tensor):
            raise VAEContractViolationError(f"Latents must be a torch.Tensor, got {type(latents).__name__}")
        if latents.ndim != 5:
            raise VAEContractViolationError(
                f"VAE latents must have 5 dimensions [B, C, T, H, W], got {latents.ndim} "
                f"with shape {tuple(latents.shape)}"
            )
        expected_channels = self.config.latent_channels
        valid_channels = (expected_channels, expected_channels + 1)
        if latents.shape[1] not in valid_channels:
            raise VAEContractViolationError(
                f"VAE latent channels mismatch: expected {expected_channels}, got {latents.shape[1]}"
            )
        if not torch.is_floating_point(latents):
            raise VAEContractViolationError("VAE latents must be floating point")

    def format_3d_grid(self, flat_tokens: Tensor, grid_shape: tuple[int, int, int]) -> Tensor:
        """Format flattened tokens [B, S, D] into 3D spatiotemporal grid [B, T', H', W', D]."""
        b, s, d = flat_tokens.shape
        t, h, w = grid_shape
        if s != t * h * w:
            raise ValueError(
                f"Token count {s} does not match grid dimensions {grid_shape} (product={t * h * w})"
            )
        return flat_tokens.reshape(b, t, h, w, d)

    def pool_spatial_tokens(
        self, flat_tokens: Tensor, grid_shape: tuple[int, int, int], factor: int
    ) -> Tensor:
        """Apply 2D spatial average pooling to flattened tokens given their (T, H, W) grid shape."""
        b, s, d = flat_tokens.shape
        t, h, w = grid_shape
        if s != t * h * w:
            raise ValueError(
                f"Token count {s} does not match grid dimensions {grid_shape} (product={t * h * w})"
            )
        grid = flat_tokens.reshape(b * t, h, w, d).permute(0, 3, 1, 2)
        pooled = nn.functional.avg_pool2d(grid, kernel_size=factor, stride=factor)
        new_h, new_w = pooled.shape[-2:]
        return pooled.permute(0, 2, 3, 1).reshape(b, t * new_h * new_w, d)

    def pool_spatial_grid(self, grid_tensor: Tensor, pool_factor: int = 2) -> Tensor:
        """Apply 2D spatial average pooling to a 3D spatiotemporal grid [B, T', H', W', D]."""
        b, t, h, w, d = grid_tensor.shape
        reshaped = grid_tensor.reshape(b * t, h, w, d).permute(0, 3, 1, 2)
        pooled = nn.functional.avg_pool2d(reshaped, kernel_size=pool_factor, stride=pool_factor)
        new_h, new_w = pooled.shape[-2:]
        return pooled.permute(0, 2, 3, 1).reshape(b, t, new_h, new_w, d)

    def compute_3d_grid_coordinates(self, grid_shape: tuple[int, int, int], batch_size: int = 1) -> Tensor:
        """Compute (t, h, w) coordinate tuples for spatiotemporal tokens.

        Returns tensor of shape [B, T' * H' * W', 3].
        """
        t_d, h_d, w_d = grid_shape
        t_c = torch.arange(t_d, device=self.device)
        h_c = torch.arange(h_d, device=self.device)
        w_c = torch.arange(w_d, device=self.device)
        grid_t, grid_h, grid_w = torch.meshgrid(t_c, h_c, w_c, indexing="ij")
        coords = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(-1, 3)
        return coords.unsqueeze(0).expand(batch_size, -1, -1)

    @torch.no_grad()
    def extract(
        self,
        rgb_frames: Tensor | None = None,
        latents: Tensor | None = None,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
        hidden_states: Tensor | None = None,
        **kwargs: Any,
    ) -> VAMExtractionOutput:
        """Standardized end-to-end extraction pipeline with pooling and grid coordinates."""
        if latents is None:
            if hidden_states is not None:
                latents = hidden_states
            elif rgb_frames is None:
                raise ValueError("Either rgb_frames or latents must be provided to extract()")
            else:
                latents = self.encode_latents(rgb_frames)

        self.validate_vae_contract(latents)
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)

        layer_features = self.forward_transformer_blocks(latents, text_conditioning, timestep)

        grid_shape = self.compute_grid_shape(latents)
        processed_by_layer: dict[int, Tensor] = {}
        grid_features_by_layer: dict[int, Tensor] = {}
        pooled_list: list[Tensor] = []

        pool_factor = self.config.pool_spatial
        apply_pool = pool_factor is not None and pool_factor > 1

        out_grid_shape = (
            (grid_shape[0], grid_shape[1] // pool_factor, grid_shape[2] // pool_factor)
            if apply_pool
            else grid_shape
        )

        for lyr in self.config.hidden_layers:
            if lyr not in layer_features:
                raise KeyError(
                    f"Hidden layer {lyr} not returned by forward_transformer_blocks. "
                    f"Available layers: {sorted(layer_features.keys())}"
                )
            feat = layer_features[lyr]  # [B, S, D]
            if apply_pool:
                feat = self.pool_spatial_tokens(feat, grid_shape, pool_factor)
            processed_by_layer[lyr] = feat
            grid_features_by_layer[lyr] = self.format_3d_grid(feat, out_grid_shape)
            pooled_list.append(feat)

        if self.config.concat_layers and len(pooled_list) > 1:
            final_features = torch.cat(pooled_list, dim=-1)
        else:
            final_features = pooled_list[-1]

        batch_size = latents.shape[0]
        coords = self.compute_3d_grid_coordinates(out_grid_shape, batch_size=batch_size)

        prov = dict(self.get_provenance())
        prov.update(
            {
                "backbone_name": self.config.backbone_name,
                "hidden_layers": list(self.config.hidden_layers),
                "unpooled_grid_shape": list(grid_shape),
                "output_grid_shape": list(out_grid_shape),
                "pool_spatial": pool_factor,
                "concat_layers": self.config.concat_layers,
                "feature_dim": final_features.shape[-1],
                "num_tokens": final_features.shape[1],
            }
        )

        return VAMExtractionOutput(
            features=final_features,
            features_by_layer=processed_by_layer,
            grid_coords=coords,
            grid_shape=out_grid_shape,
            provenance=prov,
            grid_features_by_layer=grid_features_by_layer,
        )
