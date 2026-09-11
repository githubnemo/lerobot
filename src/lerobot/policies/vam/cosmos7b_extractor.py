"""Cosmos-1.0-Diffusion-7B feature extractor with 3D spatiotemporal grid & mRoPE formatting.

Extracts intermediate representations (specifically layers 14 and 20) from
NVIDIA Cosmos-1.0-Diffusion-7B for physical AI policy learning (e.g., SmolExpert).
Supports deterministic VAE encoding with per-channel normalization, native EDM input preconditioning,
text conditioning, 3D grid structuring, mRoPE coordinate indexing, and strict LoRA weight loading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from torch import Tensor, nn

from lerobot.policies.vam.base import (
    BaseExtractorConfig,
    BaseVAMExtractor,
    VAMExtractionOutput,
)
from lerobot.policies.vam.base.native_video import CosmosEDMScaling, CosmosVAENormalizer


class Cosmos7BError(RuntimeError):
    """Base exception for Cosmos 7B extractor operations."""


@dataclass(frozen=True, slots=True)
class Cosmos7BExtractorConfig(BaseExtractorConfig):
    """Configuration for Cosmos-1.0-Diffusion-7B feature extraction."""

    backbone_name: str = "cosmos7b"
    checkpoint_path: str | Path | None = None
    vae_path: str | Path | None = Path("/home/anton/.cache/video-vam/cosmos-14b/vae")
    device: str = "cpu"
    dtype: str = "float32"  # "float32" or "bfloat16"
    hidden_layers: tuple[int, ...] = (14, 20)
    pool_spatial: int | None = None  # e.g., 2 for 2x2 spatial avg pooling
    concat_layers: bool = True  # Concatenate layer 14 & 20 along channel dimension
    state_t: int = 2
    latent_channels: int = 16
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 16
    out_channels: int = 16
    num_layers: int = 28
    num_attention_heads: int = 32
    attention_head_dim: int = 128  # 32 * 128 = 4096 hidden_dim
    hidden_dim: int = 4096
    text_embed_dim: int = 1024
    rope_scale: tuple[float, float, float] = (2.0, 1.0, 1.0)
    sigma_data: float = 0.5
    fps: int = 10

    def __post_init__(self) -> None:
        BaseExtractorConfig.__post_init__(self)
        for layer_idx in self.hidden_layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    f"hidden_layer {layer_idx} out of range for {self.num_layers}-layer Cosmos 7B"
                )
        if self.attention_head_dim * self.num_attention_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal "
                f"num_attention_heads * attention_head_dim ({self.num_attention_heads * self.attention_head_dim})"
            )


@dataclass(frozen=True, slots=True)
class Cosmos7BExtractionOutput(VAMExtractionOutput):
    """Output container for extracted Cosmos 7B intermediate representations."""


def build_cosmos7b_dummy_model(
    *,
    num_layers: int = 28,
    num_attention_heads: int = 4,
    attention_head_dim: int = 16,
    text_embed_dim: int = 32,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CosmosTransformer3DModel:
    """Construct a lightweight architecture-matched model for CPU testing and shape validation."""
    return CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        text_embed_dim=text_embed_dim,
        crossattn_proj_in_channels=text_embed_dim,
        encoder_hidden_states_channels=text_embed_dim,
        patch_size=(1, 2, 2),
        concat_padding_mask=False,
        extra_pos_embed_type=None,
    ).to(device=device, dtype=dtype)


class Cosmos7BExtractor(BaseVAMExtractor):
    """Intermediate feature extractor for NVIDIA Cosmos-1.0-Diffusion-7B."""

    def __init__(
        self,
        config: Cosmos7BExtractorConfig | None = None,
        model: CosmosTransformer3DModel | None = None,
        vae: Any | None = None,
    ) -> None:
        cfg = config or Cosmos7BExtractorConfig()
        super().__init__(cfg)
        self.config: Cosmos7BExtractorConfig = cfg
        self.edm_scaling = CosmosEDMScaling(sigma_data=cfg.sigma_data)
        self.normalizer = CosmosVAENormalizer(
            vae=vae,
            sigma_data=cfg.sigma_data,
            device=self.device,
            dtype=self.dtype,
        )

        if model is not None:
            self.transformer = model.to(device=self.device, dtype=self.dtype)
        elif self.config.checkpoint_path is not None:
            ckpt_str = str(self.config.checkpoint_path)
            self.transformer = CosmosTransformer3DModel.from_pretrained(
                ckpt_str,
                torch_dtype=self.dtype,
            ).to(self.device)
        else:
            self.transformer = CosmosTransformer3DModel(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                num_layers=self.config.num_layers,
                num_attention_heads=self.config.num_attention_heads,
                attention_head_dim=self.config.attention_head_dim,
                text_embed_dim=self.config.text_embed_dim,
                patch_size=self.config.patch_size,
                rope_scale=self.config.rope_scale,
                concat_padding_mask=False,
                extra_pos_embed_type=None,
            ).to(device=self.device, dtype=self.dtype)

        # Freeze all parameters
        self.transformer.eval()
        for param in self.transformer.parameters():
            param.requires_grad_(False)

    @property
    def vae(self) -> Any | None:
        return self.normalizer.vae

    @vae.setter
    def vae(self, value: Any | None) -> None:
        self.normalizer.vae = value

    def encode_latents(self, rgb_frames: Tensor) -> Tensor:
        """Encode raw RGB video [B, 3, T, H, W] into exact Cosmos VAE latent space [B, 16, T', H', W']."""
        if self.normalizer.vae is None:
            vae_path = self.config.vae_path
            if vae_path is not None and Path(vae_path).exists():
                self.normalizer = CosmosVAENormalizer.from_pretrained(
                    vae_path=vae_path,
                    sigma_data=self.config.sigma_data,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                raise Cosmos7BError(
                    f"VAE model not available at {vae_path} and no vae passed to Cosmos7BExtractor."
                )

        return self.normalizer.encode(rgb_frames, deterministic=True)

    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.001,
        fps: int = 10,
    ) -> dict[int, Tensor]:
        """Execute transformer forward pass with native EDM preconditioning and tap specified hidden layers."""
        b = latents.shape[0]

        # Native EDM preconditioning
        if isinstance(timestep, (int, float)):
            sigma_val = max(1e-4, float(timestep))
            sigma_t = torch.tensor([sigma_val], device=self.device, dtype=torch.float32)
        else:
            sigma_t = timestep.to(device=self.device, dtype=torch.float32)
            if sigma_t.ndim == 0:
                sigma_t = sigma_t.unsqueeze(0)
            sigma_t = sigma_t.clamp(min=1e-4)

        # c_in and c_noise from EDM scaling
        c_skip, c_out, c_in, c_noise = self.edm_scaling(sigma_t.view(-1, 1, 1, 1, 1))
        scaled_latents = (latents * c_in).to(device=self.device, dtype=self.dtype)
        t_input = c_noise.flatten().to(dtype=torch.float32)

        # Prepare text conditioning
        text_dim = getattr(self.transformer.config, "text_embed_dim", self.config.text_embed_dim)
        if text_conditioning is None:
            text_embeds = torch.zeros(b, 1, text_dim, device=self.device, dtype=self.dtype)
        elif isinstance(text_conditioning, Tensor):
            text_embeds = text_conditioning.to(device=self.device, dtype=self.dtype)
            if text_embeds.ndim == 2:
                text_embeds = text_embeds.unsqueeze(1)
        else:
            text_embeds = torch.zeros(b, 1, text_dim, device=self.device, dtype=self.dtype)

        captured_layers: dict[int, Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module: nn.Module, inp: Any, out: Tensor) -> None:
                captured_layers[layer_idx] = out.detach()

            return hook_fn

        for layer_idx in self.config.hidden_layers:
            if layer_idx < len(self.transformer.transformer_blocks):
                blk = self.transformer.transformer_blocks[layer_idx]
                hooks.append(blk.register_forward_hook(make_hook(layer_idx)))

        try:
            b, c, t_dim, h_lat, w_lat = scaled_latents.shape
            pad_mask = torch.zeros(
                (b, 1, h_lat, w_lat), dtype=scaled_latents.dtype, device=scaled_latents.device
            )
            cond_mask = torch.zeros(
                (b, 1, t_dim, h_lat, w_lat), dtype=scaled_latents.dtype, device=scaled_latents.device
            )
            self.transformer(
                hidden_states=scaled_latents,
                timestep=t_input,
                encoder_hidden_states=text_embeds,
                fps=fps,
                condition_mask=cond_mask,
                padding_mask=pad_mask,
                return_dict=True,
            )
        finally:
            for h_hook in hooks:
                h_hook.remove()

        return captured_layers

    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        b, c, t, h, w = latents.shape
        p_t, p_h, p_w = self.config.patch_size
        return (t // p_t, h // p_h, w // p_w)

    def get_provenance(self) -> dict[str, Any]:
        return {
            "model_type": "Cosmos-1.0-Diffusion-7B",
            "hidden_layers": list(self.config.hidden_layers),
            "patch_size": list(self.config.patch_size),
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @torch.no_grad()
    def extract(
        self,
        hidden_states: Tensor | None = None,
        rgb_frames: Tensor | None = None,
        latents: Tensor | None = None,
        text_conditioning: Tensor | None = None,
        timestep: Tensor | float = 0.001,
        fps: int = 10,
    ) -> Cosmos7BExtractionOutput:
        """Extract intermediate features with native EDM preconditioning."""
        inp_latents = hidden_states if hidden_states is not None else latents
        base_out = super().extract(
            rgb_frames=rgb_frames,
            latents=inp_latents,
            text_conditioning=text_conditioning,
            timestep=timestep,
            fps=fps,
        )
        return Cosmos7BExtractionOutput(
            features=base_out.features,
            features_by_layer=base_out.features_by_layer,
            grid_features_by_layer=base_out.grid_features_by_layer,
            grid_coords=base_out.grid_coords,
            grid_shape=base_out.grid_shape,
            provenance=base_out.provenance,
        )
