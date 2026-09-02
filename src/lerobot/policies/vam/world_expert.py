"""Latent diffusion expert conditioned on Cosmos layer-20 features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn
from torch.nn import functional

from ..common.vla_utils import create_sinusoidal_pos_embedding
from ..smolvla.smolvlm_with_expert import apply_rope
from .smol_expert import (
    EXPERT_WIDTH_MULTIPLIER,
    SELF_ATTN_EVERY_N_LAYERS,
    SMOLVLA_CHECKPOINT,
    SMOLVLM_CONFIG,
    CosmosPrefixAdapter,
    _build_pretrained_expert,
    _load_module_from_checkpoint,
    resolve_checkpoint,
)

WORLD_LATENT_CHANNELS = 16
WORLD_LATENT_FRAMES = 14
WORLD_LATENT_HEIGHT = 60
WORLD_LATENT_WIDTH = 80
DEFAULT_LATENT_PATCH_SIZE = 2
DEFAULT_PREFIX_HIDDEN_SIZE = 960
DEFAULT_EXPERT_HIDDEN_SIZE = int(DEFAULT_PREFIX_HIDDEN_SIZE * EXPERT_WIDTH_MULTIPLIER)
DEFAULT_KV_DIM = 320
DEFAULT_NUM_ATTENTION_HEADS = 15
DEFAULT_NUM_KEY_VALUE_HEADS = 5
DEFAULT_HEAD_DIM = 64


@dataclass(frozen=True, slots=True)
class WorldExpertPrefixKVCache:
    """Per-layer prefix K/V tensors reusable across denoising steps."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("prefix K/V cache must contain one key and value tensor per expert layer")


class LinearWorldReadout(nn.Module):
    """Shared per-spatial-position linear readout of observed Cosmos tokens."""

    def __init__(
        self,
        *,
        feature_channels: int = 2048,
        feature_frames: int = 2,
        token_height: int = 30,
        token_width: int = 40,
        latent_channels: int = WORLD_LATENT_CHANNELS,
        latent_frames: int = WORLD_LATENT_FRAMES,
        latent_height: int = WORLD_LATENT_HEIGHT,
        latent_width: int = WORLD_LATENT_WIDTH,
        latent_patch_size: int = DEFAULT_LATENT_PATCH_SIZE,
    ) -> None:
        super().__init__()
        if feature_channels <= 0 or feature_frames <= 0:
            raise ValueError("feature_channels and feature_frames must be positive")
        if token_height <= 0 or token_width <= 0:
            raise ValueError("token_height and token_width must be positive")
        if latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive")
        if latent_height % latent_patch_size or latent_width % latent_patch_size:
            raise ValueError("latent_patch_size must divide both latent spatial dimensions")
        if latent_channels <= 0 or latent_frames <= 0:
            raise ValueError("latent_channels and latent_frames must be positive")
        if feature_frames != 2:
            raise ValueError("LinearWorldReadout requires exactly two observed feature frames")
        if (token_height, token_width) != (
            latent_height // latent_patch_size,
            latent_width // latent_patch_size,
        ):
            raise ValueError("token grid must match the latent grid and patch size")

        self.feature_channels = feature_channels
        self.feature_frames = feature_frames
        self.token_height = token_height
        self.token_width = token_width
        self.input_channels = feature_channels * feature_frames
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_patch_size = latent_patch_size
        self.patch_channels = latent_frames * latent_channels * latent_patch_size * latent_patch_size
        self.norm = nn.LayerNorm(self.input_channels)
        self.readout = nn.Linear(self.input_channels, self.patch_channels)

    def _validate_tokens(self, tokens: Tensor) -> None:
        expected = (
            tokens.shape[0],
            self.feature_frames * self.token_height * self.token_width,
            self.feature_channels,
        )
        if tokens.ndim != 3 or tuple(tokens.shape) != expected:
            raise ValueError(
                f"Cosmos layer-20 tokens must have shape [B, "
                f"{self.feature_frames * self.token_height * self.token_width}, {self.feature_channels}], "
                f"got {tuple(tokens.shape)}"
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """Predict clean future latents from both observed token frames."""
        self._validate_tokens(tokens)
        batch_size = tokens.shape[0]
        features = tokens.reshape(
            batch_size,
            self.feature_frames,
            self.token_height,
            self.token_width,
            self.feature_channels,
        )
        features = features.permute(0, 2, 3, 1, 4).reshape(
            batch_size, self.token_height, self.token_width, self.input_channels
        )
        patches = self.readout(self.norm(features))
        patch = self.latent_patch_size
        return (
            patches.reshape(
                batch_size,
                self.token_height,
                self.token_width,
                self.latent_frames,
                self.latent_channels,
                patch,
                patch,
            )
            .permute(0, 4, 3, 1, 5, 2, 6)
            .reshape(
                batch_size,
                self.latent_channels,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            )
            .contiguous()
        )

    def config_dict(self) -> dict[str, Any]:
        """Return the geometry needed to reconstruct this readout."""
        return {
            "feature_channels": self.feature_channels,
            "feature_frames": self.feature_frames,
            "token_height": self.token_height,
            "token_width": self.token_width,
            "latent_channels": self.latent_channels,
            "latent_frames": self.latent_frames,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "latent_patch_size": self.latent_patch_size,
            "input_channels": self.input_channels,
            "patch_channels": self.patch_channels,
        }


class WorldExpert(nn.Module):
    """SmolExpert-shaped bidirectional diffusion transformer for future latents.

    The transformer layers are compatible with the SmolVLA expert checkpoint,
    while the latent input/output, position, timestep, and Cosmos-prefix
    projections are specific to the world-prediction task.
    """

    def __init__(
        self,
        *,
        expert: nn.Module,
        prefix_hidden_size: int = DEFAULT_PREFIX_HIDDEN_SIZE,
        expert_hidden_size: int = DEFAULT_EXPERT_HIDDEN_SIZE,
        kv_dim: int = DEFAULT_KV_DIM,
        num_attention_heads: int = DEFAULT_NUM_ATTENTION_HEADS,
        num_key_value_heads: int = DEFAULT_NUM_KEY_VALUE_HEADS,
        head_dim: int = DEFAULT_HEAD_DIM,
        self_attn_every_n_layers: int = SELF_ATTN_EVERY_N_LAYERS,
        input_channels: int = 2048,
        latent_channels: int = WORLD_LATENT_CHANNELS,
        latent_frames: int = WORLD_LATENT_FRAMES,
        latent_height: int = WORLD_LATENT_HEIGHT,
        latent_width: int = WORLD_LATENT_WIDTH,
        latent_patch_size: int = DEFAULT_LATENT_PATCH_SIZE,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        rope_theta: float = 100_000.0,
    ) -> None:
        super().__init__()
        if latent_patch_size <= 0:
            raise ValueError("latent_patch_size must be positive")
        if latent_height % latent_patch_size or latent_width % latent_patch_size:
            raise ValueError("latent_patch_size must divide both latent spatial dimensions")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("attention head count must be divisible by key/value head count")
        if kv_dim != num_key_value_heads * head_dim:
            raise ValueError("kv_dim must equal num_key_value_heads * head_dim")
        if self_attn_every_n_layers <= 0:
            raise ValueError("self_attn_every_n_layers must be positive")
        parameter = next(expert.parameters(), None)
        if parameter is None:
            raise ValueError("expert must contain parameters")
        self.expert = expert
        self.context_adapter = CosmosPrefixAdapter(input_channels, prefix_hidden_size).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.prefix_key_projection = nn.Linear(prefix_hidden_size, kv_dim).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.prefix_value_projection = nn.Linear(prefix_hidden_size, kv_dim).to(
            device=parameter.device, dtype=parameter.dtype
        )
        patch_channels = latent_channels * latent_patch_size * latent_patch_size
        token_count = (
            latent_frames * (latent_height // latent_patch_size) * (latent_width // latent_patch_size)
        )
        self.latent_in_proj = nn.Linear(patch_channels, expert_hidden_size).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.latent_pos_embedding = nn.Parameter(
            torch.empty(token_count, expert_hidden_size, device=parameter.device, dtype=parameter.dtype)
        )
        nn.init.normal_(self.latent_pos_embedding, mean=0.0, std=0.02)
        self.latent_out_proj = nn.Linear(expert_hidden_size, patch_channels).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.latent_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.latent_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size).to(
            device=parameter.device, dtype=parameter.dtype
        )
        self.prefix_hidden_size = prefix_hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.kv_dim = kv_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.latent_patch_size = latent_patch_size
        self.token_count = token_count
        self.patch_channels = patch_channels
        self.min_period = min_period
        self.max_period = max_period
        self.rope_theta = rope_theta

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = SMOLVLA_CHECKPOINT,
        *,
        vlm_config_name: str = SMOLVLM_CONFIG,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> WorldExpert:
        """Build the SmolVLA expert architecture and load its transformer weights."""
        expert, dimensions = _build_pretrained_expert(vlm_config_name=vlm_config_name, device="cpu")
        world_expert = cls(expert=expert, **dimensions, **kwargs)
        world_expert.load_pretrained_expert(checkpoint)
        return world_expert.to(device=device)

    def load_pretrained_expert(self, checkpoint: str | Path = SMOLVLA_CHECKPOINT) -> dict[str, str]:
        """Load only the transformer layers from a SmolVLA checkpoint."""
        path = resolve_checkpoint(checkpoint)
        digests: dict[str, str] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            digests.update(
                _load_module_from_checkpoint(self.expert, handle, "model.vlm_with_expert.lm_expert.")
            )
        return digests

    def _validate_latents(self, value: Tensor) -> None:
        expected = (
            value.shape[0],
            self.latent_channels,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
        )
        if value.ndim != 5 or tuple(value.shape) != expected:
            raise ValueError(
                f"future latents must have shape [B, {self.latent_channels}, "
                f"{self.latent_frames}, {self.latent_height}, {self.latent_width}], got {tuple(value.shape)}"
            )

    def _validate_sigma(self, sigma: Tensor, batch_size: int, device: torch.device) -> Tensor:
        sigma = sigma.reshape(-1).to(device=device, dtype=torch.float32)
        if (
            sigma.shape != (batch_size,)
            or not torch.isfinite(sigma).all().item()
            or (sigma <= 0).any().item()
        ):
            raise ValueError("sigma must have finite positive shape [B]")
        return sigma

    def _patchify(self, latents: Tensor) -> Tensor:
        batch_size, channels, frames, height, width = latents.shape
        patch = self.latent_patch_size
        return (
            latents.reshape(batch_size, channels, frames, height // patch, patch, width // patch, patch)
            .permute(0, 2, 3, 5, 1, 4, 6)
            .reshape(batch_size, self.token_count, self.patch_channels)
        )

    def _unpatchify(self, tokens: Tensor) -> Tensor:
        batch_size = tokens.shape[0]
        patch = self.latent_patch_size
        return (
            tokens.reshape(
                batch_size,
                self.latent_frames,
                self.latent_height // patch,
                self.latent_width // patch,
                self.latent_channels,
                patch,
                patch,
            )
            .permute(0, 4, 1, 2, 5, 3, 6)
            .reshape(
                batch_size,
                self.latent_channels,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            )
        )

    def prepare_prefix_kv(self, context: Tensor) -> WorldExpertPrefixKVCache:
        """Project Cosmos features into exact per-layer prefix K/V tensors."""
        context_hidden = self.context_adapter(context)
        prefix_key = self.prefix_key_projection(
            context_hidden.to(dtype=self.prefix_key_projection.weight.dtype)
        )
        prefix_value = self.prefix_value_projection(
            context_hidden.to(dtype=self.prefix_value_projection.weight.dtype)
        )
        batch_size, prefix_length = prefix_key.shape[:2]
        positions = torch.arange(prefix_length, device=prefix_key.device, dtype=torch.long)[None].expand(
            batch_size, -1
        )
        prefix_key = apply_rope(
            prefix_key.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim),
            positions,
            max_wavelength=self.rope_theta,
        )
        prefix_value = prefix_value.reshape(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for layer_index, layer in enumerate(self.expert.layers):
            if layer_index % self.self_attn_every_n_layers == 0:
                key, value = prefix_key, prefix_value
            else:
                key_input = prefix_key.reshape(batch_size, prefix_length, self.kv_dim)
                value_input = prefix_value.reshape(batch_size, prefix_length, self.kv_dim)
                key = layer.self_attn.k_proj(key_input.to(dtype=layer.self_attn.k_proj.weight.dtype))
                value = layer.self_attn.v_proj(value_input.to(dtype=layer.self_attn.v_proj.weight.dtype))
                key = key.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim)
                value = value.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim)
            keys.append(key)
            values.append(value)
        return WorldExpertPrefixKVCache(tuple(keys), tuple(values))

    def _attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
        output = functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return output.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)

    def _vector_field(
        self,
        scaled_noisy_latents: Tensor,
        c_noise: Tensor,
        prefix: WorldExpertPrefixKVCache,
    ) -> Tensor:
        if len(prefix.keys) != len(self.expert.layers):
            raise ValueError("prefix K/V cache depth does not match the expert")
        tokens = self._patchify(scaled_noisy_latents)
        hidden = self.latent_in_proj(tokens.to(dtype=self.latent_in_proj.weight.dtype))
        hidden = hidden + self.latent_pos_embedding[None]
        time = c_noise.reshape(-1).to(device=hidden.device, dtype=torch.float32)
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.expert_hidden_size,
            self.min_period,
            self.max_period,
            device=hidden.device,
        ).to(dtype=hidden.dtype)
        time_emb = time_emb[:, None, :].expand_as(hidden)
        hidden = self.latent_time_mlp_in(torch.cat([hidden, time_emb], dim=-1))
        hidden = functional.silu(hidden)
        hidden = self.latent_time_mlp_out(hidden)
        sequence_length = hidden.shape[1]
        batch_size = hidden.shape[0]
        for layer_index, layer in enumerate(self.expert.layers):
            prefix_key = prefix.keys[layer_index]
            prefix_value = prefix.values[layer_index]
            normalized = layer.input_layernorm(hidden.to(dtype=layer.input_layernorm.weight.dtype))
            query = layer.self_attn.q_proj(normalized.to(dtype=layer.self_attn.q_proj.weight.dtype))
            query = query.reshape(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
            if layer_index % self.self_attn_every_n_layers == 0:
                action_key = layer.self_attn.k_proj(normalized.to(dtype=layer.self_attn.k_proj.weight.dtype))
                action_value = layer.self_attn.v_proj(
                    normalized.to(dtype=layer.self_attn.v_proj.weight.dtype)
                )
                action_key = action_key.reshape(
                    batch_size, sequence_length, self.num_key_value_heads, self.head_dim
                )
                action_value = action_value.reshape(
                    batch_size, sequence_length, self.num_key_value_heads, self.head_dim
                )
                key = torch.cat([prefix_key, action_key], dim=1)
                value = torch.cat([prefix_value, action_value], dim=1)
                query_positions = torch.arange(
                    prefix_key.shape[1],
                    prefix_key.shape[1] + sequence_length,
                    device=hidden.device,
                    dtype=torch.long,
                )[None].expand(batch_size, -1)
            else:
                key, value = prefix_key, prefix_value
                query_positions = torch.arange(sequence_length, device=hidden.device, dtype=torch.long)[
                    None
                ].expand(batch_size, -1)
            query = apply_rope(query, query_positions, max_wavelength=self.rope_theta)
            attention_output = self._attention(query, key, value)
            hidden = hidden + layer.self_attn.o_proj(
                attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
            )
            residual = hidden
            hidden = layer.post_attention_layernorm(
                hidden.to(dtype=layer.post_attention_layernorm.weight.dtype)
            )
            mlp_dtype = next(layer.mlp.parameters()).dtype
            hidden = layer.mlp(hidden.to(dtype=mlp_dtype))
            hidden = hidden + residual
        hidden = self.expert.norm(hidden)
        output = self.latent_out_proj(hidden.to(dtype=self.latent_out_proj.weight.dtype))
        return self._unpatchify(output)

    def _scaling(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        from ._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling

        scaling = RectifiedFlowScaling(sigma_data=1.0, t_scaling_factor=1.0)
        sigma_5d = sigma[:, None, None, None, None].expand(sigma.shape[0], 1, self.latent_frames, 1, 1)
        return scaling(sigma_5d)

    def denoise(
        self,
        x_sigma: Tensor,
        sigma: Tensor,
        prefix_kv: WorldExpertPrefixKVCache,
    ) -> Tensor:
        """Predict rectified-flow velocity from raw ``x0 + sigma * epsilon``."""
        self._validate_latents(x_sigma)
        sigma = self._validate_sigma(sigma, x_sigma.shape[0], x_sigma.device)
        _, _, c_in, c_noise = self._scaling(sigma)
        scaled = x_sigma * c_in.to(dtype=x_sigma.dtype)
        return self._vector_field(scaled, c_noise[:, 0, 0, 0, 0], prefix_kv)

    def predict_x0(
        self,
        x_sigma: Tensor,
        sigma: Tensor,
        prefix_kv: WorldExpertPrefixKVCache,
    ) -> Tensor:
        """Convert predicted velocity to the denoiser's x0 estimate."""
        self._validate_latents(x_sigma)
        sigma = self._validate_sigma(sigma, x_sigma.shape[0], x_sigma.device)
        c_skip, c_out, _, _ = self._scaling(sigma)
        velocity = self.denoise(x_sigma, sigma, prefix_kv)
        return c_skip.to(dtype=x_sigma.dtype) * x_sigma + c_out.to(dtype=x_sigma.dtype) * velocity

    def config_dict(self) -> dict[str, Any]:
        """Return the geometry needed to reconstruct this expert."""
        return {
            "input_channels": self.input_channels,
            "prefix_hidden_size": self.prefix_hidden_size,
            "expert_hidden_size": self.expert_hidden_size,
            "kv_dim": self.kv_dim,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_layers": len(self.expert.layers),
            "self_attn_every_n_layers": self.self_attn_every_n_layers,
            "latent_channels": self.latent_channels,
            "latent_frames": self.latent_frames,
            "latent_height": self.latent_height,
            "latent_width": self.latent_width,
            "latent_patch_size": self.latent_patch_size,
            "token_count": self.token_count,
            "patch_channels": self.patch_channels,
        }
