"""FLUX.2 [klein] (9B) feature extractor with multi-reference conditioning and dual-tap points.

Integrates multi-reference conditioning for robotics observation history (e.g. t-2, t-1, t)
and optional goal frames with 4D time-space RoPE coordinates.
Guarantees causal observation extraction (no future goal frame leakage into past history).
Supports two feature extraction tapped locations:
  a) Junction between double-stream and single-stream transformer blocks.
  b) Intermediate / later step of the rectified flow generative trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from torch import Tensor, nn

from lerobot.policies.vam.base import (
    BaseExtractorConfig,
    BaseVAMExtractor,
    VAMExtractionOutput,
)
from lerobot.policies.vam.base.native_video import FluxVAENormalizer


class Flux2KleinError(RuntimeError):
    """Base exception for FLUX.2 [klein] operations."""


class Flux2KleinTapLocation(StrEnum):
    JUNCTION = "junction"
    RECTIFIED_FLOW_TRAJECTORY = "rectified_flow_trajectory"


@dataclass(frozen=True, slots=True)
class Flux2KleinExtractorConfig(BaseExtractorConfig):
    """Configuration for FLUX.2 [klein] 9B feature extraction."""

    backbone_name: str = "flux2_klein"
    checkpoint_path: str | Path | None = None
    vae_path: str | Path | None = None
    device: str = "cpu"
    dtype: str = "float32"
    hidden_layers: tuple[int, ...] = (0,)
    pool_spatial: int | None = None
    concat_layers: bool = False
    state_t: int = 2
    latent_channels: int = 128
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    patch_size: int = 1
    in_channels: int = 128
    num_layers: int = 8  # double-stream blocks
    num_single_layers: int = 48  # single-stream blocks
    attention_head_dim: int = 128
    num_attention_heads: int = 48  # 48 * 128 = 6144 hidden dimension
    hidden_dim: int = 6144
    joint_attention_dim: int = 15360
    axes_dims_rope: tuple[int, int, int, int] = (32, 32, 32, 32)
    tap_location: str = "junction"  # "junction" or "rectified_flow_trajectory"
    rf_num_steps: int = 4  # total steps in RF trajectory
    rf_tap_step: int = 2  # step index to tap (e.g. step 2 of 4 = intermediate/later)
    ref_time_scale: int = 10  # spacing factor for reference time coordinates
    goal_time_coord: int = 100  # distinct time coordinate for goal frame
    fps: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "latent_channels", self.in_channels)
        BaseExtractorConfig.__post_init__(self)
        if self.attention_head_dim * self.num_attention_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal "
                f"num_attention_heads * attention_head_dim ({self.num_attention_heads * self.attention_head_dim})"
            )
        if self.tap_location not in ("junction", "rectified_flow_trajectory"):
            raise ValueError(
                f"tap_location must be 'junction' or 'rectified_flow_trajectory', got {self.tap_location!r}"
            )
        if self.rf_tap_step < 0 or self.rf_tap_step >= self.rf_num_steps:
            raise ValueError(
                f"rf_tap_step ({self.rf_tap_step}) must be between 0 and {self.rf_num_steps - 1}"
            )


@dataclass(frozen=True, slots=True)
class MultiReferenceInputs:
    """Multi-reference conditioned input tensors for FLUX.2 transformer."""

    hidden_states: Tensor  # [B, N_target + N_ref, in_channels]
    img_ids: Tensor  # [B, N_target + N_ref, 4]
    encoder_hidden_states: Tensor  # [B, N_txt, joint_attention_dim]
    txt_ids: Tensor  # [B, N_txt, 4]
    target_tokens_count: int
    history_tokens_count: int
    goal_tokens_count: int
    token_slices: dict[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class Flux2KleinExtractionOutput(VAMExtractionOutput):
    """Output container for extracted FLUX.2 [klein] representations."""

    visual_tokens: Tensor | None = None  # [B, N_img, hidden_dim] visual sequence tokens
    text_tokens: Tensor | None = None  # [B, N_txt, hidden_dim] text sequence tokens if available
    junction_features: Tensor | None = None  # [B, N_txt + N_img, hidden_dim] full junction
    token_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    tap_location: str = "junction"
    trajectory_step: int | None = None

    @property
    def batch_size(self) -> int:
        return self.features.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.features.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]


def build_flux2_klein_dummy_model(
    *,
    num_layers: int = 2,
    num_single_layers: int = 2,
    num_attention_heads: int = 2,
    attention_head_dim: int = 16,
    in_channels: int = 16,
    joint_attention_dim: int = 32,
    axes_dims_rope: tuple[int, int, int, int] = (4, 4, 4, 4),
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Flux2Transformer2DModel:
    """Construct an architecture-matched lightweight FLUX.2 model for CPU synthetic tests."""
    return Flux2Transformer2DModel(
        patch_size=1,
        in_channels=in_channels,
        out_channels=None,
        num_layers=num_layers,
        num_single_layers=num_single_layers,
        attention_head_dim=attention_head_dim,
        num_attention_heads=num_attention_heads,
        joint_attention_dim=joint_attention_dim,
        timestep_guidance_channels=16,
        mlp_ratio=2.0,
        axes_dims_rope=axes_dims_rope,
        rope_theta=2000,
        eps=1e-6,
        guidance_embeds=True,
    ).to(device=device, dtype=dtype)


def prepare_multi_reference_conditioning(
    target_latent: Tensor,  # [B, C, H, W]
    observation_history: list[Tensor] | Tensor,  # list of [B, C, H, W] for e.g. [t-2, t-1, t]
    goal_latent: Tensor | None = None,  # [B, C, H, W] optional goal conditioning
    encoder_hidden_states: Tensor | None = None,  # [B, N_txt, joint_attention_dim]
    joint_attention_dim: int = 15360,
    ref_time_scale: int = 10,
    goal_time_coord: int = 100,
    causal_only: bool = False,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MultiReferenceInputs:
    """Format observation history (e.g. t-2, t-1, t) and optional goal into FLUX.2 multi-reference inputs.

    Guarantees that when , no future goal frame is included, preserving strict causality.
    Assigns distinct 4D time-space coordinates (T, H, W, L) to each image latent:
      - target_latent: T = 0
      - history frame k: T = ref_time_scale * (k + 1)  (e.g. 10, 20, 30)
      - goal frame: T = goal_time_coord (e.g. 100, only if not causal_only)
    """
    device = torch.device(device)
    if isinstance(observation_history, Tensor):
        if observation_history.ndim == 5:
            if observation_history.shape[1] in (16, 128):  # [B, C, T, H, W]
                frames = [observation_history[:, :, i] for i in range(observation_history.shape[2])]
            else:  # [B, T, C, H, W]
                frames = [observation_history[:, i] for i in range(observation_history.shape[1])]
        else:
            frames = [observation_history]
    else:
        frames = list(observation_history)

    target_latent = target_latent.to(device=device, dtype=dtype)
    b, c, h, w = target_latent.shape

    # 1. Pack target latent: (B, C, H, W) -> (B, H*W, C)
    target_packed = target_latent.reshape(b, c, h * w).permute(0, 2, 1)
    target_count = h * w

    # Target 4D RoPE coordinate IDs: (T=0, H, W, L=0)
    t_coord = torch.zeros(1, device=device, dtype=torch.long)
    target_ids = (
        torch.cartesian_prod(
            t_coord,
            torch.arange(h, device=device, dtype=torch.long),
            torch.arange(w, device=device, dtype=torch.long),
            torch.arange(1, device=device, dtype=torch.long),
        )
        .unsqueeze(0)
        .expand(b, -1, -1)
        .float()
    )

    all_packed = [target_packed]
    all_ids = [target_ids]
    token_slices = {"target": (0, target_count)}
    curr_offset = target_count

    # 2. Pack observation history references (e.g. t-2, t-1, t)
    history_count = 0
    for idx, ref in enumerate(frames):
        ref = ref.to(device=device, dtype=dtype)
        ref_packed = ref.reshape(b, c, h * w).permute(0, 2, 1)
        ref_t_val = int(ref_time_scale * (idx + 1))
        ref_t = torch.tensor([ref_t_val], device=device, dtype=torch.long)
        ref_ids = (
            torch.cartesian_prod(
                ref_t,
                torch.arange(h, device=device, dtype=torch.long),
                torch.arange(w, device=device, dtype=torch.long),
                torch.arange(1, device=device, dtype=torch.long),
            )
            .unsqueeze(0)
            .expand(b, -1, -1)
            .float()
        )

        all_packed.append(ref_packed)
        all_ids.append(ref_ids)
        frame_name = f"history_t_minus_{len(frames) - 1 - idx}" if idx < len(frames) - 1 else "history_t"
        token_slices[frame_name] = (curr_offset, curr_offset + (h * w))
        curr_offset += h * w
        history_count += h * w

    # 3. Optional goal frame (strictly omitted if causal_only)
    goal_count = 0
    if goal_latent is not None and not causal_only:
        goal_latent = goal_latent.to(device=device, dtype=dtype)
        goal_packed = goal_latent.reshape(b, c, h * w).permute(0, 2, 1)
        goal_t = torch.tensor([int(goal_time_coord)], device=device, dtype=torch.long)
        goal_ids = (
            torch.cartesian_prod(
                goal_t,
                torch.arange(h, device=device, dtype=torch.long),
                torch.arange(w, device=device, dtype=torch.long),
                torch.arange(1, device=device, dtype=torch.long),
            )
            .unsqueeze(0)
            .expand(b, -1, -1)
            .float()
        )

        all_packed.append(goal_packed)
        all_ids.append(goal_ids)
        token_slices["goal"] = (curr_offset, curr_offset + (h * w))
        curr_offset += h * w
        goal_count += h * w

    full_hidden_states = torch.cat(all_packed, dim=1)  # [B, N_total, C]
    full_img_ids = torch.cat(all_ids, dim=1)  # [B, N_total, 4]

    # 4. Text conditioning
    if encoder_hidden_states is None:
        txt_seq_len = 16
        encoder_hidden_states = torch.zeros(b, txt_seq_len, joint_attention_dim, device=device, dtype=dtype)
    else:
        encoder_hidden_states = encoder_hidden_states.to(device=device, dtype=dtype)
        txt_seq_len = encoder_hidden_states.shape[1]

    # Text RoPE IDs: 4D coordinates
    txt_ids = torch.zeros(b, txt_seq_len, 4, device=device)

    return MultiReferenceInputs(
        hidden_states=full_hidden_states,
        img_ids=full_img_ids,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        target_tokens_count=target_count,
        history_tokens_count=history_count,
        goal_tokens_count=goal_count,
        token_slices=token_slices,
    )


class Flux2KleinExtractor(BaseVAMExtractor):
    """FLUX.2 [klein] (9B) feature extractor with multi-reference conditioning."""

    def __init__(
        self,
        config: Flux2KleinExtractorConfig | None = None,
        model: Flux2Transformer2DModel | None = None,
        vae: Any | None = None,
    ) -> None:
        cfg = config or Flux2KleinExtractorConfig()
        super().__init__(cfg)
        self.config: Flux2KleinExtractorConfig = cfg
        self.normalizer = FluxVAENormalizer(
            vae=vae,
            device=self.device,
            dtype=self.dtype,
        )

        if model is not None:
            self.transformer = model.to(device=self.device, dtype=self.dtype)
        elif self.config.checkpoint_path is not None:
            self.transformer = Flux2Transformer2DModel.from_pretrained(
                str(self.config.checkpoint_path),
                torch_dtype=self.dtype,
            ).to(self.device)
        else:
            self.transformer = Flux2Transformer2DModel(
                patch_size=self.config.patch_size,
                in_channels=self.config.in_channels,
                num_layers=self.config.num_layers,
                num_single_layers=self.config.num_single_layers,
                attention_head_dim=self.config.attention_head_dim,
                num_attention_heads=self.config.num_attention_heads,
                joint_attention_dim=self.config.joint_attention_dim,
                axes_dims_rope=self.config.axes_dims_rope,
            ).to(device=self.device, dtype=self.dtype)

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
        """Encode raw RGB video into FLUX.2 latent space using native VAE normalizer."""
        if self.normalizer.vae is not None:
            return self.normalizer.encode(rgb_frames, deterministic=True)

        if self.config.vae_path is not None and Path(self.config.vae_path).exists():
            self.normalizer = FluxVAENormalizer.from_pretrained(
                vae_path=self.config.vae_path,
                device=self.device,
                dtype=self.dtype,
            )
            return self.normalizer.encode(rgb_frames, deterministic=True)

        return rgb_frames.to(device=self.device, dtype=self.dtype)

    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 500.0,
    ) -> dict[int, Tensor]:
        """Forward pass tapping the junction layer (or layer 0)."""
        if latents.ndim == 5:
            target = latents[:, :, -1]
            history = [latents[:, :, i] for i in range(latents.shape[2] - 1)]
        elif latents.ndim == 4:
            target = latents
            history = [latents]
        else:
            target = latents
            history = []

        cond = prepare_multi_reference_conditioning(
            target_latent=target,
            observation_history=history,
            joint_attention_dim=self.config.joint_attention_dim,
            causal_only=True,
            device=self.device,
            dtype=self.dtype,
        )
        out = self.extract_junction(cond, timestep=timestep)
        return {0: out.features}

    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        if latents.ndim == 5:
            return (latents.shape[2], latents.shape[3], latents.shape[4])
        elif latents.ndim == 4:
            return (1, latents.shape[2], latents.shape[3])
        return (1, 8, 8)

    def get_provenance(self) -> dict[str, Any]:
        return {
            "model_type": "FLUX.2-klein-9B",
            "tap_location": self.config.tap_location,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }

    @torch.no_grad()
    def extract_junction(
        self,
        cond_inputs: MultiReferenceInputs,
        timestep: Tensor | float = 500.0,
        guidance: Tensor | float = 3.5,
    ) -> Flux2KleinExtractionOutput:
        """Extract features at the junction between double-stream and single-stream blocks."""
        if isinstance(timestep, (int, float)):
            t_tensor = torch.tensor([float(timestep)], device=self.device, dtype=torch.float32)
        else:
            t_tensor = timestep.to(device=self.device, dtype=torch.float32)
        if t_tensor.ndim == 0:
            t_tensor = t_tensor.unsqueeze(0)

        if isinstance(guidance, (int, float)):
            g_tensor = torch.tensor([float(guidance)], device=self.device, dtype=torch.float32)
        else:
            g_tensor = guidance.to(device=self.device, dtype=torch.float32)
        if g_tensor.ndim == 0:
            g_tensor = g_tensor.unsqueeze(0)

        captured_double_out: list[tuple[Tensor, Tensor]] = []

        last_double_block = self.transformer.transformer_blocks[-1]

        def hook_fn(module: nn.Module, inp: Any, out: tuple[Tensor, Tensor]) -> None:
            enc_hs, hs = out
            captured_double_out.append((enc_hs.detach(), hs.detach()))

        handle = last_double_block.register_forward_hook(hook_fn)
        try:
            self.transformer(
                hidden_states=cond_inputs.hidden_states,
                encoder_hidden_states=cond_inputs.encoder_hidden_states,
                timestep=t_tensor,
                img_ids=cond_inputs.img_ids,
                txt_ids=cond_inputs.txt_ids,
                guidance=g_tensor,
                return_dict=True,
            )
        finally:
            handle.remove()

        if not captured_double_out:
            raise Flux2KleinError("Failed to capture double-stream junction output.")

        enc_hidden, visual_hidden = captured_double_out[0]
        junction_combined = torch.cat([enc_hidden, visual_hidden], dim=1)

        b, n, d = visual_hidden.shape
        grid_shape = (
            (1, int(round(n**0.5)), int(round(n**0.5))) if int(round(n**0.5)) ** 2 == n else (1, n, 1)
        )
        coords = self.compute_3d_grid_coordinates(grid_shape, batch_size=b)

        provenance = {
            "model_type": "FLUX.2-klein-9B",
            "tap_location": "junction",
            "visual_tokens": visual_hidden.shape[1],
            "text_tokens": enc_hidden.shape[1],
            "token_slices": cond_inputs.token_slices,
            "feature_dim": visual_hidden.shape[-1],
            "device": str(self.device),
        }

        return Flux2KleinExtractionOutput(
            features=visual_hidden,
            features_by_layer={0: visual_hidden},
            grid_coords=coords,
            grid_shape=grid_shape,
            provenance=provenance,
            visual_tokens=visual_hidden,
            text_tokens=enc_hidden,
            junction_features=junction_combined,
            token_slices=cond_inputs.token_slices,
            tap_location="junction",
            trajectory_step=None,
        )

    @torch.no_grad()
    def extract_rectified_flow_step(
        self,
        cond_inputs: MultiReferenceInputs,
        num_steps: int = 4,
        tap_step: int = 2,
        guidance: Tensor | float = 3.5,
    ) -> Flux2KleinExtractionOutput:
        """Extract features at an intermediate or later step of the RF generative trajectory."""
        device = self.device
        timesteps = torch.linspace(1000.0, 0.0, num_steps + 1, device=device)
        current_latents = cond_inputs.hidden_states.clone()

        tapped_output: Flux2KleinExtractionOutput | None = None

        for step_idx in range(num_steps):
            t_curr = timesteps[step_idx].view(1)
            dt = (timesteps[step_idx] - timesteps[step_idx + 1]) / 1000.0

            if step_idx == tap_step:
                tapped_cond = MultiReferenceInputs(
                    hidden_states=current_latents,
                    img_ids=cond_inputs.img_ids,
                    encoder_hidden_states=cond_inputs.encoder_hidden_states,
                    txt_ids=cond_inputs.txt_ids,
                    target_tokens_count=cond_inputs.target_tokens_count,
                    history_tokens_count=cond_inputs.history_tokens_count,
                    goal_tokens_count=cond_inputs.goal_tokens_count,
                    token_slices=cond_inputs.token_slices,
                )
                tapped_output = self.extract_junction(
                    cond_inputs=tapped_cond,
                    timestep=t_curr,
                    guidance=guidance,
                )
                break

            if isinstance(guidance, (int, float)):
                g_tensor = torch.tensor([float(guidance)], device=device, dtype=torch.float32)
            else:
                g_tensor = guidance.to(device=device, dtype=torch.float32)
            if g_tensor.ndim == 0:
                g_tensor = g_tensor.unsqueeze(0)

            out = self.transformer(
                hidden_states=current_latents,
                encoder_hidden_states=cond_inputs.encoder_hidden_states,
                timestep=t_curr,
                img_ids=cond_inputs.img_ids,
                txt_ids=cond_inputs.txt_ids,
                guidance=g_tensor,
                return_dict=True,
            )
            pred_velocity = out.sample
            current_latents = current_latents - dt * pred_velocity

        if tapped_output is None:
            raise Flux2KleinError(f"Trajectory step {tap_step} was not reached.")

        prov = dict(tapped_output.provenance)
        prov["tap_location"] = "rectified_flow_trajectory"
        prov["trajectory_step"] = tap_step
        prov["total_trajectory_steps"] = num_steps

        return Flux2KleinExtractionOutput(
            features=tapped_output.features,
            features_by_layer=tapped_output.features_by_layer,
            grid_coords=tapped_output.grid_coords,
            grid_shape=tapped_output.grid_shape,
            provenance=prov,
            visual_tokens=tapped_output.visual_tokens,
            text_tokens=tapped_output.text_tokens,
            junction_features=tapped_output.junction_features,
            token_slices=cond_inputs.token_slices,
            tap_location="rectified_flow_trajectory",
            trajectory_step=tap_step,
        )

    @torch.no_grad()
    def extract(
        self,
        cond_inputs: MultiReferenceInputs | None = None,
        timestep: Tensor | float = 500.0,
        guidance: Tensor | float = 3.5,
        rgb_frames: Tensor | None = None,
        latents: Tensor | None = None,
        hidden_states: Tensor | None = None,
        **kwargs: Any,
    ) -> Flux2KleinExtractionOutput:
        """Unified extraction dispatching according to input type and config.tap_location."""
        if cond_inputs is not None and isinstance(cond_inputs, MultiReferenceInputs):
            if self.config.tap_location == "rectified_flow_trajectory":
                return self.extract_rectified_flow_step(
                    cond_inputs=cond_inputs,
                    num_steps=self.config.rf_num_steps,
                    tap_step=self.config.rf_tap_step,
                    guidance=guidance,
                )
            return self.extract_junction(
                cond_inputs=cond_inputs,
                timestep=timestep,
                guidance=guidance,
            )

        inp_latents = hidden_states if hidden_states is not None else latents
        base_out = super().extract(
            rgb_frames=rgb_frames,
            latents=inp_latents,
            timestep=timestep,
            **kwargs,
        )
        return Flux2KleinExtractionOutput(
            features=base_out.features,
            features_by_layer=base_out.features_by_layer,
            grid_coords=base_out.grid_coords,
            grid_shape=base_out.grid_shape,
            provenance=base_out.provenance,
            visual_tokens=base_out.features,
        )
