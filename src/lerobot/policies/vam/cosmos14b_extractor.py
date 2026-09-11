"""Cosmos-1.0-Diffusion-14B feature extractor with block streaming and FP8 quantization.

Extracts intermediate representations from NVIDIA Cosmos-1.0-Diffusion-14B for
physical AI policy learning (e.g., SmolExpert). Supports:
- 14B architecture: 36 layers, 40 attention heads, head dim 128 (5120 hidden dim)
- Block streaming: streams individual transformer blocks between host CPU memory and CUDA
  to keep peak VRAM well within 24 GB GPUs (e.g. RTX 4090)
- FP8 quantization: converts linear layer weights to torch.float8_e4m3fn, reducing weight
  memory footprint from ~28.5 GB to ~14.25 GB
- Native EDM input preconditioning (c_in scaling) and deterministic AutoencoderKLCosmos
- 3D spatiotemporal grid & mRoPE coordinate indexing
- Spatial average pooling and layer concatenation (e.g. layers 18 and 30)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from torch import Tensor, nn
from torch.nn import functional

from lerobot.policies.vam.base import (
    BaseExtractorConfig,
    BaseVAMExtractor,
    VAMExtractionOutput,
)
from lerobot.policies.vam.base.native_video import CosmosEDMScaling, CosmosVAENormalizer


class Cosmos14BError(RuntimeError):
    """Base exception for Cosmos 14B extractor operations."""


class FP8Linear(nn.Module):
    """Memory-efficient Linear module storing weights in torch.float8_e4m3fn."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any = "cpu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device), requires_grad=False)
            if bias
            else None
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> FP8Linear:
        dev = linear.weight.device
        fp8_mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=dev,
        )
        fp8_mod.weight.data.copy_(linear.weight.data.to(torch.float8_e4m3fn))
        if linear.bias is not None:
            fp8_mod.bias.data.copy_(linear.bias.data.to(torch.bfloat16))
        return fp8_mod

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        w_dequant = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        out = functional.linear(x_2d, w_dequant, b)
        return out.reshape(*orig_shape[:-1], self.out_features)


def replace_linears_with_fp8(module: nn.Module) -> int:
    """Recursively replace nn.Linear with FP8Linear modules."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FP8Linear.from_linear(child))
            count += 1
        else:
            count += replace_linears_with_fp8(child)
    return count


class StreamingBlockList(nn.ModuleList):
    """Transformer block list that streams active blocks between host CPU memory and CUDA."""

    def __init__(
        self, blocks: list[nn.Module], device: str | torch.device = "cuda:0", offload_device: str = "cpu"
    ):
        super().__init__(blocks)
        self.target_device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        for b in self:
            b.to(self.offload_device)

    def __iter__(self):
        active = None
        for b in super().__iter__():
            if active is not None:
                active.to(self.offload_device)
            b.to(self.target_device)
            active = b
            yield b
        if active is not None:
            active.to(self.offload_device)


@dataclass(frozen=True, slots=True)
class Cosmos14BExtractorConfig(BaseExtractorConfig):
    """Configuration for Cosmos-1.0-Diffusion-14B feature extraction."""

    backbone_name: str = "cosmos14b"
    checkpoint_path: str | Path | None = None
    vae_path: str | Path | None = Path("/home/anton/.cache/video-vam/cosmos-14b/vae")
    device: str = "cuda:0"
    dtype: str = "bfloat16"  # "float32" or "bfloat16"
    hidden_layers: tuple[int, ...] = (18, 30)  # Tapped intermediate blocks (out of 36)
    pool_spatial: int | None = 2  # 2x2 spatial avg pooling
    concat_layers: bool = True  # Concatenate layer 18 & 30 along channel dimension
    state_t: int = 2
    latent_channels: int = 16
    extra_kwargs: dict[str, Any] = field(default_factory=dict)
    patch_size: tuple[int, int, int] = (1, 2, 2)
    in_channels: int = 17  # 16 + 1 for padding mask (concat_padding_mask=True)
    out_channels: int = 16
    num_layers: int = 36
    num_attention_heads: int = 40
    attention_head_dim: int = 128  # 40 * 128 = 5120 hidden_dim
    hidden_dim: int = 5120
    text_embed_dim: int = 1024
    rope_scale: tuple[float, float, float] = (2.0, 2.0, 2.0)
    block_streaming: bool = True  # Stream blocks to/from CPU to limit peak VRAM
    fp8_linear: bool = True  # Quantize linear layers to FP8 for memory & compute efficiency
    sigma_data: float = 0.5
    fps: int = 10

    def __post_init__(self) -> None:
        BaseExtractorConfig.__post_init__(self)
        for layer_idx in self.hidden_layers:
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(
                    f"hidden_layer {layer_idx} out of range for {self.num_layers}-layer Cosmos 14B"
                )
        if self.attention_head_dim * self.num_attention_heads != self.hidden_dim:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must equal "
                f"num_attention_heads * attention_head_dim ({self.num_attention_heads * self.attention_head_dim})"
            )


@dataclass(frozen=True, slots=True)
class Cosmos14BExtractionOutput(VAMExtractionOutput):
    """Output container for extracted Cosmos 14B intermediate representations."""


def build_cosmos14b_dummy_model(
    *,
    num_layers: int = 36,
    num_attention_heads: int = 4,
    attention_head_dim: int = 16,
    text_embed_dim: int = 32,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> CosmosTransformer3DModel:
    """Construct a lightweight architecture-matched model for testing and shape validation."""
    return CosmosTransformer3DModel(
        in_channels=17,
        out_channels=16,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        text_embed_dim=text_embed_dim,
        crossattn_proj_in_channels=text_embed_dim,
        encoder_hidden_states_channels=text_embed_dim,
        patch_size=(1, 2, 2),
        rope_scale=(2.0, 2.0, 2.0),
        concat_padding_mask=False,
        extra_pos_embed_type=None,
    ).to(device=device, dtype=dtype)


class Cosmos14BExtractor(BaseVAMExtractor):
    """Intermediate feature extractor for NVIDIA Cosmos-1.0-Diffusion-14B with streaming & FP8."""

    def __init__(
        self,
        config: Cosmos14BExtractorConfig | None = None,
        model: CosmosTransformer3DModel | None = None,
        vae: Any | None = None,
    ) -> None:
        cfg = config or Cosmos14BExtractorConfig()
        super().__init__(cfg)
        self.config: Cosmos14BExtractorConfig = cfg
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
            subfolder = "transformer" if (Path(ckpt_str) / "transformer").is_dir() else None
            if self.config.block_streaming:
                self.transformer = CosmosTransformer3DModel.from_pretrained(
                    ckpt_str,
                    subfolder=subfolder,
                    torch_dtype=self.dtype,
                )
            else:
                self.transformer = CosmosTransformer3DModel.from_pretrained(
                    ckpt_str,
                    subfolder=subfolder,
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
            ).to(device=self.device if not self.config.block_streaming else "cpu", dtype=self.dtype)

        # Apply FP8 quantization if enabled
        if self.config.fp8_linear:
            num_fp8 = replace_linears_with_fp8(self.transformer)
            print(f"[Cosmos 14B Extractor] Replaced {num_fp8} linear layers with FP8 modules.")

        # Apply block streaming if enabled
        if self.config.block_streaming and hasattr(self.transformer, "transformer_blocks"):
            blocks = list(self.transformer.transformer_blocks)
            self.transformer.transformer_blocks = StreamingBlockList(
                blocks,
                device=self.device,
                offload_device="cpu",
            )
            for name, child in self.transformer.named_children():
                if name != "transformer_blocks":
                    child.to(self.device)
            print(
                f"[Cosmos 14B Extractor] Enabled block streaming for {len(blocks)} blocks (offload to CPU)."
            )

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
                raise Cosmos14BError(
                    f"VAE model not available at {vae_path} and no vae passed to Cosmos14BExtractor."
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
        b, c, t, h, w = latents.shape

        # Native EDM preconditioning
        if isinstance(timestep, (int, float)):
            sigma_val = max(1e-4, float(timestep))
            sigma_t = torch.tensor([sigma_val], device=self.device, dtype=torch.float32)
        else:
            sigma_t = timestep.to(device=self.device, dtype=torch.float32)
            if sigma_t.ndim == 0:
                sigma_t = sigma_t.unsqueeze(0)
            sigma_t = sigma_t.clamp(min=1e-4)

        c_skip, c_out, c_in, c_noise = self.edm_scaling(sigma_t.view(-1, 1, 1, 1, 1))
        scaled_latents = (latents * c_in).to(device=self.device, dtype=self.dtype)
        t_input = c_noise.flatten().to(dtype=torch.float32)

        # Append padding channel if transformer requires 17 channels
        in_ch = getattr(self.transformer.config, "in_channels", self.config.in_channels)
        if in_ch == 17 and scaled_latents.shape[1] == 16:
            mask = torch.ones(b, 1, t, h, w, device=self.device, dtype=self.dtype)
            scaled_latents = torch.cat([scaled_latents, mask], dim=1)

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

        padding_mask = None
        if getattr(self.transformer.config, "concat_padding_mask", False):
            padding_mask = torch.ones(1, 1, h, w, device=self.device, dtype=self.dtype)

        captured_layers: dict[int, Tensor] = {}
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(module: nn.Module, inp: Any, out: Tensor) -> None:
                captured_layers[layer_idx] = out.detach().to("cpu")

            return hook_fn

        for layer_idx in self.config.hidden_layers:
            if layer_idx < len(self.transformer.transformer_blocks):
                blk = self.transformer.transformer_blocks[layer_idx]
                hooks.append(blk.register_forward_hook(make_hook(layer_idx)))

        try:
            kwargs: dict[str, Any] = {
                "hidden_states": scaled_latents,
                "timestep": t_input,
                "encoder_hidden_states": text_embeds,
                "fps": fps,
                "return_dict": True,
            }
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask
            self.transformer(**kwargs)
        finally:
            for h_hook in hooks:
                h_hook.remove()

        return {k: v.to(self.device) for k, v in captured_layers.items()}

    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        b, c, t, h, w = latents.shape
        p_t, p_h, p_w = self.config.patch_size
        return (t // p_t, h // p_h, w // p_w)

    def get_provenance(self) -> dict[str, Any]:
        return {
            "model_type": "Cosmos-1.0-Diffusion-14B",
            "hidden_layers": list(self.config.hidden_layers),
            "patch_size": list(self.config.patch_size),
            "block_streaming": self.config.block_streaming,
            "fp8_linear": self.config.fp8_linear,
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
    ) -> Cosmos14BExtractionOutput:
        """Extract intermediate features from specified layers (e.g. 18 and 30)."""
        inp_latents = hidden_states if hidden_states is not None else latents
        base_out = super().extract(
            rgb_frames=rgb_frames,
            latents=inp_latents,
            text_conditioning=text_conditioning,
            timestep=timestep,
            fps=fps,
        )
        return Cosmos14BExtractionOutput(
            features=base_out.features,
            features_by_layer=base_out.features_by_layer,
            grid_features_by_layer=base_out.grid_features_by_layer,
            grid_coords=base_out.grid_coords,
            grid_shape=base_out.grid_shape,
            provenance=base_out.provenance,
        )
