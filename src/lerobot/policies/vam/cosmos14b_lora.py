"""Quantized Video-LoRA adaptation for NVIDIA Cosmos-1.0-Diffusion-14B with native EDM objective.

Provides Quantized Low-Rank Adaptation (QLoRA / FP8) for fine-tuning the 36-block,
14.25B-parameter Cosmos DiT backbone on robot video prediction tasks. Enables training
and activation propagation within the 24 GB VRAM budget of a single NVIDIA RTX 4090:
- FP8 linear base weights (torch.float8_e4m3fn) reducing parameter footprint to ~14 GB
- Trainable low-rank adapter projections on attention modules across all 36 transformer blocks
- Native PyTorch gradient checkpointing for activation memory bounding
- Native EDM pretraining objective with weighted x0 loss (sigma_data=0.5)
- Strict metadata serialization to safetensors format
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
from torch import Tensor, nn
from torch.nn import functional

from lerobot.policies.vam.base.native_video import (
    LoRAMetadataError,
    compute_cosmos_edm_loss,
    load_native_lora,
    save_native_lora,
)
from lerobot.policies.vam.cosmos14b_extractor import FP8Linear


class Cosmos14BLoRAError(RuntimeError):
    """Exception raised during LoRA operations on Cosmos 14B."""


@dataclass(frozen=True, slots=True)
class Cosmos14BLoRAConfig:
    """Configuration for Cosmos 14B Quantized Video-LoRA adaptation."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_blocks: tuple[int, ...] = tuple(range(36))
    target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    )
    quantization: str = "fp8"  # "fp8" or "none"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
        if self.quantization not in ("fp8", "none"):
            raise ValueError(f"Unsupported quantization: {self.quantization}. Must be 'fp8' or 'none'.")


class Cosmos14BQuantizedLoRALinear(nn.Module):
    """Quantized linear base layer augmented with a low-rank adapter branch.

    Base weights are stored in torch.float8_e4m3fn for 2x memory reduction,
    while adapter parameters (lora_A, lora_B) remain in bfloat16/float32 for gradient updates.
    """

    def __init__(
        self,
        base_layer: nn.Linear | FP8Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
        quantization: str = "fp8",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.quantization = quantization
        self.adapter_dtype = dtype

        dev = base_layer.weight.device

        # Quantize or copy base weights
        if quantization == "fp8":
            if isinstance(base_layer, FP8Linear):
                self.weight = nn.Parameter(base_layer.weight.data.clone(), requires_grad=False)
                self.bias = (
                    nn.Parameter(base_layer.bias.data.clone(), requires_grad=False)
                    if base_layer.bias is not None
                    else None
                )
            else:
                self.weight = nn.Parameter(
                    base_layer.weight.data.to(torch.float8_e4m3fn),
                    requires_grad=False,
                )
                self.bias = (
                    nn.Parameter(base_layer.bias.data.to(dtype), requires_grad=False)
                    if base_layer.bias is not None
                    else None
                )
        else:
            self.weight = nn.Parameter(base_layer.weight.data.clone(), requires_grad=False)
            self.bias = (
                nn.Parameter(base_layer.bias.data.clone(), requires_grad=False)
                if base_layer.bias is not None
                else None
            )

        # LoRA parameters: rank x in_features, out_features x rank
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=dev, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize: lora_A with Kaiming uniform, lora_B with zeros so initial delta is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)

        # Ensure base weights match activation dtype (dequantize if FP8)
        w = self.weight.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        base_out = functional.linear(x_2d, w, b)

        # LoRA residual branch
        x_drop = self.dropout(x_2d.to(self.adapter_dtype))
        lora_out = (x_drop @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

        out = base_out + lora_out.to(dtype=base_out.dtype)
        return out.reshape(*orig_shape[:-1], self.out_features)

    def merge_to_base(self) -> nn.Linear:
        """Merge LoRA update into a regular nn.Linear module."""
        w_dequant = self.weight.to(torch.float32)
        delta_w = ((self.lora_B @ self.lora_A) * self.scaling).to(torch.float32)
        merged_weight = (w_dequant + delta_w).to(self.adapter_dtype)

        merged_linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.adapter_dtype,
        )
        merged_linear.weight.data.copy_(merged_weight)
        if self.bias is not None:
            merged_linear.bias.data.copy_(self.bias.data.to(self.adapter_dtype))
        return merged_linear


def inject_cosmos14b_quantized_lora(
    model: CosmosTransformer3DModel,
    config: Cosmos14BLoRAConfig | None = None,
) -> list[str]:
    """Inject Quantized LoRA into targeted attention linear projections across transformer blocks.

    Also replaces non-targeted linear layers in transformer blocks with FP8Linear if FP8
    quantization is enabled, ensuring the full model stays within 24 GB VRAM.
    """
    cfg = config or Cosmos14BLoRAConfig()
    injected_names: list[str] = []

    for block_idx in cfg.target_blocks:
        if block_idx >= len(model.transformer_blocks):
            continue
        block = model.transformer_blocks[block_idx]

        for mod_name, module in list(block.named_modules()):
            if isinstance(module, (nn.Linear, FP8Linear)):
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)

                should_target = any(mod_name.endswith(tgt) for tgt in cfg.target_modules)
                if should_target:
                    qlora_layer = Cosmos14BQuantizedLoRALinear(
                        base_layer=module,
                        rank=cfg.rank,
                        alpha=cfg.alpha,
                        dropout=cfg.dropout,
                        quantization=cfg.quantization,
                    )
                    setattr(parent, child_name, qlora_layer)
                    full_name = f"transformer_blocks.{block_idx}.{mod_name}"
                    injected_names.append(full_name)
                elif cfg.quantization == "fp8" and isinstance(module, nn.Linear):
                    fp8_layer = FP8Linear.from_linear(module)
                    setattr(parent, child_name, fp8_layer)

    # Freeze all base parameters; enable gradients only on LoRA adapters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return injected_names


def get_cosmos14b_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA adapter parameters."""
    return [
        param
        for name, param in model.named_parameters()
        if ("lora_A" in name or "lora_B" in name) and param.requires_grad
    ]


def extract_cosmos14b_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract a clean dictionary containing only LoRA adapter weights."""
    lora_sd: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.detach().cpu()
    return lora_sd


def save_cosmos14b_lora(
    model: nn.Module,
    save_path: str | Path,
    metadata: dict[str, str] | None = None,
    rank: int = 16,
    alpha: float = 16.0,
) -> None:
    """Save trained LoRA adapter weights to safetensors file with strict metadata."""
    path = Path(save_path)
    meta = metadata or {}
    r = int(meta.get("rank", rank))
    a = float(meta.get("alpha", alpha))
    model_type = meta.get("model_type", "Cosmos-1.0-Diffusion-14B-LoRA")
    save_native_lora(
        model=model,
        save_path=path,
        rank=r,
        alpha=a,
        model_type=model_type,
        extra_metadata=meta,
    )


def load_cosmos14b_lora(
    model: nn.Module,
    lora_path: str | Path,
    expected_rank: int | None = None,
    expected_alpha: float | None = None,
) -> dict[str, str]:
    """Load LoRA weights with strict rank/alpha metadata and shape validation."""
    try:
        return load_native_lora(
            model=model,
            lora_path=lora_path,
            expected_rank=expected_rank,
            expected_alpha=expected_alpha,
            strict=True,
        )
    except (LoRAMetadataError, FileNotFoundError) as e:
        raise Cosmos14BLoRAError(str(e)) from e


def merge_cosmos14b_quantized_lora(model: CosmosTransformer3DModel) -> None:
    """Permanently merge LoRA weights into base layers."""
    for block in model.transformer_blocks:
        for mod_name, module in list(block.named_modules()):
            if isinstance(module, Cosmos14BQuantizedLoRALinear):
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)
                merged = module.merge_to_base()
                setattr(parent, child_name, merged)


def compute_cosmos14b_edm_loss(
    model: CosmosTransformer3DModel,
    clean_latents: Tensor,
    encoder_hidden_states: Tensor,
    sigma: Tensor | float | None = None,
    noise: Tensor | None = None,
    padding_mask: Tensor | None = None,
    sigma_data: float = 0.5,
    fps: int = 10,
) -> Tensor:
    """Compute native Cosmos 14B EDM objective with weighted x0 reconstruction loss."""
    return compute_cosmos_edm_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=encoder_hidden_states,
        sigma=sigma,
        noise=noise,
        padding_mask=padding_mask,
        sigma_data=sigma_data,
        fps=fps,
    )


def compute_rectified_flow_loss(
    model: CosmosTransformer3DModel,
    clean_latents: Tensor,
    encoder_hidden_states: Tensor,
    timestep: Tensor | None = None,
    noise: Tensor | None = None,
    padding_mask: Tensor | None = None,
    fps: int = 10,
) -> Tensor:
    """Native diffusion objective dispatch (maps to EDM loss)."""
    return compute_cosmos14b_edm_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=encoder_hidden_states,
        sigma=timestep,
        noise=noise,
        padding_mask=padding_mask,
        fps=fps,
    )
