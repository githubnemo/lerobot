"""Video-LoRA adaptation for NVIDIA Cosmos-1.0-Diffusion-7B with native EDM objective.

Provides low-rank adaptation (LoRA) for fine-tuning the 28-block Cosmos 7B
spatiotemporal DiT backbone on video prediction tasks using the native EDM pretraining
objective, followed by downstream feature extraction for physical AI policy learning.
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


class Cosmos7BLoRAError(RuntimeError):
    """Exception raised during LoRA operations on Cosmos 7B."""


@dataclass(frozen=True, slots=True)
class Cosmos7BLoRAConfig:
    """Configuration for Cosmos 7B Video-LoRA adaptation."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_blocks: tuple[int, ...] = tuple(range(28))
    target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "net.0.proj",
        "net.2",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")


class Cosmos7BLoRALinear(nn.Module):
    """Frozen linear base layer augmented with a low-rank residual branch."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("Cosmos7BLoRALinear base_layer must be an instance of nn.Linear")
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)

        # Freeze base parameters
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        device = base_layer.weight.device
        dtype = torch.float32

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize lora_A with Kaiming uniform and lora_B with zeros so initial residual is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_layer(x)
        # Low-rank branch computed in float32 for numerical stability
        x_float = self.dropout(x.float())
        lora_out = functional.linear(functional.linear(x_float, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merge_to_base(self) -> nn.Linear:
        """Merge LoRA update into base layer weight and return standard nn.Linear."""
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged_weight = self.base_layer.weight.data + delta_w.to(
            device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
        )
        self.base_layer.weight.data.copy_(merged_weight)
        return self.base_layer


def inject_cosmos7b_lora(
    model: CosmosTransformer3DModel,
    config: Cosmos7BLoRAConfig,
) -> list[str]:
    """Inject LoRA adapters into targeted linear modules within Cosmos 7B transformer blocks."""
    injected_names: list[str] = []

    for block_idx in config.target_blocks:
        if block_idx >= len(model.transformer_blocks):
            continue
        block = model.transformer_blocks[block_idx]

        # Search for target modules within this block
        for mod_name, module in list(block.named_modules()):
            should_target = any(mod_name.endswith(tgt) for tgt in config.target_modules)
            if should_target and isinstance(module, nn.Linear):
                # Navigate parent to replace child
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)
                lora_layer = Cosmos7BLoRALinear(
                    base_layer=module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, child_name, lora_layer)
                full_name = f"transformer_blocks.{block_idx}.{mod_name}"
                injected_names.append(full_name)

    # Freeze entire backbone parameters except LoRA parameters
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return injected_names


def get_cosmos7b_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


def extract_cosmos7b_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract a clean safetensors-compatible dictionary containing only LoRA weights."""
    lora_sd: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.detach().cpu()
    return lora_sd


def save_cosmos7b_lora(
    model: nn.Module,
    save_path: str | Path,
    metadata: dict[str, str] | None = None,
    rank: int = 16,
    alpha: float = 16.0,
) -> None:
    """Save trained LoRA weights to safetensors file with strict metadata."""
    path = Path(save_path)
    meta = metadata or {}
    r = int(meta.get("rank", rank))
    a = float(meta.get("alpha", alpha))
    model_type = meta.get("model_type", "Cosmos-1.0-Diffusion-7B-LoRA")
    save_native_lora(
        model=model,
        save_path=path,
        rank=r,
        alpha=a,
        model_type=model_type,
        extra_metadata=meta,
    )


def load_cosmos7b_lora(
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
        raise Cosmos7BLoRAError(str(e)) from e


def merge_cosmos7b_lora(model: CosmosTransformer3DModel) -> None:
    """Permanently merge LoRA weights into base linear layers."""
    for block in model.transformer_blocks:
        for mod_name, module in list(block.named_modules()):
            if isinstance(module, Cosmos7BLoRALinear):
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)
                merged_linear = module.merge_to_base()
                setattr(parent, child_name, merged_linear)


def compute_cosmos7b_edm_loss(
    model: CosmosTransformer3DModel,
    clean_latents: Tensor,
    encoder_hidden_states: Tensor,
    sigma: Tensor | float | None = None,
    noise: Tensor | None = None,
    padding_mask: Tensor | None = None,
    sigma_data: float = 0.5,
    fps: int = 10,
) -> Tensor:
    """Compute native Cosmos 7B EDM objective with weighted x0 reconstruction loss."""
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
    fps: int = 10,
) -> Tensor:
    """Native diffusion objective dispatch (maps to EDM loss)."""
    return compute_cosmos7b_edm_loss(
        model=model,
        clean_latents=clean_latents,
        encoder_hidden_states=encoder_hidden_states,
        sigma=timestep,
        noise=noise,
        fps=fps,
    )
