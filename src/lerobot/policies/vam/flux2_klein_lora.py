"""LoRA adaptation setup for FLUX.2 [klein] (9B) and multi-reference conditioning.

Enables low-rank adaptation (LoRA) for FLUX.2 [klein] double-stream and single-stream
transformer blocks, fine-tuning on multi-reference robotics observation trajectories,
followed by feature extraction for SmolExpert policy training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from torch import Tensor, nn
from torch.nn import functional

from lerobot.policies.vam.base.native_video import (
    LoRAMetadataError,
    load_native_lora,
    save_native_lora,
)
from lerobot.policies.vam.flux2_klein_extractor import MultiReferenceInputs


class Flux2KleinLoRAError(RuntimeError):
    """Exception raised during LoRA operations on FLUX.2 [klein]."""


@dataclass(frozen=True, slots=True)
class Flux2KleinLoRAConfig:
    """Configuration for FLUX.2 [klein] 9B LoRA adaptation."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_double_blocks: tuple[int, ...] = tuple(range(8))
    target_single_blocks: tuple[int, ...] = tuple(range(48))
    target_modules: tuple[str, ...] = (
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")


class Flux2KleinLoRALinear(nn.Module):
    """Frozen base linear layer with trainable low-rank residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be an instance of nn.Linear")
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)

        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        device = base_layer.weight.device
        dtype = torch.float32

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_layer(x)
        x_float = self.dropout(x.float())
        lora_out = functional.linear(functional.linear(x_float, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merge_to_base(self) -> nn.Linear:
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        merged_weight = self.base_layer.weight.data + delta_w.to(
            device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
        )
        self.base_layer.weight.data.copy_(merged_weight)
        return self.base_layer


def inject_flux2_klein_lora(
    model: Flux2Transformer2DModel,
    config: Flux2KleinLoRAConfig,
) -> list[str]:
    """Inject LoRA adapters into double-stream and single-stream transformer blocks."""
    injected_names: list[str] = []

    # 1. Double-stream blocks
    for block_idx in config.target_double_blocks:
        if block_idx >= len(model.transformer_blocks):
            continue
        block = model.transformer_blocks[block_idx]
        for mod_name, module in list(block.named_modules()):
            should_target = any(mod_name.endswith(tgt) for tgt in config.target_modules)
            if should_target and isinstance(module, nn.Linear):
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)
                lora_layer = Flux2KleinLoRALinear(
                    base_layer=module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, child_name, lora_layer)
                injected_names.append(f"transformer_blocks.{block_idx}.{mod_name}")

    # 2. Single-stream blocks
    for block_idx in config.target_single_blocks:
        if block_idx >= len(model.single_transformer_blocks):
            continue
        block = model.single_transformer_blocks[block_idx]
        for mod_name, module in list(block.named_modules()):
            should_target = any(mod_name.endswith(tgt) for tgt in config.target_modules)
            if should_target and isinstance(module, nn.Linear):
                parent_name, _, child_name = mod_name.rpartition(".")
                parent = block if not parent_name else block.get_submodule(parent_name)
                lora_layer = Flux2KleinLoRALinear(
                    base_layer=module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                setattr(parent, child_name, lora_layer)
                injected_names.append(f"single_transformer_blocks.{block_idx}.{mod_name}")

    # Freeze base model, train only LoRA weights
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return injected_names


def get_flux2_klein_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA parameters."""
    return [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]


def extract_flux2_klein_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract only the LoRA state dict."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def save_flux2_klein_lora(
    model: nn.Module,
    save_path: str | Path,
    metadata: dict[str, str] | None = None,
    rank: int = 16,
    alpha: float = 16.0,
) -> None:
    """Save trained LoRA weights to safetensors file with strict metadata header."""
    path = Path(save_path)
    meta = metadata or {}
    r = int(meta.get("rank", rank))
    a = float(meta.get("alpha", alpha))
    model_type = meta.get("model_type", "FLUX.2-klein-9B-LoRA")
    save_native_lora(
        model=model,
        save_path=path,
        rank=r,
        alpha=a,
        model_type=model_type,
        extra_metadata=meta,
    )


def load_flux2_klein_lora(
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
        raise Flux2KleinLoRAError(str(e)) from e


def merge_flux2_klein_lora(model: Flux2Transformer2DModel) -> None:
    """Merge LoRA residuals permanently into base layers."""
    for block_list in (model.transformer_blocks, model.single_transformer_blocks):
        for block in block_list:
            for mod_name, module in list(block.named_modules()):
                if isinstance(module, Flux2KleinLoRALinear):
                    parent_name, _, child_name = mod_name.rpartition(".")
                    parent = block if not parent_name else block.get_submodule(parent_name)
                    merged_linear = module.merge_to_base()
                    setattr(parent, child_name, merged_linear)


def compute_multi_reference_rf_loss(
    model: Flux2Transformer2DModel,
    cond_inputs: MultiReferenceInputs,
    target_clean: Tensor,  # [B, N_target, C]
    timestep: Tensor | None = None,
    noise: Tensor | None = None,
    guidance: float = 3.5,
) -> Tensor:
    """Compute rectified flow velocity prediction loss on multi-reference conditioned prediction.

    Target velocity in normalized flow matching: v = (noise - target_clean) / 1000.0.
    """
    b = target_clean.shape[0]
    device = target_clean.device

    if noise is None:
        noise = torch.randn_like(target_clean)
    if timestep is None:
        # Sample normalized time between 0.0 and 1000.0
        timestep = torch.rand(b, device=device, dtype=torch.float32) * 1000.0

    t_norm = (timestep / 1000.0).view(b, 1, 1).to(dtype=target_clean.dtype)
    noised_target = (1.0 - t_norm) * target_clean + t_norm * noise
    target_velocity = (noise - target_clean) / 1000.0

    # Replace target portion in cond_inputs.hidden_states with noised_target
    target_count = cond_inputs.target_tokens_count
    ref_tokens = cond_inputs.hidden_states[:, target_count:]
    mixed_hidden_states = torch.cat([noised_target, ref_tokens], dim=1)

    g_tensor = torch.tensor([float(guidance)], device=device, dtype=torch.float32)

    pred_sample = model(
        hidden_states=mixed_hidden_states,
        encoder_hidden_states=cond_inputs.encoder_hidden_states,
        timestep=timestep,
        img_ids=cond_inputs.img_ids,
        txt_ids=cond_inputs.txt_ids,
        guidance=g_tensor,
        return_dict=True,
    ).sample

    # Extract target slice of prediction
    pred_target_velocity = pred_sample[:, :target_count]
    loss = functional.mse_loss(pred_target_velocity.float(), target_velocity.float())
    return loss
