"""Standardized Base Video-LoRA implementation for diffusion and flow-matching backbones.

Provides unified LoRA linear projection wrappers, standardized injection across
transformer blocks, weight extraction, and merging routines.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.nn import functional


class LoRAError(RuntimeError):
    """Exception raised during LoRA operations."""


@dataclass(frozen=True, slots=True)
class BaseVideoLoRAConfig:
    """Standard configuration contract for Video-LoRA adaptation."""

    rank: int = 16
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0")
    target_blocks: tuple[int, ...] | str = "all"
    quantization: str = "none"  # "none" or "fp8"

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be a positive integer, got {self.rank}")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError(f"LoRA alpha must be finite and positive, got {self.alpha}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0.0, 1.0), got {self.dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if self.target_blocks != "all" and not isinstance(self.target_blocks, (tuple, list)):
            raise ValueError("target_blocks must be 'all' or a tuple/list of block indices")


class BaseLoRALinear(nn.Module):
    """Unified LoRA layer supporting fp32 residual and optional fp8/fp16/bf16 base."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"BaseLoRALinear base_layer must be an instance of nn.Linear, got {type(base_layer).__name__}"
            )
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError(f"LoRA alpha must be finite and positive, got {alpha}")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)

        # Freeze base layer parameters
        for p in self.base_layer.parameters():
            p.requires_grad_(False)

        device = base_layer.weight.device
        dtype = torch.float32

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device, dtype=dtype))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialize lora_A with Kaiming uniform and lora_B with zeros so initial residual is zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_layer(x)
        # Residual branch computed in float32 for numerical stability
        x_float = self.dropout(x.float())
        lora_out = functional.linear(functional.linear(x_float, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    def merge_to_base(self) -> nn.Linear:
        """Merge LoRA update into base layer weight in place and return the base layer."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            merged_weight = self.base_layer.weight.data + delta_w.to(
                device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype
            )
            self.base_layer.weight.data.copy_(merged_weight)
        return self.base_layer


def _extract_block_index(module_name: str) -> int | None:
    """Extract block/layer index from a hierarchical module path if present."""
    parts = module_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if (
            part in ("blocks", "transformer_blocks", "layers", "block")
            and idx + 1 < len(parts)
            and parts[idx + 1].isdigit()
        ):
            return int(parts[idx + 1])
    return None


def inject_base_lora(
    model: nn.Module,
    config: BaseVideoLoRAConfig,
) -> list[str]:
    """Inject BaseLoRALinear into targeted linear modules within a model.

    Args:
        model: Target PyTorch neural network module.
        config: BaseVideoLoRAConfig specifying targets, rank, alpha, and dropout.

    Returns:
        List of injected module names.
    """
    injected_names: list[str] = []
    target_blocks = set(config.target_blocks) if config.target_blocks != "all" else None

    # Collect matching linear modules
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or isinstance(module, BaseLoRALinear):
            continue

        # Check block filter if specified
        if target_blocks is not None:
            block_idx = _extract_block_index(name)
            if block_idx is not None and block_idx not in target_blocks:
                continue

        # Check target module suffix/name match
        matches_target = any(
            name.endswith(tgt) or name.split(".")[-1] == tgt for tgt in config.target_modules
        )
        if not matches_target:
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model if not parent_name else model.get_submodule(parent_name)

        lora_layer = BaseLoRALinear(
            base_layer=module,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, child_name, lora_layer)
        injected_names.append(name)

    if not injected_names:
        raise LoRAError(
            f"No target linear modules matching {config.target_modules} found to inject LoRA into."
        )

    # Freeze base model parameters, leaving only LoRA parameters trainable
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    return injected_names


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return all trainable LoRA parameters in deterministic order."""
    params = [p for n, p in model.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
    if not params:
        raise LoRAError("No trainable LoRA parameters found in model.")
    return params


def extract_lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    """Extract a dictionary containing only LoRA parameters detached to CPU."""
    lora_sd: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.detach().cpu().contiguous()
    if not lora_sd:
        raise LoRAError("No LoRA weights found in model state.")
    return lora_sd


def save_lora_weights(
    model: nn.Module,
    save_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save trained LoRA weights as a safetensors file with sidecar metadata."""
    path = Path(save_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = extract_lora_state_dict(model)

    meta_str = {k: str(v) for k, v in (metadata or {}).items()}
    save_file(sd, str(path), metadata=meta_str)

    # Write sidecar JSON
    sidecar_path = path.with_suffix(".json")
    sidecar_data = {
        "artifact": "video_lora_checkpoint",
        "num_adapter_tensors": len(sd),
        "tensor_names": sorted(sd.keys()),
        "metadata": metadata or {},
    }
    sidecar_path.write_text(json.dumps(sidecar_data, indent=2) + "\n")


def load_lora_weights(model: nn.Module, load_path: str | Path) -> dict[str, str]:
    """Load LoRA weights from a safetensors file into an injected model."""
    path = Path(load_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA weight file not found: {path}")

    loaded_sd = load_file(str(path))
    model_sd = model.state_dict()
    status: dict[str, str] = {}

    for k, v in loaded_sd.items():
        if k not in model_sd:
            raise LoRAError(f"Checkpoint key {k!r} not found in model.")
        if model_sd[k].shape != v.shape:
            raise LoRAError(
                f"Shape mismatch for {k!r}: checkpoint has {tuple(v.shape)}, model has {tuple(model_sd[k].shape)}"
            )
        model_sd[k].copy_(v)
        status[k] = "loaded"

    return status


def merge_all_lora_to_base(model: nn.Module) -> list[str]:
    """Fuse all BaseLoRALinear residuals into their base layers and restore standard nn.Linear."""
    merged: list[str] = []
    for name, module in list(model.named_modules()):
        if isinstance(module, BaseLoRALinear):
            parent_name, _, child_name = name.rpartition(".")
            parent = model if not parent_name else model.get_submodule(parent_name)
            base_layer = module.merge_to_base()
            setattr(parent, child_name, base_layer)
            merged.append(name)

    if not merged:
        raise LoRAError("No BaseLoRALinear modules found to merge.")
    return merged
