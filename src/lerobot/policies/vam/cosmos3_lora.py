"""LoRA helpers for Cosmos 3 Edge architecture."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.policies.vam.cosmos_lora import LoRALinear


def inject_cosmos3_lora(
    transformer: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    block_indices: Sequence[int] | None = None,
) -> list[str]:
    """Wrap attention and MLP linear projections in selected Cosmos 3 layers with LoRALinear."""
    for p in transformer.parameters():
        p.requires_grad_(False)

    num_layers = len(transformer.layers)
    if block_indices is None:
        target_blocks = set(range(num_layers))
    else:
        target_blocks = set(block_indices)
        if any(idx < 0 or idx >= num_layers for idx in target_blocks):
            raise ValueError(f"block_indices must be in [0, {num_layers - 1}]")

    wrapped = []
    for i in target_blocks:
        layer = transformer.layers[i]
        for name, module in list(layer.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name, attr = name.rsplit(".", 1) if "." in name else ("", name)
                parent = layer.get_submodule(parent_name) if parent_name else layer
                if isinstance(getattr(parent, attr), LoRALinear):
                    continue
                lora_mod = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr, lora_mod)
                wrapped.append(f"layers.{i}.{name}")

    if not wrapped:
        raise ValueError("No linear layers were wrapped with LoRA")
    return wrapped


def cosmos3_lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    """Extract adapter-only state dict with clean keys."""
    state: dict[str, torch.Tensor] = {}
    for name, param in transformer.named_parameters():
        if not name.endswith(("lora_A", "lora_B")):
            continue
        clean_name = name.replace("._checkpoint_wrapped_module.", ".")
        state[clean_name] = param.detach().cpu().contiguous()
    if not state:
        raise ValueError("No LoRA parameters found in transformer")
    return state


def save_cosmos3_lora(transformer: nn.Module, path: str | Path) -> None:
    """Save adapter parameters as safetensors."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cosmos3_lora_state_dict(transformer), str(out_path))


def load_cosmos3_lora(
    transformer: nn.Module,
    path: str | Path,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    block_indices: Sequence[int] | None = None,
) -> None:
    """Inject LoRA if not present and load state dict from file."""
    state_dict = load_file(str(path))
    # Discover which layers are in state_dict if block_indices not specified
    if block_indices is None:
        layer_indices = set()
        for k in state_dict:
            parts = k.split(".")
            if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
                layer_indices.add(int(parts[1]))
        target_indices = sorted(layer_indices) if layer_indices else None
    else:
        target_indices = block_indices

    # Ensure LoRA is injected
    if not any(isinstance(m, LoRALinear) for m in transformer.modules()):
        inject_cosmos3_lora(transformer, rank=rank, alpha=alpha, block_indices=target_indices)

    # Copy weights
    model_params = dict(transformer.named_parameters())
    loaded_keys = 0
    for k, v in state_dict.items():
        if k in model_params:
            model_params[k].data.copy_(v.to(device=model_params[k].device, dtype=model_params[k].dtype))
            loaded_keys += 1
    if loaded_keys == 0:
        raise RuntimeError(f"No matching LoRA parameters could be loaded from {path}")
