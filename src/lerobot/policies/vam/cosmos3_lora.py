"""LoRA helpers for Cosmos 3 Edge architecture."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path

import safetensors
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.policies.vam.cosmos_lora import LoRALinear, freeze_base_parameters

UND_TARGET_MODULES: tuple[str, ...] = (
    "self_attn.to_q",
    "self_attn.to_k",
    "self_attn.to_v",
    "self_attn.to_out",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

GEN_TARGET_MODULES: tuple[str, ...] = (
    "self_attn.add_q_proj",
    "self_attn.add_k_proj",
    "self_attn.add_v_proj",
    "self_attn.to_add_out",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.up_proj",
    "mlp_moe_gen.down_proj",
)

DUAL_PATHWAY_TARGET_MODULES: tuple[str, ...] = UND_TARGET_MODULES + GEN_TARGET_MODULES
ALL_TARGET_MODULES: tuple[str, ...] = DUAL_PATHWAY_TARGET_MODULES


def _is_matching_target_module(name: str, target_module_set: set[str]) -> bool:
    """Return whether module name matches target_module_set."""
    if name in target_module_set:
        return True
    for t in target_module_set:
        if name == t or name.endswith(f".{t}"):
            return True
        if name.startswith(f"{t}.") or name == t:
            return True
    return False


def inject_cosmos3_lora(
    transformer: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    block_indices: Sequence[int] | None = None,
    target_modules: Sequence[str] | None = None,
) -> list[str]:
    """Wrap attention and MLP projections in selected Cosmos 3 layers with LoRALinear.

    Supports dual-pathway fine-tuning covering both the understanding pathway (:
    , , , , , , )
    and the generation pathway (: , , ,
    , , , ).

    By default (), wraps all 14 dual-pathway projections in the target blocks.
    """
    for p in transformer.parameters():
        p.requires_grad_(False)

    num_layers = len(transformer.layers)
    if block_indices is None:
        target_blocks = set(range(num_layers))
    else:
        target_blocks = set(block_indices)
        if any(idx < 0 or idx >= num_layers for idx in target_blocks):
            raise ValueError(f"block_indices must be in [0, {num_layers - 1}]")

    target_module_set = set(target_modules if target_modules is not None else DUAL_PATHWAY_TARGET_MODULES)

    wrapped = []
    for i in target_blocks:
        layer = transformer.layers[i]
        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if not _is_matching_target_module(name, target_module_set):
                continue
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


def save_cosmos3_lora(
    transformer: nn.Module,
    path: str | Path,
    *,
    rank: int | None = None,
    alpha: float | None = None,
    block_indices: Sequence[int] | None = None,
    target_modules: Sequence[str] | None = None,
) -> None:
    """Save adapter parameters as safetensors with rank/alpha/module metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = cosmos3_lora_state_dict(transformer)

    # Infer rank / alpha if not passed
    resolved_rank = rank
    resolved_alpha = alpha
    for m in transformer.modules():
        if isinstance(m, LoRALinear):
            if resolved_rank is None:
                resolved_rank = m.rank
            if resolved_alpha is None:
                resolved_alpha = m.alpha
            break

    metadata: dict[str, str] = {
        "rank": str(resolved_rank if resolved_rank is not None else 16),
        "alpha": str(resolved_alpha if resolved_alpha is not None else 32.0),
    }
    if block_indices is not None:
        metadata["block_indices"] = ",".join(map(str, sorted(block_indices)))
    if target_modules is not None:
        metadata["target_modules"] = ",".join(target_modules)

    save_file(state, str(out_path), metadata=metadata)


def load_cosmos3_lora(
    transformer: nn.Module,
    path: str | Path,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    block_indices: Sequence[int] | None = None,
    target_modules: Sequence[str] | None = None,
) -> None:
    """Inject LoRA if not present and load state dict from file with strict scope validation.

    Fails early if path does not exist.
    Fails if any expected LoRA parameter within the target scope is missing from state_dict.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint file does not exist: {file_path}")

    # Read metadata from safetensors header if present
    saved_metadata: dict[str, str] = {}
    try:
        with safetensors.safe_open(str(file_path), framework="pt") as f:
            meta = f.metadata()
            if meta:
                saved_metadata = dict(meta)
    except Exception:
        saved_metadata = {}

    loaded_rank = rank
    if "rank" in saved_metadata:
        with contextlib.suppress(ValueError):
            loaded_rank = int(saved_metadata["rank"])

    loaded_alpha = alpha
    if "alpha" in saved_metadata:
        with contextlib.suppress(ValueError):
            loaded_alpha = float(saved_metadata["alpha"])

    state_dict = load_file(str(file_path))

    # Discover layers and target modules in state_dict
    layer_indices = set()
    detected_modules = set()
    for k in state_dict:
        parts = k.split(".")
        if len(parts) > 1 and parts[0] == "layers" and parts[1].isdigit():
            layer_indices.add(int(parts[1]))
            mod_name = ".".join(parts[2:-1])
            if mod_name:
                detected_modules.add(mod_name)

    target_indices = (
        block_indices if block_indices is not None else (sorted(layer_indices) if layer_indices else None)
    )
    target_mods = (
        target_modules
        if target_modules is not None
        else (sorted(detected_modules) if detected_modules else None)
    )

    # Ensure LoRA is injected
    if not any(isinstance(m, LoRALinear) for m in transformer.modules()):
        inject_cosmos3_lora(
            transformer,
            rank=loaded_rank,
            alpha=loaded_alpha,
            block_indices=target_indices,
            target_modules=target_mods,
        )

    # Strict check: all active LoRA parameters must be present in state_dict
    active_lora_params = {
        name: param for name, param in transformer.named_parameters() if name.endswith(("lora_A", "lora_B"))
    }
    missing_keys = [name for name in active_lora_params if name not in state_dict]
    if missing_keys:
        raise RuntimeError(
            f"Missing {len(missing_keys)} required LoRA keys in {file_path}. "
            f"First missing: {missing_keys[:5]}"
        )

    # Copy weights
    for k, p in active_lora_params.items():
        p.data.copy_(state_dict[k].to(device=p.device, dtype=p.dtype))

    freeze_base_parameters(transformer)
