"""LoRA helpers for Cosmos 3 Edge architecture."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import safetensors
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from lerobot.policies.vam.cosmos_lora import LoRALinear, freeze_base_parameters
from lerobot.utils.import_utils import _peft_available, require_package

if TYPE_CHECKING or _peft_available:
    from peft import LoraConfig, PeftModel

PEFT_ADAPTER_CONFIG = "adapter_config.json"
PEFT_ADAPTER_WEIGHTS = "adapter_model.safetensors"

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
    use_dora: bool = False,
) -> list[str]:
    """Wrap attention and MLP projections in selected Cosmos 3 layers with LoRALinear (LoRA or DoRA).

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
            lora_mod = LoRALinear(module, rank=rank, alpha=alpha, use_dora=use_dora)
            setattr(parent, attr, lora_mod)
            wrapped.append(f"layers.{i}.{name}")

    if not wrapped:
        raise ValueError("No linear layers were wrapped with LoRA")

    return wrapped


def cosmos3_lora_state_dict(transformer: nn.Module) -> dict[str, torch.Tensor]:
    """Extract adapter-only state dict with clean keys (preserving any stashed pruned-pathway LoRA weights)."""
    state: dict[str, torch.Tensor] = {}
    pruned_state = getattr(transformer, "_pruned_lora_state_dict", None)
    if isinstance(pruned_state, dict):
        for clean_name, tensor in pruned_state.items():
            state[clean_name] = tensor.detach().cpu().contiguous()
    for name, param in transformer.named_parameters():
        if not name.endswith(("lora_A", "lora_B", "dora_magnitude")):
            continue
        clean_name = name.replace("._checkpoint_wrapped_module.", ".").replace("._orig_mod.", ".")
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
    use_dora: bool | None = None,
) -> None:
    """Save adapter parameters as safetensors with rank/alpha/module/dora metadata."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state = cosmos3_lora_state_dict(transformer)

    # Infer rank / alpha / use_dora if not passed
    resolved_rank = rank
    resolved_alpha = alpha
    resolved_dora = use_dora
    for m in transformer.modules():
        if isinstance(m, LoRALinear):
            if resolved_rank is None:
                resolved_rank = m.rank
            if resolved_alpha is None:
                resolved_alpha = m.alpha
            if resolved_dora is None:
                resolved_dora = m.use_dora
            break

    metadata: dict[str, str] = {
        "rank": str(resolved_rank if resolved_rank is not None else 16),
        "alpha": str(resolved_alpha if resolved_alpha is not None else 32.0),
        "use_dora": str(bool(resolved_dora)).lower(),
    }
    if block_indices is not None:
        metadata["block_indices"] = ",".join(map(str, sorted(block_indices)))
    if target_modules is not None:
        metadata["target_modules"] = ",".join(target_modules)

    save_file(state, str(out_path), metadata=metadata)


# ---- PEFT adapters -------------------------------------------------------------------------
# Same targets, init (A kaiming-uniform, B zero), scaling (alpha / r) and fp32 adapter weights on a
# bf16 base as LoRALinear, so a legacy adapter converts to PEFT format exactly.


def cosmos3_peft_target_regex(
    target_modules: Sequence[str] = DUAL_PATHWAY_TARGET_MODULES,
    block_indices: Sequence[int] | None = None,
) -> str:
    """PEFT ``target_modules`` regex (full match) selecting ``layers.<i>.<target>`` Linear layers."""
    blocks = (
        r"\d+" if block_indices is None else "(?:" + "|".join(str(i) for i in sorted(block_indices)) + ")"
    )
    modules = "|".join(re.escape(m) for m in target_modules)
    return rf"layers\.{blocks}\.(?:{modules})"


def cosmos3_lora_config(
    *,
    rank: int,
    alpha: float,
    target_modules: Sequence[str] = DUAL_PATHWAY_TARGET_MODULES,
    block_indices: Sequence[int] | None = None,
    use_dora: bool = False,
) -> LoraConfig:
    """PEFT LoraConfig equivalent to ``inject_cosmos3_lora(rank, alpha, block_indices, target_modules, use_dora)``."""
    require_package("peft", extra="peft")
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=cosmos3_peft_target_regex(target_modules, block_indices),
        init_lora_weights=True,
        use_dora=use_dora,
    )


def is_peft_adapter(path: str | Path) -> bool:
    """Whether ``path`` is a PEFT adapter directory (``adapter_config.json`` + weights)."""
    return Path(path).is_dir() and (Path(path) / PEFT_ADAPTER_CONFIG).is_file()


def cosmos3_adapter_weights_file(path: str | Path) -> Path:
    """The tensor file of a legacy adapter (the file itself) or a PEFT adapter directory."""
    path = Path(path)
    return path / PEFT_ADAPTER_WEIGHTS if is_peft_adapter(path) else path


def read_cosmos3_adapter_hyperparameters(path: str | Path) -> tuple[int, float] | None:
    """``(rank, alpha)`` recorded by the adapter itself (PEFT config or legacy safetensors metadata)."""
    path = Path(path)
    if is_peft_adapter(path):
        config = json.loads((path / PEFT_ADAPTER_CONFIG).read_text())
        return int(config["r"]), float(config["lora_alpha"])
    with safetensors.safe_open(str(path), framework="pt") as f:
        meta = f.metadata() or {}
    if "rank" in meta and "alpha" in meta:
        return int(meta["rank"]), float(meta["alpha"])
    return None


def load_cosmos3_peft_adapter(transformer: nn.Module, path: str | Path, *, merge: bool) -> nn.Module:
    """Attach a PEFT adapter to a Cosmos 3 transformer for frozen inference.

    ``merge=True`` folds ``W + (alpha / r) B A`` into the base weights (plain ``nn.Linear``, no
    adapter overhead, compile-friendly; the update is rounded to the base dtype). ``merge=False``
    keeps the fp32 adapter branch, numerically matching the legacy ``LoRALinear`` path.
    Returns the (unwrapped) transformer with every parameter frozen.
    """
    require_package("peft", extra="peft")
    if not is_peft_adapter(path):
        raise FileNotFoundError(f"Not a PEFT adapter directory: {path}")
    peft_model = PeftModel.from_pretrained(transformer, str(path), is_trainable=False)
    model = peft_model.merge_and_unload() if merge else peft_model.get_base_model()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def convert_legacy_cosmos3_lora_to_peft(
    legacy_file: str | Path, output_dir: str | Path, *, base_model_name_or_path: str | None = None
) -> Path:
    """Rewrite a ``save_cosmos3_lora`` file as a PEFT adapter directory (tensors unchanged)."""
    legacy_file, output_dir = Path(legacy_file), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    recorded = read_cosmos3_adapter_hyperparameters(legacy_file)
    if recorded is None:
        raise ValueError(f"{legacy_file} has no rank/alpha metadata; refusing to guess the scaling")
    rank, alpha = recorded
    state = load_file(str(legacy_file))
    blocks, modules = set(), set()
    for key in state:
        parts = key.split(".")
        if parts[0] != "layers" or not parts[1].isdigit() or parts[-1] not in ("lora_A", "lora_B"):
            raise ValueError(f"Unexpected legacy LoRA key: {key}")
        blocks.add(int(parts[1]))
        modules.add(".".join(parts[2:-1]))
    config = cosmos3_lora_config(
        rank=rank, alpha=alpha, target_modules=sorted(modules), block_indices=sorted(blocks)
    )
    config.base_model_name_or_path = base_model_name_or_path
    output_dir.mkdir(parents=True)
    config.save_pretrained(str(output_dir))
    save_file(
        {f"base_model.model.{key}.weight": value.contiguous() for key, value in state.items()},
        str(output_dir / PEFT_ADAPTER_WEIGHTS),
        metadata={"format": "pt"},
    )
    return output_dir


def load_cosmos3_lora(
    transformer: nn.Module,
    path: str | Path,
    *,
    rank: int = 16,
    alpha: float = 32.0,
    block_indices: Sequence[int] | None = None,
    target_modules: Sequence[str] | None = None,
    use_dora: bool | None = None,
) -> None:
    """Inject LoRA/DoRA if not present and load state dict from file with strict scope validation.

    Fails early if path does not exist.
    Fails if any expected LoRA/DoRA parameter within the target scope is missing from state_dict.
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

    has_dora_keys = any(k.endswith(".dora_magnitude") for k in state_dict)
    if use_dora is not None:
        loaded_dora = use_dora
    elif "use_dora" in saved_metadata:
        loaded_dora = saved_metadata["use_dora"].lower() in ("true", "1", "yes")
    else:
        loaded_dora = has_dora_keys

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

    # Ensure LoRA/DoRA is injected
    if not any(isinstance(m, LoRALinear) for m in transformer.modules()):
        inject_cosmos3_lora(
            transformer,
            rank=loaded_rank,
            alpha=loaded_alpha,
            block_indices=target_indices,
            target_modules=target_mods,
            use_dora=loaded_dora,
        )

    # Strict check: all active LoRA/DoRA parameters must be present in state_dict
    active_lora_params = {
        name.replace("._orig_mod.", "."): param
        for name, param in transformer.named_parameters()
        if name.endswith(("lora_A", "lora_B", "dora_magnitude"))
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

    pruned_state = getattr(transformer, "_pruned_lora_state_dict", None)
    if isinstance(pruned_state, dict):
        for k in list(pruned_state.keys()):
            if k in state_dict:
                pruned_state[k] = state_dict[k].detach().float().cpu().contiguous()

    freeze_base_parameters(transformer)
