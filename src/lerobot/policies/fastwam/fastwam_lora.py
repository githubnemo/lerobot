"""LoRA adapters for FastWAM's video expert.

Only the video branch of FastWAM's MoT layers is adapted.  The action expert
remains fully trainable, while every video-expert base parameter is frozen.
Adapters are kept in fp32 and can be saved independently of the large base
checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional

FASTWAM_VIDEO_LAYER_COUNT = 30
LORA_PROJECTION_TARGETS = (
    ("self_attn", "q"),
    ("self_attn", "k"),
    ("self_attn", "v"),
    ("self_attn", "o"),
    ("cross_attn", "q"),
    ("cross_attn", "k"),
    ("cross_attn", "v"),
    ("cross_attn", "o"),
    ("ffn", "0"),
    ("ffn", "2"),
)
FASTWAM_VIDEO_LORA_TARGETS = tuple(
    f"mot.layers.{layer}.blocks.video.{parent}.{projection}"
    for layer in range(FASTWAM_VIDEO_LAYER_COUNT)
    for parent, projection in LORA_PROJECTION_TARGETS
)


class FastWAMLoRAError(RuntimeError):
    """Base error for FastWAM LoRA contracts."""


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable fp32 low-rank residual."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float | None = None) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear base_layer must be torch.nn.Linear")
        if type(rank) is not int or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        if alpha is None:
            alpha = float(rank)
        if not math.isfinite(float(alpha)) or alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=base_layer.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=base_layer.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> nn.Parameter:
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(input)
        update = functional.linear(functional.linear(input.float(), self.lora_A), self.lora_B)
        return result + update.to(dtype=result.dtype) * self.scaling


def _adapter_name(name: str) -> bool:
    return name.endswith(("lora_A", "lora_B"))


def _video_parameter_name(name: str) -> bool:
    return name.startswith("mot.mixtures.video.") or ".blocks.video." in name


def freeze_video_expert_base_parameters(model: nn.Module) -> None:
    """Freeze video-expert weights while preserving trainable adapters."""
    for name, parameter in model.named_parameters():
        if _video_parameter_name(name):
            parameter.requires_grad_(_adapter_name(name))


def _validate_model_layout(model: nn.Module) -> None:
    mot = getattr(model, "mot", None)
    layers = getattr(mot, "layers", None)
    if layers is None or len(layers) != FASTWAM_VIDEO_LAYER_COUNT:
        actual = None if layers is None else len(layers)
        raise FastWAMLoRAError(
            "FastWAM video LoRA injection requires exactly "
            f"{FASTWAM_VIDEO_LAYER_COUNT} MoT layers, got {actual}"
        )


def _target_modules(model: nn.Module) -> Iterable[tuple[str, nn.Module]]:
    _validate_model_layout(model)
    for name in FASTWAM_VIDEO_LORA_TARGETS:
        parent_name, attribute = name.rsplit(".", 1)
        try:
            parent = model.get_submodule(parent_name)
        except AttributeError as exc:
            raise FastWAMLoRAError(f"missing FastWAM LoRA target parent `{parent_name}`") from exc
        module = getattr(parent, attribute, None)
        if module is None:
            raise FastWAMLoRAError(f"missing FastWAM LoRA target `{name}`")
        if not isinstance(module, (nn.Linear, LoRALinear)):
            raise FastWAMLoRAError(f"FastWAM LoRA target `{name}` is not linear: {type(module).__name__}")
        yield name, module


def inject_lora(model: nn.Module, *, rank: int = 16, alpha: float | None = None) -> tuple[str, ...]:
    """Inject adapters into all 300 video-expert projections.

    The target list is intentionally exact: 30 layers × (4 self-attention
    projections + 4 cross-attention projections + 2 FFN projections).  No
    action-expert module is wrapped or frozen by this function.
    """
    wrapped: list[str] = []
    for name, module in _target_modules(model):
        if isinstance(module, LoRALinear):
            if module.rank != rank or (alpha is not None and module.alpha != float(alpha)):
                raise FastWAMLoRAError(f"existing LoRA target `{name}` has incompatible rank/alpha")
            continue
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, attribute, LoRALinear(module, rank=rank, alpha=alpha))
        wrapped.append(name)

    freeze_video_expert_base_parameters(model)
    if not wrapped and not any(isinstance(module, LoRALinear) for module in model.modules()):
        raise FastWAMLoRAError("FastWAM LoRA injection did not wrap any modules")
    return tuple(wrapped)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable fp32 adapter parameters in stable module order."""
    parameters = [parameter for name, parameter in model.named_parameters() if _adapter_name(name)]
    if not parameters:
        raise FastWAMLoRAError("model has no FastWAM LoRA parameters")
    if any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in parameters):
        raise FastWAMLoRAError("all FastWAM LoRA parameters must be trainable fp32 tensors")
    return parameters


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter tensors for a compact checkpoint."""
    state = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if _adapter_name(name)
    }
    if not state:
        raise FastWAMLoRAError("model has no FastWAM LoRA state")
    return state


def save_lora_state_dict(model: nn.Module, path: str | Path) -> None:
    """Save adapter tensors as a safetensors file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(path))


def load_lora_state_dict(model: nn.Module, path: str | Path) -> None:
    """Strictly load an adapter-only safetensors checkpoint."""
    actual = load_file(str(Path(path)), device="cpu")
    expected = lora_state_dict(model)
    if set(actual) != set(expected):
        raise FastWAMLoRAError(
            f"FastWAM LoRA checkpoint keys mismatch: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )

    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, expected_tensor in expected.items():
            tensor = actual[name]
            if tuple(tensor.shape) != tuple(expected_tensor.shape) or tensor.dtype != torch.float32:
                raise FastWAMLoRAError(
                    f"FastWAM LoRA tensor {name} must have shape/dtype "
                    f"{tuple(expected_tensor.shape)}/torch.float32, got {tuple(tensor.shape)}/{tensor.dtype}"
                )
            parameters[name].copy_(tensor.to(device=parameters[name].device, dtype=torch.float32))
    freeze_video_expert_base_parameters(model)
