"""LoRA and full-clip training helpers for the Cosmos Predict2 backbone.

The vendored Cosmos model remains untouched.  This module wraps only the
attention and MLP projection ``Linear`` modules inside its 28 DiT blocks and
contains the separate 61-frame dataset/conditioning contract used by joint
video/action training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, VideoVAMDatasetConfig

LORA_TARGET_SUFFIXES = frozenset({"q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2"})
ATTENTION_TARGET_SUFFIXES = frozenset({"q_proj", "k_proj", "v_proj", "output_proj"})
MLP_TARGET_SUFFIXES = frozenset({"layer1", "layer2"})
FULL_CLIP_OFFSETS = tuple(range(-4, 57))
FULL_CLIP_LENGTH = len(FULL_CLIP_OFFSETS)
LATENT_FRAMES = 16
LATENT_CONDITIONAL_FRAMES = 2
VIDEO_SIGMA_MEDIAN = 4.0
VIDEO_SIGMA_TAIL_PROBABILITY = 0.05
VIDEO_SIGMA_TAIL_RANGE = (200.0, 100_000.0)


class CosmosLoRAError(RuntimeError):
    """Base error for LoRA training contracts."""


class LoRALinear(nn.Module):
    """A frozen linear layer with a fp32 low-rank residual."""

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
        adapter_device = base_layer.weight.device
        self.lora_A = nn.Parameter(
            torch.empty(rank, base_layer.in_features, device=adapter_device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base_layer.out_features, rank, device=adapter_device, dtype=torch.float32)
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
        return result + (update * self.scaling).to(dtype=result.dtype)


def _is_lora_target(name: str, module: nn.Module) -> bool:
    """Return whether a named module is an intended 28-block projection."""
    if not isinstance(module, nn.Linear):
        return False
    pieces = name.split(".")
    if len(pieces) < 3 or pieces[0] != "blocks" or not pieces[1].isdigit():
        return False
    if int(pieces[1]) >= 28 or pieces[-1] not in LORA_TARGET_SUFFIXES:
        return False
    if pieces[-1] in ATTENTION_TARGET_SUFFIXES:
        return "self_attn" in pieces or "cross_attn" in pieces
    return pieces[-1] in MLP_TARGET_SUFFIXES and "mlp" in pieces


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float | None = None,
    block_indices: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Wrap intended attention/MLP projections in selected Cosmos blocks.

    Wrapping is performed from the LeRobot-owned module, so the vendored source
    and its manifest are not modified.  Existing base weights retain their
    dtype; the new adapter parameters are always fp32.
    """
    if not hasattr(model, "blocks") or len(model.blocks) != 28:
        raise ValueError("Cosmos Predict2 LoRA injection requires exactly 28 DiT blocks")
    if block_indices is None:
        selected_blocks = set(range(28))
    else:
        selected_blocks = set(block_indices)
        if not selected_blocks or any(
            type(index) is not int or not 0 <= index < 28 for index in selected_blocks
        ):
            raise ValueError("block_indices must contain at least one integer in [0, 27]")
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if _is_lora_target(name, module) and int(name.split(".")[1]) in selected_blocks
    ]
    if not candidates:
        raise ValueError("no intended Cosmos attention/MLP projections were found in selected blocks")
    wrapped: list[str] = []
    for name, module in candidates:
        parent_name, attribute = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        if isinstance(getattr(parent, attribute), LoRALinear):
            continue
        setattr(parent, attribute, LoRALinear(module, rank=rank, alpha=alpha))
        wrapped.append(name)
    if not wrapped and not any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA injection did not wrap any modules")
    freeze_base_parameters(model)
    return tuple(wrapped)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_lora_file_into_base(model: nn.Module, lora_path: str | Path) -> dict[str, Any]:
    """Load a full-backbone adapter checkpoint and merge it into ``model``.

    The adapter's sidecar is part of the load contract: its rank and alpha are
    used to construct the exact wrappers before the strict adapter load.
    """
    path = Path(lora_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA weights not found: {path}")
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"LoRA provenance sidecar not found: {sidecar_path}")
    try:
        sidecar = json.loads(sidecar_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read LoRA provenance sidecar {sidecar_path}: {exc}") from exc
    lora = sidecar.get("lora")
    if not isinstance(lora, Mapping):
        raise ValueError(f"LoRA provenance sidecar has no lora object: {sidecar_path}")
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    if type(rank) is not int or rank <= 0:
        raise ValueError(f"LoRA provenance rank must be a positive integer: {sidecar_path}")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
        or float(alpha) <= 0
    ):
        raise ValueError(f"LoRA provenance alpha must be finite and positive: {sidecar_path}")
    adapter_names = inject_lora(model, rank=rank, alpha=float(alpha))
    load_lora_state_dict(model, path)
    merge_lora_into_base(model)
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rank": rank,
        "alpha": float(alpha),
        "sidecar_path": str(sidecar_path),
        "sidecar_sha256": _sha256_file(sidecar_path),
        "sidecar_metadata": sidecar,
        "adapter_modules": list(adapter_names),
        "merged": True,
    }


def freeze_base_parameters(model: nn.Module) -> None:
    """Freeze every parameter except LoRA A/B tensors."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith("lora_A") or name.endswith("lora_B"))


def merge_lora_into_base(model: nn.Module) -> tuple[str, ...]:
    """Fuse every LoRALinear residual into its base Linear in place.

    For each wrapped projection this computes ``W + (alpha / rank) * B @ A``
    in the base weight dtype, replaces the wrapper with a plain ``nn.Linear``,
    and freezes the resulting layer for inference.
    """
    candidates = [(name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)]
    if not candidates:
        raise ValueError("model has no LoRALinear modules to merge")

    merged: list[str] = []
    with torch.no_grad():
        for name, module in candidates:
            base_layer = module.base_layer
            update = (module.lora_B @ module.lora_A) * (module.alpha / module.rank)
            fused_weight = base_layer.weight + update.to(
                device=base_layer.weight.device, dtype=base_layer.weight.dtype
            )
            fused_layer = nn.Linear(
                module.in_features,
                module.out_features,
                bias=base_layer.bias is not None,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
            fused_layer.weight.copy_(fused_weight)
            if base_layer.bias is not None and fused_layer.bias is not None:
                fused_layer.bias.copy_(base_layer.bias)
            fused_layer.train(module.training)
            for parameter in fused_layer.parameters():
                parameter.requires_grad_(False)
            parent_name, attribute = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            setattr(parent, attribute, fused_layer)
            merged.append(name)
    return tuple(merged)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable LoRA parameters in stable module order."""
    parameters = [
        parameter for name, parameter in model.named_parameters() if name.endswith(("lora_A", "lora_B"))
    ]
    if not parameters:
        raise ValueError("model has no LoRA parameters")
    if any(parameter.dtype != torch.float32 or not parameter.requires_grad for parameter in parameters):
        raise ValueError("all LoRA parameters must be trainable fp32 tensors")
    return parameters


def _canonical_lora_name(name: str) -> str:
    # Activation checkpoint wrappers add this internal module segment.  Adapter
    # files must remain loadable by the unwrapped inference-time backbone.
    return name.replace("._checkpoint_wrapped_module.", ".")


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only adapter tensors with wrapper-independent names."""
    state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not name.endswith(("lora_A", "lora_B")):
            continue
        canonical_name = _canonical_lora_name(name)
        if canonical_name in state:
            raise ValueError(f"duplicate canonical LoRA parameter name: {canonical_name}")
        state[canonical_name] = parameter.detach().cpu().contiguous()
    if not state:
        raise ValueError("model has no LoRA state")
    return state


def save_lora_state_dict(model: nn.Module, path: str | Path) -> None:
    """Save the adapter-only state dict as safetensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(path))


def load_lora_state_dict(model: nn.Module, path: str | Path) -> None:
    """Strictly load an adapter-only safetensors checkpoint."""
    raw_actual = load_file(str(Path(path)), device="cpu")
    actual: dict[str, torch.Tensor] = {}
    for name, tensor in raw_actual.items():
        canonical_name = _canonical_lora_name(name)
        if canonical_name in actual:
            raise ValueError(f"duplicate canonical LoRA checkpoint key: {canonical_name}")
        actual[canonical_name] = tensor
    expected = lora_state_dict(model)
    if set(actual) != set(expected):
        raise ValueError(
            f"LoRA checkpoint keys mismatch: missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    parameters = {
        _canonical_lora_name(name): parameter
        for name, parameter in model.named_parameters()
        if name.endswith(("lora_A", "lora_B"))
    }
    with torch.no_grad():
        for name, expected_tensor in expected.items():
            tensor = actual[name]
            if tuple(tensor.shape) != tuple(expected_tensor.shape) or tensor.dtype != torch.float32:
                raise ValueError(
                    f"LoRA tensor {name} must have shape/dtype {tuple(expected_tensor.shape)}/torch.float32, "
                    f"got {tuple(tensor.shape)}/{tensor.dtype}"
                )
            parameters[name].copy_(tensor.to(device=parameters[name].device, dtype=torch.float32))
    freeze_base_parameters(model)


def full_clip_delta_timestamps(
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> dict[str, list[float]]:
    """Build a separate 61-frame camera contract without changing the causal one."""
    return {
        config.camera_key: [offset / config.fps for offset in FULL_CLIP_OFFSETS],
        config.state_key: [offset / config.fps for offset in config.state_offsets],
        config.action_key: [offset / config.fps for offset in range(config.action_chunk_size)],
    }


def full_clip_window_indices(anchor_frame: int) -> tuple[int, ...]:
    """Return the absolute frame indices requested for one full clip."""
    return tuple(anchor_frame + offset for offset in FULL_CLIP_OFFSETS)


@dataclass(frozen=True, slots=True)
class FullClipSample:
    """One expanded 61-frame sample with repeat-last padding metadata."""

    rgb_clip: torch.Tensor
    camera_is_pad: torch.Tensor
    state: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    episode_index: int
    frame_index: int
    window_indices: tuple[int, ...]
    sample: dict[str, Any]


def prepare_full_clip_sample(
    sample: Mapping[str, Any],
    *,
    frame_index: int,
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> FullClipSample:
    """Validate and reorder a full clip returned by ``LeRobotDataset``.

    LeRobot clamps out-of-episode query indices to the episode's final frame
    and emits ``camera_is_pad`` for those positions.  Those repeated frames are
    retained as model input, but their corresponding latent video loss is masked.
    """
    camera = sample.get(config.camera_key)
    camera_pad = sample.get(f"{config.camera_key}_is_pad")
    state = sample.get(config.state_key)
    action = sample.get(config.action_key)
    action_pad = sample.get(f"{config.action_key}_is_pad")
    values = (camera, camera_pad, state, action, action_pad)
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise TypeError("full-clip camera, camera_is_pad, state, action, and action_is_pad must be tensors")
    expected_camera = (
        FULL_CLIP_LENGTH,
        config.camera_shape[2],
        config.camera_shape[0],
        config.camera_shape[1],
    )
    if tuple(camera.shape) != expected_camera:
        raise ValueError(
            f"full-clip camera must have TCHW shape {expected_camera}, got {tuple(camera.shape)}"
        )
    if tuple(camera_pad.shape) != (FULL_CLIP_LENGTH,) or camera_pad.dtype != torch.bool:
        raise ValueError("full-clip camera_is_pad must have boolean shape [61]")
    if tuple(state.shape) != config.sample_state_shape:
        raise ValueError(f"state must have shape {config.sample_state_shape}, got {tuple(state.shape)}")
    if tuple(action.shape) != config.sample_action_shape:
        raise ValueError(f"action must have shape {config.sample_action_shape}, got {tuple(action.shape)}")
    if tuple(action_pad.shape) != (config.action_chunk_size,) or action_pad.dtype != torch.bool:
        raise ValueError(f"action_is_pad must have boolean shape [{config.action_chunk_size}]")
    episode_value = sample.get("episode_index", 0)
    if isinstance(episode_value, torch.Tensor):
        episode_value = episode_value.item()
    episode = int(episode_value)
    rgb_clip = camera.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    return FullClipSample(
        rgb_clip=rgb_clip,
        camera_is_pad=camera_pad.clone(),
        state=state.unsqueeze(0).contiguous(),
        target_action=action.unsqueeze(0).contiguous(),
        action_is_pad=action_pad.unsqueeze(0).contiguous(),
        episode_index=episode,
        frame_index=frame_index,
        window_indices=full_clip_window_indices(frame_index),
        sample=dict(sample),
    )


def latent_valid_mask(
    camera_is_pad: torch.Tensor,
    *,
    latent_frames: int = LATENT_FRAMES,
    pixel_frames: int = FULL_CLIP_LENGTH,
) -> torch.Tensor:
    """Map all-real temporal VAE bins to a ``[B, latent_frames]`` loss mask."""
    if camera_is_pad.ndim == 1:
        camera_is_pad = camera_is_pad.unsqueeze(0)
    if camera_is_pad.ndim != 2 or camera_is_pad.shape[1] != pixel_frames:
        raise ValueError(f"camera_is_pad must have shape [B, {pixel_frames}]")
    if latent_frames <= 0:
        raise ValueError("latent_frames must be positive")
    real_count = (~camera_is_pad).sum(dim=1)
    boundaries = torch.ceil(
        torch.linspace(0, pixel_frames, latent_frames + 1, device=camera_is_pad.device, dtype=torch.float32)
    ).to(torch.long)
    source_end = boundaries[1:] - 1
    return source_end.unsqueeze(0) < real_count.unsqueeze(1)


def sample_upstream_sigma(
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``4*exp(N(0,1))`` with the specified 5% log-uniform tail."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    sigma = VIDEO_SIGMA_MEDIAN * torch.exp(
        torch.randn(batch_size, device=device, dtype=torch.float32, generator=generator)
    )
    tail = torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
    low, high = VIDEO_SIGMA_TAIL_RANGE
    tail_sigma = torch.exp(
        torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator)
        * (math.log(high) - math.log(low))
        + math.log(low)
    )
    return torch.where(tail < VIDEO_SIGMA_TAIL_PROBABILITY, tail_sigma, sigma)


def _check_sigma(sigma: torch.Tensor, batch_size: int) -> torch.Tensor:
    sigma = sigma.reshape(-1).to(dtype=torch.float32)
    if sigma.shape != (batch_size,) or not torch.isfinite(sigma).all().item() or (sigma <= 0).any().item():
        raise ValueError("sigma must have finite positive shape [B]")
    return sigma


def build_cosmos_noised_input(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    *,
    conditional_frames: int = LATENT_CONDITIONAL_FRAMES,
    sigma_conditional: float = 1.0e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct upstream frame-replaced input and per-frame c_noise values."""
    if clean_latents.shape != noise.shape or clean_latents.ndim != 5:
        raise ValueError("clean_latents and noise must have equal [B,C,T,H,W] shapes")
    batch_size, _, time, _, _ = clean_latents.shape
    if conditional_frames <= 0 or conditional_frames > time:
        raise ValueError("conditional_frames must be inside the latent temporal dimension")
    sigma = _check_sigma(sigma, batch_size).to(device=clean_latents.device)
    from lerobot.policies.vam._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling

    scaling = RectifiedFlowScaling(sigma_data=1.0, t_scaling_factor=1.0)
    sigma_5d = sigma.view(batch_size, 1, 1, 1, 1).expand(batch_size, 1, time, 1, 1)
    _, _, c_in, c_noise = scaling(sigma_5d)
    condition_mask = torch.zeros(
        (batch_size, 1, time, clean_latents.shape[-2], clean_latents.shape[-1]),
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    condition_mask[:, :, :conditional_frames] = 1
    x_sigma = clean_latents + noise * sigma.to(dtype=clean_latents.dtype).view(batch_size, 1, 1, 1, 1)
    network_input = clean_latents * condition_mask + x_sigma * c_in.to(clean_latents.dtype) * (
        1.0 - condition_mask
    )
    sigma_conditional_5d = torch.full_like(sigma_5d, float(sigma_conditional))
    _, _, _, conditional_noise = scaling(sigma_conditional_5d)
    condition_fraction = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
    c_noise = conditional_noise * condition_fraction + c_noise * (1.0 - condition_fraction)
    return network_input, condition_mask, c_noise.squeeze(dim=[1, 3, 4])


def rectified_flow_target(clean_latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """Return the velocity target implied by vendored ``c_out=-t`` scaling."""
    if clean_latents.shape != noise.shape:
        raise ValueError("clean_latents and noise must have equal shapes")
    return noise - clean_latents


def rectified_flow_video_loss(
    prediction: torch.Tensor,
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    valid_latent_frames: torch.Tensor,
    *,
    conditional_frames: int = LATENT_CONDITIONAL_FRAMES,
    use_sigma_weights: bool = True,
) -> torch.Tensor:
    """Compute weighted rectified-flow MSE on non-conditioning, non-padding frames."""
    if prediction.shape != clean_latents.shape or prediction.shape != noise.shape or prediction.ndim != 5:
        raise ValueError("prediction, clean_latents, and noise must have equal [B,C,T,H,W] shapes")
    batch_size, _, time, _, _ = prediction.shape
    if valid_latent_frames.shape != (batch_size, time):
        raise ValueError(f"valid_latent_frames must have shape {(batch_size, time)}")
    sigma = _check_sigma(sigma, batch_size).to(device=prediction.device)
    frame_mask = valid_latent_frames.to(device=prediction.device, dtype=torch.bool).clone()
    frame_mask[:, :conditional_frames] = False
    scalar_mask = frame_mask[:, None, :, None, None]
    target = rectified_flow_target(clean_latents, noise)
    squared = (prediction.float() - target.float()).square()
    if use_sigma_weights:
        from lerobot.policies.vam._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling

        weights = RectifiedFlowScaling(sigma_data=1.0, t_scaling_factor=1.0).sigma_loss_weights(sigma)
    else:
        weights = torch.ones_like(sigma)
    weighted = squared * scalar_mask * weights.view(batch_size, 1, 1, 1, 1)
    count = scalar_mask.sum() * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
    if int(count.item()) <= 0:
        # Near-end-of-episode anchors can have only conditioning latents.
        # Keep a graph connection so the trainer can still backprop action.
        return prediction.float().sum() * 0.0
    return weighted.sum() / count


def action_context_for_arm(features: torch.Tensor, *, action_grad_to_backbone: bool) -> torch.Tensor:
    """Detach the layer-20 tap for arm B while preserving arm A gradients."""
    return features if action_grad_to_backbone else features.detach()


def capture_prediction_and_layer20(backbone: nn.Module, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one Cosmos forward and capture the pre-detach output of block 20."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 28:
        raise ValueError("Cosmos backbone must expose exactly 28 blocks")
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise TypeError("layer-20 hook output must be a tensor")
        captured.append(value)

    handle = blocks[19].register_forward_hook(hook)
    try:
        prediction = backbone(**kwargs, use_cuda_graphs=False)
    finally:
        handle.remove()
    if isinstance(prediction, tuple):
        prediction = prediction[0]
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("Cosmos backbone prediction must be a tensor")
    if not captured:
        raise RuntimeError("layer-20 hook did not observe a block output")
    return prediction, captured[0]
