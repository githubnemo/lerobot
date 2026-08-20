"""Native mimic-video World2Action decoding for the SO-101 VAM path.

The default configuration is the full native SO-101 decoder: one state token
``[B, 1, 6]``, thirty action tokens ``[B, 30, 6]``, and frozen Cosmos context
``[B, N, 2048]``.  The denoiser is trainable; context is detached at the
LeRobot boundary so no gradient can enter the frozen video extractor.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

_SUPPORTED_MODEL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

UPSTREAM_REPOSITORY = "https://github.com/mimic-video/mimic-video"
UPSTREAM_COMMIT = "e3355dbc93132b576c02f920a59b4fc18a4f5906"
NORMALIZER_SCHEMA_VERSION = 2
NORMALIZER_TENSOR_KEYS = frozenset({"state_min", "state_max", "action_min", "action_max"})
EXPECTED_NATIVE_DECODER_PARAMETERS = 499_171_958
DIAGNOSTIC_CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "robot_ready",
        "weights_only",
        "optimizer_resume_supported",
        "step",
        "manifest",
        "manifest_sha256",
        "normalizer",
        "decoder_config",
        "parameter_count",
        "optimizer",
        "autocast",
        "parameter_dtype",
        "frozen_context_dtype",
        "action_padding_semantics",
        "provenance",
    }
)
EXPECTED_DIAGNOSTIC_DECODER_CONFIG = {
    "state_shape": [1, 6],
    "action_shape": [30, 6],
    "context_channels": 2048,
    "context_layer": 20,
    "model_channels": 1024,
    "num_blocks": 24,
    "num_heads": 8,
    "full_native_capacity": True,
}


class World2ActionError(RuntimeError):
    """Base error for the LeRobot World2Action contract."""


class World2ActionValidationError(World2ActionError, ValueError):
    """An input or persisted artifact violates the strict contract."""


class World2ActionBackendError(World2ActionError):
    """The optional native decoder runtime cannot be constructed."""


def _is_float_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and torch.is_floating_point(value)


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise World2ActionValidationError(f"{name} must contain only finite values")


def _require_float(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    _require_finite(value, name)
    return value


def _strict_safetensors_state(module: nn.Module, path: Path, *, prefix: str | None = None) -> None:
    path = Path(path).expanduser()
    if path.suffix != ".safetensors":
        raise World2ActionValidationError(f"World2Action weights must be safetensors, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"World2Action checkpoint not found: {path}")
    state = load_file(str(path), device="cpu")
    if prefix is not None:
        state = {
            (key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in state.items()
        }
    expected_state = module.state_dict()
    expected = set(expected_state)
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise World2ActionValidationError(
            f"Strict World2Action checkpoint key mismatch: {json.dumps({'missing': missing, 'unexpected': unexpected})}"
        )
    for key, expected_tensor in expected_state.items():
        tensor = state[key]
        if tuple(tensor.shape) != tuple(expected_tensor.shape) or tensor.dtype != expected_tensor.dtype:
            raise World2ActionValidationError(
                f"Checkpoint tensor {key} must have shape/dtype {tuple(expected_tensor.shape)}/{expected_tensor.dtype}, "
                f"got {tuple(tensor.shape)}/{tensor.dtype}"
            )
        _require_finite(tensor, f"checkpoint tensor {key}")
    try:
        module.load_state_dict(state, strict=True, assign=True)
    except RuntimeError as exc:
        raise World2ActionValidationError(f"Strict World2Action checkpoint load failed: {exc}") from exc


def load_action_checkpoint_strict(module: nn.Module, path: str | Path, prefix: str | None = None) -> None:
    _strict_safetensors_state(module, Path(path), prefix=prefix)


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


@dataclass(frozen=True, slots=True)
class World2ActionConfig:
    """Approved native decoder capacity and flow-matching runtime settings."""

    state_shape: tuple[int, int] = (1, 6)
    action_shape: tuple[int, int] = (30, 6)
    max_horizon: int = 31
    in_channels: int = 6
    out_channels: int = 6
    model_channels: int = 1024
    num_blocks: int = 24
    num_heads: int = 8
    mlp_ratio: float = 4.0
    use_adaln_lora: bool = True
    adaln_lora_dim: int = 128
    pair_timestep_feature_rank: int = 1024
    context_channels: int = 2048
    context_layer: int = 20
    video_sigma: float = 10.0
    attention_backend: str = "torch"
    beta_alpha: float = 1.0
    beta_beta: float = 1.0
    num_denoising_steps: int = 10
    obs_dropout_train: float = 0.2
    obs_dropout_inference: float = 0.0
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    action_checkpoint_path: str | Path | None = None
    checkpoint_prefix: str | None = None

    def __post_init__(self) -> None:
        if tuple(self.state_shape) != (1, 6) or tuple(self.action_shape) != (30, 6):
            raise ValueError("The native SO-101 decoder requires state_shape=(1, 6) and action_shape=(30, 6)")
        if self.max_horizon != 31 or self.in_channels != 6 or self.out_channels != 6:
            raise ValueError(
                "The native SO-101 decoder requires max_horizon=31 and six input/output channels"
            )
        if self.context_channels != 2048:
            raise ValueError(
                "Cosmos context must remain 2048-dimensional; no projection adapter is permitted"
            )
        if self.context_layer != 20:
            raise ValueError("The approved Cosmos context layer is exactly 20")
        if self.video_sigma != 10.0:
            raise ValueError("The approved Cosmos high-noise video sigma is exactly 10")
        if self.attention_backend != "torch":
            raise ValueError("The approved runtime attention backend is PyTorch scaled_dot_product_attention")
        if self.beta_alpha != 1.0 or self.beta_beta != 1.0:
            raise ValueError("The registered action scheduler is Beta(alpha=1.0, beta=1.0)")
        if self.num_denoising_steps != 10:
            raise ValueError("The registered action scheduler has exactly ten denoising steps")
        if self.obs_dropout_train != 0.2 or self.obs_dropout_inference != 0.0:
            raise ValueError("Approved observation dropout is 0.2 in training and 0.0 in inference")
        approved_capacity = (
            self.model_channels,
            self.num_blocks,
            self.num_heads,
            self.mlp_ratio,
            self.use_adaln_lora,
            self.adaln_lora_dim,
            self.pair_timestep_feature_rank,
        )
        if approved_capacity != (1024, 24, 8, 4.0, True, 128, 1024):
            raise ValueError(
                "The native decoder capacity must remain model_channels=1024, num_blocks=24, num_heads=8, "
                "mlp_ratio=4, use_adaln_lora=True, adaln_lora_dim=128, pair_timestep_feature_rank=1024"
            )
        if self.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("Decoder dtype must be float32, float16, or bfloat16")
        if self.action_checkpoint_path is not None:
            object.__setattr__(self, "action_checkpoint_path", Path(self.action_checkpoint_path))

    @property
    def state_dim(self) -> int:
        return self.state_shape[-1]

    @property
    def action_horizon(self) -> int:
        return self.action_shape[0]


# Short name retained for callers that use the policy name rather than the
# decoder implementation name.
World2ActionDecoderConfig = World2ActionConfig


@dataclass(frozen=True, slots=True)
class ActionStateNormalizer:
    """Frozen train-split per-joint min-max statistics.

    Normalization maps the closed training range to ``[-1, 1]`` and clamps
    out-of-range values at that interval.  Denormalization also clamps its
    normalized input first; this makes the policy's saturation behavior
    explicit and prevents extrapolation beyond train-set joint ranges.

    The denormalization clamp is deliberately kept rather than made exact.
    Clamping projects a prediction onto the convex training range, and every
    reachable joint position of this robot lies inside that range, so the
    projection can only move a prediction closer to its target: it is a
    non-increasing operation on the action error, not a source of bias.
    """

    state_min: torch.Tensor
    state_max: torch.Tensor
    action_min: torch.Tensor
    action_max: torch.Tensor
    source_split: str = "train"
    state_shape: tuple[int, int] = (1, 6)
    action_shape: tuple[int, int] = (30, 6)

    def __post_init__(self) -> None:
        for name in ("state_min", "state_max", "action_min", "action_max"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (6,):
                raise World2ActionValidationError(f"{name} must be a tensor with shape [6]")
            if not torch.is_floating_point(value):
                raise World2ActionValidationError(f"{name} must be floating point")
            _require_finite(value, name)
        if self.source_split != "train":
            raise World2ActionValidationError("Normalization statistics must have source_split='train'")
        if tuple(self.state_shape) != (1, 6) or tuple(self.action_shape) != (30, 6):
            raise World2ActionValidationError("Normalizer shape provenance must match native SO-101 shapes")
        if not torch.all(self.state_max > self.state_min).item():
            raise World2ActionValidationError("state statistics contain a degenerate joint range")
        if not torch.all(self.action_max > self.action_min).item():
            raise World2ActionValidationError("action statistics contain a degenerate joint range")

    @classmethod
    def from_training_tensors(
        cls,
        state: torch.Tensor,
        action: torch.Tensor,
        *,
        action_is_pad: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        source_split: str = "train",
    ) -> ActionStateNormalizer:
        # Statistics are always computed from the training split only.
        if source_split != "train":
            raise World2ActionValidationError("Statistics may only be computed from source_split=train")
        if action_is_pad is not None and action_mask is not None:
            raise World2ActionValidationError("Pass only one of action_is_pad or action_mask")
        _validate_shape(state, (1, 6), "state training tensor", trailing=True)
        _validate_shape(action, (30, 6), "action training tensor", trailing=True)
        state_flat = state.detach().to(torch.float32).reshape(-1, 6)
        action_flat = action.detach().to(torch.float32).reshape(-1, 6)
        padding = action_is_pad if action_is_pad is not None else action_mask
        if padding is None:
            valid_action = action_flat
        else:
            padding = _validate_action_is_pad(padding, action.shape[0], action.device)
            valid_action = action_flat[~padding.reshape(-1)]
            if valid_action.numel() == 0:
                raise World2ActionValidationError("action padding mask marks every action token as padded")
        return cls(
            state_min=state_flat.amin(dim=0).cpu(),
            state_max=state_flat.amax(dim=0).cpu(),
            action_min=valid_action.amin(dim=0).cpu(),
            action_max=valid_action.amax(dim=0).cpu(),
        )

    def _apply(
        self, value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, name: str
    ) -> torch.Tensor:
        value = _require_float(value, name)
        work = value.to(torch.float32)
        scale = (maximum - minimum).to(device=value.device, dtype=torch.float32)
        result = 2.0 * (work - minimum.to(device=value.device, dtype=torch.float32)) / scale - 1.0
        return result.clamp(-1.0, 1.0).to(dtype=value.dtype)

    def _invert(
        self, value: torch.Tensor, minimum: torch.Tensor, maximum: torch.Tensor, name: str
    ) -> torch.Tensor:
        value = _require_float(value, name)
        work = value.to(torch.float32).clamp(-1.0, 1.0)
        minimum = minimum.to(device=value.device, dtype=torch.float32)
        maximum = maximum.to(device=value.device, dtype=torch.float32)
        return (((work + 1.0) * 0.5) * (maximum - minimum) + minimum).to(dtype=value.dtype)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        _validate_shape(state, self.state_shape, "state", trailing=True)
        return self._apply(state, self.state_min, self.state_max, "state")

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        _validate_shape(action, self.action_shape, "action", trailing=True)
        return self._apply(action, self.action_min, self.action_max, "action")

    def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        _validate_shape(state, self.state_shape, "normalized state", trailing=True)
        return self._invert(state, self.state_min, self.state_max, "normalized state")

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        _validate_shape(action, self.action_shape, "normalized action", trailing=True)
        return self._invert(action, self.action_min, self.action_max, "normalized action")

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema_version": NORMALIZER_SCHEMA_VERSION,
            "source_split": self.source_split,
            "state_shape": list(self.state_shape),
            "action_shape": list(self.action_shape),
            "normalization": "per_joint_min_max_to_minus_one_one",
            "clamp_policy": "normalize_and_denormalize_clamp_to_minus_one_one",
            "action_padding_policy": "exclude_action_is_pad_true_tokens",
            "dtype": "float32",
        }

    def save(
        self, path: str | Path, *, metadata_path: str | Path | None = None, overwrite: bool = False
    ) -> tuple[Path, Path]:
        """Save only tensors plus a strict JSON provenance sidecar."""
        path = Path(path).expanduser()
        sidecar = Path(metadata_path).expanduser() if metadata_path is not None else _sidecar_path(path)
        if (path.exists() or sidecar.exists()) and not overwrite:
            raise FileExistsError(f"Refusing to overwrite normalizer artifact; pass overwrite=True: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tensor_fd, tensor_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        sidecar_fd, sidecar_tmp = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
        )
        os.close(tensor_fd)
        try:
            save_file(
                {
                    "state_min": self.state_min.detach().cpu().to(torch.float32).contiguous(),
                    "state_max": self.state_max.detach().cpu().to(torch.float32).contiguous(),
                    "action_min": self.action_min.detach().cpu().to(torch.float32).contiguous(),
                    "action_max": self.action_max.detach().cpu().to(torch.float32).contiguous(),
                },
                tensor_tmp,
            )
            with os.fdopen(sidecar_fd, "w") as stream:
                json.dump(self._metadata(), stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tensor_tmp, path)
            os.replace(sidecar_tmp, sidecar)
        except BaseException:
            for temporary in (tensor_tmp, sidecar_tmp):
                with suppress(FileNotFoundError):
                    os.unlink(temporary)
            raise
        return path, sidecar

    @classmethod
    def load(cls, path: str | Path, *, metadata_path: str | Path | None = None) -> ActionStateNormalizer:
        """Load and strictly validate the safetensors/JSON pair."""
        path = Path(path).expanduser()
        sidecar = Path(metadata_path).expanduser() if metadata_path is not None else _sidecar_path(path)
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Normalizer artifact requires {path} and {sidecar}")
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != NORMALIZER_TENSOR_KEYS:
            raise World2ActionValidationError(
                f"Normalizer tensor keys must be exactly {sorted(NORMALIZER_TENSOR_KEYS)}, got {sorted(tensors)}"
            )
        try:
            metadata = json.loads(sidecar.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise World2ActionValidationError(f"Could not read normalizer metadata: {exc}") from exc
        expected = {
            "schema_version",
            "source_split",
            "state_shape",
            "action_shape",
            "normalization",
            "clamp_policy",
            "action_padding_policy",
            "dtype",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected:
            raise World2ActionValidationError("Normalizer metadata keys are malformed")
        if metadata != {
            **metadata,
            "schema_version": NORMALIZER_SCHEMA_VERSION,
            "source_split": "train",
            "state_shape": [1, 6],
            "action_shape": [30, 6],
            "normalization": "per_joint_min_max_to_minus_one_one",
            "clamp_policy": "normalize_and_denormalize_clamp_to_minus_one_one",
            "action_padding_policy": "exclude_action_is_pad_true_tokens",
            "dtype": "float32",
        }:
            raise World2ActionValidationError(
                "Normalizer metadata does not match the approved train-set contract"
            )
        return cls(
            state_min=tensors["state_min"],
            state_max=tensors["state_max"],
            action_min=tensors["action_min"],
            action_max=tensors["action_max"],
        )


# Common artifact-oriented name.
NormalizationArtifact = ActionStateNormalizer


class _IdentityNormalizer:
    source_split = "none"

    def normalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def normalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def denormalize_action(self, value: torch.Tensor) -> torch.Tensor:
        return value

    def denormalize_state(self, value: torch.Tensor) -> torch.Tensor:
        return value


@dataclass(frozen=True, slots=True)
class ParameterCountReport:
    """Parameter counts; normalization is deliberately non-trainable."""

    decoder_parameters: int
    normalizer_parameters: int
    trainable_decoder_parameters: int
    trainable_normalizer_parameters: int = 0

    @property
    def total_parameters(self) -> int:
        return self.decoder_parameters + self.normalizer_parameters

    @property
    def trainable_parameters(self) -> int:
        return self.trainable_decoder_parameters + self.trainable_normalizer_parameters

    @property
    def decoder(self) -> int:
        return self.decoder_parameters

    @property
    def normalizer(self) -> int:
        return self.normalizer_parameters


def _validate_shape(value: Any, expected: tuple[int, ...], name: str, *, trailing: bool = False) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    actual = (
        tuple(value.shape[-len(expected) :])
        if trailing and value.ndim >= len(expected)
        else tuple(value.shape)
    )
    if actual != expected:
        suffix = " (trailing dimensions)" if trailing else ""
        raise World2ActionValidationError(
            f"{name} must have shape {expected}{suffix}, got {tuple(value.shape)}"
        )
    _require_float(value, name)


def _validate_action_is_pad(mask: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2 or tuple(mask.shape) != (batch_size, 30):
        shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else type(mask).__name__
        raise World2ActionValidationError(f"action_is_pad must have boolean shape [B, 30], got {shape}")
    if mask.dtype != torch.bool:
        raise World2ActionValidationError("action_is_pad must have boolean dtype")
    mask = mask.to(device=device)
    if bool(mask.all().item()):
        raise World2ActionValidationError("action_is_pad cannot mark every action token as padded")
    return mask


class World2ActionDecoder(nn.Module):
    """LeRobot-owned loss/sampling wrapper around native World2ActionDIT."""

    def __init__(
        self,
        config: World2ActionConfig | None = None,
        *,
        denoiser: nn.Module | Any | None = None,
        normalizer: ActionStateNormalizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        super().__init__()
        self.config = config or World2ActionConfig()
        self.denoiser = denoiser if denoiser is not None else self._build_native_denoiser()
        self.normalizer = normalizer or _IdentityNormalizer()
        if scheduler is None:
            from ._vendor.cosmos_predict2.schedulers.beta_scheduler import BetaScheduler

            scheduler = BetaScheduler(
                alpha=self.config.beta_alpha,
                beta=self.config.beta_beta,
                num_denoising_steps=self.config.num_denoising_steps,
            )
        if scheduler.alpha != self.config.beta_alpha or scheduler.beta != self.config.beta_beta:
            raise ValueError("Injected scheduler alpha/beta must match the approved uniform Beta(1.0, 1.0)")
        if scheduler.num_denoising_steps != self.config.num_denoising_steps:
            raise ValueError("Injected scheduler must expose exactly ten denoising steps")
        self.scheduler = scheduler

    @staticmethod
    def _load_native_denoiser_components() -> tuple[Any, Any]:
        try:
            from ._vendor.cosmos_predict2.models.world2action_dit import World2ActionDIT
            from ._vendor.cosmos_predict2.networks.selective_activation_checkpoint import SACConfig
        except (ImportError, OSError) as exc:
            raise World2ActionBackendError(
                "The vendored native World2Action closure could not be imported"
            ) from exc
        return World2ActionDIT, SACConfig

    def _build_native_denoiser(self) -> nn.Module:
        device = torch.device(self.config.device)
        denoiser_cls, sac_config_cls = self._load_native_denoiser_components()
        constructor_kwargs = {
            "max_horizon": self.config.max_horizon,
            "in_channels": self.config.in_channels,
            "out_channels": self.config.out_channels,
            "model_channels": self.config.model_channels,
            "num_blocks": self.config.num_blocks,
            "num_heads": self.config.num_heads,
            "mlp_ratio": self.config.mlp_ratio,
            "atten_backend": self.config.attention_backend,
            "crossattn_emb_channels": self.config.context_channels,
            "use_adaln_lora": self.config.use_adaln_lora,
            "adaln_lora_dim": self.config.adaln_lora_dim,
            "pair_timestep_feature_rank": self.config.pair_timestep_feature_rank,
            "sac_config": sac_config_cls(mode="none", every_n_blocks=1),
        }
        checkpoint_path = self.config.action_checkpoint_path
        if checkpoint_path is None:
            # Scratch initialization is intentional: no compatible pretrained
            # 6D SO-101 action checkpoint exists. Do not reset torch RNG here;
            # the trainer must establish its seed before policy construction.
            try:
                model = denoiser_cls(**constructor_kwargs)
            except RuntimeError as exc:
                if "Transformer Engine" in str(exc):
                    raise World2ActionBackendError(str(exc)) from exc
                raise
            return model.to(device=device, dtype=self.config.dtype).eval()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"World2Action checkpoint not found: {checkpoint_path}")
        try:
            # Meta construction avoids allocating a second full model while
            # preserving strict loading for an explicitly supplied checkpoint.
            with torch.device("meta"):
                model = denoiser_cls(**constructor_kwargs)
        except RuntimeError as exc:
            if "Transformer Engine" in str(exc):
                raise World2ActionBackendError(str(exc)) from exc
            raise
        model = model.to_empty(device=device)
        load_action_checkpoint_strict(model, checkpoint_path, self.config.checkpoint_prefix)
        return model.to(device=device, dtype=self.config.dtype).eval()

    def _model_dtype(self, fallback: torch.dtype) -> torch.dtype:
        if isinstance(self.denoiser, nn.Module):
            for parameter in self.denoiser.parameters():
                return parameter.dtype
        return fallback

    def _validate_inputs(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        context_timestep: torch.Tensor | None,
        action: torch.Tensor | None = None,
    ) -> tuple[int, torch.Tensor | None]:
        _validate_shape(state, self.config.state_shape, "state", trailing=True)
        if state.dtype not in _SUPPORTED_MODEL_DTYPES:
            raise TypeError("state must use float16, bfloat16, or float32")
        if state.ndim != 3 or tuple(state.shape[1:]) != self.config.state_shape:
            raise World2ActionValidationError(
                f"state must have shape [B, 1, 6] and rank 3, got {tuple(state.shape)}"
            )
        if context.ndim != 3 or tuple(context.shape[2:]) != (self.config.context_channels,):
            raise World2ActionValidationError(
                f"context must have shape [B, N, {self.config.context_channels}], got {tuple(context.shape)}"
            )
        _require_float(context, "context")
        if context.dtype != torch.bfloat16:
            raise TypeError("context must use frozen Cosmos bfloat16 hidden tokens")
        if context.shape[1] <= 0 or context.shape[0] != state.shape[0]:
            raise World2ActionValidationError("context batch and token dimensions are invalid")
        if action is not None:
            _validate_shape(action, self.config.action_shape, "action", trailing=True)
            if action.ndim != 3 or tuple(action.shape[1:]) != self.config.action_shape:
                raise World2ActionValidationError(
                    f"action must have shape [B, 30, 6] and rank 3, got {tuple(action.shape)}"
                )
            if action.dtype not in _SUPPORTED_MODEL_DTYPES:
                raise TypeError("action must use float16, bfloat16, or float32")
            if action.shape[0] != state.shape[0] or action.dtype != state.dtype:
                raise World2ActionValidationError("action must be [B, 30, 6] and use the state dtype")
        if context_timestep is None:
            return state.shape[0], None
        _validate_shape(context_timestep, (state.shape[0], 1), "context_timestep", trailing=False)
        if context_timestep.ndim != 2:
            raise World2ActionValidationError("context_timestep must have shape [B, 1]")
        return state.shape[0], context_timestep

    def _context_time(
        self, context_timestep: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if context_timestep is None:
            return torch.full((batch_size, 1), self.config.video_sigma, device=device, dtype=torch.float32)
        return context_timestep.to(device=device)

    def _call_denoiser(
        self,
        state: torch.Tensor,
        action_input: torch.Tensor,
        time: torch.Tensor,
        context_time: torch.Tensor,
        context: torch.Tensor,
        *,
        obs_dropout: float,
    ) -> torch.Tensor:
        model_dtype = self._model_dtype(action_input.dtype)
        state = state.to(dtype=model_dtype)
        action_input = action_input.to(dtype=model_dtype)
        context = context.detach()
        time = time.to(device=action_input.device, dtype=model_dtype)
        if time.ndim == 1:
            time = time[:, None]
        timesteps = time.expand(-1, self.config.max_horizon)
        context_time = context_time.to(device=action_input.device, dtype=model_dtype)
        prediction = self.denoiser(
            state_B_HO_O=state,
            xt_B_HA_A=action_input,
            timesteps_B_T=timesteps,
            context_timesteps_B_1=context_time,
            crossattn_emb=context,
            obs_dropout=obs_dropout,
        )
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        if prediction.ndim != 3 or prediction.shape[0] != state.shape[0]:
            raise World2ActionValidationError("denoiser output must have rank 3 with the input batch size")
        if prediction.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise World2ActionValidationError("denoiser output must use float16, bfloat16, or float32")
        raw_shape = (self.config.max_horizon, self.config.out_channels)
        action_shape = self.config.action_shape
        if tuple(prediction.shape[1:]) == raw_shape:
            # Native DiT includes the HO state token in its output. It
            # participates in the internal sequence but has no action loss.
            observation_tokens = state.shape[1]
            prediction = prediction[:, observation_tokens:, :]
        elif tuple(prediction.shape[1:]) != action_shape:
            raise World2ActionValidationError(
                "denoiser output must have shape [B, 31, 6] (native, sliced to [B, 30, 6]) "
                "or [B, 30, 6] for an explicitly sliced injected denoiser"
            )
        _validate_shape(prediction, action_shape, "denoiser output", trailing=True)
        return prediction

    def sample_training_t(
        self,
        batch_size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the registered Beta(1, 1) training time on any test device."""
        # The official scheduler samples Beta(alpha,beta) and bounds it to
        # [0.001, 1.0]. torch.rand is exactly that Beta(1,1) law on CPU too.
        return torch.rand(batch_size, device=device, dtype=torch.float32, generator=generator) * 0.999 + 0.001

    def flow_matching_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        *,
        t: torch.Tensor | None = None,
        epsilon: torch.Tensor | None = None,
        context_timestep: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        action_is_pad: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
        obs_dropout: float | None = None,
    ) -> torch.Tensor:
        # The unmasked branch below intentionally remains ordinary MSE.
        if action_is_pad is not None and action_mask is not None:
            raise World2ActionValidationError("Pass only one of action_is_pad or action_mask")
        batch_size, context_timestep = self._validate_inputs(state, context, context_timestep, action)
        padding = action_is_pad if action_is_pad is not None else action_mask
        if padding is not None:
            padding = _validate_action_is_pad(padding, batch_size, action.device)
        state_0 = self.normalizer.normalize_state(state)
        action_0 = self.normalizer.normalize_action(action)
        device = action.device
        if t is None:
            t = self.sample_training_t(batch_size, device=device, generator=generator)
        else:
            _validate_shape(t, (batch_size,), "t", trailing=False)
            t = t.to(device=device, dtype=torch.float32)
            if not torch.all((t >= 0) & (t <= 1)).item():
                raise World2ActionValidationError("t must be in [0, 1]")
        if epsilon is None:
            epsilon = torch.randn(action_0.shape, device=device, dtype=action_0.dtype, generator=generator)
        else:
            _validate_shape(epsilon, self.config.action_shape, "epsilon", trailing=True)
            if epsilon.shape != action_0.shape or epsilon.dtype != action_0.dtype:
                raise World2ActionValidationError("epsilon must have the action shape and dtype")
        t_expanded = t[:, None, None]
        xt = (1.0 - t_expanded) * action_0 + t_expanded * epsilon
        target = epsilon - action_0
        scale = torch.sqrt((1.0 - t).square() + t.square())[:, None, None]
        dropout = self.config.obs_dropout_train if obs_dropout is None else obs_dropout
        if not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) <= 1.0:
            raise World2ActionValidationError("obs_dropout must be in [0, 1]")
        prediction = self._call_denoiser(
            state_0,
            xt / scale,
            t,
            self._context_time(context_timestep, batch_size, device),
            context,
            obs_dropout=float(dropout),
        )
        prediction_float = prediction.float()
        target_float = target.float()
        if padding is None:
            return torch.nn.functional.mse_loss(prediction_float, target_float)
        valid = (~padding).to(dtype=torch.float32).unsqueeze(-1)
        squared_error = (prediction_float - target_float).square()
        valid_scalar_count = valid.sum() * self.config.state_dim
        if not torch.isfinite(valid_scalar_count).item() or valid_scalar_count.item() <= 0:
            raise World2ActionValidationError("action_is_pad leaves no valid scalar action positions")
        return (squared_error * valid).sum() / valid_scalar_count

    compute_loss = flow_matching_loss
    loss = flow_matching_loss

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        *,
        seed: int | None = None,
        context_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Start at seeded Gaussian noise and take exactly ten official Euler steps."""
        batch_size, context_timestep = self._validate_inputs(state, context, context_timestep)
        state_0 = self.normalizer.normalize_state(state)
        device = state.device
        model_dtype = self._model_dtype(state_0.dtype)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        xt = torch.randn(
            (batch_size, *self.config.action_shape),
            device=device,
            dtype=model_dtype,
            generator=generator,
        )
        time = torch.ones(batch_size, device=device, dtype=torch.float32)
        context_time = self._context_time(context_timestep, batch_size, device)
        for _ in range(self.scheduler.num_denoising_steps):
            scale = torch.sqrt((1.0 - time).square() + time.square())[:, None, None].to(dtype=xt.dtype)
            prediction = self._call_denoiser(
                state_0,
                xt / scale,
                time,
                context_time,
                context,
                obs_dropout=self.config.obs_dropout_inference,
            )
            xt, time = self.scheduler.step(prediction, xt, time)
        return self.normalizer.denormalize_action(xt)

    sample = sample_actions
    infer = sample_actions

    def forward(
        self,
        state: torch.Tensor,
        context: torch.Tensor,
        *,
        seed: int | None = None,
        context_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.sample_actions(state, context, seed=seed, context_timestep=context_timestep)

    def parameter_count_report(self) -> ParameterCountReport:
        decoder_parameters = sum(parameter.numel() for parameter in self.denoiser.parameters())
        trainable_decoder_parameters = sum(
            parameter.numel() for parameter in self.denoiser.parameters() if parameter.requires_grad
        )
        return ParameterCountReport(
            decoder_parameters=decoder_parameters,
            normalizer_parameters=0,
            trainable_decoder_parameters=trainable_decoder_parameters,
        )


def load_diagnostic_checkpoint(
    decoder: World2ActionDecoder,
    checkpoint_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    manifest_path: str | Path,
    normalizer_path: str | Path,
    expected_step: int | None = None,
    expected_parameter_count: int = EXPECTED_NATIVE_DECODER_PARAMETERS,
) -> dict[str, Any]:
    """Strictly reopen a diagnostic-only World2Action safetensors artifact."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    metadata_path = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else checkpoint_path.with_suffix(".json")
    )
    manifest_path = Path(manifest_path).expanduser().resolve()
    normalizer_path = Path(normalizer_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Diagnostic checkpoint metadata not found: {metadata_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Diagnostic manifest not found: {manifest_path}")
    if not normalizer_path.is_file():
        raise FileNotFoundError(f"Diagnostic normalizer not found: {normalizer_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise World2ActionValidationError(f"Could not read diagnostic checkpoint metadata: {exc}") from exc
    if not isinstance(metadata, dict) or set(metadata) != DIAGNOSTIC_CHECKPOINT_METADATA_KEYS:
        raise World2ActionValidationError("Diagnostic checkpoint metadata keys are not strict")
    if metadata["schema_version"] != 1 or metadata["artifact"] != "world2action_decoder_weights":
        raise World2ActionValidationError("Diagnostic checkpoint schema/artifact is invalid")
    if metadata["status"] != "diagnostic_only_non_rollout" or metadata["robot_ready"] is not False:
        raise World2ActionValidationError("Diagnostic checkpoint must be non-rollout and robot_ready=false")
    if metadata["weights_only"] is not True or metadata["optimizer_resume_supported"] is not False:
        raise World2ActionValidationError("Diagnostic checkpoint must be weights-only and non-resumable")
    if metadata["parameter_dtype"] != "float32" or metadata["frozen_context_dtype"] != "bfloat16":
        raise World2ActionValidationError("Diagnostic checkpoint dtype provenance is invalid")
    if metadata["manifest"] != str(manifest_path) or metadata["normalizer"] != str(normalizer_path):
        raise World2ActionValidationError(
            "Diagnostic checkpoint manifest/normalizer provenance does not match"
        )
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if metadata["manifest_sha256"] != digest:
        raise World2ActionValidationError("Diagnostic checkpoint manifest SHA256 does not match")
    if type(metadata["step"]) is not int or metadata["step"] < 0:
        raise World2ActionValidationError("Diagnostic checkpoint step must be a non-negative integer")
    if expected_step is not None and metadata["step"] != expected_step:
        raise World2ActionValidationError("Diagnostic checkpoint step does not match the requested step")
    if metadata["decoder_config"] != EXPECTED_DIAGNOSTIC_DECODER_CONFIG:
        raise World2ActionValidationError(
            "Diagnostic checkpoint decoder config is not the approved native config"
        )
    expected_counts = {"decoder": expected_parameter_count, "trainable_decoder": expected_parameter_count}
    if metadata["parameter_count"] != expected_counts:
        raise World2ActionValidationError(
            "Diagnostic checkpoint parameter count is not the expected full capacity"
        )
    report = decoder.parameter_count_report()
    if (
        report.decoder_parameters != expected_parameter_count
        or report.trainable_decoder_parameters != expected_parameter_count
    ):
        raise World2ActionValidationError(
            "Constructed decoder parameter count does not match diagnostic metadata"
        )
    if decoder.config.dtype not in _SUPPORTED_MODEL_DTYPES:
        raise World2ActionValidationError("Constructed decoder dtype is unsupported")
    decoder.denoiser.to(dtype=torch.float32)
    _strict_safetensors_state(decoder.denoiser, checkpoint_path)
    decoder.denoiser.to(device=torch.device(decoder.config.device), dtype=decoder.config.dtype)
    return metadata


World2ActionModel = World2ActionDecoder
