"""VLM-free SmolVLA action expert for frozen Cosmos contexts.

SmolVLA normally feeds its action expert key/value caches produced by the VLM
prefix.  This module injects Cosmos features at that cache boundary instead:
Cosmos features are LayerNorm/Linear projected to the VLM hidden width, then a
small learned K/V projection produces the 320-wide prefix K/V consumed by the
expert.  The VLM tower is never instantiated or loaded.
"""

from __future__ import annotations

import copy
import hashlib
import math
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn

from ..common.flow_matching import euler_integrate, sample_time_beta
from ..common.vla_utils import create_sinusoidal_pos_embedding
from ..smolvla.smolvlm_with_expert import apply_rope, get_intermediate_size

SMOLVLA_CHECKPOINT = "lerobot/smolvla_base"
SMOLVLM_CONFIG = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
ACTION_HORIZON = 30
ACTION_DIM = 6
MAX_ACTION_DIM = 32
MAX_STATE_DIM = 32
EXPERT_WIDTH_MULTIPLIER = 0.75
NUM_EXPERT_LAYERS = 16
SELF_ATTN_EVERY_N_LAYERS = 2
CUDA_GRAPH_CONTEXT_SHAPE = (1, 4_800, 2_048)


def cuda_graph_shape_eligible(
    state_shape: tuple[int, ...],
    context_shape: tuple[int, ...],
    noise_shape: tuple[int, ...],
    *,
    max_action_dim: int,
    prefix_is_cached: bool,
) -> bool:
    """Return whether tensor shapes match the serialized production CUDA graph."""
    return (
        len(state_shape) in (2, 3)
        and state_shape[0] == 1
        and state_shape[-1] == ACTION_DIM
        and context_shape == CUDA_GRAPH_CONTEXT_SHAPE
        and noise_shape == (1, ACTION_HORIZON, max_action_dim)
        and prefix_is_cached
    )


@dataclass(frozen=True, slots=True)
class SmolVLANormalizer:
    """SmolVLA's train-split mean/std normalization for state and actions."""

    state_mean: Tensor
    state_std: Tensor
    action_mean: Tensor
    action_std: Tensor
    eps: float = 1e-8
    source_split: str = "train"

    def __post_init__(self) -> None:
        if self.source_split != "train":
            raise ValueError("SmolVLA normalization statistics must come from source_split='train'")
        for name in ("state_mean", "state_std", "action_mean", "action_std"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or tuple(value.shape) != (ACTION_DIM,):
                raise ValueError(f"{name} must have shape [6]")
            if not torch.is_floating_point(value) or not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must be finite floating-point data")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("normalization eps must be finite and positive")
        if not torch.all(self.state_std >= 0).item() or not torch.all(self.action_std >= 0).item():
            raise ValueError("normalization standard deviations must be non-negative")

    @classmethod
    def from_training_tensors(
        cls,
        state: Tensor,
        action: Tensor,
        *,
        action_is_pad: Tensor | None = None,
        eps: float = 1e-8,
    ) -> SmolVLANormalizer:
        """Compute population mean/std using train episodes only."""
        if state.shape[-1] != ACTION_DIM or action.shape[-1] != ACTION_DIM:
            raise ValueError("state and action tensors must end in six dimensions")
        state_flat = state.detach().to(dtype=torch.float32).reshape(-1, ACTION_DIM)
        action_flat = action.detach().to(dtype=torch.float32).reshape(-1, ACTION_DIM)
        if action_is_pad is None:
            valid_actions = action_flat
        else:
            if action_is_pad.shape != action.shape[:2] or action_is_pad.dtype is not torch.bool:
                raise ValueError("action_is_pad must have shape [B, 30] and boolean dtype")
            valid_actions = action_flat[~action_is_pad.reshape(-1)]
        if state_flat.numel() == 0 or valid_actions.numel() == 0:
            raise ValueError("cannot normalize an empty train split")
        return cls(
            state_mean=state_flat.mean(dim=0),
            state_std=state_flat.std(dim=0, unbiased=False),
            action_mean=valid_actions.mean(dim=0),
            action_std=valid_actions.std(dim=0, unbiased=False),
            eps=eps,
        )

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self.state_mean.to(state)) / (self.state_std.to(state) + self.eps)

    def normalize_action(self, action: Tensor) -> Tensor:
        return (action - self.action_mean.to(action)) / (self.action_std.to(action) + self.eps)

    def denormalize_action(self, action: Tensor) -> Tensor:
        return action * self.action_std.to(action) + self.action_mean.to(action)

    def state_dict(self) -> dict[str, Tensor]:
        return {
            "state_mean": self.state_mean.detach().cpu(),
            "state_std": self.state_std.detach().cpu(),
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
        }


def _tensor_digest(value: Tensor) -> str:
    digest = hashlib.sha256()
    value = value.detach().cpu().contiguous()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def checkpoint_tensor_sha256(checkpoint: str | Path, key: str) -> str:
    """Hash one safetensors tensor, including dtype and shape provenance."""
    path = resolve_checkpoint(checkpoint)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if key not in keys:
            raise KeyError(f"checkpoint has no tensor {key!r}")
        return _tensor_digest(handle.get_tensor(key))


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a local directory/file or a Hub model id to model.safetensors."""
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        model_file = path / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"checkpoint directory has no model.safetensors: {path}")
        return model_file
    return Path(hf_hub_download(repo_id=str(checkpoint), filename="model.safetensors"))


def _load_module_from_checkpoint(module: nn.Module, handle: Any, prefix: str) -> dict[str, str]:
    expected = module.state_dict()
    checkpoint_keys = set(handle.keys())
    missing = [prefix + name for name in expected if prefix + name not in checkpoint_keys]
    if missing:
        raise ValueError(f"pretrained SmolVLA checkpoint is missing {missing[:8]}")
    loaded: dict[str, Tensor] = {}
    digests: dict[str, str] = {}
    for name, expected_tensor in expected.items():
        key = prefix + name
        tensor = handle.get_tensor(key)
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"checkpoint tensor {key} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
        loaded[name] = tensor
        digests[key] = _tensor_digest(tensor)
    module.load_state_dict(loaded, strict=True, assign=True)
    return digests


@dataclass(frozen=True, slots=True)
class SmolExpertPrefixKVCache:
    """Per-layer prefix K/V tensors reusable across action-flow evaluations."""

    keys: tuple[Tensor, ...]
    values: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("prefix K/V cache must contain one key and value tensor per expert layer")


class _ExpertDenoiseLoop(nn.Module):
    """Fixed-shape denoise loop used by the production CUDA-graph runner."""

    def __init__(
        self, decoder: SmolExpertActionDecoder, prefix: SmolExpertPrefixKVCache, num_steps: int
    ) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError("expert CUDA graph requires a positive denoising step count")
        self.decoder = decoder
        self.prefix = prefix
        self.num_steps = num_steps
        parameter = next(decoder.parameters())
        dt = -1.0 / num_steps
        self.register_buffer(
            "time_values",
            torch.tensor(
                [1.0 + step * dt for step in range(num_steps)],
                dtype=torch.float32,
                device=parameter.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "action_mean",
            decoder.normalizer.action_mean.detach().to(device=parameter.device),
            persistent=False,
        )
        self.register_buffer(
            "action_std",
            decoder.normalizer.action_std.detach().to(device=parameter.device),
            persistent=False,
        )

    def forward(self, noise: Tensor) -> Tensor:
        x_t = noise
        batch_size = noise.shape[0]
        dt = -1.0 / self.num_steps
        for step in range(self.num_steps):
            current_time = self.time_values[step].expand(batch_size)
            velocity = self.decoder._vector_field(self.prefix, x_t, current_time)
            x_t = x_t + dt * velocity
        action = x_t[..., :ACTION_DIM]
        return (action * self.action_std.to(dtype=action.dtype) + self.action_mean.to(dtype=action.dtype)).to(
            dtype=torch.float32
        )


class _ExpertCudaGraphRunner:
    """Serialized fixed-shape CUDA graph with mutable noise and prefix buffers."""

    def __init__(self, decoder: SmolExpertActionDecoder) -> None:
        self.decoder = decoder
        self._lock = threading.Lock()
        self._graph: torch.cuda.CUDAGraph | None = None
        self._loop: _ExpertDenoiseLoop | None = None
        self._static_noise: Tensor | None = None
        self._static_prefix: SmolExpertPrefixKVCache | None = None
        self._static_output: Tensor | None = None
        self._decoder_device: torch.device | None = None
        self._decoder_dtype: torch.dtype | None = None
        self.capture_seconds: float | None = None
        self.capture_count = 0

    @staticmethod
    def _clone_prefix(prefix: SmolExpertPrefixKVCache) -> SmolExpertPrefixKVCache:
        return SmolExpertPrefixKVCache(
            tuple(key.detach().clone() for key in prefix.keys),
            tuple(value.detach().clone() for value in prefix.values),
        )

    @staticmethod
    def _same_tensor_layout(left: Tensor, right: Tensor) -> bool:
        return left.shape == right.shape and left.device == right.device and left.dtype == right.dtype

    def _matches(self, noise: Tensor, prefix: SmolExpertPrefixKVCache, num_steps: int) -> bool:
        if self._static_noise is None or self._static_prefix is None or self._loop is None:
            return False
        decoder_parameter = next(self.decoder.parameters())
        if decoder_parameter.device != self._decoder_device or decoder_parameter.dtype != self._decoder_dtype:
            return False
        if not self._same_tensor_layout(noise, self._static_noise):
            return False
        if len(prefix.keys) != len(self._static_prefix.keys):
            return False
        return (
            all(
                self._same_tensor_layout(source, target)
                and self._same_tensor_layout(source_value, target_value)
                for source, target, source_value, target_value in zip(
                    prefix.keys,
                    self._static_prefix.keys,
                    prefix.values,
                    self._static_prefix.values,
                    strict=True,
                )
            )
            and self._loop.num_steps == num_steps
        )

    def _capture(self, noise: Tensor, prefix: SmolExpertPrefixKVCache, num_steps: int) -> None:
        static_noise = noise.detach().clone()
        static_prefix = self._clone_prefix(prefix)
        loop = _ExpertDenoiseLoop(self.decoder, static_prefix, num_steps)
        stream = torch.cuda.Stream(device=noise.device)
        started = time.perf_counter()
        with torch.cuda.device(noise.device):
            current_stream = torch.cuda.current_stream()
            stream.wait_stream(current_stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                for _ in range(3):
                    loop(static_noise)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    static_output = loop(static_noise)
            stream.synchronize()
        torch.cuda.synchronize(noise.device)
        if static_output is None:
            raise RuntimeError("CUDA graph capture produced no expert output")
        self._graph = graph
        self._loop = loop
        self._static_noise = static_noise
        self._static_prefix = static_prefix
        self._static_output = static_output
        decoder_parameter = next(self.decoder.parameters())
        self._decoder_device = decoder_parameter.device
        self._decoder_dtype = decoder_parameter.dtype
        self.capture_seconds = time.perf_counter() - started
        self.capture_count += 1

    def run(self, noise: Tensor, prefix: SmolExpertPrefixKVCache, num_steps: int) -> Tensor:
        """Replay after copying fresh noise and per-layer prefix K/V buffers."""
        with self._lock:
            if not self._matches(noise, prefix, num_steps):
                self._capture(noise, prefix, num_steps)
            if self._graph is None or self._static_noise is None or self._static_prefix is None:
                raise RuntimeError("CUDA graph runner did not initialize")
            self._static_noise.copy_(noise)
            for source_key, target_key, source_value, target_value in zip(
                prefix.keys, self._static_prefix.keys, prefix.values, self._static_prefix.values, strict=True
            ):
                target_key.copy_(source_key)
                target_value.copy_(source_value)
            self._graph.replay()
            if self._static_output is None:
                raise RuntimeError("CUDA graph replay produced no expert output")
            # The graph owns this storage and mutates it on the next replay.
            return self._static_output.clone()


class CosmosPrefixAdapter(nn.Module):
    """Map arbitrary cached backbone channels to SmolVLA's VLM prefix width."""

    def __init__(self, input_channels: int = 2048, prefix_hidden_size: int = 960) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.prefix_hidden_size = prefix_hidden_size
        self.norm = nn.LayerNorm(input_channels)
        self.projection = nn.Linear(input_channels, prefix_hidden_size)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 3 or context.shape[-1] != self.input_channels:
            raise ValueError(
                f"cached context must have shape [B, tokens, {self.input_channels}], "
                f"got {tuple(context.shape)}"
            )
        # Validation runs outside autocast, so normalize the cached feature dtype at
        # this module boundary before LayerNorm/Linear consume it. Training receives
        # the same effective dtype through autocast, while checkpoint-loaded expert
        # parameters retain their pretrained dtype.
        context = context.to(dtype=self.norm.weight.dtype)
        return self.projection(self.norm(context))


def _build_pretrained_expert(
    *,
    vlm_config_name: str,
    device: torch.device | str,
) -> tuple[nn.Module, dict[str, int]]:
    """Build only the expert architecture from the VLM config, without VLM weights."""
    from transformers import AutoConfig, AutoModel

    vlm_config = AutoConfig.from_pretrained(vlm_config_name)
    text_config = copy.deepcopy(vlm_config.text_config)
    prefix_hidden_size = int(text_config.hidden_size)
    expert_hidden_size = int(prefix_hidden_size * EXPERT_WIDTH_MULTIPLIER)
    text_config.hidden_size = expert_hidden_size
    text_config.intermediate_size = get_intermediate_size(expert_hidden_size)
    text_config.num_hidden_layers = NUM_EXPERT_LAYERS
    text_config.dtype = getattr(text_config, "dtype", None) or torch.bfloat16
    expert = AutoModel.from_config(text_config).to(device=device)
    expert.embed_tokens = None
    kv_dim = int(text_config.num_key_value_heads * text_config.head_dim)
    for layer_index, layer in enumerate(expert.layers):
        if layer_index % SELF_ATTN_EVERY_N_LAYERS == 0:
            continue
        layer.self_attn.k_proj = nn.Linear(kv_dim, kv_dim, bias=bool(text_config.attention_bias)).to(
            device=device
        )
        layer.self_attn.v_proj = nn.Linear(kv_dim, kv_dim, bias=bool(text_config.attention_bias)).to(
            device=device
        )
    return expert, {
        "prefix_hidden_size": prefix_hidden_size,
        "expert_hidden_size": expert_hidden_size,
        "kv_dim": kv_dim,
        "num_attention_heads": int(text_config.num_attention_heads),
        "num_key_value_heads": int(text_config.num_key_value_heads),
        "head_dim": int(text_config.head_dim),
    }


class SmolExpertActionDecoder(nn.Module):
    """SmolVLA's pretrained action expert conditioned on Cosmos prefix K/V."""

    def __init__(
        self,
        normalizer: SmolVLANormalizer,
        *,
        expert: nn.Module,
        prefix_hidden_size: int,
        expert_hidden_size: int,
        kv_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        self_attn_every_n_layers: int = SELF_ATTN_EVERY_N_LAYERS,
        num_steps: int = 10,
        input_channels: int = 2048,
        max_action_dim: int = MAX_ACTION_DIM,
        max_state_dim: int = MAX_STATE_DIM,
        min_period: float = 4e-3,
        max_period: float = 4.0,
        rope_theta: float = 100_000.0,
    ) -> None:
        super().__init__()
        if num_steps <= 0 or max_action_dim < ACTION_DIM or max_state_dim < ACTION_DIM:
            raise ValueError("SmolVLA action dimensions and num_steps are invalid")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("attention head count must be divisible by key/value head count")
        if kv_dim != num_key_value_heads * head_dim:
            raise ValueError("kv_dim must equal num_key_value_heads * head_dim")
        self.normalizer = normalizer
        self.expert = expert
        self.context_adapter = CosmosPrefixAdapter(input_channels, prefix_hidden_size)
        self.prefix_key_projection = nn.Linear(prefix_hidden_size, kv_dim)
        self.prefix_value_projection = nn.Linear(prefix_hidden_size, kv_dim)
        self.state_proj = nn.Linear(max_state_dim, prefix_hidden_size)
        self.action_in_proj = nn.Linear(max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)
        self.prefix_hidden_size = prefix_hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.kv_dim = kv_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.num_steps = num_steps
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.min_period = min_period
        self.max_period = max_period
        self.rope_theta = rope_theta
        self.loaded_checkpoint_digests: dict[str, str] = {}
        self._cuda_graph_runner: _ExpertCudaGraphRunner | None = None
        self._cuda_graph_disabled = False
        self._cuda_graph_warning_emitted = False
        self.last_cuda_graph_capture_seconds: float | None = None
        self.cuda_graph_capture_count = 0
        self._freeze_pretrained_projections()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path = SMOLVLA_CHECKPOINT,
        *,
        normalizer: SmolVLANormalizer,
        vlm_config_name: str = SMOLVLM_CONFIG,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> SmolExpertActionDecoder:
        """Load only expert/action-head tensors from a SmolVLA checkpoint."""
        # Load safetensors into CPU modules first; assign=True preserves checkpoint dtypes, then
        # move the complete decoder to the requested accelerator.
        expert, dimensions = _build_pretrained_expert(vlm_config_name=vlm_config_name, device="cpu")
        decoder = cls(normalizer, expert=expert, **dimensions, **kwargs)
        decoder.load_pretrained_checkpoint(checkpoint)
        return decoder.to(device=device)

    @classmethod
    def from_training_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        normalizer: SmolVLANormalizer,
        expert_checkpoint: str | Path = SMOLVLA_CHECKPOINT,
        vlm_config_name: str = SMOLVLM_CONFIG,
        device: torch.device | str = "cpu",
        **kwargs: Any,
    ) -> SmolExpertActionDecoder:
        """Construct the trainer's decoder and load its custom training artifact."""
        # The custom artifact is read on CPU before the complete decoder is moved to its
        # requested device; assign=True preserves the trainer's mixed parameter dtypes.
        decoder = cls.from_pretrained(
            expert_checkpoint,
            normalizer=normalizer,
            vlm_config_name=vlm_config_name,
            device="cpu",
            **kwargs,
        )
        decoder.load_training_checkpoint(checkpoint)
        return decoder.to(device=device)

    def _freeze_pretrained_projections(self) -> None:
        # SmolVLA's state/action projection conventions are reused verbatim. The experiment trains
        # only the pretrained expert and the Cosmos prefix adapter/KV projections.
        for module in (
            self.state_proj,
            self.action_in_proj,
            self.action_out_proj,
            self.action_time_mlp_in,
            self.action_time_mlp_out,
        ):
            module.requires_grad_(False)
        self.expert.requires_grad_(True)
        self.context_adapter.requires_grad_(True)
        self.prefix_key_projection.requires_grad_(True)
        self.prefix_value_projection.requires_grad_(True)

    def load_pretrained_checkpoint(self, checkpoint: str | Path) -> dict[str, str]:
        """Strictly load the expert and SmolVLA action/state projections."""
        path = resolve_checkpoint(checkpoint)
        prefixes = (
            (self.expert, "model.vlm_with_expert.lm_expert."),
            (self.state_proj, "model.state_proj."),
            (self.action_in_proj, "model.action_in_proj."),
            (self.action_out_proj, "model.action_out_proj."),
            (self.action_time_mlp_in, "model.action_time_mlp_in."),
            (self.action_time_mlp_out, "model.action_time_mlp_out."),
        )
        digests: dict[str, str] = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for module, prefix in prefixes:
                digests.update(_load_module_from_checkpoint(module, handle, prefix))
        self.loaded_checkpoint_digests = digests
        self._freeze_pretrained_projections()
        return dict(digests)

    def load_training_checkpoint(self, checkpoint: str | Path) -> dict[str, str]:
        """Strictly load a checkpoint emitted by ``train_smolexpert_on_cosmos.py``."""
        path = resolve_checkpoint(checkpoint)
        expected = self.state_dict()
        prefix = "model."
        expected_keys = {prefix + name for name in expected}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            if missing or unexpected:
                raise ValueError(
                    "custom SmolExpert training checkpoint tensor keys mismatch: "
                    f"missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            loaded: dict[str, Tensor] = {}
            digests: dict[str, str] = {}
            for name, expected_tensor in expected.items():
                key = prefix + name
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(expected_tensor.shape):
                    raise ValueError(
                        f"checkpoint tensor {key} has shape {tuple(tensor.shape)}, "
                        f"expected {tuple(expected_tensor.shape)}"
                    )
                if tensor.dtype != expected_tensor.dtype:
                    raise ValueError(
                        f"checkpoint tensor {key} has dtype {tensor.dtype}, expected {expected_tensor.dtype}"
                    )
                loaded[name] = tensor
                digests[key] = _tensor_digest(tensor)
            self.load_state_dict(loaded, strict=True, assign=True)
        self.loaded_checkpoint_digests = {**self.loaded_checkpoint_digests, **digests}
        self._freeze_pretrained_projections()
        return dict(digests)

    def _pad_action_or_state(self, value: Tensor, width: int) -> Tensor:
        if value.shape[-1] > width:
            raise ValueError(f"last dimension {value.shape[-1]} exceeds configured width {width}")
        if value.shape[-1] == width:
            return value
        return torch.nn.functional.pad(value, (0, width - value.shape[-1]))

    def _prepare_prefix(self, state: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        if state.ndim == 3:
            state = state[:, -1, :]
        if state.ndim != 2 or state.shape[-1] != ACTION_DIM:
            raise ValueError(f"state must have shape [B, 6] or [B, 1, 6], got {tuple(state.shape)}")
        context_hidden = self.context_adapter(context)
        state_input = self._pad_action_or_state(self.normalizer.normalize_state(state), self.max_state_dim)
        state_hidden = self.state_proj(state_input.to(dtype=self.state_proj.weight.dtype))[:, None, :]
        prefix_hidden = torch.cat([context_hidden, state_hidden], dim=1)
        prefix_key = self.prefix_key_projection(
            prefix_hidden.to(dtype=self.prefix_key_projection.weight.dtype)
        )
        prefix_value = self.prefix_value_projection(
            prefix_hidden.to(dtype=self.prefix_value_projection.weight.dtype)
        )
        batch_size, prefix_length = prefix_key.shape[:2]
        positions = torch.arange(prefix_length, device=prefix_key.device, dtype=torch.long)[None].expand(
            batch_size, -1
        )
        prefix_key = apply_rope(
            prefix_key.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim),
            positions,
            max_wavelength=self.rope_theta,
        )
        return prefix_key, prefix_value.reshape(
            batch_size, prefix_length, self.num_key_value_heads, self.head_dim
        )

    @torch.no_grad()
    def prepare_prefix_kv(self, state: Tensor, context: Tensor) -> SmolExpertPrefixKVCache:
        """Prepare exact per-layer prefix K/V tensors for reuse across denoising steps.

        The uncached path projects the prefix once for the self-attention layers and
        reprojects it through each cross-attention layer's K/V projections on every
        vector-field evaluation.  This method performs those same operations once,
        without changing dtypes, RoPE positions, or tensor values used by the
        decoder.  The returned cache is valid only for this state/context pair.
        """
        prefix_key, prefix_value = self._prepare_prefix(state, context)
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for layer_index, layer in enumerate(self.expert.layers):
            is_self_attention = (
                self.self_attn_every_n_layers > 0 and layer_index % self.self_attn_every_n_layers == 0
            )
            if is_self_attention:
                key, value = prefix_key, prefix_value
            else:
                batch_size, prefix_length = prefix_key.shape[:2]
                key_input = prefix_key.reshape(batch_size, prefix_length, self.kv_dim)
                value_input = prefix_value.reshape(batch_size, prefix_length, self.kv_dim)
                key = layer.self_attn.k_proj(key_input.to(dtype=layer.self_attn.k_proj.weight.dtype))
                value = layer.self_attn.v_proj(value_input.to(dtype=layer.self_attn.v_proj.weight.dtype))
                key = key.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim)
                value = value.reshape(batch_size, prefix_length, self.num_key_value_heads, self.head_dim)
            keys.append(key)
            values.append(value)
        return SmolExpertPrefixKVCache(tuple(keys), tuple(values))

    def _attention(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=2)
        value = value.repeat_interleave(groups, dim=2)
        scores = torch.matmul(query.float().transpose(1, 2), key.float().transpose(1, 2).transpose(2, 3))
        scores = scores * (self.head_dim**-0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None], torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
        output = torch.matmul(probabilities, value.transpose(1, 2))
        return output.transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)

    def _vector_field(
        self,
        prefix: tuple[Tensor, Tensor] | SmolExpertPrefixKVCache,
        noisy_action: Tensor,
        time: Tensor,
    ) -> Tensor:
        cached_prefix = isinstance(prefix, SmolExpertPrefixKVCache)
        if not cached_prefix:
            prefix_key, prefix_value = prefix
        action_emb = self.action_in_proj(noisy_action.to(dtype=self.action_in_proj.weight.dtype))
        time_emb = create_sinusoidal_pos_embedding(
            time,
            self.expert_hidden_size,
            self.min_period,
            self.max_period,
            device=noisy_action.device,
        ).to(dtype=action_emb.dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time = self.action_time_mlp_in(torch.cat([action_emb, time_emb], dim=-1))
        action_time = torch.nn.functional.silu(action_time)
        hidden = self.action_time_mlp_out(action_time)
        action_length = hidden.shape[1]
        batch_size = hidden.shape[0]
        for layer_index, layer in enumerate(self.expert.layers):
            if cached_prefix:
                prefix_key = prefix.keys[layer_index]
                prefix_value = prefix.values[layer_index]
            # The SmolVLA action heads are checkpointed in float32 while the
            # transformer is bfloat16. Autocast reconciles this during training,
            # but validation/sampling runs without autocast; make each expert
            # submodule boundary explicit instead of relying on ambient autocast.
            normalized = layer.input_layernorm(hidden.to(dtype=layer.input_layernorm.weight.dtype))
            query = layer.self_attn.q_proj(normalized.to(dtype=layer.self_attn.q_proj.weight.dtype))
            query = query.reshape(batch_size, action_length, self.num_attention_heads, self.head_dim)
            if self.self_attn_every_n_layers > 0 and layer_index % self.self_attn_every_n_layers == 0:
                action_key = layer.self_attn.k_proj(normalized.to(dtype=layer.self_attn.k_proj.weight.dtype))
                action_value = layer.self_attn.v_proj(
                    normalized.to(dtype=layer.self_attn.v_proj.weight.dtype)
                )
                action_key = action_key.reshape(
                    batch_size, action_length, self.num_key_value_heads, self.head_dim
                )
                action_value = action_value.reshape(
                    batch_size, action_length, self.num_key_value_heads, self.head_dim
                )
                prefix_length = prefix_key.shape[1]
                key = torch.cat([prefix_key, action_key], dim=1)
                value = torch.cat([prefix_value, action_value], dim=1)
                action_causal_mask = torch.cat(
                    [
                        torch.ones(action_length, prefix_length, dtype=torch.bool, device=hidden.device),
                        torch.tril(
                            torch.ones(action_length, action_length, dtype=torch.bool, device=hidden.device)
                        ),
                    ],
                    dim=1,
                )[None].expand(batch_size, -1, -1)
                query_positions = torch.arange(
                    prefix_key.shape[1],
                    prefix_key.shape[1] + action_length,
                    device=hidden.device,
                    dtype=torch.long,
                )[None].expand(batch_size, -1)
            else:
                if cached_prefix:
                    key, value = prefix_key, prefix_value
                else:
                    key_input = prefix_key.reshape(batch_size, prefix_key.shape[1], self.kv_dim)
                    value_input = prefix_value.reshape(batch_size, prefix_value.shape[1], self.kv_dim)
                    key = layer.self_attn.k_proj(key_input.to(dtype=layer.self_attn.k_proj.weight.dtype))
                    value = layer.self_attn.v_proj(value_input.to(dtype=layer.self_attn.v_proj.weight.dtype))
                    key = key.reshape(
                        batch_size, prefix_key.shape[1], self.num_key_value_heads, self.head_dim
                    )
                    value = value.reshape(
                        batch_size, prefix_key.shape[1], self.num_key_value_heads, self.head_dim
                    )
                query_positions = torch.arange(action_length, device=hidden.device, dtype=torch.long)[
                    None
                ].expand(batch_size, -1)
                action_causal_mask = None
            query = apply_rope(query, query_positions, max_wavelength=self.rope_theta)
            attention_output = self._attention(query, key, value, action_causal_mask)
            hidden = hidden + layer.self_attn.o_proj(
                attention_output.to(dtype=layer.self_attn.o_proj.weight.dtype)
            )
            residual = hidden
            hidden = layer.post_attention_layernorm(
                hidden.to(dtype=layer.post_attention_layernorm.weight.dtype)
            )
            mlp_dtype = next(layer.mlp.parameters()).dtype
            hidden = layer.mlp(hidden.to(dtype=mlp_dtype))
            hidden = hidden + residual
        hidden = self.expert.norm(hidden)
        return self.action_out_proj(hidden.to(dtype=self.action_out_proj.weight.dtype))

    def flow_matching_loss(
        self,
        state: Tensor,
        action: Tensor,
        context: Tensor,
        *,
        t: Tensor | None = None,
        epsilon: Tensor | None = None,
        generator: torch.Generator | None = None,
        action_is_pad: Tensor | None = None,
        obs_dropout: float = 0.0,
    ) -> Tensor:
        """Compute SmolVLA's mean/std-normalized flow-matching loss."""
        if action.shape[-2:] != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(f"action must have shape [B, 30, 6], got {tuple(action.shape)}")
        if action_is_pad is not None and (
            action_is_pad.shape != action.shape[:2] or action_is_pad.dtype is not torch.bool
        ):
            raise ValueError("action_is_pad must have shape [B, 30] and boolean dtype")
        if not 0.0 <= float(obs_dropout) <= 1.0 or float(obs_dropout) != 0.0:
            raise ValueError("SmolExpertActionDecoder does not support nonzero observation dropout")
        batch_size = action.shape[0]
        action_normalized = self.normalizer.normalize_action(action)
        action_normalized = self._pad_action_or_state(action_normalized, self.max_action_dim)
        if epsilon is None:
            epsilon = torch.randn(
                action_normalized.shape,
                device=action.device,
                dtype=action_normalized.dtype,
                generator=generator,
            )
        elif epsilon.shape[-2:] == (ACTION_HORIZON, ACTION_DIM):
            epsilon = self._pad_action_or_state(epsilon, self.max_action_dim)
        elif epsilon.shape != action_normalized.shape:
            raise ValueError("epsilon must have shape [B, 30, 6] or [B, 30, 32]")
        if t is None:
            t = sample_time_beta(batch_size, action.device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)
        if t.shape != (batch_size,):
            raise ValueError("t must have shape [B]")
        t = t.to(device=action.device, dtype=torch.float32)
        x_t = (1.0 - t[:, None, None]) * action_normalized + t[:, None, None] * epsilon
        target = epsilon - action_normalized
        prediction = self._vector_field(
            self._prepare_prefix(state, context),
            x_t,
            t,
        )
        squared_error = (prediction.float() - target.float()).square()
        if action_is_pad is None:
            return squared_error.mean()
        valid = (~action_is_pad).to(dtype=squared_error.dtype).unsqueeze(-1)
        valid_count = valid.sum() * self.max_action_dim
        if valid_count.item() <= 0:
            raise ValueError("action_is_pad leaves no valid action tokens")
        return (squared_error * valid).sum() / valid_count

    def _noise_for_seed(self, batch_size: int, device: torch.device, seed: int) -> Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.randn(
            (batch_size, ACTION_HORIZON, self.max_action_dim),
            device=device,
            dtype=self.action_in_proj.weight.dtype,
            generator=generator,
        )

    def _cuda_graph_shapes_supported(
        self,
        state: Tensor,
        context: Tensor,
        noise: Tensor,
        prefix: tuple[Tensor, Tensor] | SmolExpertPrefixKVCache,
        num_steps: int,
    ) -> bool:
        if self._cuda_graph_disabled or not torch.cuda.is_available():
            return False
        if (
            not cuda_graph_shape_eligible(
                tuple(state.shape),
                tuple(context.shape),
                tuple(noise.shape),
                max_action_dim=self.max_action_dim,
                prefix_is_cached=isinstance(prefix, SmolExpertPrefixKVCache),
            )
            or state.device.type != "cuda"
            or context.device != state.device
            or noise.device != state.device
        ):
            return False
        return all(
            key.device == state.device and value.device == state.device
            for key, value in zip(prefix.keys, prefix.values, strict=True)
        )

    def _try_cuda_graph(
        self, prefix: SmolExpertPrefixKVCache, noise: Tensor, num_steps: int
    ) -> Tensor | None:
        if self._cuda_graph_runner is None:
            self._cuda_graph_runner = _ExpertCudaGraphRunner(self)
        try:
            output = self._cuda_graph_runner.run(noise, prefix, num_steps)
        except Exception as exc:  # noqa: BLE001 - inference must fall back safely
            self._cuda_graph_disabled = True
            if not self._cuda_graph_warning_emitted:
                warnings.warn(
                    "SmolExpert CUDA graph capture/replay failed; falling back to eager KV-cache "
                    f"inference: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._cuda_graph_warning_emitted = True
            return None
        self.last_cuda_graph_capture_seconds = self._cuda_graph_runner.capture_seconds
        self.cuda_graph_capture_count = self._cuda_graph_runner.capture_count
        return output

    @torch.no_grad()
    def sample_actions(
        self,
        state: Tensor,
        context: Tensor,
        *,
        seed: int | None = None,
        noise: Tensor | None = None,
        prefix_kv_cache: SmolExpertPrefixKVCache | None = None,
        use_prefix_kv_cache: bool = True,
        num_steps: int | None = None,
        use_cuda_graph: bool | None = None,
        rtc_processor: Any | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        """Sample a 30x6 raw action chunk.

        Inference prepares exact per-layer prefix K/V tensors once per call and
        reuses them across all denoising steps by default. Set
        ``use_prefix_kv_cache=False`` to run the uncached path for debugging or
        comparison. An explicitly supplied ``prefix_kv_cache`` is reused as-is
        and must have been prepared for the same state and context.
        ``num_steps`` overrides the configured default for a measurement without
        mutating the decoder. ``use_cuda_graph=None`` enables the serialized fixed-shape
        CUDA graph automatically for batch-one production inputs; pass ``False`` to opt
        out explicitly. Graph outputs are cloned before return, and fresh noise plus all
        per-layer prefix K/V tensors are copied before every replay.
        """
        device = state.device
        batch_size = state.shape[0]
        if num_steps is not None and num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if noise is None:
            noise = self._noise_for_seed(batch_size, device, 0 if seed is None else seed)
        if noise.shape != (batch_size, ACTION_HORIZON, self.max_action_dim):
            raise ValueError(f"noise must have shape [B, 30, {self.max_action_dim}]")
        if prefix_kv_cache is not None and not use_prefix_kv_cache:
            raise ValueError("prefix_kv_cache cannot be supplied when use_prefix_kv_cache=False")
        state_raw = state[:, -1, :] if state.ndim == 3 else state
        if prefix_kv_cache is not None:
            prefix = prefix_kv_cache
        elif use_prefix_kv_cache:
            # Keep the cache local to this inference call so a subsequent
            # observation/context batch always prepares a fresh prefix.
            prefix = self.prepare_prefix_kv(state_raw, context)
        else:
            prefix = self._prepare_prefix(state_raw, context)

        resolved_steps = self.num_steps if num_steps is None else num_steps
        rtc_enabled = rtc_processor is not None and rtc_processor.rtc_config.enabled
        if prev_chunk_left_over is not None:
            previous = prev_chunk_left_over
            if previous.ndim == 2:
                previous = previous.unsqueeze(0)
            if previous.ndim != 3 or previous.shape[0] != batch_size or previous.shape[-1] != ACTION_DIM:
                raise ValueError("prev_chunk_left_over must have shape [T, 6] or [B, T, 6]")
            previous = self.normalizer.normalize_action(previous.to(device=device, dtype=torch.float32))
            prev_chunk_left_over = self._pad_action_or_state(previous, self.max_action_dim).to(
                dtype=noise.dtype
            )
        graph_enabled = (
            not rtc_enabled
            and (
                use_cuda_graph is True
                or (use_cuda_graph is None and isinstance(prefix, SmolExpertPrefixKVCache))
            )
            and self._cuda_graph_shapes_supported(state, context, noise, prefix, resolved_steps)
        )
        if graph_enabled:
            graph_output = self._try_cuda_graph(prefix, noise, resolved_steps)
            if graph_output is not None:
                return graph_output

        def denoise(input_x_t: Tensor, current_time: Tensor) -> Tensor:
            return self._vector_field(prefix, input_x_t, current_time)

        sampled = euler_integrate(
            denoise,
            noise,
            resolved_steps,
            rtc_processor=rtc_processor,
            rtc_enabled=rtc_enabled,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )
        return self.normalizer.denormalize_action(sampled[..., :ACTION_DIM]).to(dtype=torch.float32)
