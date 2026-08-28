"""Frozen Cosmos-Predict2 video representation extraction for LeRobot.

The official minimal_a2a backend is imported only when a real extractor is
constructed without injected components. Importing LeRobot never imports Cosmos
or initializes distributed state.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

UPSTREAM_REPOSITORY = "https://github.com/mimic-video/mimic-video"
UPSTREAM_COMMIT = "e3355dbc93132b576c02f920a59b4fc18a4f5906"
BACKEND_NAME = "minimal_a2a"
# These are the backend strings accepted by the pinned vendored Cosmos Attention
# module. ``flash_attn_no_cp`` appears in an older branch of the vendored code,
# but the current video DiT rejects it during construction and is not exposed.
COSMOS_ATTENTION_BACKENDS = ("minimal_a2a", "torch", "transformer_engine")
COMPILE_FRIENDLY_ATTENTION_BACKEND = "torch"
VAE_INPUT_MODE_OBSERVED_PREFIX = "observed_prefix"
VAE_INPUT_MODE_LEGACY_PADDED = "legacy_padded_vae"
VAE_INPUT_MODES = (VAE_INPUT_MODE_OBSERVED_PREFIX, VAE_INPUT_MODE_LEGACY_PADDED)


class CosmosPredict2Error(RuntimeError):
    """Base error for the extractor contract."""


class CosmosPredict2BackendError(CosmosPredict2Error):
    """The official optional runtime cannot be used."""


class CosmosPredict2CheckpointError(CosmosPredict2Error):
    """A checkpoint cannot be loaded with an exact key match."""


_TE_EXTRA_STATE_KEY = re.compile(r"^blocks\.\d+\.cross_attn\.attn_op\._extra_state$")
_OFFICIAL_GENERIC_CHECKPOINT_NAME = "v2w_pretrained_cosmos.pt"


@dataclass(frozen=True, slots=True)
class CosmosPredict2ExtractorConfig:
    """Explicit runtime settings; no Hydra or external config objects are stored."""

    checkpoint_path: str | Path
    tokenizer_path: str | Path
    device: str = "cuda"
    dtype: str | torch.dtype = "bfloat16"
    hidden_layer: int = 20
    stop_after_step: int = 0
    high_noise_sigma: float = 10.0
    seed: int = 0
    backend: str = BACKEND_NAME
    attention_backend: str = BACKEND_NAME
    compile_friendly: bool = False
    torch_compile: bool = False
    compile_mode: str = "max-autotune"
    use_cuda_graphs: bool = False
    sac_mode: str = "predict2_2b_720"
    vae_input_mode: str = VAE_INPUT_MODE_OBSERVED_PREFIX
    input_frames: int = 5
    video_frames: int = 61
    input_height: int = 480
    input_width: int = 640
    latent_channels: int = 16
    latent_conditional_frames: int = 2
    state_t: int = 16
    fp8_linear: bool = False
    latent_height: int = 60
    latent_width: int = 80
    token_height: int = 30
    token_width: int = 40
    hidden_dim: int = 2048
    checkpoint_prefix: str | None = None
    random_init_seed: int | None = None
    sigma_data: float = 1.0
    sigma_conditional: float = 0.0001

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "tokenizer_path", Path(self.tokenizer_path))
        if self.random_init_seed is not None and self.random_init_seed < 0:
            raise ValueError("random_init_seed must be non-negative")
        if self.hidden_layer < 0:
            raise ValueError("hidden_layer must be non-negative")
        if self.stop_after_step != 0:
            raise ValueError("Only stop_after_step=0 (the first high-noise forward) is implemented")
        if self.high_noise_sigma <= 0:
            raise ValueError("high_noise_sigma must be positive")
        if self.backend != BACKEND_NAME:
            raise ValueError(f"Only backend={BACKEND_NAME!r} is supported")
        if self.attention_backend not in COSMOS_ATTENTION_BACKENDS:
            raise ValueError(
                "attention_backend must be one of "
                + ", ".join(repr(value) for value in COSMOS_ATTENTION_BACKENDS)
            )
        if not isinstance(self.compile_friendly, bool):
            raise ValueError("compile_friendly must be a boolean")
        if not isinstance(self.torch_compile, bool):
            raise ValueError("torch_compile must be a boolean")
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("compile_mode must be a torch.compile mode")
        if self.torch_compile:
            object.__setattr__(self, "compile_friendly", True)
        if self.torch_compile and self.fp8_linear:
            raise ValueError("torch_compile is incompatible with fp8_linear")
        if not isinstance(self.use_cuda_graphs, bool):
            raise ValueError("use_cuda_graphs must be a boolean")
        if not isinstance(self.fp8_linear, bool):
            raise ValueError("fp8_linear must be a boolean")
        if self.vae_input_mode not in VAE_INPUT_MODES:
            raise ValueError(
                "vae_input_mode must be one of " + ", ".join(repr(value) for value in VAE_INPUT_MODES)
            )
        if self.sac_mode not in {"none", "predict2_2b_720"}:
            raise ValueError("sac_mode must be 'none' or 'predict2_2b_720'")
        if (
            self.input_frames not in (1, 5)
            or self.video_frames != 61
            or (self.input_height, self.input_width) != (480, 640)
        ):
            raise ValueError("The frozen extractor observed input must be T=1 or T=5 at 480x640")
        if self.latent_conditional_frames != 2:
            raise ValueError("The frozen extractor requires exactly two conditional latent frames")
        if self.state_t not in (2, 16):
            raise ValueError("state_t must be 2 (observed-only) or 16 (observed prefix plus future)")
        if self.state_t == 2 and self.input_frames != 5:
            raise ValueError("state_t=2 observed-only extraction requires input_frames=5")
        if self.sigma_data != 1.0 or self.sigma_conditional <= 0:
            raise ValueError("The frozen extractor requires sigma_data=1 and positive sigma_conditional")


@dataclass(frozen=True, slots=True)
class CosmosPredict2Provenance:
    repository: str
    upstream_commit: str
    checkpoint_path: str
    tokenizer_path: str
    backend: str
    high_noise_sigma: float
    stop_after_step: int
    checkpoint_ignored_metadata_keys: tuple[str, ...]
    checkpoint_ignored_metadata_count: int
    noise_seed: int
    random_init_seed: int | None = None
    attention_backend: str = BACKEND_NAME
    compile_friendly: bool = False
    torch_compile: bool = False
    compile_mode: str = "max-autotune"
    use_cuda_graphs: bool = False
    vae_input_mode: str = VAE_INPUT_MODE_OBSERVED_PREFIX
    state_t: int = 16
    fp8_linear: bool = False


@dataclass(frozen=True, slots=True)
class CosmosPredict2Extraction:
    """Structured frozen-backbone representation."""

    hidden_grid: torch.Tensor
    tokens: torch.Tensor
    sigma: torch.Tensor
    grid_shape: tuple[int, int, int]
    layer: int
    provenance: CosmosPredict2Provenance


_DTYPE_NAMES = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _backbone_autocast_context(device: torch.device, dtype: torch.dtype):
    """Match official CUDA mixed-precision execution around the backbone only."""
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def arch_invariant_rand(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    # Match upstream misc.arch_invariant_rand independent of torch RNG/device.
    return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape).astype(np.float32))


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_NAMES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported extractor dtype {dtype!r}") from exc


def _freeze_module(module: Any) -> None:
    candidates = [module]
    nested = getattr(module, "model", None)
    if nested is not None:
        candidates.append(nested)
        nested_model = getattr(nested, "model", None)
        if nested_model is not None:
            candidates.append(nested_model)
    for candidate in candidates:
        if isinstance(candidate, nn.Module):
            candidate.eval()
            candidate.requires_grad_(False)


def _load_fp8_linear_components() -> tuple[Any, Any]:
    """Load the optional TE Linear and delayed-scaling recipe lazily."""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format

        if not callable(getattr(te, "Linear", None)):
            raise ImportError("transformer_engine.pytorch.Linear is unavailable")
        if not callable(getattr(te, "autocast", None)):
            raise ImportError("transformer_engine.pytorch.autocast is unavailable")
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    except Exception as exc:  # noqa: BLE001 - optional CUDA dependency is machine-specific
        raise CosmosPredict2BackendError(
            "fp8_linear=True requires Transformer Engine with pytorch.Linear and "
            f"autocast; failed to initialize it: {exc}"
        ) from exc
    return te, recipe


def _replace_linear_modules(module: nn.Module, te: Any) -> int:
    """Replace every ordinary Linear child with an equivalent TE Linear."""
    replacements = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = te.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                params_dtype=child.weight.dtype,
                device=child.weight.device,
            )
            with torch.no_grad():
                replacement.weight.copy_(child.weight)
                if child.bias is not None:
                    replacement.bias.copy_(child.bias)
            setattr(module, name, replacement)
            replacements += 1
        else:
            replacements += _replace_linear_modules(child, te)
    return replacements


@torch.library.custom_op("lerobot::cosmos_rms_norm", mutates_args=())
def _cosmos_rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Keep the native RMSNorm kernel opaque to Inductor while retaining eager semantics."""
    return torch.rms_norm(input, list(weight.shape), weight, eps)


@_cosmos_rms_norm.register_fake
def _cosmos_rms_norm_fake(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    del weight, eps
    return torch.empty_like(input)


class _CompileFriendlyRMSNorm(nn.RMSNorm):
    """Native RMSNorm with TE's inference output-dtype contract."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return _cosmos_rms_norm(input, self.weight, self.eps).to(dtype=self.weight.dtype)


def _pure_torch_rotary_pos_emb(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "sbhd",
    start_positions: torch.Tensor | None = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """Apply the inference-only bshd RoPE case without a TE custom op.

    Cosmos' video DiT calls fused RoPE with ``tensor_format="bshd"``, no
    offsets, and no context parallelism. Keeping that fixed contract explicit
    makes this replacement both compile-friendly and safer than a broad monkey
    patch of Transformer Engine.
    """
    del fused, cu_seqlens, cp_rank
    if tensor_format != "bshd" or start_positions is not None or cp_size != 1:
        raise ValueError("compile-friendly RoPE only supports bshd without offsets or context parallelism")
    freqs = freqs[: tensor.shape[1]].transpose(0, 1)
    # Keep the trig values and multiply-accumulate in FP32.  TE's fused RoPE
    # does the same before rounding back to the input dtype; casting cos/sin to
    # BF16 first measurably compounds drift across 20 transformer blocks.
    cos = freqs.cos()
    sin = freqs.sin()
    rot_dim = freqs.shape[-1]
    original = tensor[..., :rot_dim]
    if interleaved:
        even = original[..., ::2]
        odd = original[..., 1::2]
        half = torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)
    else:
        first, second = original.chunk(2, dim=-1)
        half = torch.cat((-second, first), dim=-1)
    rotated = original.float() * cos + half.float() * sin
    return torch.cat((rotated, tensor[..., rot_dim:].float()), dim=-1).to(dtype=tensor.dtype)


def make_compile_friendly_backbone(backbone: nn.Module) -> dict[str, int]:
    """Replace inference-time TE normalization and fused RoPE with torch ops.

    Linear layers are already ordinary ``torch.nn.Linear`` modules in the
    vendored model. Checkpoint loading happens before this function, so the
    replacement preserves every learned weight while removing the remaining TE
    custom operators from the executed forward.
    """
    rmsnorm_replacements = 0
    rotary_replacements = 0
    for parent in backbone.modules():
        for name, child in list(parent.named_children()):
            module_name = child.__class__.__module__
            if child.__class__.__name__ != "RMSNorm" or not module_name.startswith("transformer_engine"):
                continue
            normalized_shape = getattr(child, "normalized_shape", child.weight.shape)
            replacement = _CompileFriendlyRMSNorm(
                normalized_shape,
                eps=float(getattr(child, "eps", 1e-5)),
                elementwise_affine=True,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            replacement.weight = child.weight
            setattr(parent, name, replacement)
            rmsnorm_replacements += 1
    for module in backbone.modules():
        if hasattr(module, "_rotary_pos_emb") and callable(module._rotary_pos_emb):
            module._rotary_pos_emb = _pure_torch_rotary_pos_emb
            rotary_replacements += 1
    return {
        "rmsnorm_replacements": rmsnorm_replacements,
        "rotary_replacements": rotary_replacements,
    }


def _as_tensor_mapping(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "net"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                try:
                    return _as_tensor_mapping(nested)
                except CosmosPredict2CheckpointError:
                    pass
        if all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
            return dict(payload)
    raise CosmosPredict2CheckpointError("Checkpoint must contain a flat tensor state dict")


def _strip_checkpoint_prefix(state: dict[str, torch.Tensor], prefix: str | None) -> dict[str, torch.Tensor]:
    if prefix is None:
        candidates = [
            candidate
            for candidate in ("net.", "net_ema.")
            if state and all(key.startswith(candidate) for key in state)
        ]
        if len(candidates) > 1:
            raise CosmosPredict2CheckpointError(f"Ambiguous automatic checkpoint prefixes: {candidates}")
        prefix = candidates[0] if candidates else None
    if not prefix:
        return state
    missing_prefix = [key for key in state if not key.startswith(prefix)]
    if missing_prefix:
        raise CosmosPredict2CheckpointError(
            f"Configured checkpoint prefix {prefix!r} is missing from keys, e.g. {missing_prefix[0]!r}"
        )
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        stripped_key = key[len(prefix) :]
        if stripped_key in stripped:
            raise CosmosPredict2CheckpointError(f"Checkpoint prefix creates duplicate key {stripped_key!r}")
        stripped[stripped_key] = value
    return stripped


def _validate_te_extra_state(key: str, value: torch.Tensor) -> None:
    """Accept only inert serialized TE metadata bytes, not arbitrary objects."""
    if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1:
        raise CosmosPredict2CheckpointError(
            f"Allowed Transformer Engine metadata key {key!r} must contain a 1-D uint8 tensor"
        )


def load_checkpoint_strict(
    module: nn.Module,
    checkpoint_path: Path,
    prefix: str | None = None,
    *,
    allow_official_te_extra_state: bool = False,
) -> tuple[str, ...]:
    """Strictly load weights, allowing only the pinned inert TE metadata exception."""

    allow_missing_te_metadata = (
        Path(checkpoint_path).name != _OFFICIAL_GENERIC_CHECKPOINT_NAME
        and "bridge" in Path(checkpoint_path).name
    )
    if allow_official_te_extra_state and (
        Path(checkpoint_path).name != _OFFICIAL_GENERIC_CHECKPOINT_NAME
        and not allow_missing_te_metadata
        or prefix is not None
    ):
        raise CosmosPredict2CheckpointError(
            "Transformer Engine metadata compatibility is limited to the pinned generic checkpoint "
            "with automatic net-prefix handling"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001
        raise CosmosPredict2CheckpointError(f"Could not read checkpoint {checkpoint_path}: {exc}") from exc
    state = _strip_checkpoint_prefix(_as_tensor_mapping(payload), prefix)
    expected = set(module.state_dict())
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    ignored: list[str] = []
    if allow_official_te_extra_state and not missing:
        ignored = sorted(key for key in unexpected if _TE_EXTRA_STATE_KEY.fullmatch(key))
        for key in ignored:
            _validate_te_extra_state(key, state[key])
    missing_te = sorted(key for key in missing if _TE_EXTRA_STATE_KEY.fullmatch(key))
    if allow_missing_te_metadata:
        for key in missing_te:
            missing.remove(key)
    disallowed = sorted(set(unexpected) - set(ignored))
    if missing or disallowed:
        raise CosmosPredict2CheckpointError(
            "Strict Cosmos checkpoint key mismatch: "
            + json.dumps({"missing": missing, "unexpected": disallowed})
        )
    load_state = {key: value for key, value in state.items() if key not in ignored}
    try:
        result = module.load_state_dict(load_state, strict=False, assign=True)
        unexpected_loaded = sorted(result.unexpected_keys)
        missing_loaded = sorted(
            key
            for key in result.missing_keys
            if not (allow_missing_te_metadata and _TE_EXTRA_STATE_KEY.fullmatch(key))
        )
        if unexpected_loaded or missing_loaded:
            raise RuntimeError(f"missing={missing_loaded}, unexpected={unexpected_loaded}")
    except RuntimeError as exc:
        raise CosmosPredict2CheckpointError(f"Strict Cosmos checkpoint load failed: {exc}") from exc
    return tuple(ignored)


class CosmosPredict2Extractor:
    """Extract one frozen high-noise Video2World hidden representation.

    ``backbone`` and ``tokenizer`` are injectable for CPU contract tests. With
    neither supplied, the official minimal_a2a source closure is loaded lazily.
    """

    def __init__(
        self,
        config: CosmosPredict2ExtractorConfig,
        *,
        backbone: nn.Module | Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.dtype)
        if (backbone is None) != (tokenizer is None):
            raise ValueError("Inject both backbone and tokenizer, or neither")
        official = backbone is None
        if backbone is None:
            backbone, tokenizer, ignored_metadata_keys = self._build_official_components()
        else:
            ignored_metadata_keys = ()
        self._checkpoint_ignored_metadata_keys = tuple(ignored_metadata_keys)
        self.backbone = backbone
        self.tokenizer = tokenizer
        self._fp8_te = None
        self._fp8_recipe = None
        if self.config.fp8_linear:
            self._fp8_te, self._fp8_recipe = _load_fp8_linear_components()
            _replace_linear_modules(self.backbone, self._fp8_te)
        if self.config.compile_friendly:
            make_compile_friendly_backbone(self.backbone)
        _freeze_module(self.backbone)
        _freeze_module(self.tokenizer)
        if self.config.torch_compile and official and self.device.type == "cuda":
            if not hasattr(torch, "compile"):
                raise CosmosPredict2BackendError("torch.compile is unavailable in this PyTorch build")
            self.backbone = torch.compile(
                self.backbone,
                mode=self.config.compile_mode,
                fullgraph=True,
            )

    def _build_official_components(self) -> tuple[nn.Module, Any, tuple[str, ...]]:
        if self.device.type != "cuda":
            raise CosmosPredict2BackendError(
                "The official minimal_a2a Cosmos backend requires CUDA for its selected "
                f"PyTorch attention path; received device={self.device}. CPU tests must inject fakes."
            )
        if not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(f"Cosmos backbone checkpoint not found: {self.config.checkpoint_path}")
        if not self.config.tokenizer_path.is_file():
            raise FileNotFoundError(f"Cosmos tokenizer checkpoint not found: {self.config.tokenizer_path}")
        try:
            import_module("transformer_engine")
        except (ImportError, OSError) as exc:
            raise CosmosPredict2BackendError(
                "Transformer Engine is required for official minimal_a2a normalization and fused "
                "rotary kernels; it is not the selected attention backend."
            ) from exc
        try:
            from ._vendor.cosmos_predict2.models.text2image_dit import SACConfig
            from ._vendor.cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
            from ._vendor.cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
        except (ImportError, OSError) as exc:
            raise CosmosPredict2BackendError(
                "The vendored official Cosmos minimal_a2a source closure could not import its "
                "optional CUDA/runtime dependencies."
            ) from exc

        construction_attention_backend = (
            COMPILE_FRIENDLY_ATTENTION_BACKEND
            if self.config.compile_friendly
            else self.config.attention_backend
        )
        construction_sac_mode = "none" if self.config.compile_friendly else self.config.sac_mode
        with torch.device("meta"):
            backbone = MinimalV1LVGDiT(
                max_img_h=240,
                max_img_w=240,
                max_frames=128,
                in_channels=16,
                out_channels=16,
                patch_spatial=2,
                patch_temporal=1,
                concat_padding_mask=True,
                model_channels=2048,
                num_blocks=28,
                num_heads=16,
                atten_backend=construction_attention_backend,
                pos_emb_cls="rope3d",
                pos_emb_learnable=True,
                pos_emb_interpolation="crop",
                use_adaln_lora=True,
                adaln_lora_dim=256,
                rope_h_extrapolation_ratio=3.0,
                rope_w_extrapolation_ratio=3.0,
                rope_t_extrapolation_ratio=1.0,
                extra_per_block_abs_pos_emb=False,
                rope_enable_fps_modulation=False,
                sac_config=SACConfig(mode=construction_sac_mode, every_n_blocks=1),
            )
        backbone = backbone.to_empty(device=self.device)
        if self.config.random_init_seed is not None:
            torch.manual_seed(self.config.random_init_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.random_init_seed)

            def reset_module(module: nn.Module) -> None:
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()

            backbone.apply(reset_module)
            ignored_metadata_keys = ()
        else:
            allow_te_metadata = (
                self.config.checkpoint_path.name == _OFFICIAL_GENERIC_CHECKPOINT_NAME
                or "bridge" in self.config.checkpoint_path.name
            ) and self.config.checkpoint_prefix is None
            ignored_metadata_keys = load_checkpoint_strict(
                backbone,
                self.config.checkpoint_path,
                self.config.checkpoint_prefix,
                allow_official_te_extra_state=allow_te_metadata,
            )
        backbone = backbone.to(device=self.device, dtype=self.dtype).eval()
        tokenizer = TokenizerInterface(
            chunk_duration=81,
            vae_pth=str(self.config.tokenizer_path),
            device=str(self.device),
            dtype=self.dtype,
            is_amp=False,
            temporal_window=16,
        )
        return backbone, tokenizer, ignored_metadata_keys

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        expected = (3, self.config.input_frames, self.config.input_height, self.config.input_width)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                f"images must have shape [B, 3, T, 480, 640] with T in (1, 5), got {tuple(images.shape)}"
            )
        if images.dtype == torch.uint8:
            images = images.to(device=self.device, dtype=self.dtype) / 127.5 - 1.0
        elif not torch.is_floating_point(images):
            raise TypeError("images must be uint8 or floating point")
        if not torch.isfinite(images).all():
            raise ValueError("images contain non-finite values")
        min_value, max_value = images.amin().item(), images.amax().item()
        if min_value >= 0.0 and max_value <= 1.0:
            images = images * 2.0 - 1.0
        elif min_value >= -1.0 and max_value <= 1.0:
            pass
        else:
            raise ValueError("floating-point images must be in [0, 1] or [-1, 1]")
        images = images.to(device=self.device, dtype=self.dtype)
        if self.config.vae_input_mode == VAE_INPUT_MODE_OBSERVED_PREFIX:
            return images
        padded = torch.zeros(
            images.shape[0],
            images.shape[1],
            self.config.video_frames,
            images.shape[3],
            images.shape[4],
            device=self.device,
            dtype=self.dtype,
        )
        padded[:, :, : self.config.input_frames] = images
        return padded

    def encode_observed_pixels(self, images: torch.Tensor) -> torch.Tensor:
        """Encode only observed normalized pixels and zero-expand to ``state_t`` latents."""
        if not isinstance(images, torch.Tensor) or images.ndim != 5:
            raise ValueError("images must have shape [B, 3, T, 480, 640]")
        if tuple(images.shape[1:3]) not in {(3, 1), (3, 5)} or tuple(images.shape[3:]) != (480, 640):
            raise ValueError(
                f"observed pixels must have shape [B, 3, T, 480, 640] with T in (1, 5), got {tuple(images.shape)}"
            )
        observed_latent_frames = 1 + (images.shape[2] - 1) // 4
        expected_prefix = (
            images.shape[0],
            self.config.latent_channels,
            observed_latent_frames,
            self.config.latent_height,
            self.config.latent_width,
        )
        expected_full = (
            images.shape[0],
            self.config.latent_channels,
            self.config.state_t,
            self.config.latent_height,
            self.config.latent_width,
        )
        latent = self.tokenizer.encode(images)
        if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != expected_prefix:
            raise ValueError(
                f"observed-pixel tokenizer output must have shape {expected_prefix}, got "
                f"{tuple(latent.shape) if isinstance(latent, torch.Tensor) else type(latent)}"
            )
        latent = latent.to(device=self.device, dtype=self.dtype)
        if self.config.state_t == 2:
            if observed_latent_frames != self.config.state_t:
                raise ValueError("state_t=2 observed-only extraction requires two observed latent frames")
            return latent
        full = torch.zeros(expected_full, device=self.device, dtype=self.dtype)
        full[:, :, :observed_latent_frames] = latent
        return full

    def _encode_conditional_latents(self, images: torch.Tensor) -> torch.Tensor:
        if self.config.vae_input_mode == VAE_INPUT_MODE_OBSERVED_PREFIX:
            return self.encode_observed_pixels(images)
        latent = self.tokenizer.encode(images)
        if not isinstance(latent, torch.Tensor):
            raise TypeError("Cosmos tokenizer encode() must return a tensor")
        expected_prefix = (
            images.shape[0],
            self.config.latent_channels,
            self.config.latent_conditional_frames,
            self.config.latent_height,
            self.config.latent_width,
        )
        expected_full = (
            images.shape[0],
            self.config.latent_channels,
            self.config.state_t,
            self.config.latent_height,
            self.config.latent_width,
        )
        legacy_full = (
            images.shape[0],
            self.config.latent_channels,
            1 + (self.config.video_frames - 1) // 4,
            self.config.latent_height,
            self.config.latent_width,
        )
        accepted_shapes = (expected_prefix, expected_full)
        if self.config.state_t == 2:
            accepted_shapes += (legacy_full,)
        if tuple(latent.shape) not in accepted_shapes:
            raise ValueError(
                f"tokenizer output must have shape {expected_prefix} or {expected_full}, got {tuple(latent.shape)}"
            )
        latent = latent.to(device=self.device, dtype=self.dtype)
        if tuple(latent.shape) == expected_full:
            return latent
        if self.config.state_t == 2 and tuple(latent.shape) == legacy_full:
            return latent[:, :, : self.config.state_t]
        full = torch.zeros(expected_full, device=self.device, dtype=self.dtype)
        full[:, :, : self.config.latent_conditional_frames] = latent
        return full

    @torch.no_grad()
    def extract(
        self,
        images: torch.Tensor,
        prompt_embedding: torch.Tensor,
        *,
        noise_seed: int | None = None,
        sigma: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> CosmosPredict2Extraction:
        """Run one frozen high-noise forward with an optional per-window seed."""

        seed = self.config.seed if noise_seed is None else noise_seed
        if type(seed) is not int or seed < 0 or seed >= 2**32 - 1:
            raise ValueError("noise_seed must be an integer in [0, 2**32 - 2]")
        images = self._preprocess_images(images)
        if not isinstance(prompt_embedding, torch.Tensor):
            raise TypeError("prompt_embedding must be a torch.Tensor")
        expected_prompt = (images.shape[0], 512, 1024)
        if tuple(prompt_embedding.shape) != expected_prompt:
            raise ValueError(
                f"prompt_embedding must have shape {expected_prompt}, got {tuple(prompt_embedding.shape)}"
            )
        if not torch.is_floating_point(prompt_embedding):
            raise TypeError("prompt_embedding must be floating point")
        if not torch.isfinite(prompt_embedding).all():
            raise ValueError("prompt_embedding must contain only finite values")
        prompt_embedding = prompt_embedding.to(device=self.device, dtype=self.dtype)

        conditional = self._encode_conditional_latents(images)
        batch_size = images.shape[0]
        if sigma is None:
            sigma = torch.full(
                (batch_size,), self.config.high_noise_sigma, device=self.device, dtype=torch.float32
            )
        else:
            if tuple(sigma.shape) not in {(batch_size,), (batch_size, 1)}:
                raise ValueError(f"sigma must have shape [{batch_size}] or [{batch_size}, 1]")
            sigma = sigma.reshape(batch_size).to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(sigma).all().item() or (sigma <= 0).any().item():
                raise ValueError("sigma must contain finite positive values")
        if noise is None:
            noise = arch_invariant_rand(tuple(conditional.shape), seed).to(
                device=self.device, dtype=self.dtype
            )
        else:
            if tuple(noise.shape) != tuple(conditional.shape):
                raise ValueError(f"noise must have shape {tuple(conditional.shape)}")
            noise = noise.to(device=self.device, dtype=self.dtype)
            if not torch.isfinite(noise).all().item():
                raise ValueError("noise must contain only finite values")
        x_sigma_max = noise * sigma.to(self.dtype).view(batch_size, 1, 1, 1, 1)
        sigma_b_t = sigma.view(batch_size, 1)
        sigma_b_1_t_1_1 = sigma_b_t.view(batch_size, 1, 1, 1, 1).expand(
            batch_size, 1, self.config.state_t, 1, 1
        )
        from ._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling

        scaling = RectifiedFlowScaling(sigma_data=1.0, t_scaling_factor=1.0)
        _, _, c_in, c_noise = scaling(sigma_b_1_t_1_1)
        network_input = x_sigma_max * c_in.to(self.dtype)
        sigma_conditional = torch.full_like(sigma_b_1_t_1_1, self.config.sigma_conditional)
        _, _, _, c_noise_conditional = scaling(sigma_conditional)
        if self.config.state_t == 2:
            condition_mask = torch.ones(
                batch_size,
                1,
                self.config.state_t,
                self.config.latent_height,
                self.config.latent_width,
                device=self.device,
                dtype=self.dtype,
            )
            network_input = conditional / self.config.sigma_data
            c_noise = c_noise_conditional
        else:
            condition_mask = torch.zeros(
                batch_size,
                1,
                self.config.state_t,
                self.config.latent_height,
                self.config.latent_width,
                device=self.device,
                dtype=self.dtype,
            )
            conditional_latent_frames = self.config.latent_conditional_frames
            if self.config.vae_input_mode == VAE_INPUT_MODE_OBSERVED_PREFIX:
                conditional_latent_frames = 1 + (self.config.input_frames - 1) // 4
            condition_mask[:, :, :conditional_latent_frames] = 1
            network_input = (conditional / self.config.sigma_data) * condition_mask
            network_input = network_input + x_sigma_max * c_in.to(self.dtype) * (1.0 - condition_mask)
            condition_mask_b_1_t_1_1 = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
            c_noise = c_noise_conditional * condition_mask_b_1_t_1_1 + c_noise * (
                1.0 - condition_mask_b_1_t_1_1
            )
        padding_mask = torch.zeros(
            batch_size,
            1,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )

        # Official inference relies on CUDA autocast: timestep sinusoidal inputs
        # remain FP32, while their linear projections execute in the model dtype.
        if self.config.fp8_linear:
            if self._fp8_te is None or self._fp8_recipe is None:
                raise CosmosPredict2Error("FP8 Linear runtime was not initialized")
            backbone_context = self._fp8_te.autocast(enabled=True, recipe=self._fp8_recipe)
        else:
            backbone_context = _backbone_autocast_context(self.device, self.dtype)
        with backbone_context:
            result = self.backbone(
                x_B_C_T_H_W=network_input,
                timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]),
                crossattn_emb=prompt_embedding,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=torch.full((batch_size, 1), 10.0, device=self.device, dtype=torch.float32),
                padding_mask=padding_mask,
                data_type=self._data_type(),
                use_cuda_graphs=self.config.use_cuda_graphs,
                return_only_hidden_states_up_to=self.config.hidden_layer,
            )
        if not isinstance(result, tuple) or len(result) != 2:
            raise CosmosPredict2Error(
                "Cosmos backbone must return (prediction, hidden_states) for extraction"
            )
        _, hidden_states = result
        if not isinstance(hidden_states, (list, tuple)) or self.config.hidden_layer >= len(hidden_states):
            count = len(hidden_states) if isinstance(hidden_states, (list, tuple)) else type(hidden_states)
            raise CosmosPredict2Error(
                f"Cosmos backbone returned {count} hidden states; "
                f"layer {self.config.hidden_layer} is unavailable"
            )
        hidden = hidden_states[self.config.hidden_layer]
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("Cosmos hidden state must be a tensor")
        expected_grid = (
            batch_size,
            self.config.state_t,
            self.config.token_height,
            self.config.token_width,
            self.config.hidden_dim,
        )
        if tuple(hidden.shape) != expected_grid:
            raise ValueError(f"hidden layer must have shape {expected_grid}, got {tuple(hidden.shape)}")
        if self.config.fp8_linear and hidden.dtype != self.dtype:
            hidden = hidden.to(dtype=self.dtype)
        if hidden.dtype != self.dtype:
            raise TypeError(f"hidden layer must have dtype {self.dtype}, got {hidden.dtype}")
        if not torch.isfinite(hidden).all():
            raise ValueError("hidden layer must contain only finite values")
        hidden = hidden.detach()
        tokens = hidden.reshape(batch_size, -1, self.config.hidden_dim).contiguous()
        provenance = CosmosPredict2Provenance(
            repository=UPSTREAM_REPOSITORY,
            upstream_commit=UPSTREAM_COMMIT,
            checkpoint_path=str(self.config.checkpoint_path),
            tokenizer_path=str(self.config.tokenizer_path),
            backend=self.config.backend,
            high_noise_sigma=self.config.high_noise_sigma,
            stop_after_step=self.config.stop_after_step,
            checkpoint_ignored_metadata_keys=self._checkpoint_ignored_metadata_keys,
            checkpoint_ignored_metadata_count=len(self._checkpoint_ignored_metadata_keys),
            noise_seed=seed,
            random_init_seed=self.config.random_init_seed,
            attention_backend=self.config.attention_backend,
            compile_friendly=self.config.compile_friendly,
            torch_compile=self.config.torch_compile,
            compile_mode=self.config.compile_mode,
            use_cuda_graphs=self.config.use_cuda_graphs,
            vae_input_mode=self.config.vae_input_mode,
            state_t=self.config.state_t,
            fp8_linear=self.config.fp8_linear,
        )
        return CosmosPredict2Extraction(
            hidden_grid=hidden,
            tokens=tokens,
            sigma=sigma,
            grid_shape=(self.config.state_t, self.config.token_height, self.config.token_width),
            layer=self.config.hidden_layer,
            provenance=provenance,
        )

    def _data_type(self) -> Any:
        try:
            from ._vendor.cosmos_predict2._compat import DataType
        except ImportError:
            return "video"
        return DataType.VIDEO
