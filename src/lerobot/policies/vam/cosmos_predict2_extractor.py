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
    input_frames: int = 5
    video_frames: int = 61
    input_height: int = 480
    input_width: int = 640
    latent_channels: int = 16
    latent_conditional_frames: int = 2
    state_t: int = 16
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
        if (
            self.input_frames not in (1, 5)
            or self.video_frames != 61
            or (self.input_height, self.input_width) != (480, 640)
        ):
            raise ValueError("The frozen extractor observed input must be T=1 or T=5 at 480x640")
        if self.latent_conditional_frames != 2 or self.state_t != 16:
            raise ValueError("The frozen extractor requires two conditional latent frames and state_t=16")
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
        if backbone is None:
            backbone, tokenizer, ignored_metadata_keys = self._build_official_components()
        else:
            ignored_metadata_keys = ()
        self._checkpoint_ignored_metadata_keys = tuple(ignored_metadata_keys)
        self.backbone = backbone
        self.tokenizer = tokenizer
        _freeze_module(self.backbone)
        _freeze_module(self.tokenizer)

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
                atten_backend=self.config.backend,
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
                sac_config=SACConfig(mode="predict2_2b_720", every_n_blocks=1),
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

    def _encode_conditional_latents(self, images: torch.Tensor) -> torch.Tensor:
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
        if tuple(latent.shape) not in (expected_prefix, expected_full):
            raise ValueError(
                f"tokenizer output must have shape {expected_prefix} or {expected_full}, got {tuple(latent.shape)}"
            )
        latent = latent.to(device=self.device, dtype=self.dtype)
        if tuple(latent.shape) == expected_full:
            return latent
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
        condition_mask = torch.zeros(
            batch_size,
            1,
            self.config.state_t,
            self.config.latent_height,
            self.config.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask[:, :, : self.config.latent_conditional_frames] = 1
        network_input = (conditional / self.config.sigma_data) * condition_mask
        network_input = network_input + x_sigma_max * c_in.to(self.dtype) * (1.0 - condition_mask)
        sigma_conditional = torch.full_like(sigma_b_1_t_1_1, self.config.sigma_conditional)
        _, _, _, c_noise_conditional = scaling(sigma_conditional)
        condition_mask_b_1_t_1_1 = condition_mask.mean(dim=[1, 3, 4], keepdim=True)
        c_noise = c_noise_conditional * condition_mask_b_1_t_1_1 + c_noise * (1.0 - condition_mask_b_1_t_1_1)
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
        with _backbone_autocast_context(self.device, self.dtype):
            result = self.backbone(
                x_B_C_T_H_W=network_input,
                timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]),
                crossattn_emb=prompt_embedding,
                condition_video_input_mask_B_C_T_H_W=condition_mask,
                fps=torch.full((batch_size, 1), 10.0, device=self.device, dtype=torch.float32),
                padding_mask=padding_mask,
                data_type=self._data_type(),
                use_cuda_graphs=False,
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
