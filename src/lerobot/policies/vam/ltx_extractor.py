"""Frozen LTX-2.5 feature extraction for the Video-Action Model.

The default real backend is loaded lazily from the official LTX-2 source
closure.  CPU tests inject a latent encoder and a backbone, which keeps this
module import-safe when the optional LTX runtime and CUDA kernels are absent.

LTX-2.5 uses a normalized flow-matching sigma in ``[0, 1]``: ``1`` is the
fully noisy endpoint and ``0`` is the clean endpoint.  The extractor performs
one forward at sigma=1 and taps block 34 of 48, matching Cosmos layer 20/28 by
relative depth.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

LTX_REPOSITORY = "https://github.com/Lightricks/LTX-2"
LTX_BACKBONE_REPOSITORY = "https://huggingface.co/Lightricks/LTX-2.5"
LTX_BACKBONE_NAME = "LTX-2.5-22B-distilled"
LTX_NUM_BLOCKS = 48
LTX_HIDDEN_WIDTH = 4096
LTX_LATENT_CHANNELS = 128
LTX_TEMPORAL_COMPRESSION = 8
LTX_SPATIAL_COMPRESSION = 32
LTX_MATCHED_LAYER = 34
LTX_MATCHED_NOISE_LEVEL = 1.0


class LTXExtractorError(RuntimeError):
    """Base error for the LTX extractor contract."""


class LTXBackendError(LTXExtractorError):
    """The optional official LTX runtime cannot be constructed."""


@dataclass(frozen=True, slots=True)
class LTXExtractorConfig:
    """Explicit immutable settings for one frozen LTX feature producer."""

    checkpoint_path: str | Path = "not-loaded/ltx-2.5-transformer.safetensors"
    video_vae_path: str | Path = "not-loaded/ltx-2.5-video-vae-conv-bf16.safetensors"
    device: str = "cuda"
    dtype: str | torch.dtype = "bfloat16"
    hidden_layer: int = LTX_MATCHED_LAYER
    high_noise_sigma: float = LTX_MATCHED_NOISE_LEVEL
    seed: int = 0
    input_frames: int = 5
    input_height: int = 480
    input_width: int = 640
    padded_input_frames: int = 9
    latent_channels: int = LTX_LATENT_CHANNELS
    latent_conditional_frames: int = 2
    state_t: int = 8
    target_frame_count: int = 57
    latent_height: int = 15
    latent_width: int = 20
    hidden_dim: int = LTX_HIDDEN_WIDTH
    spatial_compression: int = LTX_SPATIAL_COMPRESSION
    temporal_compression: int = LTX_TEMPORAL_COMPRESSION
    prompt_width: int = LTX_HIDDEN_WIDTH
    fps: float = 10.0
    offload_mode: str = "cpu"
    quantization: str = "fp8-cast"
    checkpoint_sha256: str | None = None
    video_vae_sha256: str | None = None
    source_commit: str = "400fd31054597515f47125691032c04b1c3ee24e"
    model_revision: str = "6c7e5e573ac1667efc83407806fe9b0b93730e60"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "video_vae_path", Path(self.video_vae_path))
        if self.hidden_layer < 0 or self.hidden_layer >= LTX_NUM_BLOCKS:
            raise ValueError(f"hidden_layer must be in [0, {LTX_NUM_BLOCKS})")
        if not 0.0 < self.high_noise_sigma <= 1.0:
            raise ValueError("LTX high_noise_sigma must be in (0, 1]")
        if self.input_frames != 5 or (self.input_height, self.input_width) != (480, 640):
            raise ValueError("The frozen extractor input is [B, 3, 5, 480, 640]")
        if self.padded_input_frames != 9:
            raise ValueError("The causal LTX VAE requires nine frames for two latent frames")
        if self.latent_conditional_frames != 2 or self.state_t != 8:
            raise ValueError("The frozen extractor requires two conditional latent frames and state_t=8")
        if self.target_frame_count != 57 or 1 + 8 * (self.state_t - 1) != self.target_frame_count:
            raise ValueError("The matched LTX state represents 57 frames with eight latent frames")
        if (self.latent_height, self.latent_width) != (15, 20):
            raise ValueError("480x640 at 32x32 spatial compression produces a 15x20 latent grid")
        if self.hidden_dim != LTX_HIDDEN_WIDTH or self.prompt_width != LTX_HIDDEN_WIDTH:
            raise ValueError("The LTX-2.5 22B video and prompt widths are both fixed at 4096")
        if self.offload_mode not in {"none", "cpu", "disk"}:
            raise ValueError("offload_mode must be one of 'none', 'cpu', or 'disk'")
        if self.quantization not in {"none", "fp8-cast"}:
            raise ValueError("quantization must be 'none' or 'fp8-cast'")


@dataclass(frozen=True, slots=True)
class LTXProvenance:
    """Complete producer metadata attached to one extraction."""

    backbone: str
    repository: str
    source_commit: str
    model_revision: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int | None
    video_vae_path: str
    video_vae_sha256: str
    video_vae_size_bytes: int | None
    hidden_layer: int
    num_blocks: int
    layer_fraction: float
    high_noise_sigma: float
    noise_parameterization: str
    schedule: str
    noise_seed: int
    input_shape: tuple[int, ...]
    padded_input_shape: tuple[int, ...]
    latent_shape: tuple[int, ...]
    token_shape: tuple[int, ...]
    token_count: int
    hidden_width: int
    dtype: str
    quantization: str
    offload_mode: str
    token_geometry: tuple[int, int, int]
    target_frame_count: int
    conditioning_latent_frames: int
    prompt_source: str

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "backbone": self.backbone,
            "repository": self.repository,
            "source_commit": self.source_commit,
            "model_revision": self.model_revision,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "video_vae_path": self.video_vae_path,
            "video_vae_sha256": self.video_vae_sha256,
            "video_vae_size_bytes": self.video_vae_size_bytes,
            "hidden_layer": self.hidden_layer,
            "num_blocks": self.num_blocks,
            "layer_fraction": self.layer_fraction,
            "high_noise_sigma": self.high_noise_sigma,
            "noise_parameterization": self.noise_parameterization,
            "schedule": self.schedule,
            "noise_seed": self.noise_seed,
            "input_shape": list(self.input_shape),
            "padded_input_shape": list(self.padded_input_shape),
            "latent_shape": list(self.latent_shape),
            "token_shape": list(self.token_shape),
            "token_count": self.token_count,
            "hidden_width": self.hidden_width,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "offload_mode": self.offload_mode,
            "token_geometry": list(self.token_geometry),
            "target_frame_count": self.target_frame_count,
            "conditioning_latent_frames": self.conditioning_latent_frames,
            "prompt_source": self.prompt_source,
        }
        return value


@dataclass(frozen=True, slots=True)
class LTXExtraction:
    """Detached hidden representation and its producer metadata."""

    hidden_grid: torch.Tensor
    tokens: torch.Tensor
    sigma: torch.Tensor
    grid_shape: tuple[int, int, int]
    layer: int
    provenance: LTXProvenance


def arch_invariant_rand(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Match Cosmos' NumPy RandomState noise across devices and architectures."""

    return torch.from_numpy(np.random.RandomState(seed).standard_normal(shape).astype(np.float32))


def derive_window_seed(dataset_revision: str, episode_index: int, frame_index: int, global_seed: int) -> int:
    """Derive the same stable per-window seed used by the Cosmos cache builder."""

    if not isinstance(dataset_revision, str) or not dataset_revision:
        raise ValueError("dataset_revision must be a non-empty string")
    if any(type(value) is not int or value < 0 for value in (episode_index, frame_index, global_seed)):
        raise ValueError("episode_index, frame_index, and global_seed must be non-negative integers")
    material = f"{dataset_revision}\0{episode_index}\0{frame_index}\0{global_seed}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**32 - 1)


def matched_ltx_layer(
    cosmos_layer: int = 20, cosmos_blocks: int = 28, ltx_blocks: int = LTX_NUM_BLOCKS
) -> int:
    """Map a Cosmos block index to the nearest LTX block at the same depth."""

    if any(type(value) is not int or value < 0 for value in (cosmos_layer, cosmos_blocks, ltx_blocks)):
        raise ValueError("layer and block counts must be non-negative integers")
    if cosmos_blocks <= 0 or ltx_blocks <= 0 or cosmos_layer >= cosmos_blocks:
        raise ValueError("cosmos_layer must be inside a positive Cosmos network")
    return int(round(cosmos_layer * ltx_blocks / cosmos_blocks))


def matched_ltx_noise_level() -> float:
    """Return LTX's normalized equivalent of Cosmos' high-noise extraction point."""

    # LTX's flow endpoint is sigma=1.0 (pure noise); sigma=0.0 is the clean endpoint.
    return LTX_MATCHED_NOISE_LEVEL


_DTYPE_NAMES = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_NAMES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported LTX dtype {dtype!r}") from exc


def _freeze_module(module: Any) -> None:
    if isinstance(module, nn.Module):
        module.eval()
        module.requires_grad_(False)


def _file_metadata(path: Path, override: str | None) -> tuple[str, int | None]:
    if override is not None:
        size = path.stat().st_size if path.is_file() else None
        return override, size
    if not path.is_file():
        return "unavailable", None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest(), path.stat().st_size


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class _OfficialLTXRunner:
    """Call the official LTX transformer and capture one block's video stream."""

    def __init__(self, stage: Any, device: torch.device, dtype: torch.dtype) -> None:
        self.stage = stage
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        *,
        latent: torch.Tensor,
        prompt_embedding: torch.Tensor,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        positions: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        try:
            from ltx_core.model.transformer.modality import Modality
        except ImportError as exc:  # pragma: no cover - exercised only in optional runtime
            raise LTXBackendError("The official ltx-core package is not installed") from exc

        modality = Modality(
            latent=latent,
            sigma=sigma,
            timesteps=denoise_mask * sigma[:, None],
            positions=positions,
            context=prompt_embedding,
        )
        captured: list[torch.Tensor] = []

        def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            video_output = output[0] if isinstance(output, tuple) else output
            value = getattr(video_output, "x", video_output)
            if isinstance(value, torch.Tensor):
                captured.append(value.detach())

        with self.stage._transformer_ctx() as x0_model:  # official lifecycle also handles CPU offload
            velocity_model = x0_model.velocity_model
            blocks = getattr(velocity_model, "_blocks", None)
            if blocks is None:
                blocks = getattr(velocity_model, "transformer_blocks", None)
            if blocks is None:
                raise LTXBackendError("Could not locate official LTX transformer blocks for hidden capture")
            hook = blocks[layer_index].register_forward_hook(capture)
            try:
                with _autocast(self.device, self.dtype):
                    velocity_model(video=modality, audio=None, perturbations=None)
            finally:
                hook.remove()
        if len(captured) != 1:
            raise LTXBackendError(f"Expected one hidden capture at layer {layer_index}, got {len(captured)}")
        return captured[0]


class LTXExtractor:
    """Extract one frozen high-noise LTX representation.

    ``backbone`` and ``latent_encoder`` are injectable for CPU contract tests.
    With neither supplied, the official LTX-2 source closure is loaded lazily;
    prompt embeddings remain an explicit input so prompt encoding cannot see
    actions or state.
    """

    def __init__(
        self,
        config: LTXExtractorConfig,
        *,
        backbone: Any | None = None,
        latent_encoder: Any | None = None,
        prompt_encoder: Callable[[str], torch.Tensor] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = _resolve_dtype(config.dtype)
        self.prompt_encoder = prompt_encoder
        if (backbone is None) != (latent_encoder is None):
            raise ValueError("Inject both backbone and latent_encoder, or neither")
        if backbone is None:
            backbone, latent_encoder = self._build_official_components()
        self.backbone = backbone
        self.latent_encoder = latent_encoder
        _freeze_module(self.backbone)
        _freeze_module(self.latent_encoder)
        self.checkpoint_sha256, self.checkpoint_size_bytes = _file_metadata(
            config.checkpoint_path, config.checkpoint_sha256
        )
        self.video_vae_sha256, self.video_vae_size_bytes = _file_metadata(
            config.video_vae_path, config.video_vae_sha256
        )

    def _build_official_components(self) -> tuple[Any, Any]:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise LTXBackendError("The official LTX backend requires CUDA; inject fakes for CPU tests")
        if not self.config.checkpoint_path.is_file():
            raise FileNotFoundError(f"LTX transformer checkpoint not found: {self.config.checkpoint_path}")
        if not self.config.video_vae_path.is_file():
            raise FileNotFoundError(f"LTX video VAE checkpoint not found: {self.config.video_vae_path}")
        try:
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from ltx_core.model.video_vae import VAE_ENCODER_COMFY_KEYS_FILTER, VideoEncoderConfigurator
            from ltx_pipelines.utils.blocks import DiffusionStage
            from ltx_pipelines.utils.types import OffloadMode
        except (ImportError, OSError) as exc:
            raise LTXBackendError(
                "Install the official LTX-2 ltx-core and ltx-pipelines packages before real extraction"
            ) from exc

        quantization = None
        if self.config.quantization == "fp8-cast":
            try:
                from ltx_core.quantization.fp8_cast import build_policy

                quantization = build_policy(str(self.config.checkpoint_path))
            except (ImportError, OSError, RuntimeError) as exc:
                raise LTXBackendError("LTX fp8-cast quantization is unavailable on this runtime") from exc
        try:
            stage = DiffusionStage.from_checkpoint(
                str(self.config.checkpoint_path),
                dtype=self.dtype,
                device=self.device,
                quantization=quantization,
                offload_mode=OffloadMode(self.config.offload_mode),
            )
            encoder_builder = SingleGPUModelBuilder(
                model_path=str(self.config.video_vae_path),
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            )
            encoder = encoder_builder.build(device=self.device, dtype=self.dtype).eval()
        except Exception as exc:  # noqa: BLE001
            raise LTXBackendError(f"Could not build the official LTX components: {exc}") from exc
        return _OfficialLTXRunner(stage, self.device, self.dtype), encoder

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        expected = (3, self.config.input_frames, self.config.input_height, self.config.input_width)
        if images.ndim != 5 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"images must have shape [B, 3, 5, 480, 640], got {tuple(images.shape)}")
        if images.dtype == torch.uint8:
            return images.to(device=self.device, dtype=self.dtype) / 127.5 - 1.0
        if not torch.is_floating_point(images):
            raise TypeError("images must be uint8 or floating point")
        if not torch.isfinite(images).all().item():
            raise ValueError("images contain non-finite values")
        min_value, max_value = images.amin().item(), images.amax().item()
        if min_value >= 0.0 and max_value <= 1.0:
            images = images * 2.0 - 1.0
        elif min_value >= -1.0 and max_value <= 1.0:
            pass
        else:
            raise ValueError("floating-point images must be in [0, 1] or [-1, 1]")
        return images.to(device=self.device, dtype=self.dtype)

    def _encode_conditional_latents(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # LTX's causal VAE accepts 1 + 8*k frames. Repeating the last observed
        # frame is deterministic padding, not future observation leakage.
        padding = self.config.padded_input_frames - self.config.input_frames
        padded = torch.cat([images, images[:, :, -1:].expand(-1, -1, padding, -1, -1)], dim=2)
        with _autocast(self.device, self.dtype), torch.no_grad():
            latent = self.latent_encoder(padded)
        if hasattr(latent, "latent"):
            latent = latent.latent
        if not isinstance(latent, torch.Tensor):
            raise TypeError("LTX latent encoder must return a tensor")
        expected = (
            images.shape[0],
            self.config.latent_channels,
            self.config.latent_conditional_frames,
            self.config.latent_height,
            self.config.latent_width,
        )
        if tuple(latent.shape) != expected:
            raise ValueError(f"LTX VAE output must have shape {expected}, got {tuple(latent.shape)}")
        return latent.to(device=self.device, dtype=self.dtype), padded

    def _prepare_latent_state(
        self,
        conditional: torch.Tensor,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = conditional.shape[0]
        full_shape = (
            batch_size,
            self.config.latent_channels,
            self.config.state_t,
            self.config.latent_height,
            self.config.latent_width,
        )
        clean = torch.zeros(full_shape, device=self.device, dtype=self.dtype)
        clean[:, :, : self.config.latent_conditional_frames] = conditional
        denoise_grid = torch.ones(
            (batch_size, self.config.state_t, self.config.latent_height, self.config.latent_width),
            device=self.device,
            dtype=self.dtype,
        )
        denoise_grid[:, : self.config.latent_conditional_frames] = 0
        noise = arch_invariant_rand(full_shape, seed).to(device=self.device, dtype=self.dtype)
        sigma = torch.full(
            (batch_size,), self.config.high_noise_sigma, device=self.device, dtype=torch.float32
        )
        noisy = torch.lerp(clean, noise, sigma.to(self.dtype).view(batch_size, 1, 1, 1, 1))
        condition = denoise_grid[:, None]
        noisy = clean * (1.0 - condition) + noisy * condition
        tokens = (
            noisy.permute(0, 2, 3, 4, 1).reshape(batch_size, -1, self.config.latent_channels).contiguous()
        )
        mask = denoise_grid.reshape(batch_size, -1)
        positions = self._positions(batch_size, tokens.device)
        return tokens, mask, positions, sigma

    def _positions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t, h, w = self.config.state_t, self.config.latent_height, self.config.latent_width
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(t, device=device),
                torch.arange(h, device=device),
                torch.arange(w, device=device),
                indexing="ij",
            )
        )
        starts = grid.reshape(3, -1)
        ends = starts + 1
        return torch.stack((starts, ends), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    def _run_backbone(
        self,
        tokens: torch.Tensor,
        prompt_embedding: torch.Tensor,
        sigma: torch.Tensor,
        denoise_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(self.backbone, "extract_hidden"):
            return self.backbone.extract_hidden(
                tokens,
                prompt_embedding,
                sigma=sigma,
                denoise_mask=denoise_mask,
                positions=positions,
                layer_index=self.config.hidden_layer,
            )
        return self.backbone(
            latent=tokens,
            prompt_embedding=prompt_embedding,
            sigma=sigma,
            denoise_mask=denoise_mask,
            positions=positions,
            layer_index=self.config.hidden_layer,
        )

    def extract(
        self,
        images: torch.Tensor,
        prompt_embedding: torch.Tensor | None = None,
        *,
        prompt: str | None = None,
        noise_seed: int | None = None,
    ) -> LTXExtraction:
        """Run one frozen sigma=1 forward with an optional causal prompt encoder."""

        seed = self.config.seed if noise_seed is None else noise_seed
        if type(seed) is not int or seed < 0 or seed >= 2**32 - 1:
            raise ValueError("noise_seed must be an integer in [0, 2**32 - 2]")
        images = self._preprocess_images(images)
        prompt_source = "embedding"
        if prompt_embedding is None:
            if prompt is None or self.prompt_encoder is None:
                raise ValueError("Provide prompt_embedding or both prompt and prompt_encoder")
            prompt_embedding = self.prompt_encoder(prompt)
            prompt_source = "prompt_encoder"
        if not isinstance(prompt_embedding, torch.Tensor) or prompt_embedding.ndim != 3:
            raise TypeError("prompt_embedding must have shape [B, sequence, 4096]")
        expected_batch = images.shape[0]
        if (
            prompt_embedding.shape[0] != expected_batch
            or prompt_embedding.shape[-1] != self.config.prompt_width
        ):
            raise ValueError(
                f"prompt_embedding must have shape [B, sequence, {self.config.prompt_width}], got {tuple(prompt_embedding.shape)}"
            )
        if not torch.is_floating_point(prompt_embedding) or not torch.isfinite(prompt_embedding).all().item():
            raise TypeError("prompt_embedding must be finite floating point")
        prompt_embedding = prompt_embedding.to(device=self.device, dtype=self.dtype)
        conditional, padded = self._encode_conditional_latents(images)
        tokens, denoise_mask, positions, sigma = self._prepare_latent_state(conditional, seed)
        with torch.no_grad(), _autocast(self.device, self.dtype):
            hidden = self._run_backbone(tokens, prompt_embedding, sigma, denoise_mask, positions)
        if not isinstance(hidden, torch.Tensor):
            raise TypeError("LTX backbone hidden state must be a tensor")
        expected_tokens = (
            expected_batch * self.config.state_t * self.config.latent_height * self.config.latent_width
        )
        if hidden.ndim == 3:
            expected = (expected_batch, expected_tokens // expected_batch, self.config.hidden_dim)
            if tuple(hidden.shape) != expected:
                raise ValueError(f"LTX hidden tokens must have shape {expected}, got {tuple(hidden.shape)}")
            tokens_out = hidden
            hidden_grid = hidden.reshape(
                expected_batch,
                self.config.state_t,
                self.config.latent_height,
                self.config.latent_width,
                self.config.hidden_dim,
            )
        elif hidden.ndim == 5:
            expected = (
                expected_batch,
                self.config.state_t,
                self.config.latent_height,
                self.config.latent_width,
                self.config.hidden_dim,
            )
            if tuple(hidden.shape) != expected:
                raise ValueError(f"LTX hidden grid must have shape {expected}, got {tuple(hidden.shape)}")
            hidden_grid = hidden
            tokens_out = hidden.reshape(expected_batch, -1, self.config.hidden_dim)
        else:
            raise ValueError(f"LTX hidden state must be rank 3 or 5, got rank {hidden.ndim}")
        if hidden.dtype != self.dtype:
            raise TypeError(f"LTX hidden state must have dtype {self.dtype}, got {hidden.dtype}")
        if not torch.isfinite(hidden).all().item():
            raise ValueError("LTX hidden state must contain only finite values")
        hidden_grid = hidden_grid.detach().contiguous()
        tokens_out = tokens_out.detach().contiguous()
        checkpoint_identity = str(self.config.checkpoint_path)
        provenance = LTXProvenance(
            backbone=LTX_BACKBONE_NAME,
            repository=LTX_BACKBONE_REPOSITORY,
            source_commit=self.config.source_commit,
            model_revision=self.config.model_revision,
            checkpoint_path=checkpoint_identity,
            checkpoint_sha256=self.checkpoint_sha256,
            checkpoint_size_bytes=self.checkpoint_size_bytes,
            video_vae_path=str(self.config.video_vae_path),
            video_vae_sha256=self.video_vae_sha256,
            video_vae_size_bytes=self.video_vae_size_bytes,
            hidden_layer=self.config.hidden_layer,
            num_blocks=LTX_NUM_BLOCKS,
            layer_fraction=self.config.hidden_layer / LTX_NUM_BLOCKS,
            high_noise_sigma=self.config.high_noise_sigma,
            noise_parameterization="normalized_rectified_flow_sigma",
            schedule="LTX-2.5 distilled sigma schedule; extraction at sigma=1.0",
            noise_seed=seed,
            input_shape=tuple(images.shape),
            padded_input_shape=tuple(padded.shape),
            latent_shape=(
                images.shape[0],
                self.config.latent_channels,
                self.config.state_t,
                self.config.latent_height,
                self.config.latent_width,
            ),
            token_shape=tuple(tokens_out.shape),
            token_count=tokens_out.shape[1],
            hidden_width=self.config.hidden_dim,
            dtype=str(self.dtype).removeprefix("torch."),
            quantization=self.config.quantization,
            offload_mode=self.config.offload_mode,
            token_geometry=(self.config.state_t, self.config.latent_height, self.config.latent_width),
            target_frame_count=self.config.target_frame_count,
            conditioning_latent_frames=self.config.latent_conditional_frames,
            prompt_source=prompt_source,
        )
        return LTXExtraction(
            hidden_grid=hidden_grid,
            tokens=tokens_out,
            sigma=sigma,
            grid_shape=(self.config.state_t, self.config.latent_height, self.config.latent_width),
            layer=self.config.hidden_layer,
            provenance=provenance,
        )

    def extract_window(
        self,
        images: torch.Tensor,
        prompt_embedding: torch.Tensor | None,
        *,
        dataset_revision: str,
        episode_index: int,
        frame_index: int,
        global_seed: int,
        prompt: str | None = None,
    ) -> LTXExtraction:
        """Extract with the canonical dataset/episode/frame-derived noise seed."""

        return self.extract(
            images,
            prompt_embedding,
            prompt=prompt,
            noise_seed=derive_window_seed(dataset_revision, episode_index, frame_index, global_seed),
        )
