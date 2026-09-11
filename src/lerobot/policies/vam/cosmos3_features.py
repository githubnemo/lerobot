"""Cosmos 3 Edge feature extraction with native VAE encoding, 3D mRoPE, and gen_seq visual routing.

Extracts intermediate representations (specifically layer 20 = after 20 transformer blocks, block index 19)
from NVIDIA Cosmos 3 Edge for physical AI policy learning (SmolExpert). Conforms to BaseVAMExtractor
contract with exact native VAE encoding (argmax posterior, per-channel mean/inv_std normalization,
no silent missing stats), correct visual routing through gen_seq (und_seq reserved for text prompt tokens),
real dataset FPS 10 mRoPE modulation with unified_3d_mrope_temporal_modality_margin (15000), safe hook-based
tapping with early-exit truncation, and honest provenance tracking.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from torch import Tensor, nn

from lerobot.policies.vam.base import (
    BaseExtractorConfig,
    BaseVAMExtractor,
    VAMExtractionOutput,
)


class Cosmos3FeatureError(RuntimeError):
    """Base error for Cosmos 3 feature extraction operations."""


class Cosmos3EarlyExitError(Exception):
    """Control-flow exception to safely truncate forward execution after tapping target layer."""


Cosmos3EarlyExitException = Cosmos3EarlyExitError


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 digest of a file on disk."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_digest(path: str | Path) -> str:
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin", ".model"}
    )
    if not files:
        raise FileNotFoundError(f"No checkpoint files at {root}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(sha256_file(file)))
    return digest.hexdigest()


def compute_cosmos3_cache_key(
    *,
    dataset_repo: str,
    dataset_revision: str,
    episodes: Sequence[int],
    stride: int,
    clip_frames: int,
    hidden_layer: int,
    fps: float = 10.0,
    prompt: str = "take cube out of box",
    max_samples: int | None = None,
    checkpoint_sha256: str | None = None,
    lora_sha256: str | None = None,
    preprocessing_version: str = "cosmos3_native_v2",
) -> str:
    """Deterministic cache key encoding all data, model, and preprocessing invariants."""
    ep_str = ",".join(map(str, sorted(episodes)))
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    key_material = (
        f"{dataset_repo}|{dataset_revision}|{ep_str}|{stride}|{clip_frames}|"
        f"{hidden_layer}|{fps:.2f}|{prompt_hash}|{max_samples}|"
        f"{checkpoint_sha256 or 'base'}|{lora_sha256 or 'none'}|{preprocessing_version}"
    )
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def encode_cosmos3_video(
    vae: AutoencoderKLWan,
    rgb_frames: Tensor,
    *,
    sample_mode: str = "argmax",
    latents_mean: Tensor | Sequence[float] | None = None,
    latents_std: Tensor | Sequence[float] | None = None,
) -> Tensor:
    """Native Wan2.2 VAE encoding matching Cosmos3OmniPipeline._encode_video bit-for-bit.

    Args:
        vae: AutoencoderKLWan model instance.
        rgb_frames: Tensor [B, 3, T, H, W] with uint8 [0, 255] or float [0, 1] or float [-1, 1].
        sample_mode: Posterior sampling mode ("argmax" for deterministic mode, "sample" for stochastic).
        latents_mean: Per-channel latent mean (default: vae.config.latents_mean).
        latents_std: Per-channel latent std (default: vae.config.latents_std).

    Returns:
        Normalized latents [B, z_dim, T_lat, H_lat, W_lat].
    """
    if rgb_frames.ndim not in (4, 5) or rgb_frames.shape[1] != 3:
        raise ValueError("RGB must be BCHW or BCTHW with three channels")
    if sample_mode not in {"argmax", "sample"}:
        raise ValueError("Invalid posterior mode")
    if not torch.isfinite(rgb_frames).all():
        raise ValueError("RGB must be finite")
    if rgb_frames.dtype != torch.uint8 and (rgb_frames.min() < 0 or rgb_frames.max() > 1):
        raise ValueError("Floating RGB must be [0,1]; retain uint8 for [0,255] input")
    param = next(vae.parameters(), None)
    target_device = param.device if param is not None else rgb_frames.device
    target_dtype = getattr(vae, "dtype", rgb_frames.dtype)
    x = rgb_frames.to(dtype=target_dtype, device=target_device)

    # Explicit RGB normalization contract
    if rgb_frames.dtype == torch.uint8 or x.max() > 1.0:
        x = x / 255.0
        x = (x * 2.0) - 1.0
    elif x.min() >= 0.0:
        x = (x * 2.0) - 1.0

    if x.ndim == 4:
        x = x.unsqueeze(2)

    with torch.no_grad():
        encoder_output = vae.encode(x)
        if hasattr(encoder_output, "latent_dist"):
            if sample_mode == "argmax":
                raw_mu = encoder_output.latent_dist.mode()
            else:
                raw_mu = encoder_output.latent_dist.sample()
        elif hasattr(encoder_output, "latents"):
            raw_mu = encoder_output.latents
        else:
            raise Cosmos3FeatureError("Could not access latents from VAE encoder output")

    mean_vec = (
        latents_mean
        if latents_mean is not None
        else getattr(getattr(vae, "config", None), "latents_mean", None)
    )
    std_vec = (
        latents_std if latents_std is not None else getattr(getattr(vae, "config", None), "latents_std", None)
    )

    if mean_vec is None or std_vec is None:
        raise ValueError(
            "VAE config is missing latents_mean or latents_std. "
            "Cosmos 3 VAE requires explicit per-channel normalization statistics."
        )

    mean_t = (
        mean_vec.to(device=raw_mu.device, dtype=raw_mu.dtype)
        if isinstance(mean_vec, Tensor)
        else torch.tensor(mean_vec, device=raw_mu.device, dtype=raw_mu.dtype)
    )
    std_t = (
        std_vec.to(device=raw_mu.device, dtype=raw_mu.dtype)
        if isinstance(std_vec, Tensor)
        else torch.tensor(std_vec, device=raw_mu.device, dtype=raw_mu.dtype)
    )
    if (
        mean_t.numel() != raw_mu.shape[1]
        or std_t.numel() != raw_mu.shape[1]
        or not torch.isfinite(std_t).all()
        or not torch.all(std_t > 0)
    ):
        raise ValueError("Invalid per-channel VAE statistics")
    inv_std = 1.0 / std_t
    normalized = (raw_mu - mean_t.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)
    return normalized


def patchify_and_pack_cosmos3_vision_latents(
    latents: Tensor,
    patch_size: int = 2,
) -> tuple[Tensor, tuple[int, int, int]]:
    """Patchify and flatten vision latents [B, C, T, H, W] -> [B, T*(H/p)*(W/p), C*p*p].

    Returns packed tensor and the patch grid shape (T, patch_h, patch_w).
    """
    if latents.ndim != 5:
        raise ValueError(f"latents must have 5 dimensions [B, C, T, H, W], got shape {latents.shape}")
    b, c, t, h, w = latents.shape
    p = patch_size
    h_padded = ((h + p - 1) // p) * p
    w_padded = ((w + p - 1) // p) * p

    if h_padded != h or w_padded != w:
        padded = torch.zeros((b, c, t, h_padded, w_padded), device=latents.device, dtype=latents.dtype)
        padded[:, :, :, :h, :w] = latents
        latents = padded

    patch_h = h_padded // p
    patch_w = w_padded // p

    # Reshape: [B, C, T, patch_h, p, patch_w, p] -> [B, T, patch_h, patch_w, p, p, C] -> [B, S, C*p*p]
    x_p = latents.view(b, c, t, patch_h, p, patch_w, p)
    x_p = x_p.permute(0, 2, 3, 5, 4, 6, 1).contiguous()
    packed = x_p.view(b, t * patch_h * patch_w, p * p * c)
    return packed, (t, patch_h, patch_w)


def prepare_cosmos3_mrope_and_seq(
    transformer: nn.Module,
    text_token_embeds: Tensor,
    packed_vision_tokens: Tensor,
    grid_shape: tuple[int, int, int],
    *,
    fps: float = 10.0,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Build und_seq (text tokens), gen_seq (vision tokens), and 3D mRoPE rotary embeddings.

    Visual route invariant: Clean observation tokens ALWAYS route through gen_seq.
    und_seq is strictly reserved for text understanding prefix tokens (und_len = num_text_tokens).
    Preserves unified_3d_mrope_temporal_modality_margin (15000) and enable_fps_modulation flags.
    """
    target_device = device if device is not None else text_token_embeds.device
    target_dtype = dtype if dtype is not None else text_token_embeds.dtype

    num_text_tokens = (
        text_token_embeds.shape[1] if text_token_embeds.ndim == 3 else text_token_embeds.shape[0]
    )
    grid_t, patch_h, patch_w = grid_shape

    config = getattr(transformer, "config", None)
    enable_fps = getattr(config, "enable_fps_modulation", True)
    modality_margin = getattr(config, "unified_3d_mrope_temporal_modality_margin", 15000)
    base_fps_val = float(getattr(config, "base_fps", base_fps))
    reset_spatial = getattr(config, "unified_3d_mrope_reset_spatial_ids", True)

    # 1. 3D mRoPE position IDs for text and vision
    text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
        num_tokens=num_text_tokens,
        temporal_offset=0,
        use_float_positions=enable_fps,
    )

    vision_temporal_offset = next_mrope_offset + modality_margin
    effective_fps = fps if enable_fps else None

    vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
        grid_t=grid_t,
        grid_h=patch_h,
        grid_w=patch_w,
        temporal_offset=vision_temporal_offset,
        reset_spatial_indices=reset_spatial,
        fps=effective_fps,
        base_fps=base_fps_val,
        temporal_compression_factor=temporal_compression_factor,
    )

    position_ids = torch.cat([text_mrope_ids, vision_mrope_ids], dim=1).to(device=target_device)

    # 2. Rotary embeddings
    cos, sin = transformer.rotary_emb(
        position_ids=position_ids.unsqueeze(1),
        device=target_device,
        dtype=target_dtype,
    )
    cos = cos.squeeze(0)
    sin = sin.squeeze(0)

    und_len = num_text_tokens
    rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])

    # 3. Sequences
    und_seq = text_token_embeds.to(device=target_device, dtype=target_dtype)
    if und_seq.ndim == 3 and und_seq.shape[0] == 1:
        und_seq = und_seq.squeeze(0)

    gen_seq = packed_vision_tokens.to(device=target_device, dtype=target_dtype)
    if gen_seq.ndim == 3 and gen_seq.shape[0] == 1:
        gen_seq = gen_seq.squeeze(0)

    return und_seq, gen_seq, rotary_emb


def tap_cosmos3_layers(
    transformer: nn.Module,
    und_seq: Tensor,
    gen_seq: Tensor,
    rotary_emb: tuple[Tensor, Tensor, Tensor, Tensor],
    target_layers: Sequence[int] = (20,),
    *,
    early_exit: bool = True,
) -> dict[int, Tensor]:
    """Execute transformer decoder blocks and tap gen_seq visual representations.

    Definition: Layer N means the representation after N transformer blocks (i.e. block index N - 1).
    For target_layers=(20,), taps the output after block index 19 (first 20 blocks).
    Uses forward hooks with safe removal in try/finally block, and safe early-exit truncation.
    """
    captured: dict[int, Tensor] = {}
    hooks = []

    # Map layer number N (1..num_layers) to block index N - 1
    target_block_map: dict[int, int] = {}
    for lyr in target_layers:
        blk_idx = lyr - 1 if lyr >= 1 else 0
        target_block_map[blk_idx] = lyr

    max_block_idx = max(target_block_map.keys())

    def make_hook(block_idx: int, layer_num: int):
        def hook_fn(module: nn.Module, inputs: Any, outputs: Any) -> None:
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                _, g_out = outputs[0], outputs[1]
                captured[layer_num] = g_out.detach().unsqueeze(0)
            elif isinstance(outputs, Tensor):
                captured[layer_num] = outputs.detach().unsqueeze(0)
            if early_exit and block_idx == max_block_idx:
                raise Cosmos3EarlyExitException("Reached target layer")

        return hook_fn

    for blk_idx, lyr_num in target_block_map.items():
        if 0 <= blk_idx < len(transformer.layers):
            blk = transformer.layers[blk_idx]
            hooks.append(blk.register_forward_hook(make_hook(blk_idx, lyr_num)))

    curr_und = und_seq
    curr_gen = gen_seq

    try:
        if early_exit:
            try:
                for i in range(max_block_idx + 1):
                    curr_und, curr_gen = transformer.layers[i](curr_und, curr_gen, rotary_emb)
            except Cosmos3EarlyExitException:
                pass
        else:
            for layer in transformer.layers:
                curr_und, curr_gen = layer(curr_und, curr_gen, rotary_emb)
    finally:
        for h in hooks:
            h.remove()

    return captured


@dataclass(frozen=True, slots=True)
class Cosmos3ExtractorConfig(BaseExtractorConfig):
    """Configuration for Cosmos 3 Edge feature extraction."""

    backbone_name: str = "cosmos3-edge"
    checkpoint_path: str | Path | None = None
    vae_path: str | Path | None = None
    device: str = "cpu"
    dtype: str = "bfloat16"
    hidden_layers: tuple[int, ...] = (20,)  # Layer 20 = after 20 blocks (block index 19)
    pool_spatial: int | None = None
    concat_layers: bool = False
    state_t: int = 2
    latent_channels: int = 48
    latent_patch_size: int = 2
    hidden_dim: int = 2048
    num_layers: int = 28
    fps: float = 10.0
    base_fps: float = 24.0
    temporal_compression_factor: int = 4
    prompt: str = "take cube out of box"
    lora_checkpoint: str | Path | None = None
    lora_rank: int = 16
    lora_alpha: float = 32.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        BaseExtractorConfig.__post_init__(self)
        for layer_idx in self.hidden_layers:
            if layer_idx < 1 or layer_idx > self.num_layers:
                raise ValueError(f"hidden_layer {layer_idx} out of range [1, {self.num_layers}] for Cosmos 3")


@dataclass(frozen=True, slots=True)
class Cosmos3ExtractionOutput(VAMExtractionOutput):
    """Standard output container for Cosmos 3 Edge extracted features."""


class Cosmos3FeatureExtractor(BaseVAMExtractor):
    """Unified Cosmos 3 Edge feature extractor conforming to BaseVAMExtractor contract."""

    def __init__(
        self,
        config: Cosmos3ExtractorConfig,
        transformer: nn.Module | None = None,
        vae: AutoencoderKLWan | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        super().__init__(config)
        self.config: Cosmos3ExtractorConfig = config
        self._torch_dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float32

        # 1. Initialize or assign Transformer
        if transformer is not None:
            self.transformer = transformer.to(device=self.device, dtype=self._torch_dtype)
        elif config.checkpoint_path is not None:
            trans_path = Path(config.checkpoint_path)
            if (trans_path / "transformer").is_dir():
                trans_path = trans_path / "transformer"
            self.transformer = Cosmos3OmniTransformer.from_pretrained(
                str(trans_path),
                torch_dtype=self._torch_dtype,
            ).to(self.device)
        else:
            raise Cosmos3FeatureError("Must provide transformer instance or checkpoint_path")

        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        # 2. Load LoRA if requested
        if config.lora_checkpoint is not None:
            from lerobot.policies.vam.cosmos3_lora import load_cosmos3_lora

            load_cosmos3_lora(
                self.transformer,
                config.lora_checkpoint,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
            )

        # 3. Initialize or assign VAE
        if vae is not None:
            self.vae = vae.to(device=self.device, dtype=self._torch_dtype)
        elif config.vae_path is not None or config.checkpoint_path is not None:
            v_path = (
                Path(config.vae_path) if config.vae_path is not None else Path(config.checkpoint_path) / "vae"
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                str(v_path),
                torch_dtype=self._torch_dtype,
            ).to(self.device)
        else:
            self.vae = None

        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)

        # 4. Initialize text prompt embedding cache
        self.prompt_text = config.prompt
        self.tokenizer = tokenizer
        self._cached_prompt_embeds: Tensor | None = None
        self._init_prompt_tokens()

    def _init_prompt_tokens(self) -> None:
        if self.tokenizer is None:
            if self.config.checkpoint_path is None:
                raise Cosmos3FeatureError("An explicit tokenizer is required with an injected transformer")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.config.checkpoint_path) / "text_tokenizer"), local_files_only=True
            )
        prompt_tokens = self.tokenizer(self.prompt_text).input_ids
        prompt_input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self._cached_prompt_embeds = self.transformer.embed_tokens(prompt_input_ids)

    def encode_latents(self, rgb_frames: Tensor) -> Tensor:
        """Encode RGB video [B, 3, T, H, W] into normalized VAE latents [B, 48, T_lat, H_lat, W_lat]."""
        if self.vae is None:
            raise Cosmos3FeatureError("VAE model is required for encode_latents()")
        return encode_cosmos3_video(self.vae, rgb_frames, sample_mode="argmax")

    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
    ) -> dict[int, Tensor]:
        """Forward latents through Cosmos 3 Transformer and tap requested layer representations."""

        if latents.shape[0] != 1:
            raise ValueError("Packed Cosmos 3 extraction currently requires batch size one")
        # 1. Patchify latents and project to transformer hidden dim
        packed_latent, grid_shape = patchify_and_pack_cosmos3_vision_latents(
            latents, patch_size=self.config.latent_patch_size
        )
        proj_vision = self.transformer.proj_in(packed_latent.to(dtype=self._torch_dtype, device=self.device))

        # 2. Prepare text embeddings
        if isinstance(text_conditioning, Tensor):
            text_embeds = text_conditioning.to(device=self.device, dtype=self._torch_dtype)
        elif self._cached_prompt_embeds is not None:
            text_embeds = self._cached_prompt_embeds
        else:
            raise Cosmos3FeatureError("No text conditioning or cached prompt embeddings available")

        # 3. Assemble und_seq, gen_seq and rotary embeddings
        und_seq, gen_seq, rotary_emb = prepare_cosmos3_mrope_and_seq(
            transformer=self.transformer,
            text_token_embeds=text_embeds,
            packed_vision_tokens=proj_vision,
            grid_shape=grid_shape,
            fps=self.config.fps,
            base_fps=self.config.base_fps,
            temporal_compression_factor=self.config.temporal_compression_factor,
            device=self.device,
            dtype=self._torch_dtype,
        )

        # 4. Tap requested hidden layers
        tapped = tap_cosmos3_layers(
            self.transformer,
            und_seq=und_seq,
            gen_seq=gen_seq,
            rotary_emb=rotary_emb,
            target_layers=self.config.hidden_layers,
            early_exit=True,
        )

        return tapped

    def compute_grid_shape(self, latents: Tensor) -> tuple[int, int, int]:
        """Compute patch grid shape (T_lat, H_lat//p, W_lat//p)."""
        _, _, t, h, w = latents.shape
        p = self.config.latent_patch_size
        return (t, math.ceil(h / p), math.ceil(w / p))

    def get_provenance(self) -> dict[str, Any]:
        prov: dict[str, Any] = {
            "model_type": "Cosmos3-Edge",
            "hidden_layers": list(self.config.hidden_layers),
            "latent_patch_size": self.config.latent_patch_size,
            "fps": self.config.fps,
            "base_fps": self.config.base_fps,
            "device": str(self.device),
            "dtype": str(self.config.dtype),
            "prompt": self.prompt_text,
        }
        if self.config.checkpoint_path:
            chk = Path(self.config.checkpoint_path)
            prov["checkpoint_path"] = str(chk)
            chk_file = chk / "transformer" / "diffusion_pytorch_model.safetensors"
            if chk_file.is_file():
                prov["transformer_sha256"] = sha256_file(chk_file)
        if self.config.lora_checkpoint:
            prov["lora_checkpoint"] = str(self.config.lora_checkpoint)
            if Path(self.config.lora_checkpoint).is_file():
                prov["lora_sha256"] = sha256_file(self.config.lora_checkpoint)
            prov["lora_rank"] = self.config.lora_rank
            prov["lora_alpha"] = self.config.lora_alpha
        return prov

    @torch.no_grad()
    def extract(
        self,
        rgb_frames: Tensor | None = None,
        latents: Tensor | None = None,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
        hidden_states: Tensor | None = None,
        **kwargs: Any,
    ) -> Cosmos3ExtractionOutput:
        base_out = super().extract(
            rgb_frames=rgb_frames,
            latents=latents,
            text_conditioning=text_conditioning,
            timestep=timestep,
            hidden_states=hidden_states,
            **kwargs,
        )
        return Cosmos3ExtractionOutput(
            features=base_out.features,
            features_by_layer=base_out.features_by_layer,
            grid_features_by_layer=base_out.grid_features_by_layer,
            grid_coords=base_out.grid_coords,
            grid_shape=base_out.grid_shape,
            provenance=base_out.provenance,
        )
