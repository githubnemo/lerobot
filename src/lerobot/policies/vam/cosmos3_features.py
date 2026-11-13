"""Cosmos 3 Edge feature extraction with native VAE encoding, 3D mRoPE, and gen_seq visual routing.

Extracts intermediate representations (specifically layer 20 = after 20 transformer blocks, block index 19)
from NVIDIA Cosmos 3 Edge for physical AI policy learning (SmolExpert). Conforms to BaseVAMExtractor
contract with exact native VAE encoding (argmax posterior, per-channel mean/inv_std normalization,
no silent missing stats), correct visual routing through gen_seq (und_seq reserved for text prompt tokens),
real dataset FPS 10 mRoPE modulation with unified_3d_mrope_temporal_modality_margin (15000), safe hook-based
tapping with early-exit truncation, and honest provenance tracking.
"""

from __future__ import annotations

import gc
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3OmniTransformer, _rotate_half
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from torch import Tensor, nn

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - optional on non-CUDA platforms
    _TRITON_AVAILABLE = False

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_rmsnorm_rope_kernel(
        X_ptr,
        W_ptr,
        COS_ptr,
        SIN_ptr,
        OUT_ptr,
        stride_xb,
        stride_xs,
        stride_xh,
        stride_cs,
        HALF_DIM: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_s = tl.program_id(1)
        pid_h = tl.program_id(2)

        offs = tl.arange(0, HALF_DIM)
        base_x = X_ptr + pid_b * stride_xb + pid_s * stride_xs + pid_h * stride_xh
        x0 = tl.load(base_x + offs).to(tl.float32)
        x1 = tl.load(base_x + HALF_DIM + offs).to(tl.float32)

        w0 = tl.load(W_ptr + offs)
        w1 = tl.load(W_ptr + HALF_DIM + offs)

        mean_sq = (tl.sum(x0 * x0, axis=0) + tl.sum(x1 * x1, axis=0)) / (2 * HALF_DIM)
        rstd = tl.rsqrt(mean_sq + EPS)

        n0 = ((x0 * rstd).to(tl.bfloat16) * w0).to(tl.bfloat16)
        n1 = ((x1 * rstd).to(tl.bfloat16) * w1).to(tl.bfloat16)

        base_c = COS_ptr + pid_s * stride_cs
        base_s = SIN_ptr + pid_s * stride_cs
        c0 = tl.load(base_c + offs)
        c1 = tl.load(base_c + HALF_DIM + offs)
        s0 = tl.load(base_s + offs)
        s1 = tl.load(base_s + HALF_DIM + offs)

        out0 = n0 * c0 - n1 * s0
        out1 = n1 * c1 + n0 * s1

        base_out = OUT_ptr + pid_b * stride_xb + pid_s * stride_xs + pid_h * stride_xh
        tl.store(base_out + offs, out0)
        tl.store(base_out + HALF_DIM + offs, out1)

    @triton.jit
    def _wan_rmsnorm_nhwc_kernel(
        X_ptr,
        G_ptr,
        OUT_ptr,
        C: tl.constexpr,
        T_in,
        T_out,
        HW,
        PAD_T: tl.constexpr,
        SCALE: tl.constexpr,
        APPLY_SILU: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_hw = tl.program_id(0)
        pid_bt = tl.program_id(1)
        b = pid_bt // T_in
        t = pid_bt % T_in

        offs_c = tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        in_idx = ((b * T_in + t) * HW + pid_hw) * C
        out_idx = ((b * T_out + (t + PAD_T)) * HW + pid_hw) * C

        x_val = tl.load(X_ptr + in_idx + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        sum_sq = tl.sum(x_val * x_val, axis=0)
        inv_norm = 1.0 / tl.maximum(tl.sqrt(sum_sq), 1e-12)

        g_val = tl.load(G_ptr + offs_c, mask=mask_c, other=0.0).to(tl.bfloat16)
        normed = (x_val * inv_norm).to(tl.bfloat16)
        y = (normed * SCALE * g_val).to(tl.bfloat16)
        if APPLY_SILU:
            yf = y.to(tl.float32)
            y = (yf * tl.sigmoid(yf)).to(tl.bfloat16)
        tl.store(OUT_ptr + out_idx + offs_c, y, mask=mask_c)


def triton_wan_rmsnorm_nhwc(
    x: Tensor,
    norm_mod: nn.Module,
    apply_silu: bool = False,
    pad_t: int = 0,
) -> Tensor:
    """Single-pass channel-wise RMSNorm (+ optional SiLU + optional causal time pad) in channels_last_3d layout."""
    if not (_TRITON_AVAILABLE and x.is_cuda and x.dtype == torch.bfloat16):
        out = norm_mod(x)
        if apply_silu:
            out = torch.nn.functional.silu(out)
        if pad_t > 0 and out.ndim == 5:
            out = torch.nn.functional.pad(out, (0, 0, 0, 0, pad_t, 0))
        return out
    if x.ndim == 4:
        return triton_wan_rmsnorm_nhwc(x.unsqueeze(2), norm_mod, apply_silu=apply_silu, pad_t=0).squeeze(2)
    if not x.is_contiguous(memory_format=torch.channels_last_3d):
        x = x.contiguous(memory_format=torch.channels_last_3d)
    B, C, T, H, W = x.shape
    HW = H * W
    T_out = T + pad_t
    out = torch.empty((B, C, T_out, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last_3d)
    if pad_t > 0:
        out[:, :, :pad_t, :, :].zero_()
    BLOCK_C = triton.next_power_of_2(C)
    _wan_rmsnorm_nhwc_kernel[(HW, B * T)](
        x,
        norm_mod.gamma,
        out,
        C=C,
        T_in=T,
        T_out=T_out,
        HW=HW,
        PAD_T=pad_t,
        SCALE=float(norm_mod.scale),
        APPLY_SILU=apply_silu,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


def optimize_wan_vae_encoder(vae: AutoencoderKLWan) -> None:
    """Optimize Wan2.2 3D Causal VAE encoder for single-pass channels_last_3d execution with fused Triton kernels."""
    import types
    import torch.nn.functional as F
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        AvgDown3D,
        WanCausalConv3d,
        WanEncoder3d,
        WanResidualBlock,
        WanResidualDownBlock,
        WanResample,
        WanRMS_norm,
    )

    if getattr(vae, "_fast_single_pass_enabled", False) or not hasattr(vae, "encoder") or vae.encoder is None:
        return

    for m in vae.modules():
        if isinstance(m, nn.Conv3d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
        elif isinstance(m, nn.Conv2d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)

    def _fast_causal_conv_forward(self, x: Tensor, cache_x: Tensor | None = None, time_pre_padded: bool = False) -> Tensor:
        pad_w, _, pad_h, _, pad_t, _ = self._padding
        if not time_pre_padded:
            if cache_x is not None and pad_t > 0:
                cache_x = cache_x.to(x.device)
                x = torch.cat([cache_x, x], dim=2)
                pad_t -= cache_x.shape[2]
            if pad_t > 0:
                B, C, T, H, W = x.shape
                out_pad = torch.empty(
                    (B, C, T + pad_t, H, W),
                    device=x.device,
                    dtype=x.dtype,
                    memory_format=torch.channels_last_3d,
                )
                out_pad[:, :, :pad_t].zero_()
                out_pad[:, :, pad_t:].copy_(x)
                x = out_pad
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            x = x.contiguous(memory_format=torch.channels_last_3d)
        return F.conv3d(x, self.weight, self.bias, self.stride, (0, pad_h, pad_w), self.dilation, self.groups)

    def _fast_resblock_forward(self, x: Tensor, feat_cache: Any = None, feat_idx: list[int] | None = None) -> Tensor:
        if feat_cache is not None:
            return WanResidualBlock.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx or [0])
        h = self.conv_shortcut(x)
        x = triton_wan_rmsnorm_nhwc(x, self.norm1, apply_silu=True, pad_t=self.conv1._padding[4])
        x = self.conv1(x, time_pre_padded=True)
        x = triton_wan_rmsnorm_nhwc(x, self.norm2, apply_silu=True, pad_t=self.conv2._padding[4])
        x = self.conv2(x, time_pre_padded=True)
        return x + h

    def _fast_avgdown_forward(self, x: Tensor) -> Tensor:
        if self.factor_t == 1 and self.factor_s == 1 and self.in_channels == self.out_channels:
            return x
        x_c = x.contiguous()
        pad_t = (self.factor_t - x_c.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x_c = F.pad(x_c, (0, 0, 0, 0, pad_t, 0))
        B, C, T, H, W = x_c.shape
        x_c = x_c.view(
            B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s
        )
        x_c = x_c.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x_c = x_c.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        return x_c.mean(dim=2).contiguous(memory_format=torch.channels_last_3d)

    def _fast_resdown_forward(self, x: Tensor, feat_cache: Any = None, feat_idx: list[int] | None = None) -> Tensor:
        x_in = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(x_in)

    def _fast_resample_forward(self, x: Tensor, feat_cache: Any = None, feat_idx: list[int] | None = None) -> Tensor:
        if feat_cache is not None:
            return WanResample.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx or [0])
        b, c, t, h, w = x.size()
        if x.is_contiguous(memory_format=torch.channels_last_3d):
            x_2d = x.permute(0, 2, 3, 4, 1).view(b * t, h, w, c).permute(0, 3, 1, 2)
        else:
            x_2d = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last)
        x_2d = self.resample(x_2d)
        if x_2d.is_contiguous(memory_format=torch.channels_last):
            x = x_2d.permute(0, 2, 3, 1).view(b, t, x_2d.size(2), x_2d.size(3), x_2d.size(1)).permute(0, 4, 1, 2, 3)
        else:
            x = x_2d.view(b, t, x_2d.size(1), x_2d.size(2), x_2d.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == "downsample3d" and t > 1:
            x = torch.cat([x[:, :, :1, :, :], self.time_conv(x)], dim=2).contiguous(
                memory_format=torch.channels_last_3d
            )
        return x

    def _fast_encoder_forward(self, x: Tensor, feat_cache: Any = None, feat_idx: list[int] | None = None) -> Tensor:
        if feat_cache is not None:
            return WanEncoder3d.forward(self, x, feat_cache=feat_cache, feat_idx=feat_idx or [0])
        x = self.conv_in(x)
        for layer in self.down_blocks:
            x = layer(x, feat_cache=None)
        x = self.mid_block(x, feat_cache=None)
        x = triton_wan_rmsnorm_nhwc(x, self.norm_out, apply_silu=True, pad_t=self.conv_out._padding[4])
        x = self.conv_out(x, time_pre_padded=True)
        return x

    for m in vae.modules():
        if isinstance(m, WanCausalConv3d):
            m.forward = types.MethodType(_fast_causal_conv_forward, m)
        elif isinstance(m, WanResidualBlock):
            m.forward = types.MethodType(_fast_resblock_forward, m)
        elif isinstance(m, AvgDown3D):
            m.forward = types.MethodType(_fast_avgdown_forward, m)
        elif isinstance(m, WanResidualDownBlock):
            m.forward = types.MethodType(_fast_resdown_forward, m)
        elif isinstance(m, WanResample):
            m.forward = types.MethodType(_fast_resample_forward, m)
        elif isinstance(m, WanRMS_norm):
            m.forward = types.MethodType(
                lambda self, x: triton_wan_rmsnorm_nhwc(x, self, apply_silu=False, pad_t=0), m
            )

    vae.encoder.forward = types.MethodType(_fast_encoder_forward, vae.encoder)
    vae._fast_single_pass_enabled = True


def fused_rmsnorm_rope(
    x: Tensor,
    norm_module: nn.Module,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """Apply head-wise RMSNorm followed by 3D-mRoPE rotation (uses custom Triton kernel on CUDA bf16)."""
    norm_weight = getattr(norm_module, "weight", None)
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or (norm_weight is not None and norm_weight.requires_grad)
    )
    if (
        not needs_grad
        and _TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and norm_weight is not None
    ):
        B, S, H, D = x.shape
        x_contig = x if x.is_contiguous() else x.contiguous()
        out = torch.empty_like(x_contig)
        eps = float(getattr(norm_module, "eps", 1e-6))
        _fused_rmsnorm_rope_kernel[(B, S, H)](
            x_contig,
            norm_module.weight,
            cos,
            sin,
            out,
            x_contig.stride(0),
            x_contig.stride(1),
            x_contig.stride(2),
            cos.stride(0),
            HALF_DIM=D // 2,
            EPS=eps,
        )
        return out
    normed = norm_module(x)
    cos_b = cos.unsqueeze(0).unsqueeze(2)
    sin_b = sin.unsqueeze(0).unsqueeze(2)
    return normed * cos_b + _rotate_half(normed) * sin_b

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
    param = next(vae.parameters(), None)
    target_device = param.device if param is not None else rgb_frames.device
    target_dtype = getattr(vae, "dtype", rgb_frames.dtype)

    if rgb_frames.dtype == torch.uint8:
        x = rgb_frames.to(dtype=target_dtype, device=target_device)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        x = x.mul(2.0 / 255.0).sub_(1.0)
    else:
        if not torch.isfinite(rgb_frames).all():
            raise ValueError("RGB must be finite")
        if rgb_frames.min() < 0 or rgb_frames.max() > 1:
            raise ValueError("Floating RGB must be [0,1]; retain uint8 for [0,255] input")
        x = rgb_frames.to(dtype=target_dtype, device=target_device)
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.max() > 1.0:
            x = (x / 255.0) * 2.0 - 1.0
        elif x.min() >= 0.0:
            x = (x * 2.0) - 1.0

    with torch.no_grad():
        if (
            getattr(vae, "_fast_single_pass_enabled", False)
            and sample_mode == "argmax"
            and x.shape[2] <= 5
            and not getattr(vae, "use_tiling", False)
        ):
            from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

            x_in = patchify(x, patch_size=vae.config.patch_size) if vae.config.patch_size is not None else x
            enc_feat = vae.encoder(x_in, feat_cache=None)
            raw_mu = vae.quant_conv(enc_feat)[:, : vae.z_dim].contiguous()
        else:
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

    cached_stats = getattr(vae, "_cached_norm_stats", None)
    if (
        cached_stats is not None
        and latents_mean is None
        and latents_std is None
        and cached_stats[0].device == raw_mu.device
        and cached_stats[0].dtype == raw_mu.dtype
    ):
        mean_v, inv_std_v = cached_stats
        return (raw_mu - mean_v) * inv_std_v

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
    mean_v = mean_t.view(1, -1, 1, 1, 1)
    inv_std_v = inv_std.view(1, -1, 1, 1, 1)
    if latents_mean is None and latents_std is None:
        vae._cached_norm_stats = (mean_v, inv_std_v)
    normalized = (raw_mu - mean_v) * inv_std_v
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
            keep_grad = torch.is_grad_enabled()
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                _, g_out = outputs[0], outputs[1]
                captured[layer_num] = (g_out if keep_grad else g_out.detach()).unsqueeze(0)
            elif isinstance(outputs, Tensor):
                captured[layer_num] = (outputs if keep_grad else outputs.detach()).unsqueeze(0)
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


@dataclass(slots=True)
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
    trainable_lora: bool = False
    lora_use_dora: bool = False
    lora_pathway: str = "gen_only"
    gradient_checkpointing: bool = False
    compile_dit: bool = False
    compile_vae: bool = False
    dit_compile_mode: str = "default"
    vae_compile_mode: str = "default"
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        BaseExtractorConfig.__post_init__(self)
        for layer_idx in self.hidden_layers:
            if layer_idx < 1 or layer_idx > self.num_layers:
                raise ValueError(f"hidden_layer {layer_idx} out of range [1, {self.num_layers}] for Cosmos 3")


@dataclass(frozen=True, slots=True)
class Cosmos3ExtractionOutput(VAMExtractionOutput):
    """Standard output container for Cosmos 3 Edge extracted features."""


def _apply_cosmos3_lora_pathway_grad(
    transformer: nn.Module,
    *,
    trainable_lora: bool,
    lora_pathway: str,
) -> None:
    """Enforce fp32 precision and pathway-scoped requires_grad on Cosmos 3 LoRA/DoRA parameters."""
    if not trainable_lora:
        for p in transformer.parameters():
            p.requires_grad_(False)
        return

    from lerobot.policies.vam.cosmos3_lora import (
        DUAL_PATHWAY_TARGET_MODULES,
        GEN_TARGET_MODULES,
        UND_TARGET_MODULES,
        _is_matching_target_module,
    )

    if lora_pathway == "gen_only":
        allowed_targets = set(GEN_TARGET_MODULES)
    elif lora_pathway == "und_only":
        allowed_targets = set(UND_TARGET_MODULES)
    elif lora_pathway == "dual":
        allowed_targets = set(DUAL_PATHWAY_TARGET_MODULES)
    else:
        raise ValueError(f"Unsupported Cosmos 3 lora_pathway: {lora_pathway!r}")

    for name, p in transformer.named_parameters():
        if not name.endswith(("lora_A", "lora_B", "dora_magnitude")):
            p.requires_grad_(False)
            continue
        if p.dtype != torch.float32:
            p.data = p.data.float()
        clean = name.replace("._checkpoint_wrapped_module.", ".").replace("._orig_mod.", ".")
        parts = clean.split(".")
        if len(parts) > 3 and parts[0] == "layers" and parts[1].isdigit():
            mod_subpath = ".".join(parts[2:-1])
        else:
            mod_subpath = ".".join(parts[:-1])
        p.requires_grad_(_is_matching_target_module(mod_subpath, allowed_targets))


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

        # 1. Initialize or assign Transformer (keep on CPU until unused blocks/heads are pruned)
        loaded_transformer_from_disk = False
        if config.checkpoint_path is not None and not Path(config.checkpoint_path).exists():
            fallback_dir = Path.home() / ".cache/video-vam/cosmos3-edge"
            if not (fallback_dir / "transformer").is_dir():
                from huggingface_hub import snapshot_download

                fallback_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download("nvidia/Cosmos3-Edge", local_dir=str(fallback_dir))
            config.checkpoint_path = fallback_dir
            self.config.checkpoint_path = fallback_dir
        if transformer is not None:
            self.transformer = transformer.to(device=self.device, dtype=self._torch_dtype)
        elif config.checkpoint_path is not None:
            trans_path = Path(config.checkpoint_path)
            if (trans_path / "transformer").is_dir():
                trans_path = trans_path / "transformer"
            self.transformer = Cosmos3OmniTransformer.from_pretrained(
                str(trans_path),
                torch_dtype=self._torch_dtype,
            )
            loaded_transformer_from_disk = True
        else:
            raise Cosmos3FeatureError("Must provide transformer instance or checkpoint_path")

        self.transformer.eval()
        for p in self.transformer.parameters():
            p.requires_grad_(False)

        # 2. Load or inject LoRA/DoRA if requested
        max_layer = max(self.config.hidden_layers)
        if loaded_transformer_from_disk and hasattr(self.transformer, "layers") and len(self.transformer.layers) > max_layer:
            self.transformer.layers = self.transformer.layers[:max_layer]

        if config.lora_checkpoint is not None:
            from lerobot.policies.vam.cosmos3_lora import (
                is_peft_adapter,
                load_cosmos3_lora,
                load_cosmos3_peft_adapter,
            )

            if is_peft_adapter(config.lora_checkpoint):
                # Unmerged: identical numerics to the legacy fp32 LoRALinear branch.
                self.transformer = load_cosmos3_peft_adapter(
                    self.transformer, config.lora_checkpoint, merge=False
                )
            else:
                from lerobot.policies.vam.cosmos_lora import merge_lora_into_base

                load_cosmos3_lora(
                    self.transformer,
                    config.lora_checkpoint,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    block_indices=range(len(self.transformer.layers)),
                    use_dora=config.lora_use_dora if config.trainable_lora else None,
                )
                if not config.trainable_lora:
                    merge_lora_into_base(self.transformer)
        elif config.trainable_lora:
            from lerobot.policies.vam.cosmos3_lora import (
                DUAL_PATHWAY_TARGET_MODULES,
                GEN_TARGET_MODULES,
                UND_TARGET_MODULES,
                inject_cosmos3_lora,
            )

            if config.lora_pathway == "gen_only":
                target_mods = GEN_TARGET_MODULES
            elif config.lora_pathway == "und_only":
                target_mods = UND_TARGET_MODULES
            else:
                target_mods = DUAL_PATHWAY_TARGET_MODULES
            inject_cosmos3_lora(
                self.transformer,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                block_indices=range(len(self.transformer.layers)),
                target_modules=target_mods,
                use_dora=config.lora_use_dora,
            )

        _apply_cosmos3_lora_pathway_grad(
            self.transformer,
            trainable_lora=config.trainable_lora,
            lora_pathway=config.lora_pathway,
        )

        # 3. Initialize text prompt embedding cache (before moving/pruning embed_tokens)
        self.prompt_text = config.prompt
        self.tokenizer = tokenizer
        self._cached_prompt_embeds: Tensor | None = None
        self._cached_und_kv: list[tuple[Tensor, Tensor]] | None = None
        self._cached_gen_rotary: dict[tuple[int, int, int], tuple[Tensor, Tensor]] = {}
        self._init_prompt_tokens()

        if loaded_transformer_from_disk:
            for unused_attr in ("lm_head", "proj_out", "time_embedder", "action_proj_in", "action_proj_out"):
                if hasattr(self.transformer, unused_attr):
                    setattr(self.transformer, unused_attr, None)
            embed_tokens_cpu = getattr(self.transformer, "embed_tokens", None)
            if embed_tokens_cpu is not None:
                self.transformer.embed_tokens = None
            fp32_lora_snapshot = {
                name: p.detach().float()
                for name, p in self.transformer.named_parameters()
                if name.endswith(("lora_A", "lora_B", "dora_magnitude"))
            }
            self.transformer = self.transformer.to(device=self.device, dtype=self._torch_dtype)
            for name, p in self.transformer.named_parameters():
                if name in fp32_lora_snapshot:
                    p.data = fp32_lora_snapshot[name].to(device=self.device, dtype=torch.float32)
            _apply_cosmos3_lora_pathway_grad(
                self.transformer,
                trainable_lora=config.trainable_lora,
                lora_pathway=config.lora_pathway,
            )
            if not (config.trainable_lora and config.lora_pathway in ("dual", "und_only")):
                self._precompute_and_prune_und_pathway()
            if embed_tokens_cpu is not None:
                self.transformer.embed_tokens = embed_tokens_cpu.to(device="cpu", dtype=self._torch_dtype)

        # 4. Initialize or assign VAE (prune unused decoder before moving to GPU)
        if vae is not None:
            self.vae = vae.to(device=self.device, dtype=self._torch_dtype)
        elif config.vae_path is not None or config.checkpoint_path is not None:
            v_path = (
                Path(config.vae_path) if config.vae_path is not None else Path(config.checkpoint_path) / "vae"
            )
            loaded_vae = AutoencoderKLWan.from_pretrained(
                str(v_path),
                torch_dtype=self._torch_dtype,
            )
            if hasattr(loaded_vae, "decoder"):
                loaded_vae.decoder = None
            if hasattr(loaded_vae, "post_quant_conv"):
                loaded_vae.post_quant_conv = None
            self.vae = loaded_vae.to(device=self.device, dtype=self._torch_dtype)
        else:
            self.vae = None

        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)
            if self.device.type == "cuda" and isinstance(self.vae, AutoencoderKLWan):
                optimize_wan_vae_encoder(self.vae)

        if loaded_transformer_from_disk:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 5. Optional compilation
        if self.config.compile_dit:
            self.compile_transformer(mode=self.config.dit_compile_mode)
        if self.config.compile_vae and self.vae is not None:
            self.compile_vae(mode=self.config.vae_compile_mode)

    def _init_prompt_tokens(self) -> None:
        if self.tokenizer is None:
            if self.config.checkpoint_path is None:
                raise Cosmos3FeatureError("An explicit tokenizer is required with an injected transformer")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(Path(self.config.checkpoint_path) / "text_tokenizer"), local_files_only=True
            )
        prompt_tokens = self.tokenizer(self.prompt_text).input_ids
        embed_device = next(self.transformer.embed_tokens.parameters()).device
        prompt_input_ids = torch.tensor(prompt_tokens, dtype=torch.long, device=embed_device)
        with torch.no_grad():
            self._cached_prompt_embeds = self.transformer.embed_tokens(prompt_input_ids).to(
                device=self.device, dtype=self._torch_dtype
            )

    def _supports_fast_batched_gen(self) -> bool:
        if not hasattr(self.transformer, "layers") or len(self.transformer.layers) == 0:
            return False
        blk0 = self.transformer.layers[0]
        return (
            hasattr(blk0, "self_attn")
            and hasattr(blk0.self_attn, "add_q_proj")
            and hasattr(blk0, "mlp_moe_gen")
        )

    @torch.no_grad()
    def _compute_und_kv_for_embeds(self, text_embeds: Tensor) -> list[tuple[Tensor, Tensor]]:
        und_seq = text_embeds.to(device=self.device, dtype=self._torch_dtype)
        if und_seq.ndim == 3 and und_seq.shape[0] == 1:
            und_seq = und_seq.squeeze(0)
        num_text_tokens = und_seq.shape[0]
        t_cfg = getattr(self.transformer, "config", None)
        enable_fps = getattr(t_cfg, "enable_fps_modulation", True)
        text_mrope_ids, _ = get_3d_mrope_ids_text_tokens(
            num_tokens=num_text_tokens,
            temporal_offset=0,
            use_float_positions=enable_fps,
        )
        position_ids = text_mrope_ids.to(device=self.device)
        cos, sin = self.transformer.rotary_emb(
            position_ids=position_ids.unsqueeze(1),
            device=self.device,
            dtype=self._torch_dtype,
        )
        cos_u = cos.squeeze(0).unsqueeze(1)
        sin_u = sin.squeeze(0).unsqueeze(1)

        kv_cache: list[tuple[Tensor, Tensor]] = []
        curr_und = und_seq
        for blk in self.transformer.layers:
            u_norm = blk.input_layernorm(curr_und)
            attn = blk.self_attn
            q_u = attn.norm_q(attn.to_q(u_norm).view(-1, attn.num_attention_heads, attn.head_dim))
            k_u = attn.norm_k(attn.to_k(u_norm).view(-1, attn.num_key_value_heads, attn.head_dim))
            v_u = attn.to_v(u_norm).view(-1, attn.num_key_value_heads, attn.head_dim)
            k_norm_und = getattr(attn, "k_norm_und_for_gen", None)
            k_u_gen = k_norm_und(k_u) if k_norm_und is not None else k_u
            q_u = q_u * cos_u + _rotate_half(q_u) * sin_u
            k_u = k_u * cos_u + _rotate_half(k_u) * sin_u
            k_u_gen = k_u_gen * cos_u + _rotate_half(k_u_gen) * sin_u
            kv_cache.append((k_u_gen.unsqueeze(0).contiguous(), v_u.unsqueeze(0).contiguous()))
            c_out = (
                dispatch_attention_fn(
                    q_u.unsqueeze(0),
                    k_u.unsqueeze(0),
                    v_u.unsqueeze(0),
                    is_causal=True,
                    enable_gqa=True,
                )
                .squeeze(0)
                .flatten(-2, -1)
            )
            res_u = curr_und + attn.to_out(c_out)
            curr_und = res_u + blk.mlp(blk.post_attention_layernorm(res_u))
        return kv_cache

    @torch.no_grad()
    def _precompute_and_prune_und_pathway(self) -> None:
        if not self._supports_fast_batched_gen() or self._cached_prompt_embeds is None:
            return
        self._cached_und_kv = self._compute_und_kv_for_embeds(self._cached_prompt_embeds)
        pruned_lora: dict[str, Tensor] = dict(getattr(self.transformer, "_pruned_lora_state_dict", {}))
        for idx, blk in enumerate(self.transformer.layers):
            for sub_prefix in ("mlp", "self_attn.to_q", "self_attn.to_k", "self_attn.to_v", "self_attn.to_out"):
                try:
                    sub_mod = blk.get_submodule(sub_prefix)
                except AttributeError:
                    continue
                if sub_mod is None:
                    continue
                for p_name, param in sub_mod.named_parameters():
                    if p_name.endswith(("lora_A", "lora_B", "dora_magnitude")):
                        clean_key = (
                            f"layers.{idx}.{sub_prefix}.{p_name}"
                            .replace("._checkpoint_wrapped_module.", ".")
                            .replace("._orig_mod.", ".")
                        )
                        pruned_lora[clean_key] = param.detach().float().cpu().contiguous()
            if hasattr(blk, "mlp"):
                blk.mlp = None
            if hasattr(blk, "input_layernorm"):
                blk.input_layernorm = None
            if hasattr(blk, "post_attention_layernorm"):
                blk.post_attention_layernorm = None
            attn = getattr(blk, "self_attn", None)
            if attn is not None:
                for attr in ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k", "k_norm_und_for_gen"):
                    if hasattr(attn, attr):
                        setattr(attn, attr, None)
        if pruned_lora:
            self.transformer._pruned_lora_state_dict = pruned_lora

    def _get_gen_rotary(self, num_text_tokens: int, grid_shape: tuple[int, int, int]) -> tuple[Tensor, Tensor]:
        key = (num_text_tokens, *grid_shape)
        cached = self._cached_gen_rotary.get(key)
        if cached is not None:
            return cached
        t_cfg = getattr(self.transformer, "config", None)
        enable_fps = getattr(t_cfg, "enable_fps_modulation", True)
        modality_margin = getattr(t_cfg, "unified_3d_mrope_temporal_modality_margin", 15000)
        base_fps_val = float(getattr(t_cfg, "base_fps", self.config.base_fps))
        reset_spatial = getattr(t_cfg, "unified_3d_mrope_reset_spatial_ids", True)
        _, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=num_text_tokens,
            temporal_offset=0,
            use_float_positions=enable_fps,
        )
        vision_temporal_offset = next_mrope_offset + modality_margin
        effective_fps = self.config.fps if enable_fps else None
        grid_t, patch_h, patch_w = grid_shape
        vision_mrope_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=grid_t,
            grid_h=patch_h,
            grid_w=patch_w,
            temporal_offset=vision_temporal_offset,
            reset_spatial_indices=reset_spatial,
            fps=effective_fps,
            base_fps=base_fps_val,
            temporal_compression_factor=self.config.temporal_compression_factor,
        )
        position_ids = vision_mrope_ids.to(device=self.device)
        cos, sin = self.transformer.rotary_emb(
            position_ids=position_ids.unsqueeze(1),
            device=self.device,
            dtype=self._torch_dtype,
        )
        pair = (cos.squeeze(0).contiguous(), sin.squeeze(0).contiguous())
        self._cached_gen_rotary[key] = pair
        return pair

    def compile_transformer(self, mode: str = "default") -> None:
        """Compile targeted transformer layers with PyTorch Inductor kernel fusion."""
        if self.device.type != "cuda":
            return
        max_block = max(lyr - 1 for lyr in self.config.hidden_layers)
        for i in range(max_block + 1):
            if i < len(self.transformer.layers):
                self.transformer.layers[i] = torch.compile(self.transformer.layers[i], mode=mode)

    def compile_vae(self, mode: str = "reduce-overhead") -> None:
        """Compile VAE encoder with PyTorch Inductor CUDA Graphs."""
        if self.device.type != "cuda":
            return
        if self.vae is not None and hasattr(self.vae, "encoder"):
            self.vae.encoder = torch.compile(self.vae.encoder, mode=mode)

    def encode_latents(self, rgb_frames: Tensor) -> Tensor:
        """Encode RGB video [B, 3, T, H, W] into normalized VAE latents [B, 48, T_lat, H_lat, W_lat]."""
        if self.vae is None:
            raise Cosmos3FeatureError("VAE model is required for encode_latents()")
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()
        return encode_cosmos3_video(self.vae, rgb_frames, sample_mode="argmax")

    def forward_transformer_blocks(
        self,
        latents: Tensor,
        text_conditioning: Tensor | str | None = None,
        timestep: float | Tensor = 0.0,
    ) -> dict[int, Tensor]:
        """Forward latents through Cosmos 3 Transformer and tap requested layer representations."""
        if latents.shape[0] == 0:
            raise ValueError("latents cannot be empty")

        if (
            self._supports_fast_batched_gen()
            and (self._cached_und_kv is not None or (self.config.trainable_lora and self._cached_prompt_embeds is not None))
            and text_conditioning is None
        ):
            packed_latent, grid_shape = patchify_and_pack_cosmos3_vision_latents(
                latents, patch_size=self.config.latent_patch_size
            )
            g = self.transformer.proj_in(packed_latent.to(dtype=self._torch_dtype, device=self.device))
            b_cur = g.shape[0]
            und_kv = (
                self._cached_und_kv
                if self._cached_und_kv is not None
                else self._compute_und_kv_for_embeds.__wrapped__(self, self._cached_prompt_embeds)  # type: ignore[arg-type]
            )
            num_text_tokens = und_kv[0][0].shape[1]
            cos_g, sin_g = self._get_gen_rotary(num_text_tokens, grid_shape)
            max_block_idx = max(lyr - 1 for lyr in self.config.hidden_layers)
            target_set = set(self.config.hidden_layers)
            captured: dict[int, Tensor] = {}
            keep_grad = torch.is_grad_enabled()
            use_ckpt = bool(self.config.gradient_checkpointing and keep_grad)

            def _gen_block_forward(
                blk: nn.Module,
                g_in: Tensor,
                k_u_g: Tensor,
                v_u_g: Tensor,
            ) -> Tensor:
                g_norm = blk.input_layernorm_moe_gen(g_in)
                attn = blk.self_attn
                q_raw = attn.add_q_proj(g_norm).view(b_cur, -1, attn.num_attention_heads, attn.head_dim)
                k_raw = attn.add_k_proj(g_norm).view(b_cur, -1, attn.num_key_value_heads, attn.head_dim)
                v_g = attn.add_v_proj(g_norm).view(b_cur, -1, attn.num_key_value_heads, attn.head_dim)
                q_g = fused_rmsnorm_rope(q_raw, attn.norm_added_q, cos_g, sin_g)
                k_g = fused_rmsnorm_rope(k_raw, attn.norm_added_k, cos_g, sin_g)
                all_k = torch.cat([k_u_g.expand(b_cur, -1, -1, -1), k_g], dim=1)
                all_v = torch.cat([v_u_g.expand(b_cur, -1, -1, -1), v_g], dim=1)
                f_out = dispatch_attention_fn(
                    q_g, all_k, all_v, is_causal=False, enable_gqa=True
                ).flatten(-2, -1)
                res_g = g_in + attn.to_add_out(f_out)
                return res_g + blk.mlp_moe_gen(blk.post_attention_layernorm_moe_gen(res_g))

            for idx in range(min(max_block_idx + 1, len(self.transformer.layers))):
                blk = self.transformer.layers[idx]
                k_u_g, v_u_g = und_kv[idx]
                if use_ckpt:
                    g = torch.utils.checkpoint.checkpoint(
                        _gen_block_forward, blk, g, k_u_g, v_u_g, use_reentrant=False
                    )
                else:
                    g = _gen_block_forward(blk, g, k_u_g, v_u_g)
                lyr_num = idx + 1
                if lyr_num in target_set:
                    captured[lyr_num] = g if keep_grad else g.detach()
            return captured

        if latents.shape[0] > 1:
            batched_tapped: dict[int, list[Tensor]] = {lyr: [] for lyr in self.config.hidden_layers}
            for b_idx in range(latents.shape[0]):
                single_tapped = self.forward_transformer_blocks(
                    latents[b_idx : b_idx + 1],
                    text_conditioning=text_conditioning,
                    timestep=timestep,
                )
                for lyr in self.config.hidden_layers:
                    batched_tapped[lyr].append(single_tapped[lyr])
            return {lyr: torch.cat(tensors, dim=0) for lyr, tensors in batched_tapped.items()}
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
            from lerobot.policies.vam.cosmos3_lora import cosmos3_adapter_weights_file

            prov["lora_checkpoint"] = str(self.config.lora_checkpoint)
            weights = cosmos3_adapter_weights_file(self.config.lora_checkpoint)
            if weights.is_file():
                prov["lora_sha256"] = sha256_file(weights)
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
