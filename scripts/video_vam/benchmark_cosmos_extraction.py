#!/usr/bin/env python3
"""Benchmark Cosmos-2B feature extraction and optional SmolVLA decoding.

This benchmark keeps the historical comparison contract explicit: its default
``legacy_padded_vae`` arm zero-pads RGB ``[B, 3, 5, 480, 640]`` input to 61 frames,
then runs the VAE, DiT through layer 20, and production ``pool2`` transform. Pass
``--vae-input-mode observed_prefix`` to measure the new default path. Run it only
after sourcing ``cosmos_cuda_env.sh``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.policies.vam.context_transform import apply_context_transform
from lerobot.policies.vam.cosmos_predict2_extractor import (
    COSMOS_ATTENTION_BACKENDS,
    VAE_INPUT_MODE_LEGACY_PADDED,
    VAE_INPUT_MODES,
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
    arch_invariant_rand,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    SMOLVLA_CHECKPOINT,
    SMOLVLM_CONFIG,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)

DEFAULT_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_PROMPT = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/cosmos-extraction-benchmark")
DEFAULT_DECODER_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/runs/smolexpert-cosmos/train-converge/best.safetensors"
)
DEFAULT_NORMALIZER = Path(
    "/home/anton/.cache/video-vam/runs/smolexpert-cosmos/train-converge/normalizer.safetensors"
)
ARM_CHOICES = (
    "baseline",
    "compile",
    "compile_max",
    "compile_regional",
    "compile_regional_max",
    "fp8",
    "vae_prefix",
    "int8",
    "int8_no_compile",
    "int8_dynamic",
    "int8_dynamic_no_compile",
    "float8",
    "float8_no_compile",
    "cuda_graph",
    "decoder_sweep",
    "context_cache",
    "expert_compile",
    "all",
)
COMPILE_ARMS = ("compile", "compile_max", "compile_regional", "compile_regional_max")
QUANTIZATION_ARMS = (
    "int8",
    "int8_no_compile",
    "int8_dynamic",
    "int8_dynamic_no_compile",
    "float8",
    "float8_no_compile",
)
COMPILE_QUANTIZATION_ARMS = ("int8", "int8_dynamic", "float8")
# Keep the queued ``--arm all`` path limited to arms that are implemented and
# safe to run unattended. Rejected/experimental arms remain explicitly selectable.
ALL_ARMS = ("baseline", "compile", "compile_regional", "fp8", "vae_prefix")
DECODER_STEP_SWEEP = (10, 5, 3, 2, 1)
EXPERT_COMPILE_VARIANTS = (
    "eager_kv_cache",
    "compile_reduce_overhead",
    "compile_max_autotune",
    "cuda_graph",
)
EXPERT_COMPILE_STEPS = 10
EXPERT_COMPILE_FIVE_STEP_VARIANT = 5
EXPERT_COMPILE_MAX_ABS_TOLERANCE = 1e-2
# These are deliberately strict enough to reject a semantic change, while
# allowing at most a few BF16 round-off ulps in downstream tensors. The VAE
# latent threshold is much tighter because the causal prefix should execute
# the same operations as the first two frames of the padded stream.
CORRECTNESS_COSINE_THRESHOLD = 0.9999
VAE_PREFIX_TOLERANCES = {
    "latent_first_two": {"max_abs": 1e-4, "rmse": 2e-5, "cosine": 1.0 - 1e-7},
    "layer20_hidden_grid": {"max_abs": 1e-2, "rmse": 2e-3, "cosine": 1.0 - 1e-6},
    "pool2_features": {"max_abs": 1e-2, "rmse": 2e-3, "cosine": 1.0 - 1e-6},
    "actions": {"max_abs": 1e-4, "rmse": 2e-5, "cosine": 1.0 - 1e-7},
}
MEASURED_STAGE_NAMES = (
    "preprocessing",
    "vae_encode",
    "dit_forward_layer20",
    "feature_reduction_pool2",
)


class ArmUnavailableError(RuntimeError):
    """An optional optimization arm cannot run in this environment."""


class _TimedBackbone(nn.Module):
    """Synchronize around one backbone call and store elapsed seconds."""

    def __init__(self, module: nn.Module, device: torch.device, samples: list[float]) -> None:
        super().__init__()
        self.module = module
        self.device = device
        self.samples = samples

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        _synchronize(self.device)
        started = time.perf_counter()
        result = self.module(*args, **kwargs)
        _synchronize(self.device)
        self.samples.append(time.perf_counter() - started)
        return result


class _ContextBackbone(nn.Module):
    """Run a backbone call inside a supplied autocast-like context."""

    def __init__(self, module: nn.Module, context_factory: Callable[[], Any]) -> None:
        super().__init__()
        self.module = module
        self.context_factory = context_factory

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self.context_factory():
            return self.module(*args, **kwargs)


class _HiddenLayerDtypeBackbone(nn.Module):
    """Restore the extractor's BF16 hidden-state contract after torchao linears."""

    def __init__(self, module: nn.Module, *, hidden_layer: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.module = module
        self.hidden_layer = hidden_layer
        self.dtype = dtype

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        result = self.module(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != 2:
            return result
        prediction, hidden_states = result
        if not isinstance(hidden_states, (list, tuple)) or self.hidden_layer >= len(hidden_states):
            return result
        adapted = list(hidden_states)
        hidden = adapted[self.hidden_layer]
        if isinstance(hidden, torch.Tensor) and hidden.dtype != self.dtype:
            adapted[self.hidden_layer] = hidden.to(dtype=self.dtype)
        adapted_states = tuple(adapted) if isinstance(hidden_states, tuple) else adapted
        return prediction, adapted_states


class _CudaGraphBackbone(nn.Module):
    """Capture and replay one fixed-shape DiT call without Dynamo."""

    def __init__(self, module: nn.Module, *, warmups: int = 3) -> None:
        super().__init__()
        self.module = module
        self.warmups = warmups
        self.graph = torch.cuda.CUDAGraph()
        self.static_kwargs: dict[str, Any] | None = None
        self.static_output: Any = None
        self.capture_seconds: float | None = None

    def _capture(self, kwargs: dict[str, Any]) -> None:
        if self.static_kwargs is not None:
            return
        tensor_kwargs = {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in kwargs.items()
        }
        stream = torch.cuda.Stream(device=next(self.module.parameters()).device)
        started = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmups):
                self.module(**kwargs)
            stream.synchronize()
            self.static_kwargs = tensor_kwargs
            with torch.cuda.graph(self.graph, stream=stream):
                self.static_output = self.module(**self.static_kwargs)
        stream.synchronize()
        torch.cuda.synchronize()
        self.capture_seconds = time.perf_counter() - started

    def forward(self, **kwargs: Any) -> Any:
        self._capture(kwargs)
        if self.static_kwargs is None:
            raise RuntimeError("CUDA graph capture did not initialize static inputs")
        if self.capture_seconds is not None and self.graph is not None:
            for key, value in kwargs.items():
                static_value = self.static_kwargs[key]
                if isinstance(value, torch.Tensor):
                    static_value.copy_(value)
            self.graph.replay()
        return self.static_output


class _ExpertDenoiseLoop(nn.Module):
    """Unroll a fixed-step Smol expert denoise loop for compile or graph capture."""

    def __init__(self, decoder: SmolExpertActionDecoder, prefix: Any, num_steps: int) -> None:
        super().__init__()
        if num_steps <= 0:
            raise ValueError("expert denoise loop requires a positive step count")
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
            "action_std", decoder.normalizer.action_std.detach().to(device=parameter.device), persistent=False
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        batch_size = noise.shape[0]
        dt = -1.0 / self.num_steps
        x_t = noise
        for step in range(self.num_steps):
            current_time = self.time_values[step].expand(batch_size)
            velocity = self.decoder._vector_field(self.prefix, x_t, current_time)
            x_t = x_t + dt * velocity
        action = x_t[..., :ACTION_DIM]
        return (action * self.action_std.to(dtype=action.dtype) + self.action_mean.to(dtype=action.dtype)).to(
            dtype=torch.float32
        )


class _ExpertCudaGraph:
    """Capture and replay a complete fixed-shape expert denoise loop."""

    def __init__(self, module: nn.Module, *, warmups: int = 3) -> None:
        if not torch.cuda.is_available():
            raise ArmUnavailableError("expert CUDA graph capture requires CUDA")
        self.module = module
        self.warmups = warmups
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_noise: torch.Tensor | None = None
        self.static_output: torch.Tensor | None = None
        self.capture_seconds: float | None = None

    def _capture(self, noise: torch.Tensor) -> None:
        if self.static_noise is not None:
            return
        self.static_noise = noise.detach().clone()
        stream = torch.cuda.Stream(device=noise.device)
        started = time.perf_counter()
        with torch.cuda.device(noise.device):
            current_stream = torch.cuda.current_stream()
            stream.wait_stream(current_stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                for _ in range(self.warmups):
                    self.module(self.static_noise)
                stream.synchronize()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.static_output = self.module(self.static_noise)
            stream.synchronize()
        torch.cuda.synchronize(noise.device)
        self.capture_seconds = time.perf_counter() - started

    def __call__(self, noise: torch.Tensor) -> torch.Tensor:
        self._capture(noise)
        if self.static_noise is None or self.static_output is None or self.graph is None:
            raise RuntimeError("expert CUDA graph capture did not initialize static buffers")
        if noise.shape != self.static_noise.shape or noise.device != self.static_noise.device:
            raise ValueError("expert CUDA graph replay requires the captured noise shape and device")
        self.static_noise.copy_(noise)
        self.graph.replay()
        return self.static_output


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _dtype_name(value: torch.dtype | Any) -> str:
    return str(value).removeprefix("torch.")


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark inputs and arm selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=ARM_CHOICES,
        default="baseline",
        help="Optimization arm to measure; all runs only stable implemented arms.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=COSMOS_ATTENTION_BACKENDS,
        default="minimal_a2a",
        help="Vendored Cosmos attention dispatch used when constructing the backbone.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_DECODER_CHECKPOINT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument(
        "--expert-checkpoint",
        default=SMOLVLA_CHECKPOINT,
        help="Pretrained SmolVLA checkpoint used to construct the custom expert architecture.",
    )
    parser.add_argument(
        "--vlm-config",
        default=SMOLVLM_CONFIG,
        help="SmolVLM config used by the trainer's VLM-free expert constructor.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="Measured iterations (minimum: 20).")
    parser.add_argument("--warmups", type=int, default=3, help="Unmeasured warmup iterations (minimum: 3).")
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vae-input-mode",
        choices=VAE_INPUT_MODES,
        default=VAE_INPUT_MODE_LEGACY_PADDED,
        help=(
            "VAE input contract for benchmark arms (legacy_padded_vae preserves the historical "
            "baseline; use observed_prefix for the new default path)."
        ),
    )
    parser.add_argument("--decoder-steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-decoder",
        action="store_true",
        help="Measure only extraction and pool2; otherwise include the 30-action decoder chunk.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark for this process.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Convert padded [B,C,T,H,W] frames to channels_last_3d before VAE encoding.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate safety-critical benchmark settings before loading a model."""
    if args.arm not in ARM_CHOICES:
        raise ValueError(f"unknown arm {args.arm!r}")
    if args.vae_input_mode not in VAE_INPUT_MODES:
        raise ValueError(f"unknown VAE input mode {args.vae_input_mode!r}")
    if args.arm == "vae_prefix" and args.vae_input_mode != VAE_INPUT_MODE_LEGACY_PADDED:
        raise ValueError("--arm vae_prefix requires --vae-input-mode legacy_padded_vae")
    if args.arm == "expert_compile" and args.no_decoder:
        raise ValueError("--arm expert_compile requires the SmolVLA decoder")
    if args.arm == "expert_compile" and args.decoder_steps != EXPERT_COMPILE_STEPS:
        raise ValueError("--arm expert_compile requires --decoder-steps 10")
    if args.attention_backend not in COSMOS_ATTENTION_BACKENDS:
        raise ValueError(f"unknown attention backend {args.attention_backend!r}")
    if args.iterations < 20:
        raise ValueError("--iterations must be at least 20 for a stable p90")
    if args.warmups < 3:
        raise ValueError("--warmups must be at least 3")
    if args.episode < 0 or args.seed < 0:
        raise ValueError("--episode and --seed must be non-negative")
    if args.sigma <= 0:
        raise ValueError("--sigma must be positive")
    if args.decoder_steps <= 0:
        raise ValueError("--decoder-steps must be positive")
    if args.device != "cuda":
        raise ValueError("the real Cosmos benchmark requires --device cuda")


def summarize_samples(samples: Sequence[float]) -> dict[str, float | int]:
    """Return median/p90 milliseconds for a non-empty timing sample."""
    if not samples:
        raise ValueError("cannot summarize an empty timing sample")
    values = [float(value) for value in samples]
    if any(value < 0 for value in values):
        raise ValueError("timings cannot be negative")
    p90 = statistics.quantiles(values, n=100, method="inclusive")[89] if len(values) > 1 else values[0]
    return {
        "median_ms": 1000.0 * statistics.median(values),
        "p90_ms": 1000.0 * p90,
        "samples": len(values),
    }


def output_drift(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Return absolute and RMS output drift for two fixed decoder outputs."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"outputs must have equal shapes, got {tuple(reference.shape)} and {tuple(candidate.shape)}"
        )
    difference = candidate.float() - reference.float()
    if not torch.isfinite(difference).all().item():
        raise ValueError("output drift received non-finite decoder outputs")
    return {
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
    }


def tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    # Compare equal-shaped tensors, including a scale-independent cosine score.
    if reference.shape != candidate.shape:
        raise ValueError(
            f"tensors must have equal shapes, got {tuple(reference.shape)} and {tuple(candidate.shape)}"
        )
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    if not torch.isfinite(difference).all().item():
        raise ValueError("tensor comparison received non-finite values")
    reference_flat = reference_float.reshape(-1)
    candidate_flat = candidate_float.reshape(-1)
    # Accumulate the cosine in FP64 on the reporting side; a 10M-element FP32
    # reduction can round a materially different representation to exactly 1.
    reference_cosine = reference_flat.double()
    candidate_cosine = candidate_flat.double()
    denominator = reference_cosine.norm() * candidate_cosine.norm()
    cosine = (
        torch.ones((), device=reference.device, dtype=torch.float64)
        if denominator == 0
        else torch.dot(reference_cosine, candidate_cosine) / denominator
    )
    return {
        "shape": list(reference.shape),
        "max_abs": float(difference.abs().max().item()),
        "mean_abs": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "cosine": float(cosine.clamp(-1.0, 1.0).item()),
    }


def summarize_tensor_comparisons(samples: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    # Aggregate repeated tensor comparisons without hiding worst absolute drift.
    if not samples:
        raise ValueError("cannot summarize empty tensor comparisons")
    return {
        "max_abs": max(float(sample["max_abs"]) for sample in samples),
        "mean_abs": statistics.fmean(float(sample["mean_abs"]) for sample in samples),
        "rmse": statistics.fmean(float(sample["rmse"]) for sample in samples),
        "cosine": statistics.fmean(float(sample["cosine"]) for sample in samples),
        "min_cosine": min(float(sample["cosine"]) for sample in samples),
        "samples": len(samples),
    }


def _within_tolerance(metrics: Mapping[str, float | int], tolerance: Mapping[str, float]) -> bool:
    # Apply strict max/RMS/cosine acceptance limits to one comparison.
    return (
        metrics["max_abs"] <= tolerance["max_abs"]
        and metrics["rmse"] <= tolerance["rmse"]
        and metrics["min_cosine"] >= tolerance["cosine"]
    )


def _summarize_drift(samples: Sequence[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-call output drift without hiding the worst absolute error."""
    if not samples:
        raise ValueError("cannot summarize empty output-drift samples")
    return {
        "max_abs": max(sample["max_abs"] for sample in samples),
        "mean_abs": statistics.fmean(sample["mean_abs"] for sample in samples),
        "rmse": statistics.fmean(sample["rmse"] for sample in samples),
    }


def arms_for_request(arm: str) -> tuple[str, ...]:
    """Expand the CLI arm while keeping rejected arms out of ``all``."""
    if arm not in ARM_CHOICES:
        raise ValueError(f"unknown arm {arm!r}")
    return ALL_ARMS if arm == "all" else (arm,)


def optimization_inventory() -> dict[str, Any]:
    """Report optional optimization libraries without initializing a model."""
    te_version = _package_version("transformer-engine")
    fp8: dict[str, Any] = {
        "implemented": False,
        "package": "transformer-engine",
        "version": te_version,
        "path": "transformer_engine.pytorch.fp8_autocast",
    }
    if te_version is None:
        fp8["reason"] = "Transformer Engine is not installed."
    else:
        try:
            import transformer_engine.pytorch as te
            from transformer_engine.common.recipe import DelayedScaling, Format

            if not hasattr(te, "fp8_autocast"):
                raise ImportError("Transformer Engine has no fp8_autocast")
            DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
            fp8["implemented"] = True
            fp8["reason"] = "Transformer Engine FP8 autocast is available."
        except Exception as exc:  # noqa: BLE001 - report environment-specific import failures
            fp8["reason"] = f"Transformer Engine is installed but its FP8 API cannot initialize: {exc}"
    torchao_version = _package_version("torchao")
    bitsandbytes_version = _package_version("bitsandbytes")
    torchao_configs: list[str] = []
    int8_reason = "torchao is not installed"
    if torchao_version:
        try:
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                Int8DynamicActivationInt8WeightConfig,
                Int8WeightOnlyConfig,
            )

            del Float8DynamicActivationFloat8WeightConfig
            del Int8DynamicActivationInt8WeightConfig
            del Int8WeightOnlyConfig
            torchao_configs = [
                "Int8WeightOnlyConfig",
                "Int8DynamicActivationInt8WeightConfig",
                "Float8DynamicActivationFloat8WeightConfig",
            ]
            int8_reason = "torchao quantization configs are importable; arms are validated by this run"
        except Exception as exc:  # noqa: BLE001 - report optional dependency failures
            int8_reason = f"torchao is installed but quantization configs cannot import: {exc}"
    attention_backends = {
        "minimal_a2a": {
            "implemented": True,
            "description": "Vendored MinimalA2AAttnOp; dispatches the pinned SDPA attention helper.",
        },
        "torch": {
            "implemented": True,
            "description": "Vendored torch_attention_op using torch scaled_dot_product_attention.",
        },
        "transformer_engine": {
            "implemented": True,
            "description": "Vendored Transformer Engine DotProductAttention; requires Transformer Engine.",
        },
    }
    return {
        "vae_prefix": {
            "implemented": True,
            "benchmark_only": True,
            "description": "Compare observed_prefix with the explicit legacy_padded_vae 61-frame encode.",
            "production_default": "observed_prefix",
            "legacy_compatibility_mode": "legacy_padded_vae",
            "torch_compile_encode": "not enabled until correctness and graph-stability validation",
        },
        "attention_backends": attention_backends,
        "torch_compile": {
            "implemented": True,
            "arms": list(COMPILE_ARMS),
            "compile_friendly_model": (
                "native torch RMSNorm (opaque eager-math custom op) + native bshd RoPE + "
                "torch SDPA + SAC disabled"
            ),
        },
        "cuda_graph": {
            "implemented": True,
            "benchmark_arm": "cuda_graph",
            "strategy": "manual fixed-shape DiT capture; bypasses the vendored unsupported helper",
        },
        "expert_compile": {
            "implemented": True,
            "benchmark_arm": "expert_compile",
            "variants": list(EXPERT_COMPILE_VARIANTS),
            "cache_scope": "one fixed pool2 context/state pair per benchmark run",
            "correctness_max_abs_tolerance": EXPERT_COMPILE_MAX_ABS_TOLERANCE,
        },
        "fp8": fp8,
        "int8": {
            "implemented": bool(torchao_configs),
            "torchao_version": torchao_version,
            "bitsandbytes_version": bitsandbytes_version,
            "configs": torchao_configs,
            "reason": int8_reason,
            "compile_friendly_required": True,
            "cuda_validation": "per-arm correctness and timing report",
        },
    }


def _make_fp8_backbone(backbone: nn.Module) -> tuple[nn.Module, dict[str, Any]]:
    """Wrap the DiT in Transformer Engine's delayed-scaling FP8 autocast."""
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format

        fp8_autocast = te.fp8_autocast
        recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    except Exception as exc:  # noqa: BLE001 - convert optional dependency failures to arm status
        raise ArmUnavailableError(f"FP8 setup failed: {exc}") from exc
    wrapped = _ContextBackbone(
        backbone,
        lambda: fp8_autocast(enabled=True, fp8_recipe=recipe),
    )
    return wrapped, {
        "provider": "transformer_engine",
        "path": "transformer_engine.pytorch.fp8_autocast",
        "recipe": "DelayedScaling(HYBRID, amax_history_len=16, amax_compute_algo=max)",
    }


def _prepare_quantized_backbone(backbone: nn.Module, arm: str) -> tuple[nn.Module, dict[str, Any]]:
    """Apply a torchao quantizer to the compile-friendly pure-PyTorch DiT."""
    try:
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )

        config_by_arm = {
            "int8": ("Int8WeightOnlyConfig", Int8WeightOnlyConfig()),
            "int8_no_compile": ("Int8WeightOnlyConfig", Int8WeightOnlyConfig()),
            "int8_dynamic": (
                "Int8DynamicActivationInt8WeightConfig",
                Int8DynamicActivationInt8WeightConfig(),
            ),
            "int8_dynamic_no_compile": (
                "Int8DynamicActivationInt8WeightConfig",
                Int8DynamicActivationInt8WeightConfig(),
            ),
            "float8": (
                "Float8DynamicActivationFloat8WeightConfig",
                Float8DynamicActivationFloat8WeightConfig(),
            ),
            "float8_no_compile": (
                "Float8DynamicActivationFloat8WeightConfig",
                Float8DynamicActivationFloat8WeightConfig(),
            ),
        }
        config_name, config = config_by_arm[arm]
        quantize_(backbone, config)
    except Exception as exc:  # noqa: BLE001 - optional quantizer is machine-specific
        raise ArmUnavailableError(f"{arm} torchao quantization setup failed: {exc}") from exc

    metadata: dict[str, Any] = {
        "implemented": True,
        "component": "compile-friendly pure-PyTorch Cosmos DiT",
        "quantizer": "torchao",
        "config": config_name,
        "compiled": False,
        "output_dtype_adapter": "hidden layer 20 cast to bfloat16",
    }
    if arm in COMPILE_QUANTIZATION_ARMS:
        try:
            candidate = torch.compile(backbone, mode="max-autotune", fullgraph=True)
        except Exception as exc:  # noqa: BLE001 - backend availability is machine-specific
            raise ArmUnavailableError(f"{arm} torch.compile setup failed: {exc}") from exc
        metadata.update(
            {
                "compiled": True,
                "strategy": "full",
                "mode": "max-autotune",
                "fullgraph": True,
                "compile_friendly": True,
                "deferred_until_first_forward": True,
            }
        )
        return _HiddenLayerDtypeBackbone(candidate, hidden_layer=20, dtype=torch.bfloat16), metadata
    return _HiddenLayerDtypeBackbone(backbone, hidden_layer=20, dtype=torch.bfloat16), metadata


def _prepare_arm_backbone(backbone: nn.Module, arm: str) -> tuple[nn.Module, dict[str, Any]]:
    if arm == "baseline":
        return backbone, {"implemented": True, "description": "eager frozen Cosmos DiT"}
    if arm in COMPILE_ARMS:
        if not hasattr(torch, "compile"):
            raise ArmUnavailableError("torch.compile is unavailable in this PyTorch build")
        mode = {
            "compile": "reduce-overhead",
            "compile_max": "max-autotune",
            "compile_regional": "default",
            "compile_regional_max": "max-autotune-no-cudagraphs",
        }[arm]
        try:
            if arm in {"compile_regional", "compile_regional_max"}:
                if not hasattr(backbone, "blocks"):
                    raise ValueError("backbone has no transformer blocks")
                for index, block in enumerate(backbone.blocks):
                    backbone.blocks[index] = torch.compile(block, mode=mode, fullgraph=True)
                return backbone, {
                    "implemented": True,
                    "component": "transformer blocks",
                    "strategy": "regional",
                    "blocks": len(backbone.blocks),
                    "mode": mode,
                    "fullgraph": True,
                    "compile_friendly": True,
                    "deferred_until_first_forward": True,
                }
            compiled = torch.compile(backbone, mode=mode, fullgraph=True)
        except Exception as exc:  # noqa: BLE001 - backend availability is machine-specific
            raise ArmUnavailableError(f"torch.compile setup failed: {exc}") from exc
        return compiled, {
            "implemented": True,
            "component": "whole DiT",
            "strategy": "full",
            "mode": mode,
            "fullgraph": True,
            "compile_friendly": True,
            "deferred_until_first_forward": True,
        }
    if arm in QUANTIZATION_ARMS:
        return _prepare_quantized_backbone(backbone, arm)
    if arm == "cuda_graph":
        if not torch.cuda.is_available():
            raise ArmUnavailableError("CUDA graph capture requires CUDA")
        return _CudaGraphBackbone(backbone), {
            "implemented": True,
            "component": "DiT forward",
            "strategy": "manual_fixed_shape_capture",
            "warmups_on_side_stream": 3,
            "capture_deferred_until_first_warmup": True,
        }
    if arm == "fp8":
        return _make_fp8_backbone(backbone)
    if arm in QUANTIZATION_ARMS:
        raise ValueError(f"quantization arm {arm!r} must be prepared before this branch")
    if arm == "cuda_graph":
        raise ArmUnavailableError(
            "CUDA graph capture is unsupported by the vendored Cosmos path; this opt-in arm is not measured"
        )
    raise ValueError(f"arm {arm!r} must be resolved before preparing a backbone")


def _install_stage_timers(
    extractor: CosmosPredict2Extractor,
    backbone: nn.Module,
    device: torch.device,
    stage_samples: dict[str, list[float]],
    *,
    channels_last: bool,
) -> tuple[Callable[..., Any], Callable[..., Any], nn.Module]:
    """Install synchronized timers and return the original extractor members."""
    original_preprocess = extractor._preprocess_images
    original_encode = extractor._encode_conditional_latents
    extractor_patch: Any = extractor

    def timed_preprocess(images: torch.Tensor) -> torch.Tensor:
        _synchronize(device)
        started = time.perf_counter()
        result = original_preprocess(images)
        if channels_last:
            result = result.contiguous(memory_format=torch.channels_last_3d)
        _synchronize(device)
        stage_samples["preprocessing"].append(time.perf_counter() - started)
        return result

    def timed_encode(images: torch.Tensor) -> torch.Tensor:
        _synchronize(device)
        started = time.perf_counter()
        result = original_encode(images)
        _synchronize(device)
        stage_samples["vae_encode"].append(time.perf_counter() - started)
        return result

    timed_backbone = _TimedBackbone(backbone, device, stage_samples["dit_forward_layer20"])
    extractor_patch._preprocess_images = timed_preprocess
    extractor_patch._encode_conditional_latents = timed_encode
    extractor.backbone = timed_backbone
    return original_preprocess, original_encode, timed_backbone


def _measure_correctness(
    extractor: CosmosPredict2Extractor,
    candidate: nn.Module,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor,
    reference_context: torch.Tensor,
) -> dict[str, Any]:
    """Compare a candidate's pool2 output with one eager-TE reference."""
    original_backbone = extractor.backbone
    extractor.backbone = candidate
    try:
        with torch.inference_mode():
            extraction = extractor.extract(images, prompt_embedding, noise=noise, sigma=sigma)
        candidate_context = apply_context_transform(extraction.tokens, "pool2").detach().cpu()
    finally:
        extractor.backbone = original_backbone
    metrics = tensor_comparison(reference_context, candidate_context)
    metrics["threshold_cosine"] = CORRECTNESS_COSINE_THRESHOLD
    metrics["passed"] = bool(metrics["cosine"] > CORRECTNESS_COSINE_THRESHOLD)
    return metrics


def _run_one(
    extractor: CosmosPredict2Extractor,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
    stage_samples: dict[str, list[float]],
    end_to_end_samples: list[float],
    decoder_samples: list[float],
    noise_seed: int,
    measure: bool,
) -> None:
    """Run one extraction, pool2 reduction, and optional decoder chunk."""
    _synchronize(device)
    started = time.perf_counter()
    extraction = extractor.extract(
        images,
        prompt_embedding,
        noise_seed=noise_seed,
        sigma=sigma,
    )
    _synchronize(device)
    reduction_started = time.perf_counter()
    context = apply_context_transform(extraction.tokens, "pool2")
    _synchronize(device)
    stage_samples["feature_reduction_pool2"].append(time.perf_counter() - reduction_started)
    extraction_elapsed = time.perf_counter() - started
    if decoder is not None:
        if state is None or decoder_noise is None:
            raise ValueError("decoder timing requires state and decoder noise")
        _synchronize(device)
        decoder_started = time.perf_counter()
        decoder.sample_actions(state, context, noise=decoder_noise)
        _synchronize(device)
        decoder_samples.append(time.perf_counter() - decoder_started)
    _synchronize(device)
    full_elapsed = time.perf_counter() - started
    if measure:
        end_to_end_samples.append(extraction_elapsed)
        stage_samples.setdefault("full_end_to_end", []).append(full_elapsed)


def _measure_candidate(
    extractor: CosmosPredict2Extractor,
    candidate: nn.Module,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
    warmups: int,
    iterations: int,
    noise_seed: int,
    channels_last: bool,
) -> dict[str, Any]:
    stage_samples: dict[str, list[float]] = {name: [] for name in MEASURED_STAGE_NAMES}
    stage_samples["full_end_to_end"] = []
    end_to_end_samples: list[float] = []
    decoder_samples: list[float] = []
    extractor_patch: Any = extractor
    original_preprocess, original_encode, original_timed_backbone = _install_stage_timers(
        extractor,
        candidate,
        device,
        stage_samples,
        channels_last=channels_last,
    )
    try:
        for _ in range(warmups):
            _run_one(
                extractor,
                images,
                prompt_embedding,
                sigma,
                device=device,
                state=state,
                decoder=decoder,
                decoder_noise=decoder_noise,
                stage_samples=stage_samples,
                end_to_end_samples=end_to_end_samples,
                decoder_samples=decoder_samples,
                noise_seed=noise_seed,
                measure=False,
            )
        setup_seconds = (
            stage_samples["dit_forward_layer20"][0] if stage_samples["dit_forward_layer20"] else None
        )
        for samples in stage_samples.values():
            samples.clear()
        for _ in range(iterations):
            _run_one(
                extractor,
                images,
                prompt_embedding,
                sigma,
                device=device,
                state=state,
                decoder=decoder,
                decoder_noise=decoder_noise,
                stage_samples=stage_samples,
                end_to_end_samples=end_to_end_samples,
                decoder_samples=decoder_samples,
                noise_seed=noise_seed,
                measure=True,
            )
    finally:
        extractor_patch._preprocess_images = original_preprocess
        extractor_patch._encode_conditional_latents = original_encode
        extractor.backbone = original_timed_backbone.module
    stage_timings = {name: summarize_samples(stage_samples[name]) for name in MEASURED_STAGE_NAMES}
    extraction_timing = summarize_samples(end_to_end_samples)
    full_timing = summarize_samples(stage_samples["full_end_to_end"])
    decoder_timing = summarize_samples(decoder_samples) if decoder is not None else None
    return {
        "stage_timings_ms": stage_timings,
        "extraction_end_to_end_ms": extraction_timing,
        "decoder_included": decoder is not None,
        "decoder_inference_ms": decoder_timing,
        "full_end_to_end_ms": full_timing,
        "_setup_seconds": setup_seconds,
    }


def _assemble_full_condition(extractor: CosmosPredict2Extractor, latent: torch.Tensor) -> torch.Tensor:
    """Expand a tokenizer result into the extractor's fixed 16-frame tensor."""
    config = extractor.config
    expected_prefix = (
        latent.shape[0],
        config.latent_channels,
        config.latent_conditional_frames,
        config.latent_height,
        config.latent_width,
    )
    expected_full = (
        latent.shape[0],
        config.latent_channels,
        config.state_t,
        config.latent_height,
        config.latent_width,
    )
    if tuple(latent.shape) == expected_full:
        return latent
    if tuple(latent.shape) != expected_prefix:
        raise ValueError(
            f"tokenizer result must have shape {expected_prefix} or {expected_full}, got {tuple(latent.shape)}"
        )
    full = torch.zeros(expected_full, device=latent.device, dtype=latent.dtype)
    full[:, :, : config.latent_conditional_frames] = latent
    return full


def _timed_tokenizer_encode(
    extractor: CosmosPredict2Extractor,
    video: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    _synchronize(device)
    started = time.perf_counter()
    tokenizer = extractor.tokenizer
    assert tokenizer is not None
    latent = tokenizer.encode(video)
    _synchronize(device)
    return latent, time.perf_counter() - started


def _extract_with_condition(
    extractor: CosmosPredict2Extractor,
    images: torch.Tensor,
    prepared_images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    condition: torch.Tensor,
    noise: torch.Tensor,
) -> Any:
    """Run the production extractor with a precomputed condition tensor."""
    original_preprocess = extractor._preprocess_images
    original_encode = extractor._encode_conditional_latents
    extractor_patch: Any = extractor
    extractor_patch._preprocess_images = lambda _images: prepared_images
    extractor_patch._encode_conditional_latents = lambda _images: condition
    try:
        return extractor.extract(images, prompt_embedding, noise_seed=0, sigma=sigma, noise=noise)
    finally:
        extractor_patch._preprocess_images = original_preprocess
        extractor_patch._encode_conditional_latents = original_encode


def _run_conditioned_path(
    extractor: CosmosPredict2Extractor,
    images: torch.Tensor,
    prepared_images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    condition: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
) -> dict[str, Any]:
    """Run one fixed condition through layer 20, pool2, and the decoder."""
    path_started = time.perf_counter()
    _synchronize(device)
    dit_started = time.perf_counter()
    extraction = _extract_with_condition(
        extractor,
        images,
        prepared_images,
        prompt_embedding,
        sigma,
        condition,
        noise,
    )
    _synchronize(device)
    dit_elapsed = time.perf_counter() - dit_started
    pool_started = time.perf_counter()
    context = apply_context_transform(extraction.tokens, "pool2")
    _synchronize(device)
    pool_elapsed = time.perf_counter() - pool_started
    decoder_elapsed = None
    action = None
    if decoder is not None:
        if state is None or decoder_noise is None:
            raise ValueError("decoder timing requires state and decoder noise")
        _synchronize(device)
        decoder_started = time.perf_counter()
        action = decoder.sample_actions(state, context, noise=decoder_noise)
        _synchronize(device)
        decoder_elapsed = time.perf_counter() - decoder_started
    _synchronize(device)
    path_elapsed = time.perf_counter() - path_started
    return {
        "extraction": extraction,
        "context": context,
        "action": action,
        "dit_elapsed": dit_elapsed,
        "pool_elapsed": pool_elapsed,
        "decoder_elapsed": decoder_elapsed,
        "extraction_elapsed": path_elapsed if decoder_elapsed is None else path_elapsed - decoder_elapsed,
        "full_elapsed": path_elapsed,
    }


def _run_vae_prefix_once(
    extractor: CosmosPredict2Extractor,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
    noise: torch.Tensor,
    channels_last: bool,
    timings: dict[str, list[float]],
    comparisons: dict[str, list[dict[str, Any]]],
    measure: bool,
) -> None:
    """Measure padded/prefix VAE paths and compare every downstream result."""
    _synchronize(device)
    preprocess_started = time.perf_counter()
    prepared_images = extractor._preprocess_images(images)
    if channels_last:
        prepared_images = prepared_images.contiguous(memory_format=torch.channels_last_3d)
    _synchronize(device)
    preprocess_elapsed = time.perf_counter() - preprocess_started

    padded_path_started = time.perf_counter()
    padded_latent, padded_vae_elapsed = _timed_tokenizer_encode(extractor, prepared_images, device)
    padded_condition = _assemble_full_condition(extractor, padded_latent)
    padded_path = _run_conditioned_path(
        extractor,
        images,
        prepared_images,
        prompt_embedding,
        sigma,
        padded_condition,
        noise,
        device=device,
        state=state,
        decoder=decoder,
        decoder_noise=decoder_noise,
    )
    padded_path["full_elapsed"] = time.perf_counter() - padded_path_started
    padded_path["extraction_elapsed"] = (
        padded_path["full_elapsed"]
        if padded_path["decoder_elapsed"] is None
        else padded_path["full_elapsed"] - padded_path["decoder_elapsed"]
    )

    prefix_images = prepared_images[:, :, : extractor.config.input_frames]
    if channels_last:
        prefix_images = prefix_images.contiguous(memory_format=torch.channels_last_3d)
    prefix_path_started = time.perf_counter()
    prefix_latent, prefix_vae_elapsed = _timed_tokenizer_encode(extractor, prefix_images, device)
    prefix_condition = _assemble_full_condition(extractor, prefix_latent)
    prefix_path = _run_conditioned_path(
        extractor,
        images,
        prepared_images,
        prompt_embedding,
        sigma,
        prefix_condition,
        noise,
        device=device,
        state=state,
        decoder=decoder,
        decoder_noise=decoder_noise,
    )
    prefix_path["full_elapsed"] = time.perf_counter() - prefix_path_started
    prefix_path["extraction_elapsed"] = (
        prefix_path["full_elapsed"]
        if prefix_path["decoder_elapsed"] is None
        else prefix_path["full_elapsed"] - prefix_path["decoder_elapsed"]
    )

    if not measure:
        return
    timings["preprocessing"].append(preprocess_elapsed)
    timings["padded_vae_encode"].append(padded_vae_elapsed)
    timings["prefix_vae_encode"].append(prefix_vae_elapsed)
    timings["padded_layer20"].append(padded_path["dit_elapsed"])
    timings["prefix_layer20"].append(prefix_path["dit_elapsed"])
    timings["padded_pool2"].append(padded_path["pool_elapsed"])
    timings["prefix_pool2"].append(prefix_path["pool_elapsed"])
    timings["padded_extraction"].append(preprocess_elapsed + padded_path["extraction_elapsed"])
    timings["prefix_extraction"].append(preprocess_elapsed + prefix_path["extraction_elapsed"])
    timings["padded_full"].append(preprocess_elapsed + padded_path["full_elapsed"])
    timings["prefix_full"].append(preprocess_elapsed + prefix_path["full_elapsed"])
    if decoder is not None:
        timings["padded_decoder"].append(float(padded_path["decoder_elapsed"]))
        timings["prefix_decoder"].append(float(prefix_path["decoder_elapsed"]))

    comparisons["latent_first_two"].append(
        tensor_comparison(padded_latent[:, :, : extractor.config.latent_conditional_frames], prefix_latent)
    )
    comparisons["layer20_hidden_grid"].append(
        tensor_comparison(padded_path["extraction"].hidden_grid, prefix_path["extraction"].hidden_grid)
    )
    comparisons["pool2_features"].append(tensor_comparison(padded_path["context"], prefix_path["context"]))
    if decoder is not None:
        comparisons["actions"].append(tensor_comparison(padded_path["action"], prefix_path["action"]))


def _measure_vae_prefix(
    extractor: CosmosPredict2Extractor,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
    warmups: int,
    iterations: int,
    noise_seed: int,
    channels_last: bool,
) -> dict[str, Any]:
    """Benchmark 61-frame versus five-frame VAE encoding on identical inputs."""
    config = extractor.config
    full_shape = (
        images.shape[0],
        config.latent_channels,
        config.state_t,
        config.latent_height,
        config.latent_width,
    )
    noise = arch_invariant_rand(full_shape, noise_seed).to(device=device, dtype=extractor.dtype)
    timing_names = (
        "preprocessing",
        "padded_vae_encode",
        "prefix_vae_encode",
        "padded_layer20",
        "prefix_layer20",
        "padded_pool2",
        "prefix_pool2",
        "padded_extraction",
        "prefix_extraction",
        "padded_full",
        "prefix_full",
        "padded_decoder",
        "prefix_decoder",
    )
    timings: dict[str, list[float]] = {name: [] for name in timing_names}
    comparisons: dict[str, list[dict[str, Any]]] = {
        "latent_first_two": [],
        "layer20_hidden_grid": [],
        "pool2_features": [],
        "actions": [],
    }
    for _ in range(warmups):
        _run_vae_prefix_once(
            extractor,
            images,
            prompt_embedding,
            sigma,
            device=device,
            state=state,
            decoder=decoder,
            decoder_noise=decoder_noise,
            noise=noise,
            channels_last=channels_last,
            timings=timings,
            comparisons=comparisons,
            measure=False,
        )
    for samples in timings.values():
        samples.clear()
    for comparison_samples in comparisons.values():
        comparison_samples.clear()
    for _ in range(iterations):
        _run_vae_prefix_once(
            extractor,
            images,
            prompt_embedding,
            sigma,
            device=device,
            state=state,
            decoder=decoder,
            decoder_noise=decoder_noise,
            noise=noise,
            channels_last=channels_last,
            timings=timings,
            comparisons=comparisons,
            measure=True,
        )

    prefix_stage_timings = {
        "preprocessing": summarize_samples(timings["preprocessing"]),
        "vae_encode": summarize_samples(timings["prefix_vae_encode"]),
        "dit_forward_layer20": summarize_samples(timings["prefix_layer20"]),
        "feature_reduction_pool2": summarize_samples(timings["prefix_pool2"]),
    }
    decoder_timing = summarize_samples(timings["prefix_decoder"]) if decoder is not None else None
    latent_first_two_summary = summarize_tensor_comparisons(comparisons["latent_first_two"])
    layer20_summary = summarize_tensor_comparisons(comparisons["layer20_hidden_grid"])
    pool2_summary = summarize_tensor_comparisons(comparisons["pool2_features"])
    actions_summary = summarize_tensor_comparisons(comparisons["actions"]) if decoder is not None else None
    compared_outputs = [
        "latent_first_two",
        "layer20_hidden_grid",
        "pool2_features",
        *(["actions"] if decoder is not None else []),
    ]
    comparison_summaries = {
        "latent_first_two": latent_first_two_summary,
        "layer20_hidden_grid": layer20_summary,
        "pool2_features": pool2_summary,
    }
    if actions_summary is not None:
        comparison_summaries["actions"] = actions_summary
    comparison_report = {
        "padded_pixel_shape": [
            images.shape[0],
            images.shape[1],
            config.video_frames,
            images.shape[3],
            images.shape[4],
        ],
        "prefix_pixel_shape": [
            images.shape[0],
            images.shape[1],
            config.input_frames,
            images.shape[3],
            images.shape[4],
        ],
        "padded_latent_shape": [
            images.shape[0],
            config.latent_channels,
            config.state_t,
            config.latent_height,
            config.latent_width,
        ],
        "prefix_latent_shape": [
            images.shape[0],
            config.latent_channels,
            config.latent_conditional_frames,
            config.latent_height,
            config.latent_width,
        ],
        "timings_ms": {
            "padded_61_frame_vae_encode": summarize_samples(timings["padded_vae_encode"]),
            "prefix_observed_vae_encode": summarize_samples(timings["prefix_vae_encode"]),
            "padded_layer20_forward": summarize_samples(timings["padded_layer20"]),
            "prefix_layer20_forward": summarize_samples(timings["prefix_layer20"]),
            "padded_pool2": summarize_samples(timings["padded_pool2"]),
            "prefix_pool2": summarize_samples(timings["prefix_pool2"]),
            "padded_extraction_end_to_end": summarize_samples(timings["padded_extraction"]),
            "prefix_extraction_end_to_end": summarize_samples(timings["prefix_extraction"]),
            "padded_full_end_to_end": summarize_samples(timings["padded_full"]),
            "prefix_full_end_to_end": summarize_samples(timings["prefix_full"]),
            "padded_decoder": summarize_samples(timings["padded_decoder"]) if decoder is not None else None,
            "prefix_decoder": summarize_samples(timings["prefix_decoder"]) if decoder is not None else None,
        },
        "vae_median_speedup": float(statistics.median(timings["padded_vae_encode"]))
        / float(statistics.median(timings["prefix_vae_encode"])),
        "extraction_median_speedup": float(statistics.median(timings["padded_extraction"]))
        / float(statistics.median(timings["prefix_extraction"])),
        "full_median_speedup": float(statistics.median(timings["padded_full"]))
        / float(statistics.median(timings["prefix_full"])),
        "latent_first_two": {
            **latent_first_two_summary,
            "padded_first_two_shape": comparisons["latent_first_two"][0]["shape"],
            "prefix_shape": [
                images.shape[0],
                config.latent_channels,
                config.latent_conditional_frames,
                config.latent_height,
                config.latent_width,
            ],
        },
        "layer20_hidden_grid": layer20_summary,
        "pool2_features": pool2_summary,
        "actions": actions_summary,
        "strict_tolerances": VAE_PREFIX_TOLERANCES,
        "compared_outputs": compared_outputs,
    }
    equivalence = {
        name: _within_tolerance(comparison_summaries[name], VAE_PREFIX_TOLERANCES[name])
        for name in compared_outputs
    }
    comparison_report["equivalence_by_output"] = equivalence
    # A decoder-free probe cannot authorize a production representation change;
    # action equivalence must be measured whenever the decoder is available.
    comparison_report["equivalent_under_strict_tolerances"] = decoder is not None and all(
        equivalence.values()
    )
    comparison_report["production_path"] = (
        "safe_opt_in_candidate"
        if comparison_report["equivalent_under_strict_tolerances"]
        else ("requires_decoder_validation" if decoder is None else "do_not_implement")
    )
    return {
        "stage_timings_ms": prefix_stage_timings,
        "extraction_end_to_end_ms": summarize_samples(timings["prefix_extraction"]),
        "decoder_included": decoder is not None,
        "decoder_inference_ms": decoder_timing,
        "full_end_to_end_ms": summarize_samples(timings["prefix_full"]),
        "vae_prefix_comparison": comparison_report,
    }


def _measure_decoder_step_sweep(
    decoder: SmolExpertActionDecoder,
    state: torch.Tensor,
    context: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure SmolVLA Euler-step alternatives against the fixed 10-step output."""
    timings: dict[int, list[float]] = {num_steps: [] for num_steps in DECODER_STEP_SWEEP}
    drift_samples: dict[int, list[dict[str, float]]] = {num_steps: [] for num_steps in DECODER_STEP_SWEEP}
    for _ in range(warmups):
        for num_steps in DECODER_STEP_SWEEP:
            decoder.sample_actions(state, context, noise=noise, num_steps=num_steps)
    output_shape: list[int] | None = None
    for _ in range(iterations):
        reference: torch.Tensor | None = None
        for num_steps in DECODER_STEP_SWEEP:
            _synchronize(device)
            started = time.perf_counter()
            output = decoder.sample_actions(state, context, noise=noise, num_steps=num_steps)
            _synchronize(device)
            timings[num_steps].append(time.perf_counter() - started)
            output_shape = list(output.shape)
            if num_steps == 10:
                reference = output.detach()
            elif reference is not None:
                drift_samples[num_steps].append(output_drift(reference, output))
    if output_shape is None:
        raise ValueError("decoder step sweep produced no outputs")
    return {
        "reference_steps": 10,
        "output_shape": output_shape,
        "steps": [
            {
                "num_steps": num_steps,
                "timing_ms": summarize_samples(timings[num_steps]),
                "output_drift_vs_10": (
                    {"max_abs": 0.0, "mean_abs": 0.0, "rmse": 0.0}
                    if num_steps == 10
                    else _summarize_drift(drift_samples[num_steps])
                ),
            }
            for num_steps in DECODER_STEP_SWEEP
        ],
    }


def _measure_context_kv_cache(
    decoder: SmolExpertActionDecoder,
    state: torch.Tensor,
    context: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Compare reusable per-layer prefix K/V against the exact uncached path."""
    _synchronize(device)
    cache_started = time.perf_counter()
    prefix_kv_cache = decoder.prepare_prefix_kv(state, context)
    _synchronize(device)
    cache_preparation_ms = 1000.0 * (time.perf_counter() - cache_started)

    for _ in range(warmups):
        decoder.sample_actions(state, context, noise=noise)
        decoder.sample_actions(state, context, noise=noise, prefix_kv_cache=prefix_kv_cache)

    uncached_samples: list[float] = []
    cached_samples: list[float] = []
    drift_samples: list[dict[str, float]] = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        uncached_output = decoder.sample_actions(state, context, noise=noise)
        _synchronize(device)
        uncached_samples.append(time.perf_counter() - started)

        _synchronize(device)
        started = time.perf_counter()
        cached_output = decoder.sample_actions(state, context, noise=noise, prefix_kv_cache=prefix_kv_cache)
        _synchronize(device)
        cached_samples.append(time.perf_counter() - started)
        drift_samples.append(output_drift(uncached_output, cached_output))

    uncached_timing = summarize_samples(uncached_samples)
    cached_timing = summarize_samples(cached_samples)
    return {
        "decoder_steps": decoder.num_steps,
        "layers_cached": len(prefix_kv_cache.keys),
        "cache_scope": "one fixed state/context pair across all measured denoising calls",
        "cache_preparation_ms": cache_preparation_ms,
        "uncached_decoder_inference_ms": uncached_timing,
        "cached_decoder_inference_ms": cached_timing,
        "median_speedup": float(uncached_timing["median_ms"]) / float(cached_timing["median_ms"]),
        "output_drift_vs_uncached": _summarize_drift(drift_samples),
    }


def _build_expert_runner(
    decoder: SmolExpertActionDecoder,
    state: torch.Tensor,
    context: torch.Tensor,
    prefix: Any,
    noise_steps: int,
    mode: str,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], str | None]:
    """Build one eager, compiled, or manual-graph fixed-step expert runner."""
    if mode == "eager":
        return (
            lambda noise: decoder.sample_actions(
                state, context, noise=noise, prefix_kv_cache=prefix, num_steps=noise_steps
            ),
            None,
        )
    loop = _ExpertDenoiseLoop(decoder, prefix, noise_steps)
    if mode in {"compile_reduce_overhead", "compile_max_autotune"}:
        compile_mode = {
            "compile_reduce_overhead": "reduce-overhead",
            "compile_max_autotune": "max-autotune",
        }[mode]
        return torch.compile(loop, mode=compile_mode, fullgraph=True), "compile"
    if mode == "cuda_graph":
        return _ExpertCudaGraph(loop), "cuda_graph"
    raise ValueError(f"unknown expert compile runner mode: {mode!r}")


def _measure_expert_compile_variant(
    runner: Callable[[torch.Tensor], torch.Tensor],
    noise: torch.Tensor,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
    setup_kind: str | None,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Measure one fixed expert runner after synchronized warmups."""
    warmup_elapsed: list[float] = []
    output: torch.Tensor | None = None
    for _ in range(warmups):
        _synchronize(device)
        started = time.perf_counter()
        output = runner(noise)
        _synchronize(device)
        warmup_elapsed.append(time.perf_counter() - started)
    samples: list[float] = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter()
        output = runner(noise)
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    if output is None:
        raise ValueError("expert compile runner produced no output")
    setup_seconds = None
    if setup_kind == "compile":
        setup_seconds = warmup_elapsed[0]
    elif setup_kind == "cuda_graph":
        setup_seconds = getattr(runner, "capture_seconds", None)
    return {
        "timing_ms": summarize_samples(samples),
        "setup_seconds": setup_seconds,
    }, output.detach().clone()


def _measure_expert_compile(
    decoder: SmolExpertActionDecoder,
    state: torch.Tensor,
    context: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Compare eager/cache, compile, and full-loop CUDA graph expert inference."""
    expected_context_shape = (1, 4_800, 2_048)
    if tuple(context.shape) != expected_context_shape:
        raise ValueError(
            f"expert compile context must have shape {expected_context_shape}, got {tuple(context.shape)}"
        )
    if state.shape[0] != 1 or state.shape[-1] != 6:
        raise ValueError(
            f"expert compile state must have shape [1, 6] or [1, 1, 6], got {tuple(state.shape)}"
        )
    if tuple(noise.shape) != (1, 30, decoder.max_action_dim):
        raise ValueError(
            f"expert compile noise must have shape [1, 30, {decoder.max_action_dim}], got {tuple(noise.shape)}"
        )

    _synchronize(device)
    cache_started = time.perf_counter()
    prefix = decoder.prepare_prefix_kv(state, context)
    _synchronize(device)
    cache_preparation_ms = 1000.0 * (time.perf_counter() - cache_started)
    variant_specs = (
        ("eager_kv_cache", "eager", {"strategy": "eager", "kv_cache": True}),
        (
            "compile_reduce_overhead",
            "compile_reduce_overhead",
            {"strategy": "torch.compile", "mode": "reduce-overhead", "fullgraph": True, "kv_cache": True},
        ),
        (
            "compile_max_autotune",
            "compile_max_autotune",
            {"strategy": "torch.compile", "mode": "max-autotune", "fullgraph": True, "kv_cache": True},
        ),
        (
            "cuda_graph",
            "cuda_graph",
            {"strategy": "manual_cuda_graph", "full_loop_capture": True, "kv_cache": True},
        ),
    )
    variants: list[dict[str, Any]] = []
    outputs: dict[str, torch.Tensor] = {}
    runners: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
    with torch.inference_mode():
        for name, mode, optimization in variant_specs:
            try:
                runner, setup_kind = _build_expert_runner(
                    decoder, state, context, prefix, EXPERT_COMPILE_STEPS, mode
                )
                runners[name] = runner
                measured, output = _measure_expert_compile_variant(
                    runner,
                    noise,
                    device=device,
                    warmups=warmups,
                    iterations=iterations,
                    setup_kind=setup_kind,
                )
                record = {
                    "name": name,
                    "status": "ok",
                    "optimization": optimization,
                    "timing_ms": measured["timing_ms"],
                }
                if measured["setup_seconds"] is not None:
                    record["setup_seconds"] = measured["setup_seconds"]
                if not isinstance(output, torch.Tensor):
                    raise TypeError("expert runner returned no tensor output")
                variants.append(record)
                outputs[name] = output
            except Exception as exc:  # noqa: BLE001 - keep one unavailable optimization from hiding others
                variants.append(
                    {
                        "name": name,
                        "status": "unavailable",
                        "optimization": optimization,
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

    baseline_record = next(item for item in variants if item["name"] == "eager_kv_cache")
    if baseline_record["status"] != "ok":
        raise ArmUnavailableError(f"eager expert baseline failed: {baseline_record['message']}")
    baseline_output = outputs["eager_kv_cache"]
    for record in variants:
        output = outputs.get(record["name"])
        if output is None:
            continue
        drift = output_drift(baseline_output, output)
        drift["threshold_max_abs"] = EXPERT_COMPILE_MAX_ABS_TOLERANCE
        drift["passed"] = bool(drift["max_abs"] < EXPERT_COMPILE_MAX_ABS_TOLERANCE)
        record["correctness_vs_eager_kv_cache"] = drift

    passing = [
        record
        for record in variants
        if record["status"] == "ok" and record.get("correctness_vs_eager_kv_cache", {}).get("passed", False)
    ]
    winner_record = min(passing, key=lambda item: float(item["timing_ms"]["median_ms"]))
    winner_name = winner_record["name"]
    mode_by_name = {name: mode for name, mode, _ in variant_specs}
    five_step_report: dict[str, Any]
    try:
        winner_mode = mode_by_name[winner_name]
        five_runner, five_setup_kind = _build_expert_runner(
            decoder, state, context, prefix, EXPERT_COMPILE_FIVE_STEP_VARIANT, winner_mode
        )
        runners[f"{winner_name}_5_steps"] = five_runner
        with torch.inference_mode():
            eager_five_runner, _ = _build_expert_runner(
                decoder, state, context, prefix, EXPERT_COMPILE_FIVE_STEP_VARIANT, "eager"
            )
            eager_five_output = eager_five_runner(noise).detach().clone()
            five_measured, five_output = _measure_expert_compile_variant(
                five_runner,
                noise,
                device=device,
                warmups=warmups,
                iterations=iterations,
                setup_kind=five_setup_kind,
            )
        five_vs_eager = output_drift(eager_five_output, five_output)
        five_vs_eager["threshold_max_abs"] = EXPERT_COMPILE_MAX_ABS_TOLERANCE
        five_vs_eager["passed"] = bool(five_vs_eager["max_abs"] < EXPERT_COMPILE_MAX_ABS_TOLERANCE)
        five_step_report = {
            "status": "ok",
            "winner_variant": winner_name,
            "timing_ms": five_measured["timing_ms"],
            "setup_seconds": five_measured["setup_seconds"],
            "correctness_vs_eager_5_steps": five_vs_eager,
            "output_drift_vs_eager_10_steps": output_drift(baseline_output, five_output),
        }
    except Exception as exc:  # noqa: BLE001 - preserve the ten-step result if the optional probe fails
        five_step_report = {
            "status": "unavailable",
            "winner_variant": winner_name,
            "message": f"{type(exc).__name__}: {exc}",
        }

    production_api = _measure_production_expert_api(
        decoder, state, context, noise, device=device, warmups=warmups, iterations=iterations
    )
    graph_record = next(item for item in variants if item["name"] == "cuda_graph")
    graph_decision = {
        "requested": True,
        "decision": "captured" if graph_record["status"] == "ok" else "not_feasible",
        "reason": (
            "fixed [1, 30, 32] noise, [1, 4801, 8, 40] per-layer cache, and 10-step loop are static"
            if graph_record["status"] == "ok"
            else graph_record.get("message", "capture failed")
        ),
    }
    return {
        "context_shape": list(context.shape),
        "state_shape": list(state.shape),
        "noise_shape": list(noise.shape),
        "decoder_steps": EXPERT_COMPILE_STEPS,
        "cache_layers": len(prefix.keys),
        "cache_preparation_ms": cache_preparation_ms,
        "correctness_max_abs_tolerance": EXPERT_COMPILE_MAX_ABS_TOLERANCE,
        "variants": variants,
        "winner_variant": winner_name,
        "winner_timing_ms": winner_record["timing_ms"],
        "five_step_winner": five_step_report,
        "cuda_graph_decision": graph_decision,
        "production_api": production_api,
    }


def _measure_production_expert_api(
    decoder: SmolExpertActionDecoder,
    state: torch.Tensor,
    context: torch.Tensor,
    noise: torch.Tensor,
    *,
    device: torch.device,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure the public ``sample_actions`` API with fresh graph input copies."""
    _synchronize(device)
    cache_started = time.perf_counter()
    prefix = decoder.prepare_prefix_kv(state, context)
    _synchronize(device)
    cache_preparation_ms = 1000.0 * (time.perf_counter() - cache_started)

    def run(use_cuda_graph: bool) -> torch.Tensor:
        return decoder.sample_actions(
            state,
            context,
            noise=noise,
            prefix_kv_cache=prefix,
            num_steps=EXPERT_COMPILE_STEPS,
            use_cuda_graph=use_cuda_graph,
        )

    def measure(use_cuda_graph: bool) -> tuple[dict[str, Any], torch.Tensor]:
        for _ in range(warmups):
            _synchronize(device)
            output = run(use_cuda_graph)
            _synchronize(device)
        samples: list[float] = []
        for _ in range(iterations):
            _synchronize(device)
            started = time.perf_counter()
            output = run(use_cuda_graph)
            _synchronize(device)
            samples.append(time.perf_counter() - started)
        timing = summarize_samples(samples)
        combined = {
            "median_ms": cache_preparation_ms + float(timing["median_ms"]),
            "p90_ms": cache_preparation_ms + float(timing["p90_ms"]),
            "samples": timing["samples"],
        }
        return {"timing_ms": timing, "combined_with_kv_preparation_ms": combined}, output.detach().clone()

    with torch.inference_mode():
        eager_measured, eager_output = measure(False)
        graph_measured, graph_output = measure(True)

        state_second = state + 0.5
        context_second = context + torch.tensor(0.125, device=context.device, dtype=context.dtype)
        noise_second = -noise
        second_prefix = decoder.prepare_prefix_kv(state_second, context_second)
        eager_second = decoder.sample_actions(
            state_second,
            context_second,
            noise=noise_second,
            prefix_kv_cache=second_prefix,
            num_steps=EXPERT_COMPILE_STEPS,
            use_cuda_graph=False,
        )
        graph_second = decoder.sample_actions(
            state_second,
            context_second,
            noise=noise_second,
            prefix_kv_cache=second_prefix,
            num_steps=EXPERT_COMPILE_STEPS,
            use_cuda_graph=True,
        )
    graph_active = decoder.cuda_graph_capture_count > 0 and not decoder._cuda_graph_disabled
    return {
        "cache_preparation_ms": cache_preparation_ms,
        "eager_kv": {
            **eager_measured,
            "status": "ok",
            "use_cuda_graph": False,
        },
        "cuda_graph": {
            **graph_measured,
            "status": "ok" if graph_active else "fallback_eager",
            "use_cuda_graph": True,
            "capture_seconds": decoder.last_cuda_graph_capture_seconds,
        },
        "correctness_two_observations": {
            "first": output_drift(eager_output, graph_output),
            "second": output_drift(eager_second, graph_second),
            "outputs_differ": bool(not torch.equal(eager_output, eager_second)),
            "eager_observation_delta": output_drift(eager_output, eager_second),
        },
        "fresh_buffer_copy_scope": "noise plus all per-layer prefix K/V tensors copied before every graph replay",
    }


def _latency_metrics(result: dict[str, Any]) -> dict[str, float | None]:
    extraction_ms = float(result["extraction_end_to_end_ms"]["median_ms"])
    full_ms = float(result["full_end_to_end_ms"]["median_ms"])
    decoder = result.get("decoder_inference_ms")
    decoder_ms = float(decoder["median_ms"]) if decoder else None
    return {
        "extraction_hz": 1000.0 / extraction_ms,
        "decoder_hz": None if decoder_ms is None else 1000.0 / decoder_ms,
        "chunk_hz_30_steps": 30_000.0 / full_ms,
        "chunk_hz_30_steps_p90": 30_000.0 / float(result["full_end_to_end_ms"]["p90_ms"]),
    }


def _success_result(
    requested_arm: str,
    effective_arm: str,
    arm_metadata: dict[str, Any],
    measured: dict[str, Any],
    *,
    extractor_load_seconds: float,
    decoder_load_seconds: float | None,
    dtypes: dict[str, str],
    message: str | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    result = {
        "arm": requested_arm,
        "effective_arm": effective_arm,
        "status": status,
        "message": message,
        "optimization": arm_metadata,
        "extractor_load_seconds": extractor_load_seconds,
        "decoder_load_seconds": decoder_load_seconds,
        "dtypes": dtypes,
        **measured,
    }
    result["latency"] = _latency_metrics(result)
    return result


def _unavailable_result(
    arm: str,
    status: str,
    message: str,
    *,
    optimization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "effective_arm": None,
        "status": status,
        "message": message,
        "optimization": optimization or {"implemented": False},
        "extractor_load_seconds": None,
        "decoder_load_seconds": None,
        "dtypes": {},
    }


def build_output_payload(
    *,
    arm_results: list[dict[str, Any]],
    hardware: dict[str, Any],
    configuration: dict[str, Any],
    input_metadata: dict[str, Any],
    optimization_inventory_data: dict[str, Any],
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the stable, JSON-safe benchmark output schema."""
    payload = {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "hardware": hardware,
        "configuration": configuration,
        "input": input_metadata,
        "optimization_inventory": optimization_inventory_data,
        "arms": arm_results,
    }
    validate_output_payload(payload)
    return payload


def validate_output_payload(payload: dict[str, Any]) -> None:
    """Validate the fields consumed by the markdown renderer and downstream analysis."""
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark output schema_version must be 1")
    if not isinstance(payload.get("arms"), list) or not payload["arms"]:
        raise ValueError("benchmark output must contain at least one arm")
    required = {"arm", "effective_arm", "status", "message", "optimization", "dtypes"}
    for arm in payload["arms"]:
        missing = required - set(arm)
        if missing:
            raise ValueError(f"benchmark arm is missing fields: {sorted(missing)}")
        if arm["status"] in {"ok", "fallback_baseline"}:
            for field in (
                "stage_timings_ms",
                "extraction_end_to_end_ms",
                "full_end_to_end_ms",
                "latency",
            ):
                if field not in arm:
                    raise ValueError(f"measured arm is missing {field!r}")


def _timing_cell(arm: dict[str, Any], field: str) -> str:
    timing: Any = arm
    for part in field.split("."):
        if not isinstance(timing, dict):
            return "—"
        timing = timing.get(part)
    if not timing:
        return "—"
    return f"{float(timing['median_ms']):.2f} / {float(timing['p90_ms']):.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact summary table from the JSON payload."""
    validate_output_payload(payload)
    hardware = payload["hardware"]
    configuration = payload["configuration"]
    lines = [
        "# Cosmos-2B feature-extraction latency benchmark",
        "",
        f"Generated: `{payload['generated_at_utc']}`  ",
        f"GPU: `{hardware.get('gpu_name', 'unknown')}`; PyTorch `{hardware.get('torch_version', 'unknown')}`; "
        f"CUDA `{hardware.get('cuda_version', 'unknown')}`  ",
        f"Iterations: `{configuration.get('iterations')}` after `{configuration.get('warmups')}` warmups; "
        "timings are median / p90 milliseconds.",
        "",
        "The extraction end-to-end column includes preprocessing, VAE, DiT, and production pool2 reduction. "
        "Full end-to-end includes the optional 30-action SmolVLA expert chunk.",
        "",
        "| Arm | Status | Preprocess | VAE | DiT layer 20 | Pool2 | Extraction E2E | Decoder | Full E2E | 30-step Hz |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        latency = arm.get("latency", {})
        hz = latency.get("chunk_hz_30_steps")
        hz_cell = "—" if hz is None else f"{float(hz):.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(arm["arm"]),
                    str(arm["status"]),
                    _timing_cell(arm, "stage_timings_ms.preprocessing"),
                    _timing_cell(arm, "stage_timings_ms.vae_encode"),
                    _timing_cell(arm, "stage_timings_ms.dit_forward_layer20"),
                    _timing_cell(arm, "stage_timings_ms.feature_reduction_pool2"),
                    _timing_cell(arm, "extraction_end_to_end_ms"),
                    _timing_cell(arm, "decoder_inference_ms"),
                    _timing_cell(arm, "full_end_to_end_ms"),
                    hz_cell,
                ]
            )
            + " |"
        )
        if arm.get("message"):
            lines.append(f"\n`{arm['arm']}`: {arm['message']}\n")
        correctness = arm.get("correctness_vs_eager_te")
        if correctness:
            lines.append(
                f"`{arm['arm']}` pool2 vs eager TE: cosine {float(correctness['cosine']):.7f}, "
                f"max abs {float(correctness['max_abs']):.6g}, passed={correctness['passed']}\n"
            )
        optimization = arm.get("optimization", {})
        compile_seconds = optimization.get("compile_warmup_seconds")
        if compile_seconds is not None:
            lines.append(
                f"`{arm['arm']}` first-forward compile: {float(compile_seconds):.2f} s"
                + (
                    f"; mode={optimization['mode']}, strategy={optimization['strategy']}\n"
                    if optimization.get("mode") and optimization.get("strategy")
                    else "\n"
                )
            )
        capture_seconds = optimization.get("capture_seconds")
        if capture_seconds is not None:
            lines.append(f"`{arm['arm']}` manual CUDA graph capture: {float(capture_seconds):.2f} s\n")
        expert_compile = arm.get("expert_compile")
        if expert_compile:
            lines.append(
                f"`expert_compile` fixed pool2/state decoder benchmark; cache preparation: "
                f"{float(expert_compile['cache_preparation_ms']):.2f} ms\n"
            )
            for variant in expert_compile["variants"]:
                line = f"`{variant['name']}` status={variant['status']}"
                if variant.get("timing_ms"):
                    line += (
                        f", p50/p90={float(variant['timing_ms']['median_ms']):.2f}/"
                        f"{float(variant['timing_ms']['p90_ms']):.2f} ms"
                    )
                if variant.get("correctness_vs_eager_kv_cache"):
                    correctness = variant["correctness_vs_eager_kv_cache"]
                    line += (
                        f", drift max abs={float(correctness['max_abs']):.6g}, passed={correctness['passed']}"
                    )
                if variant.get("message"):
                    line += f": {variant['message']}"
                lines.append(line + "\n")
            lines.append(
                f"`expert_compile` winner: `{expert_compile['winner_variant']}`; "
                f"5-step status={expert_compile['five_step_winner']['status']}\n"
            )
            graph_decision = expert_compile["cuda_graph_decision"]
            lines.append(
                f"`expert_compile` CUDA graph decision: `{graph_decision['decision']}` — "
                f"{graph_decision['reason']}\n"
            )
            production = expert_compile.get("production_api")
            if production:
                for name in ("eager_kv", "cuda_graph"):
                    variant = production[name]
                    lines.append(
                        f"production `{name}` ({variant['status']}): "
                        f"{float(variant['timing_ms']['median_ms']):.2f}/"
                        f"{float(variant['timing_ms']['p90_ms']):.2f} ms; "
                        f"KV+denoise {float(variant['combined_with_kv_preparation_ms']['median_ms']):.2f}/"
                        f"{float(variant['combined_with_kv_preparation_ms']['p90_ms']):.2f} ms\n"
                    )
                two_obs = production["correctness_two_observations"]
                lines.append(
                    f"production two-observation drift: "
                    f"{float(two_obs['first']['max_abs']):.6g}, "
                    f"{float(two_obs['second']['max_abs']):.6g}; "
                    f"outputs_differ={two_obs['outputs_differ']}\n"
                )
        sweep = arm.get("decoder_step_sweep")
        if sweep:
            summary = ", ".join(
                f"{item['num_steps']} steps: {float(item['timing_ms']['median_ms']):.2f} ms "
                f"(drift RMSE {float(item['output_drift_vs_10']['rmse']):.6g})"
                for item in sweep["steps"]
            )
            lines.append(f"`{arm['arm']}` decoder sweep: {summary}\n")
        cache = arm.get("context_kv_cache")
        if cache:
            lines.append(
                f"`{arm['arm']}` cached/uncached decoder median: "
                f"{float(cache['cached_decoder_inference_ms']['median_ms']):.2f} / "
                f"{float(cache['uncached_decoder_inference_ms']['median_ms']):.2f} ms; "
                f"drift RMSE: {float(cache['output_drift_vs_uncached']['rmse']):.6g}\n"
            )
        vae_prefix = arm.get("vae_prefix_comparison")
        if vae_prefix:
            timings = vae_prefix["timings_ms"]
            lines.append(
                f"`{arm['arm']}` VAE p50/p90 (61 frames -> 5 frames): "
                f"{float(timings['padded_61_frame_vae_encode']['median_ms']):.2f}/"
                f"{float(timings['padded_61_frame_vae_encode']['p90_ms']):.2f} -> "
                f"{float(timings['prefix_observed_vae_encode']['median_ms']):.2f}/"
                f"{float(timings['prefix_observed_vae_encode']['p90_ms']):.2f} ms; "
                f"speedup {float(vae_prefix['vae_median_speedup']):.2f}x; "
                f"equivalence={vae_prefix['equivalent_under_strict_tolerances']}\n"
            )
    lines.extend(
        [
            "",
            "## Runtime notes",
            "",
            f"- Extractor dtype: `{payload['input'].get('extractor_dtype', 'unknown')}`; "
            f"input: `{payload['input'].get('rgb_shape')}`; pool2 output: `{payload['input'].get('pool2_shape')}`.",
            f"- cudnn benchmark: `{configuration.get('cudnn_benchmark')}`; channels-last: `{configuration.get('channels_last')}`.",
            f"- Cosmos attention backend: `{configuration.get('attention_backend', 'unknown')}`; "
            f"CUDA graphs requested: `{configuration.get('cuda_graphs_requested', False)}`.",
            f"- Benchmark VAE input mode: `{configuration.get('vae_input_mode', 'legacy_padded_vae')}`; "
            "new online/cache extraction defaults to `observed_prefix`, while "
            "`legacy_padded_vae` preserves the historical 61-frame input.",
            "- Int8/Float8 arms use torchao on the compile-friendly pure-PyTorch DiT; unavailable or failed "
            "variants remain explicitly reported in the JSON optimization inventory.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and Markdown benchmark reports."""
    validate_output_payload(payload)
    output_dir.expanduser().mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "cosmos_extraction_benchmark.json"
    markdown_path = output_dir / "cosmos_extraction_benchmark.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(payload))
    return json_path, markdown_path


def _load_decoder(args: argparse.Namespace, device: torch.device) -> tuple[SmolExpertActionDecoder, float]:
    from safetensors.torch import load_file

    if not args.decoder_checkpoint.is_file():
        raise FileNotFoundError(f"decoder checkpoint not found: {args.decoder_checkpoint}")
    if not args.normalizer.is_file():
        raise FileNotFoundError(f"decoder normalizer not found: {args.normalizer}")
    normalizer_tensors = load_file(str(args.normalizer), device="cpu")
    expected_normalizer_keys = {"state_mean", "state_std", "action_mean", "action_std"}
    if set(normalizer_tensors) != expected_normalizer_keys:
        raise ValueError(
            "SmolVLA normalizer tensor keys mismatch: "
            f"missing={sorted(expected_normalizer_keys - set(normalizer_tensors))}, "
            f"unexpected={sorted(set(normalizer_tensors) - expected_normalizer_keys)}"
        )
    normalizer = SmolVLANormalizer(**normalizer_tensors)
    started = time.perf_counter()
    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        args.decoder_checkpoint,
        normalizer=normalizer,
        expert_checkpoint=args.expert_checkpoint,
        vlm_config_name=args.vlm_config,
        device=device,
        num_steps=args.decoder_steps,
    )
    _synchronize(device)
    decoder.eval()
    return decoder, time.perf_counter() - started


def _load_extractor(
    args: argparse.Namespace,
    device: torch.device,
    *,
    use_cuda_graphs: bool = False,
    compile_friendly: bool = False,
) -> tuple[CosmosPredict2Extractor, float]:
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=args.checkpoint.expanduser(),
        tokenizer_path=args.tokenizer.expanduser(),
        device=str(device),
        dtype="bfloat16",
        high_noise_sigma=args.sigma,
        seed=args.seed,
        hidden_layer=20,
        stop_after_step=0,
        attention_backend=args.attention_backend,
        compile_friendly=compile_friendly,
        use_cuda_graphs=use_cuda_graphs,
        vae_input_mode=args.vae_input_mode,
    )
    _synchronize(device)
    started = time.perf_counter()
    extractor = CosmosPredict2Extractor(config)
    _synchronize(device)
    return extractor, time.perf_counter() - started


def _measure_arm(
    args: argparse.Namespace,
    arm: str,
    *,
    device: torch.device,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    state: torch.Tensor | None,
    decoder: SmolExpertActionDecoder | None,
    decoder_noise: torch.Tensor | None,
    decoder_load_seconds: float | None,
    reference_context: torch.Tensor | None,
    correctness_noise: torch.Tensor,
) -> dict[str, Any]:
    if arm in {"decoder_sweep", "context_cache", "expert_compile"} and (
        state is None or decoder is None or decoder_noise is None
    ):
        return _unavailable_result(
            arm,
            "unavailable",
            f"{arm} requires the SmolVLA decoder; remove --no-decoder and provide decoder artifacts.",
        )

    if arm == "expert_compile":
        assert decoder is not None and state is not None and decoder_noise is not None
        if reference_context is None:
            raise ArmUnavailableError("expert_compile requires the fixed eager pool2 context")
        measured = _measure_expert_compile(
            decoder,
            state,
            reference_context.to(device=device, dtype=torch.bfloat16),
            decoder_noise,
            device=device,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        winner_timing = measured["winner_timing_ms"]
        winner_median = float(winner_timing["median_ms"])
        winner_p90 = float(winner_timing["p90_ms"])
        return {
            "arm": arm,
            "effective_arm": arm,
            "status": "ok",
            "message": "decoder-only fixed pool2/state benchmark; Cosmos extraction omitted",
            "optimization": {
                "implemented": True,
                "component": "SmolVLA action decoder",
                "fixed_context": True,
                "variants": list(EXPERT_COMPILE_VARIANTS),
            },
            "extractor_load_seconds": None,
            "decoder_load_seconds": decoder_load_seconds,
            "dtypes": {"context": "bfloat16", "decoder": _dtype_name(next(decoder.parameters()).dtype)},
            "stage_timings_ms": {},
            "extraction_end_to_end_ms": None,
            "decoder_inference_ms": winner_timing,
            "full_end_to_end_ms": winner_timing,
            "latency": {
                "extraction_hz": None,
                "decoder_hz": 1000.0 / winner_median,
                "chunk_hz_30_steps": 30_000.0 / winner_median,
                "chunk_hz_30_steps_p90": 30_000.0 / winner_p90,
            },
            "expert_compile": measured,
        }

    extractor, extractor_load_seconds = _load_extractor(
        args,
        device,
        compile_friendly=arm in COMPILE_ARMS or arm in QUANTIZATION_ARMS,
    )
    if arm == "vae_prefix":
        measured = _measure_vae_prefix(
            extractor,
            images,
            prompt_embedding,
            sigma,
            device=device,
            state=state,
            decoder=decoder,
            decoder_noise=decoder_noise,
            warmups=args.warmups,
            iterations=args.iterations,
            noise_seed=args.seed,
            channels_last=args.channels_last,
        )
        return _success_result(
            arm,
            arm,
            {
                "implemented": True,
                "component": "Cosmos VAE encode",
                "padded_input": "61 frames",
                "prefix_input": "5 observed frames",
                "production_default": False,
                "production_path_enabled": False,
            },
            measured,
            extractor_load_seconds=extractor_load_seconds,
            decoder_load_seconds=decoder_load_seconds,
            dtypes=_runtime_dtypes(extractor, decoder),
        )
    original_backbone = extractor.backbone
    try:
        if arm in {"decoder_sweep", "context_cache"}:
            candidate = original_backbone
            arm_metadata = {
                "implemented": True,
                "component": "SmolVLA action decoder",
                "backbone_arm": "baseline",
            }
        else:
            try:
                candidate, arm_metadata = _prepare_arm_backbone(original_backbone, arm)
            except ArmUnavailableError:
                raise
        try:
            measured = _measure_candidate(
                extractor,
                candidate,
                images,
                prompt_embedding,
                sigma,
                device=device,
                state=state,
                decoder=decoder,
                decoder_noise=decoder_noise,
                warmups=args.warmups,
                iterations=args.iterations,
                noise_seed=args.seed,
                channels_last=args.channels_last,
            )
        except Exception as exc:
            raise ArmUnavailableError(f"{arm} arm failed during warmup/measurement: {exc}") from exc
        setup_seconds = measured.pop("_setup_seconds", None)
        if arm in COMPILE_ARMS or arm in COMPILE_QUANTIZATION_ARMS:
            arm_metadata["compile_warmup_seconds"] = setup_seconds
        if isinstance(candidate, _CudaGraphBackbone):
            arm_metadata["capture_seconds"] = candidate.capture_seconds
        if reference_context is not None:
            measured["correctness_vs_eager_te"] = _measure_correctness(
                extractor,
                candidate,
                images,
                prompt_embedding,
                sigma,
                correctness_noise,
                reference_context,
            )
        result = _success_result(
            arm,
            arm,
            arm_metadata,
            measured,
            extractor_load_seconds=extractor_load_seconds,
            decoder_load_seconds=decoder_load_seconds,
            dtypes=_runtime_dtypes(extractor, decoder),
        )
        if arm in {"decoder_sweep", "context_cache"}:
            assert decoder is not None and state is not None and decoder_noise is not None
            extraction = extractor.extract(
                images,
                prompt_embedding,
                noise_seed=args.seed,
                sigma=sigma,
            )
            context = apply_context_transform(extraction.tokens, "pool2")
            if arm == "decoder_sweep":
                result["decoder_step_sweep"] = _measure_decoder_step_sweep(
                    decoder,
                    state,
                    context,
                    decoder_noise,
                    device=device,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
            else:
                result["context_kv_cache"] = _measure_context_kv_cache(
                    decoder,
                    state,
                    context,
                    decoder_noise,
                    device=device,
                    warmups=args.warmups,
                    iterations=args.iterations,
                )
        return result
    finally:
        extractor.backbone = original_backbone


def _runtime_dtypes(
    extractor: CosmosPredict2Extractor,
    decoder: SmolExpertActionDecoder | None,
) -> dict[str, str]:
    dtypes = {"extractor": _dtype_name(extractor.dtype)}
    if decoder is not None:
        dtypes["decoder"] = _dtype_name(next(decoder.parameters()).dtype)
    return dtypes


def _load_eager_te_reference(
    args: argparse.Namespace,
    device: torch.device,
    images: torch.Tensor,
    prompt_embedding: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Load one eager TE model and retain only its pool2 reference on CPU."""
    extractor, load_seconds = _load_extractor(args, device, compile_friendly=False)
    with torch.inference_mode():
        extraction = extractor.extract(images, prompt_embedding, noise=noise, sigma=sigma)
    reference_context = apply_context_transform(extraction.tokens, "pool2").detach().cpu()
    del extraction, extractor
    gc.collect()
    torch.cuda.empty_cache()
    _synchronize(device)
    return reference_context, load_seconds


def benchmark(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    """Run selected arms and write reports; returns payload and error indicator."""
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script never runs the real benchmark on CPU")
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

    from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
    from scripts.video_vam.smoke_test_cosmos_extractor import load_real_sample

    _, prepared = load_real_sample(
        root=args.root.expanduser(),
        episode_index=args.episode,
        frame_index=args.frame_index,
        window_start=args.window_start,
    )
    prompt_artifact = load_prompt_embedding(args.prompt.expanduser())
    prompt_embedding = prompt_artifact.embedding
    images = prepared.rgb_history
    state = prepared.state.to(device=device, dtype=torch.float32) if not args.no_decoder else None
    sigma = torch.full((1,), args.sigma, device=device, dtype=torch.float32)
    correctness_noise = arch_invariant_rand((images.shape[0], 16, 16, 60, 80), args.seed).to(
        device=device, dtype=torch.bfloat16
    )
    reference_context, reference_load_seconds = _load_eager_te_reference(
        args,
        device,
        images,
        prompt_embedding,
        sigma,
        correctness_noise,
    )
    print(f"Eager-TE correctness reference loaded in {reference_load_seconds:.2f}s", flush=True)
    decoder: SmolExpertActionDecoder | None = None
    decoder_load_seconds: float | None = None
    decoder_noise: torch.Tensor | None = None
    if not args.no_decoder:
        decoder, decoder_load_seconds = _load_decoder(args, device)
        decoder_dtype = next(decoder.parameters()).dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        decoder_noise = torch.randn(
            (1, 30, decoder.max_action_dim),
            device=device,
            dtype=decoder_dtype,
            generator=generator,
        )

    inventory = optimization_inventory()
    inventory["attention_backend_selected"] = args.attention_backend
    selected_arms = arms_for_request(args.arm)
    results: list[dict[str, Any]] = []
    had_error = False
    for arm in selected_arms:
        print(f"\n=== arm={arm} ===", flush=True)
        try:
            result = _measure_arm(
                args,
                arm,
                device=device,
                images=images,
                prompt_embedding=prompt_embedding,
                sigma=sigma,
                state=state,
                decoder=decoder,
                decoder_noise=decoder_noise,
                decoder_load_seconds=decoder_load_seconds,
                reference_context=reference_context,
                correctness_noise=correctness_noise,
            )
        except ArmUnavailableError as exc:
            message = str(exc)
            print(f"WARNING: {message}", flush=True)
            result = _unavailable_result(arm, "unavailable", message)
        except Exception as exc:  # noqa: BLE001 - preserve other arm reports in --arm all
            had_error = True
            message = f"{type(exc).__name__}: {exc}"
            print(f"ERROR: {message}", flush=True)
            result = _unavailable_result(arm, "error", message)
        results.append(result)
        if result["status"] in {"ok", "fallback_baseline"}:
            if result.get("extraction_end_to_end_ms") is None:
                print(
                    f"{arm}: decoder winner median={result['decoder_inference_ms']['median_ms']:.2f} ms, "
                    f"30-step Hz={result['latency']['chunk_hz_30_steps']:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"{arm}: extraction median={result['extraction_end_to_end_ms']['median_ms']:.2f} ms, "
                    f"full median={result['full_end_to_end_ms']['median_ms']:.2f} ms, "
                    f"30-step Hz={result['latency']['chunk_hz_30_steps']:.2f}",
                    flush=True,
                )

    hardware = {
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda or "unavailable",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": str(device),
    }
    input_metadata = {
        "rgb_shape": list(images.shape),
        "rgb_dtype": _dtype_name(images.dtype),
        "prompt_shape": list(prompt_embedding.shape),
        "prompt_dtype": _dtype_name(prompt_embedding.dtype),
        "extractor_dtype": "bfloat16",
        "attention_backend": args.attention_backend,
        "hidden_layer": 20,
        "video_frames": 61,
        "pool2_shape": [1, 4800, 2048],
        "sigma": args.sigma,
        "vae_input_mode": args.vae_input_mode,
        "semantic_contract": (
            "5 observed frames -> 2 latent prefix -> zero-expand to 16 -> VAE/DiT layer 20 -> pool2"
            if args.vae_input_mode == "observed_prefix"
            else "5 observed frames -> 61 padded frames -> VAE -> DiT layer 20 -> pool2"
        ),
    }
    configuration = {
        "arm_requested": args.arm,
        "arms_measured": list(selected_arms),
        "iterations": args.iterations,
        "warmups": args.warmups,
        "seed": args.seed,
        "sigma": args.sigma,
        "decoder_included": not args.no_decoder,
        "decoder_steps": None if args.no_decoder else args.decoder_steps,
        "decoder_checkpoint": None if args.no_decoder else str(args.decoder_checkpoint.expanduser()),
        "normalizer": None if args.no_decoder else str(args.normalizer.expanduser()),
        "expert_checkpoint": None if args.no_decoder else args.expert_checkpoint,
        "vlm_config": None if args.no_decoder else args.vlm_config,
        "context_transform": "pool2",
        "vae_input_mode": args.vae_input_mode,
        "normalization": "SmolVLA MEAN_STD from train episodes 0-31",
        "cudnn_benchmark": bool(args.cudnn_benchmark),
        "channels_last": bool(args.channels_last),
        "attention_backend": args.attention_backend,
        "compile_friendly_requested": args.arm in COMPILE_ARMS or args.arm in QUANTIZATION_ARMS,
        "cuda_graphs_requested": args.arm == "cuda_graph",
        "decoder_step_sweep": list(DECODER_STEP_SWEEP) if args.arm == "decoder_sweep" else None,
        "expert_compile_variants": list(EXPERT_COMPILE_VARIANTS) if args.arm == "expert_compile" else None,
        "decoder_load_seconds": decoder_load_seconds,
        "synchronization": "torch.cuda.synchronize before and after each stage and end-to-end interval",
    }
    payload = build_output_payload(
        arm_results=results,
        hardware=hardware,
        configuration=configuration,
        input_metadata=input_metadata,
        optimization_inventory_data=inventory,
    )
    json_path, markdown_path = write_outputs(args.output_dir.expanduser().resolve(), payload)
    print(f"\nJSON report: {json_path}", flush=True)
    print(f"Markdown report: {markdown_path}", flush=True)
    return payload, had_error


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        _, had_error = benchmark(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 2 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
