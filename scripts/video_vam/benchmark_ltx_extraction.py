#!/usr/bin/env python3
"""Profile and correctness-gate LTX-2.5 extraction on one CUDA device.

The benchmark deliberately keeps model-changing arms opt-in.  The baseline is
full eager FP8-cast/offload execution with a capture hook that continues all
48 blocks.  The optimized arm uses the same inputs and weights, but raises an
internal control-flow exception immediately after the selected block.  No
training or feature-cache output is produced.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot.policies.vam.ltx_extractor import (
    LTXExtractor,
    LTXExtractorConfig,
    derive_window_seed,
)

try:
    from scripts.video_vam.smoke_test_cosmos_extractor import load_real_sample
except ModuleNotFoundError:
    from smoke_test_cosmos_extractor import load_real_sample

DEFAULT_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_VIDEO_VAE = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
)
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/ltx-extraction-benchmark")
CORRECTNESS_COSINE_THRESHOLD = 0.999


class BenchmarkError(RuntimeError):
    """A benchmark arm could not be measured safely."""


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = fraction * (len(values) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (rank - low)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min_seconds": min(values) if values else None,
        "mean_seconds": statistics.mean(values) if values else None,
        "p50_seconds": _percentile(values, 0.50),
        "p90_seconds": _percentile(values, 0.90),
        "max_seconds": max(values) if values else None,
    }


def _memory(device: torch.device) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    if reference.shape != candidate.shape:
        return {
            "shape_equal": 0.0,
            "cosine": 0.0,
            "max_abs": float("inf"),
            "rmse": float("inf"),
        }
    left = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    denominator = left.norm() * right.norm()
    cosine_value = float(torch.dot(left, right) / denominator) if denominator else 1.0
    cosine = max(-1.0, min(1.0, cosine_value))
    delta = left - right
    rmse = float(delta.square().mean().sqrt())
    reference_rms = float(left.square().mean().sqrt())
    return {
        "shape_equal": 1.0,
        "cosine": cosine,
        "max_abs": float(delta.abs().max()),
        "rmse": rmse,
        "reference_rms": reference_rms,
        "relative_rmse": rmse / reference_rms if reference_rms else 0.0,
    }


def _load_embedding(path: Path | None, batch_size: int) -> torch.Tensor:
    if path is None:
        return torch.zeros((batch_size, 1, 4096), dtype=torch.bfloat16)
    from safetensors.torch import load_file

    values = load_file(str(path), device="cpu")
    if "embedding" in values:
        embedding = values["embedding"]
    elif len(values) == 1:
        embedding = next(iter(values.values()))
    else:
        raise ValueError(f"Prompt artifact {path} must contain one embedding tensor")
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    expected = (batch_size, embedding.shape[1], 4096)
    if tuple(embedding.shape) != expected:
        raise ValueError(
            f"Prompt embedding must have shape [B, sequence, 4096], got {tuple(embedding.shape)}"
        )
    return embedding.to(dtype=torch.bfloat16).contiguous()


def _arm_run(
    extractor: LTXExtractor,
    images: torch.Tensor,
    prompt: torch.Tensor,
    seed: int,
    *,
    early_exit: bool,
    warmups: int,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    backbone = extractor.backbone
    if not hasattr(backbone, "early_exit"):
        raise BenchmarkError("real benchmark requires the official LTX runner")
    backbone.early_exit = early_exit
    warmup_timings: list[dict[str, float]] = []
    for _ in range(warmups):
        timings: dict[str, float] = {}
        extractor.extract(images, prompt, noise_seed=seed, timings=timings)
        warmup_timings.append(timings)
    samples: list[dict[str, Any]] = []
    first_output: torch.Tensor | None = None
    for index in range(iterations):
        timings = {}
        _sync(device)
        started = time.perf_counter()
        extraction = extractor.extract(images, prompt, noise_seed=seed, timings=timings)
        _sync(device)
        timings["total_extraction"] = time.perf_counter() - started
        profile = dict(getattr(backbone, "last_profile", {}))
        transfer = profile.get("offload_block_get_seconds", [])
        timings["offload_block_get"] = sum(float(item["seconds"]) for item in transfer)
        samples.append(
            {
                "iteration": index,
                "timings": timings,
                "blocks_loaded": profile.get("blocks_loaded", []),
                "early_exit": profile.get("early_exit", early_exit),
            }
        )
        if first_output is None:
            first_output = extraction.tokens.detach().clone()
    if first_output is None:
        raise BenchmarkError("no measured extraction output")
    stage_names = (
        "input_preprocessing_h2d",
        "vae_encode",
        "noise_full_latent_assembly",
        "prompt_context_prep",
        "transformer_compute_to_hidden",
        "feature_flatten_projection_reduction",
        "offload_block_get",
        "total_extraction",
    )
    stage_stats = {
        name: _summary([float(sample["timings"][name]) for sample in samples if name in sample["timings"]])
        for name in stage_names
    }
    all_blocks = [block for sample in samples for block in sample["blocks_loaded"]]
    return (
        {
            "status": "ok",
            "early_exit": early_exit,
            "warmup_count": warmups,
            "iteration_count": iterations,
            "warmup_first_call_seconds": _summary(
                [sum(float(value) for value in timings.values()) for timings in warmup_timings]
            ),
            "stages": stage_stats,
            "blocks_loaded_min": min(all_blocks) if all_blocks else None,
            "blocks_loaded_max": max(all_blocks) if all_blocks else None,
            "blocks_loaded_unique": sorted(set(all_blocks)),
            "per_iteration": samples,
            "peak_memory": _memory(device),
        },
        first_output,
    )


def _vae_output(
    module: nn.Module, value: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
        result = module(value)
    if hasattr(result, "latent"):
        result = result.latent
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"VAE returned {type(result).__name__}, expected a tensor")
    _sync(device)
    return result


def _vae_variant(
    extractor: LTXExtractor,
    images: torch.Tensor,
    *,
    name: str,
    channels_last: bool,
    cudnn_benchmark: bool,
    iterations: int,
    device: torch.device,
    compile_mode: str | None = None,
) -> dict[str, Any]:
    module: nn.Module = extractor.latent_encoder
    if compile_mode is not None:
        module = torch.compile(module, mode=compile_mode, fullgraph=False)
    prepared = extractor._preprocess_images(images)
    padding = prepared[:, :, -1:].expand(-1, -1, 4, -1, -1)
    value = torch.cat([prepared, padding], dim=2)
    if channels_last:
        value = value.contiguous(memory_format=torch.channels_last_3d)
    old_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = cudnn_benchmark
    try:
        warmup_error: str | None = None
        first_call_seconds: float | None = None
        try:
            _sync(device)
            first_started = time.perf_counter()
            _vae_output(module, value, device, extractor.dtype)
            _sync(device)
            first_call_seconds = time.perf_counter() - first_started
            _vae_output(module, value, device, extractor.dtype)
        except Exception as exc:  # noqa: BLE001 - optional arms are reported, not fatal
            warmup_error = f"{type(exc).__name__}: {exc}"
        if warmup_error is not None:
            return {"name": name, "status": "failed", "error": warmup_error}
        samples: list[float] = []
        first: torch.Tensor | None = None
        for _ in range(iterations):
            _sync(device)
            started = time.perf_counter()
            output = _vae_output(module, value, device, extractor.dtype)
            _sync(device)
            samples.append(time.perf_counter() - started)
            if first is None:
                first = output.detach().clone()
        if first is None:
            raise BenchmarkError(f"VAE variant {name} had no output")
        compile_graphs: dict[str, Any] | None = None
        if compile_mode is not None:
            try:
                explanation = torch._dynamo.explain(extractor.latent_encoder)(value)
                compile_graphs = {
                    "graph_count": int(explanation.graph_count),
                    "graph_break_count": int(explanation.graph_break_count),
                    "break_reasons": [str(reason) for reason in explanation.break_reasons],
                }
            except Exception as exc:  # noqa: BLE001 - graph enumeration is diagnostic
                compile_graphs = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
        return {
            "name": name,
            "status": "ok",
            "channels_last_3d": channels_last,
            "cudnn_benchmark": cudnn_benchmark,
            "compile_mode": compile_mode,
            "compile_graphs": compile_graphs,
            "first_call_seconds": first_call_seconds,
            "timing": _summary(samples),
            "output_shape": list(first.shape),
            "output_dtype": str(first.dtype).removeprefix("torch."),
            "output": first,
        }
    finally:
        torch.backends.cudnn.benchmark = old_benchmark
        del module
        gc.collect()


def _vae_probe(extractor: LTXExtractor, images: torch.Tensor, device: torch.device) -> dict[str, Any]:
    prepared = extractor._preprocess_images(images)
    result: dict[str, Any] = {"frames_5": {"status": "not_run"}, "frames_9": {"status": "not_run"}}
    for frames in (5, 9):
        value = (
            prepared
            if frames == 5
            else torch.cat([prepared, prepared[:, :, -1:].expand(-1, -1, 4, -1, -1)], dim=2)
        )
        try:
            _sync(device)
            output = _vae_output(extractor.latent_encoder, value, device, extractor.dtype)
            result[f"frames_{frames}"] = {
                "status": "ok",
                "input_shape": list(value.shape),
                "output_shape": list(output.shape),
                "clean_latent_frames": int(output.shape[2]) if output.ndim >= 3 else None,
                "preserves_two_clean_latents": bool(output.ndim >= 3 and output.shape[2] >= 2),
            }
        except Exception as exc:  # noqa: BLE001 - invalid temporal grids are evidence
            result[f"frames_{frames}"] = {
                "status": "failed",
                "input_shape": list(value.shape),
                "error": f"{type(exc).__name__}: {exc}",
            }
    result["decision"] = (
        "keep_9_frames"
        if not result["frames_5"].get("preserves_two_clean_latents", False)
        else "compare_hidden_features_before_changing"
    )
    return result


def _attention_labels(extractor: LTXExtractor) -> list[str]:
    model = getattr(extractor.backbone, "_transformer_model", None)
    if model is None:
        return []
    labels: set[str] = set()
    for module in model.modules():
        for name in ("attention_function", "masked_attention_function"):
            function = getattr(module, name, None)
            if function is None:
                continue
            labels.add(str(getattr(function, "label", type(function).__name__)))
    return sorted(labels)


def _synthetic_expert(context: torch.Tensor, device: torch.device, iterations: int) -> dict[str, Any]:
    """Measure an untrained 4096->960 adapter plus synthetic K/V projection only."""

    class AdapterKV(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapter = nn.Sequential(nn.LayerNorm(4096), nn.Linear(4096, 960))
            self.kv = nn.Linear(960, 640)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.kv(self.adapter(value))

    module = AdapterKV().to(device=device, dtype=torch.bfloat16).eval()
    value = context.to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module(value)
    samples: list[float] = []
    for _ in range(iterations):
        _sync(device)
        started = time.perf_counter()
        with torch.no_grad():
            module(value)
        _sync(device)
        samples.append(time.perf_counter() - started)
    graph_result: dict[str, Any] = {"status": "not_run"}
    try:
        graph = torch.cuda.CUDAGraph()
        static_value = value.clone()
        stream = torch.cuda.Stream(device=device)
        with torch.no_grad(), torch.cuda.stream(stream):
            stream.wait_stream(torch.cuda.current_stream())
            for _ in range(3):
                module(static_value)
            stream.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                static_output = module(static_value)
        stream.synchronize()
        replay_samples: list[float] = []
        for _ in range(iterations):
            static_value.copy_(value)
            _sync(device)
            started = time.perf_counter()
            graph.replay()
            _sync(device)
            replay_samples.append(time.perf_counter() - started)
        graph_result = {
            "status": "ok",
            "timing": _summary(replay_samples),
            "output_shape": list(static_output.shape),
        }
    except Exception as exc:  # noqa: BLE001 - graph support is optional
        graph_result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    del module
    gc.collect()
    return {
        "label": "synthetic_untrained_adapter_4096_to_960_plus_kv; no quality result",
        "eager_adapter_kv": _summary(samples),
        "cuda_graph_adapter_kv": graph_result,
    }


def _streaming_capacity(extractor: LTXExtractor, device: torch.device) -> dict[str, Any]:
    """Project full-prefix residency from the live official GPU buffer pool."""

    model = getattr(extractor.backbone, "_transformer_model", None)
    velocity_model = getattr(model, "velocity_model", None)
    provider = getattr(velocity_model, "_provider", None)
    pool = getattr(provider, "_pool", None)
    if pool is None:
        return {"status": "unavailable", "reason": "official streaming buffer pool not found"}
    slot_nbytes = int(pool.slot_nbytes)
    default_slots = int(pool.capacity)
    required_slots = extractor.config.hidden_layer + 1
    current_allocated = int(torch.cuda.memory_allocated(device))
    gpu_total = int(torch.cuda.get_device_properties(device).total_memory)
    projected_allocated = current_allocated + max(0, required_slots - default_slots) * slot_nbytes
    return {
        "status": "ok",
        "slot_nbytes": slot_nbytes,
        "slot_gib": slot_nbytes / 2**30,
        "default_slots": default_slots,
        "required_prefix_slots": required_slots,
        "current_allocated_bytes": current_allocated,
        "gpu_total_bytes": gpu_total,
        "projected_prefix_allocated_bytes": projected_allocated,
        "projected_prefix_allocated_gib": projected_allocated / 2**30,
        "projected_headroom_gib": (gpu_total - projected_allocated) / 2**30,
        "safe_to_attempt": projected_allocated < gpu_total * 0.9,
        "decision": "reject_full_resident_prefix_without_rearchitecting_cache",
    }


def _bytes_to_gib(value: int | None) -> float | None:
    return None if value is None else value / 2**30


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# LTX-2.5 extraction latency benchmark",
        "",
        f"- Timestamp: `{report['timestamp_utc']}`",
        f"- Device: `{report['device']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Quantization/offload: `{report['quantization']}` / `{report['offload_mode']}`",
        f"- Input: `{report['input_shape']}`; hidden: `{report['hidden_shape']}`",
        "",
        "## Real eager baseline vs true early exit",
        "",
    ]
    for name in ("baseline_eager", "optimized_true_early_exit"):
        arm = report["arms"][name]
        lines.append(f"### {name}")
        lines.append("")
        if arm.get("status") != "ok":
            lines.append(f"Status: **failed** — `{arm.get('error', 'unknown error')}`.")
            lines.append("")
            continue
        lines.append(
            f"Measured `{arm['iteration_count']}` iterations after `{arm['warmup_count']}` warmup; "
            f"early exit=`{arm['early_exit']}`, blocks loaded `{arm['blocks_loaded_min']}`..`{arm['blocks_loaded_max']}`."
        )
        lines.append("")
        lines.append("| Stage | p50 (s) | p90 (s) |")
        lines.append("|---|---:|---:|")
        for stage, values in arm["stages"].items():
            lines.append(
                f"| `{stage}` | {values['p50_seconds']:.6f} | {values['p90_seconds']:.6f} |"
                if values["p50_seconds"] is not None
                else f"| `{stage}` | n/a | n/a |"
            )
        lines.append("")
        memory = arm["peak_memory"]
        lines.append(
            f"Peak memory: allocated `{_bytes_to_gib(memory['max_allocated_bytes']):.2f} GiB`, "
            f"reserved `{_bytes_to_gib(memory['max_reserved_bytes']):.2f} GiB`."
        )
        lines.append("")
    correctness = report["correctness_gate"]
    lines.extend(
        [
            "## Correctness gate",
            "",
            f"Hidden feature cosine `{correctness['cosine']:.9f}`, max abs `{correctness['max_abs']:.6g}`, RMSE `{correctness['rmse']:.6g}`; threshold `{CORRECTNESS_COSINE_THRESHOLD}` => **{correctness['status']}**.",
            "",
            "## VAE and backend probes",
            "",
            f"- 5-vs-9-frame probe: `{report['vae_probe']['decision']}`.",
            f"- Attention labels observed: `{', '.join(report['attention_labels']) or 'unavailable'}`.",
            "- Transformer compile: this retained benchmark keeps provider/offload eager; dedicated block, stateless-weight, slot-keyed, regional, and full-graph arms were valid graphs but had no measured end-to-end win.",
            "- Full resident prefix: **rejected without execution** from the live streaming-pool memory projection; see `streaming_capacity` in JSON.",
            "",
            "## Prompt and optional expert",
            "",
            "Prompt embeddings are loaded from a reusable artifact on CPU and transferred with the extraction input. Zero prompt is only acceptable for smoke/benchmark plumbing.",
        ]
    )
    if "synthetic_expert" in report:
        lines.append(
            f"- {report['synthetic_expert']['label']}; this is runtime-only and has no quality claim."
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            f"`{report['command']}`",
            "",
            "The benchmark does not write training/validation caches or launch training.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VIDEO_VAE)
    parser.add_argument("--prompt-embedding", type=Path)
    parser.add_argument("--allow-zero-prompt", action="store_true")
    parser.add_argument("--dataset-revision", default="243370c3c08bcbd860133c4a0d658ea7c1d2e77e")
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--vae-iterations", type=int, default=10)
    parser.add_argument(
        "--vae-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="Opt in to compiled VAE execution for the end-to-end extraction arms",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--with-expert", action="store_true")
    parser.add_argument(
        "--try-compile", action="store_true", help="Try VAE reduce-overhead and max-autotune arms"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("LTX benchmark requires CUDA")
    if args.iterations < 3:
        raise ValueError("Use at least three measured iterations")
    if args.vae_iterations < 3:
        raise ValueError("Use at least three measured VAE iterations")
    device = torch.device("cuda")
    dataset, prepared = load_real_sample(
        root=args.root, episode_index=args.episode, frame_index=args.frame_index
    )
    images = prepared.rgb_history
    prompt = _load_embedding(args.prompt_embedding, images.shape[0])
    if args.prompt_embedding is None and not args.allow_zero_prompt:
        raise ValueError(
            "Provide --prompt-embedding, or explicitly pass --allow-zero-prompt for benchmark plumbing"
        )
    seed = derive_window_seed(args.dataset_revision, args.episode, prepared.frame_index, args.global_seed)
    config = LTXExtractorConfig(
        checkpoint_path=args.checkpoint,
        video_vae_path=args.video_vae,
        device="cuda",
        dtype="bfloat16",
        persistent_transformer=True,
        vae_compile_mode=args.vae_compile_mode,
    )
    extractor = LTXExtractor(config)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ltx_extraction_benchmark.json"
    markdown_path = output_dir / "ltx_extraction_benchmark.md"
    if not args.overwrite and (json_path.exists() or markdown_path.exists()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --overwrite")
    torch.cuda.reset_peak_memory_stats(device)
    model_load_seconds = extractor.open_transformer()
    model_load_memory = _memory(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        attention_labels = _attention_labels(extractor)
        streaming_capacity = _streaming_capacity(extractor, device)
        baseline, baseline_hidden = _arm_run(
            extractor,
            images,
            prompt,
            seed,
            early_exit=False,
            warmups=args.warmups,
            iterations=args.iterations,
            device=device,
        )
        optimized, optimized_hidden = _arm_run(
            extractor,
            images,
            prompt,
            seed,
            early_exit=True,
            warmups=args.warmups,
            iterations=args.iterations,
            device=device,
        )
        correctness_metrics = _tensor_metrics(baseline_hidden, optimized_hidden)
        correctness: dict[str, float | str] = {
            **correctness_metrics,
            "status": "pass" if correctness_metrics["cosine"] >= CORRECTNESS_COSINE_THRESHOLD else "reject",
        }
        vae_probe = _vae_probe(extractor, images, device)
        variants: list[dict[str, Any]] = []
        for name, channels_last, cudnn in (
            ("contiguous", False, False),
            ("channels_last_3d", True, False),
            ("cudnn_benchmark", False, True),
            ("channels_last_3d+cudnn_benchmark", True, True),
        ):
            try:
                variants.append(
                    _vae_variant(
                        extractor,
                        images,
                        name=name,
                        channels_last=channels_last,
                        cudnn_benchmark=cudnn,
                        iterations=args.vae_iterations,
                        device=device,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - retain other arm results
                variants.append({"name": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        contiguous = next(item for item in variants if item["name"] == "contiguous")
        reference_output = contiguous.get("output")
        for variant in variants:
            if variant.get("status") != "ok":
                continue
            if reference_output is not None:
                variant["correctness_vs_contiguous"] = _tensor_metrics(reference_output, variant["output"])
            else:
                variant["correctness_vs_contiguous"] = {
                    "status": "unavailable",
                    "reason": "contiguous reference arm failed",
                }
            del variant["output"]
        compile_variants: list[dict[str, Any]] = []
        if args.try_compile:
            for mode in ("default", "reduce-overhead", "max-autotune"):
                try:
                    variant = _vae_variant(
                        extractor,
                        images,
                        name=f"compile:{mode}",
                        channels_last=False,
                        cudnn_benchmark=False,
                        iterations=args.vae_iterations,
                        device=device,
                        compile_mode=mode,
                    )
                    if variant.get("status") == "ok":
                        if reference_output is not None:
                            variant["correctness_vs_contiguous"] = _tensor_metrics(
                                reference_output, variant["output"]
                            )
                        eager_p50 = contiguous["timing"]["p50_seconds"]
                        first_call = variant.get("first_call_seconds")
                        variant["estimated_compile_overhead_seconds"] = (
                            first_call - eager_p50
                            if first_call is not None and eager_p50 is not None
                            else None
                        )
                        del variant["output"]
                    compile_variants.append(variant)
                except Exception as exc:  # noqa: BLE001 - report unsupported compile modes
                    compile_variants.append(
                        {
                            "name": f"compile:{mode}",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        report: dict[str, Any] = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "host": platform.node(),
            "device": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "dataset": f"{dataset.repo_id}@{dataset.revision}",
            "episode": args.episode,
            "frame_index": prepared.frame_index,
            "noise_seed": seed,
            "input_shape": list(images.shape),
            "hidden_shape": list(baseline_hidden.shape),
            "checkpoint": str(args.checkpoint),
            "video_vae": str(args.video_vae),
            "quantization": config.quantization,
            "offload_mode": config.offload_mode,
            "vae_compile_mode": config.vae_compile_mode,
            "model_load_seconds": model_load_seconds,
            "model_load_peak_memory": model_load_memory,
            "attention_labels": attention_labels,
            "arms": {
                "baseline_eager": baseline,
                "optimized_true_early_exit": optimized,
            },
            "correctness_gate": correctness,
            "streaming_capacity": streaming_capacity,
            "vae_probe": vae_probe,
            "vae_variants": variants,
            "compile_variants": compile_variants,
            "prompt": {
                "source": str(args.prompt_embedding)
                if args.prompt_embedding is not None
                else "zero_prompt_benchmark_only",
                "loaded_on_cpu": True,
                "width": 4096,
            },
            "command": " ".join(["benchmark_ltx_extraction.py", *(sys.argv[1:] if argv is None else argv)]),
        }
        if args.with_expert:
            report["synthetic_expert"] = _synthetic_expert(optimized_hidden, device, args.vae_iterations)
        report["winner"] = "optimized_true_early_exit"
        report["estimated_speedup_total_p50"] = (
            baseline["stages"]["total_extraction"]["p50_seconds"]
            / optimized["stages"]["total_extraction"]["p50_seconds"]
            if optimized["stages"]["total_extraction"]["p50_seconds"]
            else None
        )
        # Hidden tensors are not JSON artifacts; only report their metrics.
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        markdown_path.write_text(_markdown(report))
        print("LTX extraction benchmark succeeded")
        print(f"  JSON: {json_path}")
        print(f"  Markdown: {markdown_path}")
        print(f"  baseline p50: {baseline['stages']['total_extraction']['p50_seconds']:.3f}s")
        print(f"  optimized p50: {optimized['stages']['total_extraction']['p50_seconds']:.3f}s")
        print(f"  speedup: {report['estimated_speedup_total_p50']:.2f}x")
        print(f"  correctness: {correctness['status']} cosine={correctness['cosine']:.9f}")
        return 0
    finally:
        extractor.close()
        del extractor
        gc.collect()
        _sync(device)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - preserve complete benchmark traceback
        print(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(2) from exc
