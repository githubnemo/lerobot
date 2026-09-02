#!/usr/bin/env python3
"""Time GPU-resident int4 LTX extract at state_t=2 vs the CPU-offload FP8 baseline.

No robot. Same 5-frame window and prompt as cache construction. INT4 uses torchao
weight-only on Linear layers after a CPU BF16 load so the 22B transformer can
stay on a 24 GiB GPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.vam.ltx_extractor import LTXExtractor, LTXExtractorConfig, derive_window_seed
from lerobot.policies.vam.ltx_prompt_embedding import load_ltx_prompt_artifact
from scripts.video_vam.smoke_test_cosmos_extractor import load_real_sample

DEFAULT_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_VAE = Path("/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors")
DEFAULT_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/ltx-int4-resident-latency")
DATASET_REVISION = "243370c3c08bcbd860133c4a0d658ea7c1d2e77e"


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = fraction * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min_seconds": min(values) if values else None,
        "mean_seconds": statistics.mean(values) if values else None,
        "p50_seconds": _percentile(values, 0.50),
        "p90_seconds": _percentile(values, 0.90),
        "max_seconds": max(values) if values else None,
    }


def _ms(summary: dict[str, float | int | None], key: str = "p50_seconds") -> str:
    value = summary.get(key)
    return "n/a" if value is None else f"{1000.0 * float(value):.1f}"


def _measure(
    *,
    state_t: int,
    quantization: str,
    offload_mode: str,
    images: torch.Tensor,
    prompt: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cuda")
    config = LTXExtractorConfig(
        checkpoint_path=args.checkpoint,
        video_vae_path=args.video_vae,
        device="cuda",
        dtype="bfloat16",
        hidden_layer=34,
        high_noise_sigma=1.0,
        seed=args.global_seed,
        input_frames=5,
        padded_input_frames=9,
        state_t=state_t,
        offload_mode=offload_mode,
        quantization=quantization,
        persistent_transformer=True,
        explicit_prefix_execution=True,
        vae_compile_mode=None,
    )
    extractor = LTXExtractor(config)
    torch.cuda.reset_peak_memory_stats(device)
    load_seconds = extractor.open_transformer()
    expected_tokens = state_t * 15 * 20
    for _ in range(args.warmups):
        extractor.extract(images, prompt, noise_seed=seed)
    samples: list[float] = []
    stage_samples: dict[str, list[float]] = {}
    hidden_shape: list[int] | None = None
    try:
        for _ in range(args.iterations):
            timings: dict[str, float] = {}
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            extraction = extractor.extract(images, prompt, noise_seed=seed, timings=timings)
            torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
            for key, value in timings.items():
                stage_samples.setdefault(key, []).append(float(value))
            hidden_shape = list(extraction.tokens.shape)
            if extraction.tokens.shape[1] != expected_tokens:
                raise RuntimeError(
                    "state_t=%s produced %s tokens, expected %s"
                    % (state_t, extraction.tokens.shape[1], expected_tokens)
                )
        return {
            "status": "ok",
            "state_t": state_t,
            "quantization": quantization,
            "offload_mode": offload_mode,
            "token_count": expected_tokens,
            "hidden_shape": hidden_shape,
            "model_load_seconds": load_seconds,
            "warmup_count": args.warmups,
            "iteration_count": args.iterations,
            "total": _summary(samples),
            "stages": {name: _summary(values) for name, values in stage_samples.items()},
            "peak_memory": {
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            },
        }
    finally:
        try:
            extractor.close()
        except Exception as close_exc:
            print("close failed (non-fatal): %s" % close_exc, flush=True)
        del extractor
        gc.collect()
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _, prepared = load_real_sample(root=args.root, episode_index=args.episode, frame_index=args.frame_index)
    images = prepared.rgb_history
    prompt = load_ltx_prompt_artifact(args.prompt.expanduser().resolve()).embedding
    seed = derive_window_seed(DATASET_REVISION, args.episode, prepared.frame_index, args.global_seed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ltx_int4_resident_latency.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError("refusing to overwrite %s; pass --overwrite" % json_path)

    arms: list[dict[str, Any]] = []
    print("measuring int4 GPU-resident state_t=2", flush=True)
    try:
        arm = _measure(
            state_t=2,
            quantization="int4",
            offload_mode="none",
            images=images,
            prompt=prompt,
            seed=seed,
            args=args,
        )
        dit = arm["stages"].get("transformer_compute_to_hidden", {})
        vae = arm["stages"].get("vae_encode", {})
        print(
            "int4 T=2 tokens=%s p50_ms=%s dit_p50_ms=%s vae_p50_ms=%s peak_gib=%.2f load_s=%.1f"
            % (
                arm["token_count"],
                _ms(arm["total"]),
                _ms(dit),
                _ms(vae),
                arm["peak_memory"]["max_allocated_bytes"] / 2**30,
                arm["model_load_seconds"],
            ),
            flush=True,
        )
        arms.append(arm)
    except Exception as exc:  # noqa: BLE001
        print("int4 arm failed: %s" % exc, flush=True)
        traceback.print_exc()
        arms.append({"status": "failed", "error": "{}: {}".format(type(exc).__name__, exc)})

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "episode": args.episode,
        "frame_index": prepared.frame_index,
        "cpu_offload_fp8_t2_p50_ms_reference": 1228.3,
        "arms": arms,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote %s" % json_path, flush=True)
    return 0 if arms and arms[0].get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
