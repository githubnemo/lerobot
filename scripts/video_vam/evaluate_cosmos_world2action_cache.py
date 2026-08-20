#!/usr/bin/env python3
"""Evaluate every cached training sample with a diagnostic World2Action checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.vam.cosmos_cache_dataset import CosmosFeatureCacheDataset, manifest_sha256
from lerobot.policies.vam.world2action import (
    ActionStateNormalizer,
    World2ActionConfig,
    World2ActionDecoder,
    load_diagnostic_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=1000)
    parser.add_argument("--seed", type=int, required=True, help="Recorded deterministic evaluation seed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _require_finite(value: torch.Tensor | float, name: str) -> None:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} is non-finite")


def _autocast(device: torch.device):
    if device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 autocast is required for diagnostic evaluation")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _masked_physical_mse(
    prediction: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (30, 6):
        raise ValueError("physical action tensors must both have shape [B, 30, 6]")
    if tuple(action_is_pad.shape) != tuple(prediction.shape[:2]) or action_is_pad.dtype != torch.bool:
        raise ValueError("action_is_pad must have boolean shape [B, 30]")
    valid = (~action_is_pad).unsqueeze(-1).to(torch.float32)
    count = valid.sum() * prediction.shape[-1]
    if count.item() <= 0:
        raise ValueError("action_is_pad leaves no valid positions")
    result = ((prediction.float() - target.float()).square() * valid).sum() / count
    _require_finite(result, "physical action MSE")
    return result


def _sample_inputs(item: Any, device: torch.device, seed: int):
    action = item.target_action.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    epsilon = torch.randn(action.shape, device=device, dtype=action.dtype, generator=generator)
    state = item.state.to(device=device, dtype=torch.float32)
    context = item.context.to(device=device)
    padding = item.action_is_pad.to(device=device)
    t = torch.full((action.shape[0],), 0.5, device=device, dtype=torch.float32)
    return state, action, context, padding, t, epsilon


@torch.no_grad()
def evaluate_entries(
    decoder: World2ActionDecoder,
    items: Iterable[Any],
    *,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    decoder.eval()
    results = []
    for index, item in enumerate(items):
        sample_seed = seed + index
        state, action, context, padding, t, epsilon = _sample_inputs(item, device, sample_seed)
        with _autocast(device):
            flow_loss = decoder.flow_matching_loss(
                state,
                action,
                context,
                t=t,
                epsilon=epsilon,
                action_is_pad=padding,
            )
            sampled = decoder.sample_actions(state, context, seed=sample_seed)
        physical_mse = _masked_physical_mse(sampled, action, padding)
        _require_finite(flow_loss, "fixed flow loss")
        results.append(
            {
                "sample_id": item.sample_id,
                "evaluation_seed": sample_seed,
                "fixed_flow_loss": float(flow_loss.float().item()),
                "sampled_action_physical_mse": float(physical_mse.item()),
            }
        )
    if not results:
        raise ValueError("diagnostic manifest contains no entries")
    return results


def _stats(values: list[float]) -> dict[str, float]:
    for value in values:
        _require_finite(value, "metric")
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    flow = [result["fixed_flow_loss"] for result in results]
    physical = [result["sampled_action_physical_mse"] for result in results]
    flow_stats = _stats(flow)
    physical_stats = _stats(physical)
    rmse = math.sqrt(physical_stats["mean"])
    _require_finite(rmse, "physical RMSE")
    return {
        "count": len(results),
        "fixed_flow_loss": flow_stats,
        "sampled_action_physical_mse": physical_stats,
        "sampled_action_physical_rmse": rmse,
    }


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite evaluation JSON; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with open(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed < 0 or args.expected_step < 0:
        raise ValueError("seed and expected-step must be non-negative")
    if args.device != "cuda":
        raise ValueError("Diagnostic evaluation requires CUDA for the full native decoder")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no model or workload is run on CPU")
    checkpoint = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    normalizer_path = args.normalizer.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    normalizer = ActionStateNormalizer.load(normalizer_path)
    config = World2ActionConfig(device=args.device, dtype=torch.bfloat16)
    decoder = World2ActionDecoder(config, normalizer=normalizer)
    metadata = load_diagnostic_checkpoint(
        decoder,
        checkpoint,
        manifest_path=manifest_path,
        normalizer_path=normalizer_path,
        expected_step=args.expected_step,
    )
    dataset = CosmosFeatureCacheDataset(manifest_path, shuffle_seed=0)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    results = evaluate_entries(decoder, dataset, device=device, seed=args.seed)
    elapsed = time.perf_counter() - started
    aggregate = aggregate_results(results)
    payload = {
        "schema_version": 1,
        "artifact": "world2action_diagnostic_evaluation",
        "status": "diagnostic_only_non_rollout",
        "robot_ready": False,
        "checkpoint": str(checkpoint),
        "checkpoint_step": metadata["step"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256(manifest_path),
        "normalizer": str(normalizer_path),
        "evaluation_seed": args.seed,
        "aggregate": aggregate,
        "samples": results,
        "runtime": {
            "seconds": elapsed,
            "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda"
            else 0,
        },
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "python_version": platform.python_version(),
            "autocast": "cuda_bfloat16",
        },
    }
    _require_finite(torch.tensor(aggregate["fixed_flow_loss"]["mean"]), "aggregate flow loss")
    _require_finite(torch.tensor(aggregate["sampled_action_physical_mse"]["mean"]), "aggregate physical MSE")
    _atomic_json(output_path, payload, overwrite=args.overwrite)
    return payload


def main(argv: list[str] | None = None) -> int:
    try:
        payload = evaluate(parse_args(argv))
        print(json.dumps({"output": payload["artifact"], "count": payload["aggregate"]["count"]}))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
