#!/usr/bin/env python3
"""Evaluate a frozen-LTX World2Action checkpoint under action-RMSE protocol 1.0."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

from lerobot.policies.vam.ltx_action import LTXContextAdapter
from lerobot.policies.vam.ltx_feature_cache import (
    LTXFeatureCacheDataset,
    load_ltx_cache_manifest,
    sha256_file,
)
from lerobot.policies.vam.vam_split import load_vam_split
from lerobot.policies.vam.world2action import World2ActionConfig, World2ActionDecoder
from scripts.video_vam.train_cosmos_world2action import convert_decoder_parameters_to_fp32
from scripts.video_vam.train_ltx_world2action import (
    DEFAULT_NORMALIZER,
    DEFAULT_SPLIT,
    DEFAULT_VAL_MANIFEST,
    evaluate,
    load_model,
)
from scripts.video_vam.train_ltx_world2action_tiny import load_established_normalizer

DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/runs/ltx25-frozen-pool2-w2a-plateau-20260825/best.safetensors"
)
DEFAULT_OUTPUT = Path(
    "/home/anton/.cache/video-vam/runs/ltx25-frozen-pool2-w2a-plateau-20260825/protocol_eval.json"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required")
    manifest_path = args.val_manifest.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    normalizer_path = args.normalizer.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing existing evaluator output: {output}")
    manifest = load_ltx_cache_manifest(manifest_path)
    split = load_vam_split(split_path, manifest, allow_partial=True)
    if len(manifest.entries) != 88 or len(split.validation_probe) != 88:
        raise ValueError("protocol 1.0 requires exactly 88 ordered held-out validation anchors")
    metadata_path = checkpoint.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("val_manifest_sha256") != sha256_file(manifest_path) or metadata.get(
        "split_sha256"
    ) != sha256_file(split_path):
        raise ValueError("checkpoint cache/split provenance does not match evaluator inputs")
    normalizer, normalizer_source = load_established_normalizer(normalizer_path)
    device = torch.device("cuda")
    adapter = LTXContextAdapter().to(device=device, dtype=torch.float32)
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cuda", dtype=torch.bfloat16), normalizer=normalizer
    )
    convert_decoder_parameters_to_fp32(decoder)
    load_model(checkpoint, decoder, adapter)
    torch.cuda.reset_peak_memory_stats()
    metrics = evaluate(
        decoder,
        adapter,
        LTXFeatureCacheDataset(manifest),
        manifest.entries,
        split,
        normalizer,
        device=device,
        batch_size=args.batch_size,
    )
    if not all(
        math.isfinite(float(metrics[key]))
        for key in (
            "val_fixed_flow_loss_normalized",
            "val_action_rmse_degrees",
            "val_action_rmse_normalized",
        )
    ):
        raise FloatingPointError("evaluator produced a non-finite metric")
    payload = {
        "protocol_version": "1.0",
        "metric_type": "frozen-protocol global masked physical action RMSE in degrees",
        "backend": "frozen LTX-2.5 block-34 pool2 + trainable adapter + World2Action",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": metadata["step"],
        "wandb_url": metadata.get("wandb_url"),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset": manifest.payload["dataset"],
        "split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
            "train_episodes": list(split.train_episodes),
            "validation_episodes": list(split.val_episodes),
            "probe_seed": split.probe_seed,
        },
        "ordered_validation_ids": [
            {
                "sample_id": entry.sample_id,
                "episode_index": entry.episode_index,
                "frame_index": entry.frame_index,
            }
            for entry in manifest.entries
        ],
        "normalizer": {
            "path": str(normalizer_path),
            "sha256": sha256_file(normalizer_path),
            "source": normalizer_source,
        },
        "action_contract": {
            "horizon": 30,
            "dimensions": 6,
            "units": "degrees",
            "padding": "action_is_pad=true excluded",
            "aggregation": "sqrt(global masked squared-error sum / global valid scalar count)",
            "sampler_seed": "persisted per-sample split validation probe",
        },
        "metrics": metrics,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "metric_names_are_distinct": True,
        "val_flow_equals_val_rmse": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("PROTOCOL_EVAL=" + json.dumps(payload["metrics"], sort_keys=True), flush=True)
    print(f"PROTOCOL_EVAL_OUTPUT={output}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
