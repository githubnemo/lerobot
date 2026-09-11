#!/usr/bin/env python3
"""Extract pure 600-token Layer-20 vision representations from Cosmos 3 Edge into unified cache format."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos3_features import (
    Cosmos3ExtractorConfig,
    Cosmos3FeatureExtractor,
    checkpoint_digest,
    compute_cosmos3_cache_key,
    sha256_file,
)

try:
    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample
except ModuleNotFoundError:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample

DEFAULT_CHECKPOINT_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--lora-checkpoint", type=Path, default=None, help="Path to trained LoRA safetensors")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--hidden-layer", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--prompt", type=str, default="take cube out of box")
    parser.add_argument("--dataset-repo-id", type=str, default="hubnemo/cube_out_of_box_dataset")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    if args.stride <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        raise ValueError("Stride and sample limit must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    device = torch.device(args.device)

    # Fail early if LoRA checkpoint path was provided but does not exist
    if args.lora_checkpoint is not None:
        lora_path = Path(args.lora_checkpoint).expanduser().resolve()
        if not lora_path.is_file():
            raise FileNotFoundError(f"Specified LoRA checkpoint not found: {lora_path}")
    else:
        lora_path = None

    t0_load = time.time()
    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=str(args.dataset_root),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    print(f"Initializing Cosmos3FeatureExtractor (layer={args.hidden_layer}, device={device})...")
    config = Cosmos3ExtractorConfig(
        backbone_name="cosmos3-edge",
        checkpoint_path=checkpoint_dir,
        hidden_layers=(args.hidden_layer,),
        device=str(device),
        dtype="bfloat16",
        fps=10.0,
        base_fps=24.0,
        prompt=args.prompt,
        lora_checkpoint=lora_path,
        lora_rank=args.rank,
        lora_alpha=args.alpha,
    )
    extractor = Cosmos3FeatureExtractor(config)
    load_seconds = time.time() - t0_load

    # Checkpoint and LoRA SHA-256 provenance
    transformer_sha = checkpoint_digest(checkpoint_dir)
    lora_sha = sha256_file(lora_path) if lora_path is not None else None
    prompt_hash = hashlib.sha256(args.prompt.encode("utf-8")).hexdigest()

    cache_key = compute_cosmos3_cache_key(
        dataset_repo=args.dataset_repo_id,
        dataset_revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        episodes=args.episodes,
        stride=args.stride,
        clip_frames=5,
        hidden_layer=args.hidden_layer,
        fps=10.0,
        prompt=args.prompt,
        max_samples=args.max_samples,
        checkpoint_sha256=transformer_sha,
        lora_sha256=lora_sha,
    )

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    t0_extract = time.time()

    for ep in args.episodes:
        row = dataset.meta.episodes[ep]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        # Causal window of 5 frames (from_idx + 4 .. to_idx)
        frames = list(range(from_idx + 4, to_idx, args.stride))
        for f in frames:
            if args.max_samples is not None and len(entries) >= args.max_samples:
                break
            filename = f"episode-{ep:04d}-frame-{f:06d}.safetensors"
            out_file = output_dir / filename

            sample = dataset[_relative_index(dataset, f)]
            prepared = prepare_sample(sample, frame_index=f, config=CUBE_OUT_OF_BOX_CONTRACT)
            rgb = prepared.rgb_history.to(device=device)  # [1, 3, 5, 480, 640]

            with torch.no_grad():
                out = extractor.extract(rgb_frames=rgb)
                context = out.features.cpu().squeeze(0).contiguous()  # [600, 2048]

            state = prepared.state.cpu().contiguous()
            target_action = prepared.target_action.cpu().contiguous()
            action_is_pad = prepared.action_is_pad.cpu().contiguous()

            tensors = {
                "context": context,
                "state": state,
                "target_action": target_action,
                "action_is_pad": action_is_pad,
            }
            save_file(tensors, str(out_file))

            file_size = out_file.stat().st_size
            total_bytes += file_size
            file_sha = sha256_file(out_file)

            entries.append(
                {
                    "sample_id": f"episode-{ep:04d}-frame-{f:06d}",
                    "episode_index": ep,
                    "frame_index": f,
                    "window_indices": list(range(f - 4, f + 1)),
                    "safetensors": out_file.name,
                    "bytes": file_size,
                    "safetensors_sha256": file_sha,
                }
            )

            if len(entries) % 50 == 0 or len(entries) == 1:
                print(f"Extracted {len(entries)} samples: ep {ep}, frame {f}, context shape: {context.shape}")

    extraction_seconds = time.time() - t0_extract

    # Write unified manifest conforming to train_smolexpert.py contract
    manifest_payload = {
        "schema_version": 1,
        "dataset": {
            "repo_id": args.dataset_repo_id,
            "revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
        },
        "subset": {
            "episodes": list(args.episodes),
            "stride": args.stride,
        },
        "provenance": {
            "builder": "scripts/video_vam/extract_cosmos3_edge_pure_vision.py",
            "backbone": "cosmos3-edge",
            "cache_key": cache_key,
            "hidden_layer": args.hidden_layer,
            "context_tokens": 600,
            "context_dim": 2048,
            "fps": 10.0,
            "base_fps": 24.0,
            "prompt": args.prompt,
            "prompt_sha256": prompt_hash,
            "checkpoint_path": str(checkpoint_dir),
            "transformer_sha256": transformer_sha,
            "lora_checkpoint": str(lora_path) if lora_path else None,
            "lora_sha256": lora_sha,
            "lora_rank": args.rank if lora_path else None,
            "lora_alpha": args.alpha if lora_path else None,
        },
        "runtime": {
            "load_seconds": round(load_seconds, 2),
            "extraction_seconds": round(extraction_seconds, 2),
            "device": str(device),
            "dtype": "bfloat16",
        },
        "entries": entries,
        "total_bytes": total_bytes,
    }

    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with open(temporary_manifest, "w") as mf:
        json.dump(manifest_payload, mf, indent=2)
    temporary_manifest.replace(manifest_path)

    print(
        f"SUCCESS: Extracted {len(entries)} items to {output_dir} in {extraction_seconds:.1f}s. "
        f"Manifest saved to {manifest_path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
