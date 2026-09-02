#!/usr/bin/env python3
"""Extract Cosmos3-Edge Layer-20 spatiotemporal feature caches."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos3_edge_extractor import Cosmos3EdgeConfig, Cosmos3EdgeExtractor
from lerobot.policies.vam.cosmos_cache_dataset import sha256_file
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prompt", default="take cube out of box")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # device = torch.device(args.device)

    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root="/home/anton/.cache/video-vam/cube-out-of-box-dataset",
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    print("Initializing Cosmos3EdgeExtractor...")
    extractor = Cosmos3EdgeExtractor(Cosmos3EdgeConfig(device=args.device, dtype="bfloat16"))

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    start_time = time.time()

    for ep in args.episodes:
        row = dataset.meta.episodes[ep]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        # First valid causal frame is from_idx + 4
        frames = list(range(from_idx + 4, to_idx, args.stride))
        for f in frames:
            if args.max_samples is not None and len(entries) >= args.max_samples:
                break
            filename = f"episode-{ep:04d}-frame-{f:06d}.safetensors"
            out_file = output_dir / filename

            sample = dataset[_relative_index(dataset, f)]
            prepared = prepare_sample(sample, frame_index=f, config=CUBE_OUT_OF_BOX_CONTRACT)
            rgb = prepared.rgb_history  # [1, 3, 5, 480, 640]

            with torch.no_grad():
                # Extract Layer-20 features: [N, 2048]
                features = extractor.extract_features(rgb, prompt=args.prompt)
                # Reshape to [1, N, 2048]
                if features.ndim == 2:
                    features = features.unsqueeze(0)

            tensors = {
                "context": features.cpu().contiguous().to(torch.bfloat16),
                "state": prepared.state.cpu().contiguous(),
                "target_action": prepared.target_action.cpu().contiguous(),
                "action_is_pad": prepared.action_is_pad.cpu().contiguous(),
            }
            save_file(tensors, str(out_file))

            file_size = out_file.stat().st_size
            total_bytes += file_size
            entries.append(
                {
                    "episode_index": ep,
                    "frame_index": f,
                    "safetensors": filename,
                    "bytes": file_size,
                    "safetensors_sha256": sha256_file(out_file),
                    "noise_seed": 0,
                }
            )
            print(f"cached {ep}/{f}: {filename} (context shape={features.shape})")

    # Write manifest
    manifest_payload = {
        "schema_version": 1,
        "dataset_repo_id": CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        "dataset_revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
        "context_transform": "none",
        "output_tokens": entries[0]["bytes"] if entries else 0,
        "provenance": {
            "backbone": "Cosmos3-Edge",
            "hidden_layer": 20,
            "context_channels": 2048,
            "high_noise_sigma": 80.0,
            "context_storage": f"detached bfloat16 [B, {features.shape[1]}, 2048]",
        },
        "entries": entries,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as m:
        json.dump(manifest_payload, m, indent=2)

    print(f"\nExtracted {len(entries)} samples in {time.time() - start_time:.1f}s -> {manifest_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
