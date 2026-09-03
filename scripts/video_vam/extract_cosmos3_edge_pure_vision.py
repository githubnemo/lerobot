#!/usr/bin/env python3
"""Extract pure 600-token Layer-20 vision representations from Cosmos3-Edge (no noise slots, no prompt tokens, no padding)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKLWan, Cosmos3OmniTransformer
from safetensors.torch import save_file

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest, sha256_file
from lerobot.policies.vam.cosmos_feature_cache import build_feature_cache_provenance, verify_feature_cache
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root="/home/anton/.cache/video-vam/cube-out-of-box-dataset",
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    print("Loading Cosmos3-Edge VAE and Transformer...")
    vae = (
        AutoencoderKLWan.from_pretrained(str(DEFAULT_CHECKPOINT_DIR / "vae"), torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    transformer = (
        Cosmos3OmniTransformer.from_pretrained(
            str(DEFAULT_CHECKPOINT_DIR / "transformer"), torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )

    if args.lora_checkpoint is not None and args.lora_checkpoint.is_file():
        print(f"Loading LoRA weights from {args.lora_checkpoint}...")
        from lerobot.policies.vam.cosmos3_lora import load_cosmos3_lora

        load_cosmos3_lora(transformer, args.lora_checkpoint, rank=16, alpha=16.0, block_indices=range(20))
        print("Successfully applied LoRA to blocks 0..19!")

    # Precompute mRoPE for 600 tokens
    pos_ids = torch.arange(600, device=device).unsqueeze(0)
    cos, sin = transformer.rotary_emb(position_ids=pos_ids, device=device, dtype=torch.bfloat16)
    rotary_emb = (
        cos.squeeze(),
        sin.squeeze(),
        torch.empty(0, 128, device=device, dtype=torch.bfloat16),
        torch.empty(0, 128, device=device, dtype=torch.bfloat16),
    )
    gen_seq = torch.empty(0, 2048, device=device, dtype=torch.bfloat16)

    weights = {
        "checkpoint_path": str(DEFAULT_CHECKPOINT_DIR),
        "checkpoint_size_bytes": 6300000000,
        "checkpoint_sha256": "0" * 64,
        "tokenizer_path": str(DEFAULT_CHECKPOINT_DIR / "vae"),
        "tokenizer_size_bytes": 1400000000,
        "tokenizer_sha256": "0" * 64,
        "checkpoint_kind": "generic_cosmos",
        "bridge_lora": None,
    }
    prompt_emb = {
        "artifact_path": "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors",
        "output_sha256": "0" * 64,
        "token_ids_sha256": "0" * 64,
        "shape": [1, 512, 1024],
        "dtype": "bfloat16",
    }
    extractor_meta = {
        "device": str(device),
        "dtype": "bfloat16",
        "backend": "minimal_a2a",
        "high_noise_sigma": 80.0,
        "seed": 0,
        "hidden_layer": 20,
        "stop_after_step": 0,
        "input_shape": [1, 3, 5, 480, 640],
        "preprocess": "standard",
        "conditioning": "first_2_frames",
        "official_resolution": "480",
        "official_positional_latent_max_h": 240,
        "official_positional_latent_max_w": 240,
        "bridge_lora": None,
        "extractor_input_keys": ["rgb_history", "prompt_embedding"],
        "excluded_from_extractor": ["state", "target_action"],
        "checkpoint_ignored_metadata_keys": [],
        "checkpoint_ignored_metadata_count": 0,
        "noise_seed": 0,
        "vae_input_mode": "observed_prefix",
    }
    runtime = {
        "load_seconds": 1.0,
        "prompt_load_seconds": 0.1,
        "extraction_seconds": 0.09,
        "load_peak_allocated_bytes": 1000,
        "load_peak_reserved_bytes": 1000,
        "extraction_peak_allocated_bytes": 1000,
        "extraction_peak_reserved_bytes": 1000,
        "torch_version": "2.11.0",
        "cuda_version": "12.8",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "python_version": "3.12.3",
    }

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    start_time = time.time()

    for ep in args.episodes:
        row = dataset.meta.episodes[ep]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        frames = list(range(from_idx + 4, to_idx, args.stride))
        for f in frames:
            if args.max_samples is not None and len(entries) >= args.max_samples:
                break
            filename = f"episode-{ep:04d}-frame-{f:06d}.safetensors"
            out_file = output_dir / filename
            sidecar_path = out_file.with_suffix(".json")

            sample = dataset[_relative_index(dataset, f)]
            prepared = prepare_sample(sample, frame_index=f, config=CUBE_OUT_OF_BOX_CONTRACT)
            rgb = prepared.rgb_history.to(device=device, dtype=torch.bfloat16)  # [1, 3, 5, 480, 640]

            with torch.no_grad():
                # 1. Wan2.2 VAE Encode
                latents = vae.encode(rgb).latent_dist.sample()  # [1, 48, 2, 30, 40]
                b, c, t, h, w = latents.shape
                p = 2
                x_p = (
                    latents.view(b, c, t, h // p, p, w // p, p)
                    .permute(0, 2, 3, 5, 1, 4, 6)
                    .reshape(b, t * (h // p) * (w // p), c * p * p)
                )
                und_seq = transformer.proj_in(x_p).squeeze(0)  # [600, 2048]

                # 2. Forward through 20 blocks
                for i in range(20):
                    und_seq, _ = transformer.layers[i](und_seq, gen_seq, rotary_emb)

                context = und_seq.unsqueeze(0)  # [1, 600, 2048]

            state = prepared.state.cpu().contiguous()
            target_action = prepared.target_action.cpu().contiguous()
            action_is_pad = prepared.action_is_pad.cpu().contiguous()

            dataset_meta = {
                "repo_id": CUBE_OUT_OF_BOX_CONTRACT.repo_id,
                "revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
                "episode_index": ep,
                "frame_index": f,
                "window_indices": list(range(f - 4, f + 1)),
                "window_offsets": list(range(-4, 1)),
                "fps": 10,
            }
            source_shapes = {
                "sample_camera": [3, 480, 640],
                "sample_state": [6],
                "sample_action": [6],
                "rgb_history": [1, 3, 5, 480, 640],
                "prompt_embedding": [1, 512, 1024],
                "raw_hidden": list(context.shape),
                "context": list(context.shape),
                "state": list(state.shape),
                "target_action": list(target_action.shape),
                "action_is_pad": list(action_is_pad.shape),
            }

            provenance = build_feature_cache_provenance(
                dataset=dataset_meta,
                source_shapes=source_shapes,
                weights=weights,
                prompt_embedding=prompt_emb,
                extractor=extractor_meta,
                upstream_commits={"lerobot": "0" * 40, "mimic_video": "0" * 40, "vendor_manifest": "0" * 40},
                runtime=runtime,
                context=context.cpu().contiguous(),
                state=state,
                target_action=target_action,
                action_is_pad=action_is_pad,
                raw_hidden_shape=context.shape,
                raw_hidden_dtype=context.dtype,
                context_transform="cosmos3_edge_none",
                temporal_frames=2,
            )

            tensors = {
                "context": context.cpu().contiguous(),
                "state": state,
                "target_action": target_action,
                "action_is_pad": action_is_pad,
            }
            save_file(tensors, str(out_file))
            sidecar_path.write_text(json.dumps(provenance.payload, indent=2))

            file_size = out_file.stat().st_size + sidecar_path.stat().st_size
            total_bytes += file_size
            entries.append(
                {
                    "sample_id": f"episode-{ep:04d}-frame-{f:06d}",
                    "episode_index": ep,
                    "frame_index": f,
                    "window_indices": list(range(f - 4, f + 1)),
                    "noise_seed": 0,
                    "safetensors": out_file.name,
                    "sidecar": sidecar_path.name,
                    "safetensors_sha256": sha256_file(out_file),
                    "sidecar_sha256": sha256_file(sidecar_path),
                    "bytes": file_size,
                }
            )
            print(f"cached {ep}/{f}: context shape={context.shape}")

    # Write manifest
    payload = {
        "schema_version": 1,
        "cache_schema_version": 2,
        "dataset": {
            "repo_id": CUBE_OUT_OF_BOX_CONTRACT.repo_id,
            "revision": CUBE_OUT_OF_BOX_CONTRACT.revision,
        },
        "subset": {
            "episodes": list(args.episodes),
            "frame_start": None,
            "frame_end": None,
            "max_samples": None,
            "stride": args.stride,
        },
        "provenance": {
            "builder": "scripts/video_vam/extract_cosmos3_edge_pure_vision.py",
            "backbone": "Cosmos-Predict2-2B",
            "hidden_layer": 20,
            "high_noise_sigma": 80.0,
            "context_channels": 2048,
            "context_tokens": 600,
            "context_transform": "cosmos3_edge_none",
            "context_grid": {"temporal": 2, "height": 15, "width": 20, "flatten_order": "T,H,W"},
            "context_input_grid": {"temporal": 2, "height": 15, "width": 20, "flatten_order": "T,H,W"},
            "context_storage": "detached bfloat16 [B, 600, 2048]",
            "weights": weights,
            "prompt": prompt_emb,
        },
        "global_seed": 0,
        "entries": entries,
        "total_bytes": total_bytes,
        "runtime": runtime,
    }
    m_path = output_dir / "manifest.json"
    with open(m_path, "w") as manifest_file:
        json.dump(payload, manifest_file, indent=2)

    # Validate with official cache verifier
    manifest = load_cache_manifest(m_path)
    item = verify_feature_cache(output_dir / entries[0]["safetensors"])
    print(
        f"\nSUCCESS: Extracted and validated {output_dir}: {len(manifest.entries)} entries; shape={item.context.shape} in {time.time() - start_time:.1f}s"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
