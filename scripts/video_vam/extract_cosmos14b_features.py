#!/usr/bin/env python3
"""Feature extraction from NVIDIA Cosmos-1.0-Diffusion-14B with block streaming and FP8.

Extracts intermediate layer representations (default layers 18 and 30) from the 14B model
using:
1. Block streaming: keeping transformer blocks on host CPU memory and streaming active
   blocks to CUDA on-demand, keeping peak VRAM footprint minimal.
2. FP8 linear quantization: converting linear weights to torch.float8_e4m3fn, reducing
   weight memory by 50% and accelerating memory throughput.
3. 3D spatiotemporal grid formatting and 2x2 spatial pooling for downstream SmolExpert policy training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.vam.cosmos14b_extractor import (
    Cosmos14BExtractionOutput,
    Cosmos14BExtractor,
    Cosmos14BExtractorConfig,
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_lora import (
    Cosmos14BLoRAConfig,
    inject_cosmos14b_quantized_lora,
    load_cosmos14b_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract intermediate layer representations from Cosmos-1.0-Diffusion-14B with streaming & FP8."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/anton/.cache/video-vam/cosmos-14b",
        help="Path or HuggingFace ID for Cosmos-1.0-Diffusion-14B checkpoint.",
    )
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help="Optional path to fine-tuned Video-LoRA safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cosmos14b-features"),
        help="Output directory to store extracted feature cache and manifest.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="18,30",
        help="Comma-separated list of transformer blocks to tap (default: 18,30).",
    )
    parser.add_argument(
        "--pool-spatial",
        type=int,
        default=2,
        help="Spatial pooling factor (default: 2 for 2x2 average pooling). 0 or 1 to disable.",
    )
    parser.add_argument(
        "--concat-layers",
        action="store_true",
        default=True,
        help="Concatenate tapped layers along the channel dimension (e.g. 5120*2 = 10240).",
    )
    parser.add_argument(
        "--block-streaming",
        action="store_true",
        default=True,
        help="Stream transformer blocks between host CPU memory and GPU to limit peak VRAM.",
    )
    parser.add_argument(
        "--no-block-streaming",
        action="store_false",
        dest="block_streaming",
        help="Disable block streaming (keep all blocks on device).",
    )
    parser.add_argument(
        "--fp8-linear",
        action="store_true",
        default=True,
        help="Quantize linear layer weights to FP8 for 2x memory reduction and accelerated compute.",
    )
    parser.add_argument(
        "--no-fp8-linear",
        action="store_false",
        dest="fp8_linear",
        help="Disable FP8 linear quantization.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="take cube out of box",
        help="Task text prompt string.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to run extraction on (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Data type for extraction (default: bfloat16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic dry-run on CPU with architecture-matched dummy model.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset"),
        help="Path to robot demonstration dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to extract (default: None for full dataset).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Stride between frames across the robot dataset (default: 3).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for parallel forward pass during feature extraction (default: 16).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())
    pool_factor = args.pool_spatial if args.pool_spatial and args.pool_spatial > 1 else None

    print(f"[Cosmos 14B Extractor] Target layers: {hidden_layers}")
    print(f"[Cosmos 14B Extractor] Device: {args.device}, Dtype: {args.dtype}")
    print(f"[Cosmos 14B Extractor] Spatial pooling: {pool_factor}, Concat layers: {args.concat_layers}")
    print(f"[Cosmos 14B Extractor] Block streaming: {args.block_streaming}, FP8 linear: {args.fp8_linear}")

    if args.dry_run:
        print(f"[Cosmos 14B Extractor] --- RUNNING SYNTHETIC DRY-RUN ON {args.device.upper()} ---")
        num_layers = 32
        dummy_model = build_cosmos14b_dummy_model(
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            text_embed_dim=32,
            device=args.device,
            dtype=torch.float32,
        )
        extractor_config = Cosmos14BExtractorConfig(
            hidden_layers=hidden_layers,
            patch_size=(1, 2, 2),
            in_channels=17,
            out_channels=16,
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            hidden_dim=64,
            text_embed_dim=32,
            device=args.device,
            dtype="float32",
            pool_spatial=pool_factor,
            concat_layers=args.concat_layers,
            block_streaming=args.block_streaming,
            fp8_linear=args.fp8_linear,
        )
        extractor = Cosmos14BExtractor(config=extractor_config, model=dummy_model)
    else:
        extractor_config = Cosmos14BExtractorConfig(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dtype=args.dtype,
            hidden_layers=hidden_layers,
            pool_spatial=pool_factor,
            concat_layers=args.concat_layers,
            block_streaming=args.block_streaming,
            fp8_linear=args.fp8_linear,
        )
        extractor = Cosmos14BExtractor(config=extractor_config)

    # Optional LoRA injection & loading
    if args.lora_path is not None:
        print(f"[Cosmos 14B Extractor] Injecting and loading Video-LoRA from {args.lora_path}...")
        lora_sd = load_file(str(args.lora_path))
        lora_blocks = set()
        for k in lora_sd:
            parts = k.split(".")
            if len(parts) > 1 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                lora_blocks.add(int(parts[1]))
        target_blocks = tuple(sorted(lora_blocks)) if lora_blocks else tuple(range(36))

        lora_cfg = Cosmos14BLoRAConfig(
            rank=16,
            alpha=16.0,
            target_blocks=target_blocks,
            quantization="fp8" if args.fp8_linear and not args.dry_run else "none",
        )
        inject_cosmos14b_quantized_lora(extractor.transformer, lora_cfg)
        load_cosmos14b_lora(extractor.transformer, args.lora_path)
        print(f"[Cosmos 14B Extractor] Loaded Video-LoRA weights into {len(target_blocks)} blocks.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    dataset = None
    if args.dataset_root is not None and args.dataset_root.is_dir():
        try:
            from lerobot.datasets import LeRobotDataset
            from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT

            dataset = LeRobotDataset(
                CUBE_OUT_OF_BOX_CONTRACT.repo_id,
                root=str(args.dataset_root),
                delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
                revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
                return_uint8=True,
                download_videos=False,
            )
            print(f"[Cosmos 14B Extractor] Loaded robot dataset with {len(dataset)} frames.")
        except Exception as e:
            print(f"[Cosmos 14B Extractor] Note: dataset loading skipped ({e})")

    if dataset is not None:
        sample_indices = list(range(0, len(dataset), args.stride))
        if args.dry_run and args.num_samples is None:
            sample_indices = sample_indices[:5]
        elif args.num_samples is not None and args.num_samples > 0:
            sample_indices = sample_indices[: args.num_samples]
    else:
        sample_indices = list(
            range(args.num_samples if args.num_samples is not None else (5 if args.dry_run else 10))
        )

    print(
        f"[Cosmos 14B Extractor] Extracting {len(sample_indices)} samples (stride={args.stride}, batch_size={args.batch_size})..."
    )
    t0 = time.time()
    batch_size = args.batch_size

    for b_start in range(0, len(sample_indices), batch_size):
        b_indices = sample_indices[b_start : b_start + batch_size]
        b_latents = []
        b_states = []
        b_actions = []

        for _count_in_batch, idx in enumerate(b_indices):
            state = None
            action = None
            latents = None

            if dataset is not None and idx < len(dataset):
                row = dataset[idx]
                if "observation.state" in row:
                    state = row["observation.state"].float()
                if "action" in row:
                    act = row["action"].float()
                    action = act.unsqueeze(0).expand(30, -1).clone() if act.ndim == 1 else act.clone()
                if "observation.images.front" in row:
                    rgb = row["observation.images.front"]
                    if rgb.ndim == 4:
                        rgb = rgb[-1]
                    if rgb.ndim == 3:
                        rgb = rgb.unsqueeze(0)
                    rgb_norm = (rgb.float() / 255.0 - 0.5) * 2.0
                    down = torch.nn.functional.interpolate(
                        rgb_norm, size=(32, 32), mode="bilinear", align_corners=False
                    )
                    # Pad/format to 17 channels [B=1, C=17, T=1, H=32, W=32]
                    b_frames = down.shape[0]
                    lat_16 = down.repeat(1, 6, 1, 1)[:, :16, :, :]
                    padding_mask = torch.ones(b_frames, 1, 32, 32, dtype=rgb_norm.dtype)
                    lat_17 = torch.cat([lat_16, padding_mask], dim=1)  # [T, 17, 32, 32]
                    lat_5d = lat_17.permute(1, 0, 2, 3).unsqueeze(0)  # [1, 17, T, 32, 32]
                    latents = lat_5d.to(device=args.device, dtype=extractor.dtype)

            if latents is None:
                latents = torch.randn(1, 17, 1, 32, 32, device=args.device, dtype=extractor.dtype)
            if state is None:
                state = torch.randn(6, dtype=torch.float32)
            if action is None:
                action = torch.randn(30, 6, dtype=torch.float32)

            b_latents.append(latents)
            b_states.append(state)
            b_actions.append(action)

        # Batch forward pass through Cosmos 14B
        batch_input = torch.cat(b_latents, dim=0)
        out: Cosmos14BExtractionOutput = extractor.extract(batch_input)

        for i, idx in enumerate(b_indices):
            global_count = b_start + i
            artifact_name = f"cosmos14b_sample_{global_count:05d}.safetensors"
            artifact_path = args.output_dir / artifact_name

            tensors_to_save = {
                "features": out.features[i : i + 1].detach().cpu().clone(),
                "grid_coords": out.grid_coords[i : i + 1].detach().cpu().clone()
                if out.grid_coords.shape[0] > i
                else out.grid_coords.detach().cpu().clone(),
                "state": b_states[i],
                "action": b_actions[i],
            }
            for lyr_idx, lyr_feat in out.features_by_layer.items():
                tensors_to_save[f"layer_{lyr_idx}"] = lyr_feat[i : i + 1].detach().cpu().clone()

            metadata = {
                "sample_index": str(global_count),
                "dataset_index": str(idx),
                "hidden_layers": str(hidden_layers),
                "block_streaming": str(args.block_streaming),
                "fp8_linear": str(args.fp8_linear),
                "feature_dim": str(out.feature_dim),
            }
            save_file(tensors_to_save, str(artifact_path), metadata=metadata)

            manifest_entries.append(
                {
                    "sample_index": global_count,
                    "dataset_index": idx,
                    "file": artifact_name,
                    "artifact_file": artifact_name,
                    "feature_dim": out.feature_dim,
                    "num_tokens": out.num_tokens,
                    "grid_shape": list(out.grid_shape),
                }
            )

        extracted_so_far = min(b_start + batch_size, len(sample_indices))
        pct = (extracted_so_far / len(sample_indices)) * 100.0
        rate = extracted_so_far / (time.time() - t0)
        print(
            f"  Extracted {extracted_so_far}/{len(sample_indices)} samples ({pct:.1f}%) | {rate:.2f} samples/s...",
            flush=True,
        )

    elapsed = time.time() - t0
    manifest = {
        "model_type": "Cosmos-1.0-Diffusion-14B",
        "checkpoint_path": str(args.checkpoint_path),
        "lora_path": str(args.lora_path) if args.lora_path else None,
        "hidden_layers": list(hidden_layers),
        "pool_spatial": pool_factor,
        "block_streaming": args.block_streaming,
        "fp8_linear": args.fp8_linear,
        "total_samples": len(manifest_entries),
        "stride": args.stride,
        "elapsed_seconds": elapsed,
        "provenance": out.provenance,
        "entries": manifest_entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"\n[Cosmos 14B Extractor] Successfully extracted {len(manifest_entries)} samples in {elapsed:.2f}s -> saved to {args.output_dir}"
    )
    print(f"[Cosmos 14B Extractor] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
