#!/usr/bin/env python3
"""Raw feature extraction from intermediate layers of NVIDIA Cosmos-1.0-Diffusion-7B.

Supports tapping layers 14 and 20, 3D spatiotemporal grid & mRoPE formatting,
and text conditioning support. Can save extracted representations to safetensors
cache manifests for downstream SmolExpert policy training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.vam.cosmos7b_extractor import (
    Cosmos7BExtractionOutput,
    Cosmos7BExtractor,
    Cosmos7BExtractorConfig,
    build_cosmos7b_dummy_model,
)
from lerobot.policies.vam.cosmos7b_lora import (
    Cosmos7BLoRAConfig,
    inject_cosmos7b_lora,
    load_cosmos7b_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract intermediate layer representations from Cosmos-1.0-Diffusion-7B."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path or HuggingFace ID for Cosmos-1.0-Diffusion-7B checkpoint.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional path to fine-tuned Video-LoRA safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cosmos7b-features"),
        help="Output directory to store extracted feature cache and manifest.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="14,20",
        help="Comma-separated list of transformer blocks to tap (default: 14,20).",
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
        help="Concatenate tapped layers along the channel dimension (e.g. 4096*2 = 8192).",
    )
    parser.add_argument(
        "--text-conditioning",
        type=str,
        default=None,
        help="Optional path to precomputed prompt embeddings (.safetensors).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="take cube out of box",
        help="Task text prompt string when precomputed embeddings are not provided.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run extraction on (default: cuda).",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    hidden_layers = tuple(int(x.strip()) for x in args.hidden_layers.split(",") if x.strip())
    pool_factor = args.pool_spatial if args.pool_spatial and args.pool_spatial > 1 else None

    print(f"[Cosmos 7B Extractor] Target layers: {hidden_layers}")
    print(f"[Cosmos 7B Extractor] Device: {args.device}, Dtype: {args.dtype}")
    print(f"[Cosmos 7B Extractor] Spatial pooling: {pool_factor}, Concat layers: {args.concat_layers}")

    if args.dry_run:
        print(f"[Cosmos 7B Extractor] --- RUNNING SYNTHETIC DRY-RUN ON {args.device.upper()} ---")
        num_layers = 22  # Covers layers 14 and 20 and matches Video-LoRA
        # Build lightweight dummy model matching Cosmos 7B block structure and names
        dummy_model = build_cosmos7b_dummy_model(
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            text_embed_dim=32,
            device=args.device,
            dtype=torch.float32,
        )
        hidden_dim = 4 * 16  # 64
        extractor_config = Cosmos7BExtractorConfig(
            hidden_layers=hidden_layers,
            patch_size=(1, 2, 2),
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            hidden_dim=hidden_dim,
            text_embed_dim=32,
            pool_spatial=pool_factor,
            concat_layers=args.concat_layers,
            device=args.device,
            dtype="float32",
        )
        extractor = Cosmos7BExtractor(config=extractor_config, model=dummy_model)
    else:
        extractor_config = Cosmos7BExtractorConfig(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dtype=args.dtype,
            hidden_layers=hidden_layers,
            patch_size=(1, 2, 2),
            pool_spatial=pool_factor,
            concat_layers=args.concat_layers,
        )
        extractor = Cosmos7BExtractor(config=extractor_config)

    # Optional LoRA injection & loading
    if args.lora_path is not None:
        print(f"[Cosmos 7B Extractor] Injecting and loading Video-LoRA from {args.lora_path}...")
        lora_sd = load_file(str(args.lora_path))
        lora_blocks = set()
        for k in lora_sd.keys():
            parts = k.split(".")
            if len(parts) > 1 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                lora_blocks.add(int(parts[1]))
        target_blocks = tuple(sorted(lora_blocks)) if lora_blocks else hidden_layers

        lora_cfg = Cosmos7BLoRAConfig(
            rank=16,
            alpha=16.0,
            target_blocks=target_blocks,
        )
        inject_cosmos7b_lora(extractor.transformer, lora_cfg)
        load_cosmos7b_lora(extractor.transformer, args.lora_path)
        print("[Cosmos 7B Extractor] LoRA weights loaded successfully.")

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
            print(
                f"[Cosmos 7B Extractor] Loaded robot dataset with {len(dataset)} frames from {args.dataset_root}"
            )
        except Exception as e:
            print(f"[Cosmos 7B Extractor] Note: Dataset loading skipped ({e})")

    if dataset is not None:
        sample_indices = list(range(0, len(dataset), args.stride))
        if args.dry_run and args.num_samples is None:
            sample_indices = sample_indices[:5]
        elif args.num_samples is not None and args.num_samples > 0:
            sample_indices = sample_indices[: args.num_samples]
    else:
        sample_indices = list(
            range(args.num_samples if args.num_samples is not None else (5 if args.dry_run else 3))
        )

    print(f"[Cosmos 7B Extractor] Extracting {len(sample_indices)} samples (stride={args.stride})...")
    t0 = time.time()
    for count, idx in enumerate(sample_indices):
        state = None
        action = None
        sample_latents = None

        if dataset is not None and idx < len(dataset):
            row = dataset[idx]
            if "observation.state" in row:
                state = row["observation.state"].float()
            if "action" in row:
                act = row["action"].float()
                if act.ndim == 1:
                    action = act.unsqueeze(0).expand(30, -1).clone()
                else:
                    action = act.clone()
            if "observation.images.front" in row:
                rgb = row["observation.images.front"]
                if rgb.ndim == 3:
                    rgb = rgb.unsqueeze(0).expand(4, -1, -1, -1)
                elif rgb.shape[0] >= 4:
                    rgb = rgb[:4]
                rgb_norm = (rgb.float() / 255.0 - 0.5) * 2.0
                down = torch.nn.functional.interpolate(
                    rgb_norm, size=(32, 32), mode="bilinear", align_corners=False
                )
                p = down.view(4, 3, 16, 2, 16, 2).permute(0, 1, 3, 5, 2, 4).reshape(4, 12, 16, 16)
                sample_latents = (
                    torch.nn.functional.pad(p, (0, 0, 0, 0, 0, 4))
                    .permute(1, 0, 2, 3)
                    .unsqueeze(0)
                    .to(device=args.device, dtype=extractor.dtype)
                )

        if sample_latents is None:
            if args.dry_run:
                sample_latents = torch.randn(1, 16, 4, 8, 8, device=args.device, dtype=torch.float32)
            else:
                sample_latents = torch.randn(1, 16, 4, 16, 16, device=args.device, dtype=extractor.dtype)

        if args.dry_run:
            sample_text = torch.randn(1, 8, 32, device=args.device, dtype=torch.float32)
        else:
            torch.manual_seed(abs(hash(args.prompt)) % (2**31 - 1))
            sample_text = torch.randn(
                1, 16, extractor_config.text_embed_dim, device=args.device, dtype=extractor.dtype
            )

        out: Cosmos7BExtractionOutput = extractor.extract(
            hidden_states=sample_latents,
            text_conditioning=sample_text,
            timestep=0.0,
            fps=24,
        )

        artifact_name = f"cosmos7b_sample_{count:05d}.safetensors"
        artifact_path = args.output_dir / artifact_name

        if state is None:
            state = torch.randn(6, dtype=torch.float32)
        if action is None:
            action = torch.randn(30, 6, dtype=torch.float32)

        tensors_to_save = {
            "features": out.features.detach().cpu().clone(),
            "grid_coords": out.grid_coords.detach().cpu().clone(),
            "state": state,
            "action": action,
        }
        for lyr, feat in out.features_by_layer.items():
            tensors_to_save[f"layer_{lyr}_flat"] = feat.detach().cpu().clone()
            tensors_to_save[f"layer_{lyr}_grid"] = out.grid_features_by_layer[lyr].detach().cpu().clone()

        metadata = {
            "sample_index": str(count),
            "token_count": str(out.num_tokens),
            "feature_dim": str(out.feature_dim),
            "grid_shape": json.dumps(out.grid_shape),
            "tapped_layers": json.dumps(list(hidden_layers)),
        }
        save_file(tensors_to_save, str(artifact_path), metadata=metadata)

        manifest_entries.append(
            {
                "sample_index": count,
                "dataset_index": idx,
                "artifact_file": artifact_name,
                "num_tokens": out.num_tokens,
                "feature_dim": out.feature_dim,
                "grid_shape": list(out.grid_shape),
            }
        )

    manifest = {
        "model": "Cosmos-1.0-Diffusion-7B",
        "tapped_layers": list(hidden_layers),
        "concat_layers": args.concat_layers,
        "pool_spatial": pool_factor,
        "total_samples": len(manifest_entries),
        "entries": manifest_entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"[Cosmos 7B Extractor] Completed extraction of {len(manifest_entries)} samples in {elapsed:.2f}s "
        f"-> saved to {args.output_dir}"
    )
    print(f"[Cosmos 7B Extractor] Manifest saved to {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
