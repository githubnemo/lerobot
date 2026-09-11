#!/usr/bin/env python3
"""Standardized Protocol 1.0 Feature Cache Builder for Video Action Models (VAM).

Extracts and caches frozen/adapted spatiotemporal representations from video backbones
(Cosmos 7B, Cosmos 14B, FLUX.2 klein, Cosmos 3 Edge, etc.) on robot trajectories
strictly conforming to Protocol 1.0:
  - Disjoint episode splits (Train: 0-31, Val: 32-39; 88 fixed validation anchors).
  - True VAE encoding contract (no heuristic downsampling mocks).
  - Exact action masking: persists `action_is_pad`, proprioceptive `state`, and 30-step `action`.
  - Cryptographic validation via SHA-256 hashes in atomic manifest manifests.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import Tensor

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.base import (
    PROTOCOL_1_0_TRAIN_EPISODES,
    BaseVAMExtractor,
    ProtocolViolationError,
    enforce_protocol_1_0_split,
)
from lerobot.policies.vam.cosmos7b_extractor import (
    Cosmos7BExtractor,
    Cosmos7BExtractorConfig,
    build_cosmos7b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_extractor import (
    Cosmos14BExtractor,
    Cosmos14BExtractorConfig,
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.flux2_klein_extractor import (
    Flux2KleinExtractor,
    Flux2KleinExtractorConfig,
    build_flux2_klein_dummy_model,
    prepare_multi_reference_conditioning,
)

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_VAE_PATH = Path("/home/anton/.cache/video-vam/cosmos-14b/vae")
DEFAULT_COSMOS14B_PATH = Path("/home/anton/.cache/video-vam/cosmos-14b")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hexadecimal digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_episode_range(value: str) -> tuple[int, ...]:
    """Parse comma-separated list and ranges like '0-31' or '0,1,2,32-39'."""
    episodes: set[int] = set()
    parts = value.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            if start < 0 or end < start:
                raise ValueError(f"Invalid episode range: {part}")
            for ep in range(start, end + 1):
                episodes.add(ep)
        else:
            episodes.add(int(part))
    if not episodes or min(episodes) < 0:
        raise ValueError("Episode selection must contain non-negative integers")
    return tuple(sorted(episodes))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        choices=["cosmos7b", "cosmos14b", "flux2_klein"],
        required=True,
        help="Target video action model backbone",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Path to backbone checkpoint (defaults to cached standard weights if present)",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=None,
        help="Path to VAE checkpoint directory",
    )
    parser.add_argument("--protocol", choices=("protocol1", "scale100"), default="protocol1")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-embedding-path", type=Path)
    parser.add_argument("--prompt", default=CUBE_OUT_OF_BOX_CONTRACT.task)
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        help="Hugging Face repo ID for dataset (default: canonical 40-episode dataset)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to LeRobot cube_out_of_box_dataset directory",
    )
    parser.add_argument(
        "--train-episodes",
        type=str,
        default="0-31",
        help="Train episode range under Protocol 1.0 (default: 0-31)",
    )
    parser.add_argument(
        "--val-episodes",
        type=str,
        default="32-39",
        help="Validation episode range under Protocol 1.0 (default: 32-39)",
    )
    parser.add_argument(
        "--eval2-episodes",
        type=str,
        default=None,
        help="Optional Eval-Set 2 episode range (e.g. 90-99 for 100-episode benchmark)",
    )
    parser.add_argument("--train-stride", type=int, default=3, help="Stride for training episodes")
    parser.add_argument(
        "--val-stride", type=int, default=20, help="Stride for validation episodes (88 anchors)"
    )
    parser.add_argument(
        "--eval2-stride", type=int, default=20, help="Stride for Eval-Set 2 episodes (default: 20)"
    )
    parser.add_argument("--pool-spatial", type=int, default=2, help="Spatial average pooling factor")
    parser.add_argument(
        "--lora-weights",
        "--lora-path",
        dest="lora_weights",
        type=Path,
        default=None,
        help="Optional path to fine-tuned Video-LoRA safetensors weights to adapt extractor",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Extraction batch size (default: 1)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Target cache directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run lightweight dry-run without loading full models"
    )
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="Cap training samples (e.g. for testing)"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None, help="Cap validation samples (e.g. for testing)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached files")
    parser.add_argument("--resume", action="store_true", help="Resume extraction skipping existing files")
    return parser.parse_args(argv)


def build_extractor(
    backbone: str,
    checkpoint_path: Path | None,
    vae_path: Path | None,
    device: str,
    dtype: str,
    pool_spatial: int | None,
    dry_run: bool = False,
    lora_weights: Path | None = None,
) -> BaseVAMExtractor:
    """Instantiate and initialize the standardized BaseVAMExtractor implementation."""
    if not dry_run and checkpoint_path is None and backbone != "cosmos14b":
        raise ValueError("Production extraction requires --checkpoint-path")
    if lora_weights is not None and not lora_weights.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found: {lora_weights}")
    dev = torch.device(device)

    if backbone == "cosmos7b":
        if dry_run:
            dummy = build_cosmos7b_dummy_model(
                num_layers=22,
                num_attention_heads=4,
                attention_head_dim=16,
                text_embed_dim=32,
                device=device,
                dtype=torch.float32,
            )
            cfg = Cosmos7BExtractorConfig(
                device=device,
                dtype="float32",
                hidden_layers=(14, 20),
                num_layers=22,
                num_attention_heads=4,
                attention_head_dim=16,
                hidden_dim=64,
                text_embed_dim=32,
                pool_spatial=pool_spatial,
                concat_layers=True,
            )
            extractor = Cosmos7BExtractor(config=cfg, model=dummy)
        else:
            cfg = Cosmos7BExtractorConfig(
                checkpoint_path=checkpoint_path,
                vae_path=vae_path,
                device=device,
                dtype=dtype,
                hidden_layers=(14, 20),
                pool_spatial=pool_spatial,
                concat_layers=True,
            )
            extractor = Cosmos7BExtractor(config=cfg)

        if lora_weights is not None and lora_weights.is_file():
            print(f"[Cosmos 7B] Injecting and loading Video-LoRA weights from {lora_weights}...")
            from lerobot.policies.vam.cosmos7b_lora import (
                Cosmos7BLoRAConfig,
                inject_cosmos7b_lora,
                load_cosmos7b_lora,
            )

            lora_cfg = Cosmos7BLoRAConfig(rank=16, alpha=16.0)
            inject_cosmos7b_lora(extractor.transformer, lora_cfg)
            load_cosmos7b_lora(extractor.transformer, lora_weights)
            extractor.config.extra_kwargs["lora_weights"] = str(lora_weights)
            print("[Cosmos 7B] Successfully loaded Video-LoRA weights.")
        return extractor

    elif backbone == "cosmos14b":
        ckpt = checkpoint_path or (DEFAULT_COSMOS14B_PATH if DEFAULT_COSMOS14B_PATH.is_dir() else None)
        if dry_run:
            dummy = build_cosmos14b_dummy_model(
                num_layers=32,
                num_attention_heads=4,
                attention_head_dim=16,
                text_embed_dim=32,
                device=device,
                dtype=torch.float32,
            )
            cfg = Cosmos14BExtractorConfig(
                device=device,
                dtype="float32",
                hidden_layers=(18, 30),
                num_layers=32,
                num_attention_heads=4,
                attention_head_dim=16,
                hidden_dim=64,
                text_embed_dim=32,
                pool_spatial=pool_spatial,
                concat_layers=True,
                block_streaming=False,
                fp8_linear=False,
            )
            extractor = Cosmos14BExtractor(config=cfg, model=dummy)
        else:
            if ckpt is None:
                raise FileNotFoundError("Cosmos14B checkpoint is required outside --dry-run")
            cfg = Cosmos14BExtractorConfig(
                checkpoint_path=ckpt,
                vae_path=vae_path,
                device=device,
                dtype=dtype,
                hidden_layers=(18, 30),
                pool_spatial=pool_spatial,
                concat_layers=True,
                block_streaming=True,
                fp8_linear=True,
            )
            extractor = Cosmos14BExtractor(config=cfg)

        if lora_weights is not None and lora_weights.is_file():
            print(f"[Cosmos 14B] Injecting and loading Video-LoRA weights from {lora_weights}...")
            from safetensors.torch import load_file

            from lerobot.policies.vam.cosmos14b_lora import (
                Cosmos14BLoRAConfig,
                inject_cosmos14b_quantized_lora,
                load_cosmos14b_lora,
            )

            lora_sd = load_file(str(lora_weights))
            lora_blocks = set()
            for k in lora_sd:
                parts = k.split(".")
                if len(parts) > 1 and parts[0] == "transformer_blocks" and parts[1].isdigit():
                    lora_blocks.add(int(parts[1]))
            target_blocks = (
                tuple(sorted(lora_blocks)) if lora_blocks else tuple(range(36 if not dry_run else 32))
            )

            lora_cfg = Cosmos14BLoRAConfig(
                rank=16,
                alpha=16.0,
                target_blocks=target_blocks,
                quantization="fp8" if cfg.fp8_linear and not dry_run and dev.type == "cuda" else "none",
            )
            inject_cosmos14b_quantized_lora(extractor.transformer, lora_cfg)
            load_cosmos14b_lora(extractor.transformer, lora_weights)
            extractor.config.extra_kwargs["lora_weights"] = str(lora_weights)
            print(f"[Cosmos 14B] Successfully loaded Video-LoRA weights into {len(target_blocks)} blocks.")
        return extractor

    elif backbone == "flux2_klein":
        if dry_run:
            dummy = build_flux2_klein_dummy_model(
                num_layers=2,
                num_single_layers=2,
                num_attention_heads=2,
                attention_head_dim=16,
                in_channels=16,
                joint_attention_dim=32,
                axes_dims_rope=(4, 4, 4, 4),
                device=device,
                dtype=torch.float32,
            )
            cfg = Flux2KleinExtractorConfig(
                device=device,
                dtype="float32",
                in_channels=16,
                num_layers=2,
                num_single_layers=2,
                attention_head_dim=16,
                num_attention_heads=2,
                hidden_dim=32,
                joint_attention_dim=32,
                axes_dims_rope=(4, 4, 4, 4),
                tap_location="junction",
            )
            extractor = Flux2KleinExtractor(config=cfg, model=dummy)
        else:
            cfg = Flux2KleinExtractorConfig(
                checkpoint_path=checkpoint_path,
                vae_path=vae_path,
                device=device,
                dtype=dtype,
                num_layers=8,
                num_single_layers=0,
                tap_location="junction",
            )
            extractor = Flux2KleinExtractor(config=cfg)

        if lora_weights is not None and lora_weights.is_file():
            print(f"[FLUX.2 klein] Injecting and loading Video-LoRA weights from {lora_weights}...")
            from lerobot.policies.vam.flux2_klein_lora import (
                Flux2KleinLoRAConfig,
                inject_flux2_klein_lora,
                load_flux2_klein_lora,
            )

            lora_cfg = Flux2KleinLoRAConfig(
                rank=16,
                alpha=16.0,
                target_double_blocks=tuple(range(cfg.num_layers if not dry_run else 2)),
                target_single_blocks=tuple(range(cfg.num_single_layers if not dry_run else 2)),
            )
            inject_flux2_klein_lora(extractor.transformer, lora_cfg)
            load_flux2_klein_lora(extractor.transformer, lora_weights)
            extractor.config.extra_kwargs["lora_weights"] = str(lora_weights)
            print("[FLUX.2 klein] Successfully loaded Video-LoRA weights.")
        return extractor

    raise ValueError(f"Unknown backbone: {backbone}")


def extract_split(
    extractor: BaseVAMExtractor,
    dataset: LeRobotDataset | None,
    episodes: Sequence[int],
    stride: int,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    split_name: str,
    batch_size: int = 8,
    max_samples: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> Path:
    """Extract and write feature cache artifacts and manifest for one episode split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    if dataset is None and not dry_run:
        raise ValueError("Production extraction requires a real dataset")
    if stride <= 0 or batch_size <= 0 or (max_samples is not None and max_samples <= 0):
        raise ValueError("Stride, batch size and sample limits must be positive")
    if resume and overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if resume:
        raise ValueError("Unverified legacy cache resume is disabled; use a new output directory")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Existing cache: {manifest_path}; use a new directory or explicit --overwrite")

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    t0 = time.time()

    # Determine frame indices for each episode in split
    target_tasks: list[tuple[int, int]] = []
    if dataset is not None:
        for ep in episodes:
            row = dataset.meta.episodes[ep]
            from_idx = int(row["dataset_from_index"])
            to_idx = int(row["dataset_to_index"])
            # CUBE_OUT_OF_BOX_CONTRACT history_offsets = (-4, -3, -2, -1, 0), so start at from_idx + 4
            for f in range(from_idx + 4, to_idx, stride):
                target_tasks.append((ep, f))
    else:
        # Synthetic mock tasks for testing without local dataset
        for ep in episodes:
            for f in range(4, 4 + stride * 2, stride):
                target_tasks.append((ep, f))

    if max_samples is not None:
        target_tasks = target_tasks[:max_samples]

    print(
        f"[{split_name.upper()}] Processing {len(target_tasks)} frames across episodes {episodes[:3]}...{episodes[-1:]} (batch_size={batch_size})"
    )

    for b_start in range(0, len(target_tasks), batch_size):
        b_tasks = target_tasks[b_start : b_start + batch_size]
        pending_items: list[tuple[int, int, str, Path]] = []
        b_rgb: list[Tensor] = []
        b_state: list[Tensor] = []
        b_action: list[Tensor] = []
        b_pad: list[Tensor] = []

        for ep, frame in b_tasks:
            filename = f"episode-{ep:04d}-frame-{frame:06d}.safetensors"
            out_file = output_dir / filename

            if out_file.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite cache artifact {out_file}")
            pending_items.append((ep, frame, filename, out_file))

            if dataset is not None and not dry_run:
                sample = dataset[frame]
                camera = sample[CUBE_OUT_OF_BOX_CONTRACT.camera_key]
                state = sample[CUBE_OUT_OF_BOX_CONTRACT.state_key].float()
                action = sample[CUBE_OUT_OF_BOX_CONTRACT.action_key].float()
                action_is_pad = sample[f"{CUBE_OUT_OF_BOX_CONTRACT.action_key}_is_pad"].bool()

                if camera.ndim != 4 or camera.shape[:2] != (5, 3):
                    raise ValueError(f"Expected actual five-frame TCHW history, got {camera.shape}")
                rgb_ready = camera.permute(1, 0, 2, 3).contiguous()

                b_rgb.append(rgb_ready)
                b_state.append(state)
                b_action.append(action)
                b_pad.append(action_is_pad)
            else:
                b_state.append(torch.randn(6, dtype=torch.float32))
                b_action.append(torch.randn(30, 6, dtype=torch.float32))
                b_pad.append(torch.zeros(30, dtype=torch.bool))

        if not pending_items:
            continue

        # Forward pass through extractor
        if dataset is not None and not dry_run:
            batch_rgb_tensor = torch.stack(b_rgb, dim=0).to(device=device, dtype=dtype)  # [B, 3, 5, 256, 256]
            if isinstance(extractor, Flux2KleinExtractor):
                lat = extractor.encode_latents(batch_rgb_tensor)
                b_out_list: list[Tensor] = []
                b_coords_list: list[Tensor] = []
                for i in range(lat.shape[0]):
                    target = lat[i : i + 1, :, -1]
                    hist = [lat[i : i + 1, :, k] for k in range(min(3, lat.shape[2] - 1))]
                    cond = prepare_multi_reference_conditioning(
                        target_latent=target,
                        observation_history=hist,
                        joint_attention_dim=extractor.config.joint_attention_dim,
                        encoder_hidden_states=extractor.cache_text_embedding,
                        device=device,
                        dtype=dtype,
                    )
                    out_single = extractor.extract(cond_inputs=cond)
                    b_out_list.append(out_single.features.squeeze(0))
                    b_coords_list.append(out_single.grid_coords.squeeze(0))
                batch_features = torch.stack(b_out_list, dim=0)
                batch_coords = torch.stack(b_coords_list, dim=0)
            else:
                out = extractor.extract(
                    rgb_frames=batch_rgb_tensor,
                    text_conditioning=extractor.cache_text_embedding.expand(
                        batch_rgb_tensor.shape[0], -1, -1
                    ),
                    timestep=0.001,
                    fps=CUBE_OUT_OF_BOX_CONTRACT.fps,
                )
                batch_features = out.features
                batch_coords = out.grid_coords
        else:
            c_in = getattr(extractor.config, "in_channels", 16)
            mock_lat = torch.randn(len(pending_items), c_in, 1, 8, 8, device=device, dtype=extractor.dtype)
            if isinstance(extractor, Flux2KleinExtractor):
                b_out_list = []
                b_coords_list = []
                for i in range(len(pending_items)):
                    target = mock_lat[i : i + 1, :, 0]
                    cond = prepare_multi_reference_conditioning(
                        target_latent=target,
                        observation_history=[target],
                        joint_attention_dim=extractor.config.joint_attention_dim,
                        device=device,
                        dtype=extractor.dtype,
                    )
                    out_single = extractor.extract(cond_inputs=cond)
                    b_out_list.append(out_single.features.squeeze(0))
                    b_coords_list.append(out_single.grid_coords.squeeze(0))
                batch_features = torch.stack(b_out_list, dim=0)
                batch_coords = torch.stack(b_coords_list, dim=0)
            else:
                out = extractor.extract(latents=mock_lat)
                batch_features = out.features
                batch_coords = out.grid_coords

        for i, (ep, frame, filename, out_file) in enumerate(pending_items):
            feat_i = batch_features[i].detach().cpu().to(dtype=torch.float32).contiguous()
            coords_i = batch_coords[i].detach().cpu().contiguous()
            state_i = b_state[i].detach().cpu().to(dtype=torch.float32).contiguous()
            act_i = b_action[i].detach().cpu().to(dtype=torch.float32).contiguous()
            pad_i = b_pad[i].detach().cpu().to(dtype=torch.bool).contiguous()

            tensors_to_save = {
                "context": feat_i,
                "features": feat_i.clone(),
                "state": state_i,
                "action": act_i,
                "target_action": act_i.clone(),
                "action_is_pad": pad_i,
                "grid_coords": coords_i,
            }

            save_file(tensors_to_save, str(out_file))
            s_hash = sha256_file(out_file)
            size = out_file.stat().st_size
            total_bytes += size

            entries.append(
                {
                    "sample_id": f"episode-{ep:04d}-frame-{frame:06d}",
                    "episode_index": ep,
                    "frame_index": frame,
                    "window_indices": list(range(frame - 4, frame + 1)),
                    "safetensors": filename,
                    "safetensors_sha256": s_hash,
                    "bytes": size,
                    "num_tokens": feat_i.shape[0],
                    "feature_dim": feat_i.shape[-1],
                }
            )

        extracted_count = len(entries)
        rate = extracted_count / max(0.1, time.time() - t0)
        print(
            f"[{split_name.upper()}] Processed {extracted_count}/{len(target_tasks)} ({rate:.1f} samples/s)..."
        )

    if not entries:
        raise ValueError("Selection produced no cache entries")
    # Write atomic manifest.json
    manifest = {
        "schema_version": 1,
        "cache_schema_version": 1,
        "backbone": extractor.config.backbone_name,
        "split_name": split_name,
        "episodes": list(episodes),
        "stride": stride,
        "total_samples": len(entries),
        "total_bytes": total_bytes,
        "provenance": dict(extractor.get_provenance()),
        "entries": entries,
    }
    manifest["provenance"].update(getattr(extractor, "cache_feature_identity", {}))
    manifest["dataset"] = dict(getattr(extractor, "cache_dataset_identity", {}))
    manifest["protocol"] = getattr(extractor, "cache_protocol", "protocol1")
    manifest["dry_run"] = dry_run
    manifest["preprocessing_version"] = "native-history-v2"
    manifest["global_seed"] = getattr(extractor, "cache_seed", 0)
    temporary = manifest_path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, manifest_path)

    print(
        f"[{split_name.upper()}] Successfully cached {len(entries)} samples to {manifest_path} in {time.time() - t0:.1f}s"
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_eps = parse_episode_range(args.train_episodes)
    val_eps = parse_episode_range(args.val_episodes)
    eval2_eps = parse_episode_range(args.eval2_episodes) if args.eval2_episodes else ()

    # 1. Enforce Protocol 1.0 Guard
    enforce_protocol_1_0_split(train_eps, val_eps, protocol=args.protocol)
    if eval2_eps and (args.protocol != "scale100" or not set(eval2_eps) <= set(range(90, 100))):
        raise ProtocolViolationError("Eval-2 requires scale100 and episodes 90..99")
    expected_repo = (
        CUBE_OUT_OF_BOX_CONTRACT.repo_id if args.protocol == "protocol1" else "Orellius/cube_out_of_box_v2"
    )
    if not args.dry_run and args.dataset_repo_id != expected_repo:
        raise ProtocolViolationError("Dataset repo does not match selected protocol")
    if not args.dry_run and args.protocol == "scale100" and not args.dataset_revision:
        raise ValueError("Scale100 requires a pinned --dataset-revision")
    torch.manual_seed(args.seed)
    if eval2_eps:
        overlap_train = sorted(set(train_eps) & set(eval2_eps))
        if overlap_train:
            raise ProtocolViolationError(
                f"DATA LEAKAGE: Eval-Set 2 overlaps with train episodes: {overlap_train}"
            )
        overlap_val = sorted(set(val_eps) & set(eval2_eps))
        if overlap_val:
            raise ProtocolViolationError(
                f"DATA LEAKAGE: Eval-Set 2 overlaps with val episodes: {overlap_val}"
            )
        canonical_train = set(PROTOCOL_1_0_TRAIN_EPISODES)
        leaked_train = sorted(set(eval2_eps) & canonical_train)
        if leaked_train:
            raise ProtocolViolationError(
                f"PROTOCOL VIOLATION: Eval-Set 2 contains canonical train episodes: {leaked_train}"
            )

    print("=" * 70)
    print(
        f"Protocol 1.0 Split Validated: Train {train_eps[0]}..{train_eps[-1]} ({len(train_eps)} eps), "
        f"Val {val_eps[0]}..{val_eps[-1]} ({len(val_eps)} eps)"
    )
    if eval2_eps:
        print(f"Eval-Set 2 Validated: {eval2_eps[0]}..{eval2_eps[-1]} ({len(eval2_eps)} eps)")
    print(f"Backbone: {args.backbone} | Output: {output_dir}")
    print("=" * 70)

    # 2. Instantiate Dataset
    dataset = None
    if not args.dry_run:
        repo_id = args.dataset_repo_id
        root_to_use = None
        if args.dataset_root and args.dataset_root.is_dir() and repo_id == CUBE_OUT_OF_BOX_CONTRACT.repo_id:
            root_to_use = str(args.dataset_root)

        print(
            f"Loading LeRobot dataset '{repo_id}' {'from ' + root_to_use if root_to_use else '(standard cache)'}..."
        )
        dataset = LeRobotDataset(
            repo_id,
            root=root_to_use,
            delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
            revision=CUBE_OUT_OF_BOX_CONTRACT.revision
            if repo_id == CUBE_OUT_OF_BOX_CONTRACT.repo_id
            else args.dataset_revision,
            download_videos=False,
        )

    if not args.dry_run and not args.text_embedding_path:
        raise ValueError("Production extraction requires --text-embedding-path with matching prompt metadata")
    resolved_checkpoint = args.checkpoint_path or (
        DEFAULT_COSMOS14B_PATH if args.backbone == "cosmos14b" else None
    )
    resolved_vae = args.vae_path or (resolved_checkpoint / "vae" if resolved_checkpoint else None)
    if not args.dry_run and (resolved_vae is None or not resolved_vae.is_dir()):
        raise FileNotFoundError(f"Native VAE checkpoint missing: {resolved_vae}")
    # 3. Instantiate Standardized Extractor
    print(f"Building {args.backbone} extractor on {args.device} ({args.dtype})...")
    extractor = build_extractor(
        backbone=args.backbone,
        checkpoint_path=resolved_checkpoint,
        vae_path=resolved_vae,
        device=args.device,
        dtype=args.dtype,
        pool_spatial=args.pool_spatial,
        dry_run=args.dry_run,
        lora_weights=args.lora_weights,
    )

    extractor.cache_dataset_identity = {
        "repo_id": expected_repo,
        "revision": CUBE_OUT_OF_BOX_CONTRACT.revision
        if args.protocol == "protocol1"
        else (args.dataset_revision or "synthetic"),
    }
    extractor.cache_protocol = args.protocol
    extractor.cache_seed = args.seed
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32

    if not args.dry_run:
        from lerobot.policies.vam.base.native_video import load_and_validate_prompt_embedding

        extractor.cache_text_embedding, text_provenance = load_and_validate_prompt_embedding(
            args.text_embedding_path,
            args.prompt,
            getattr(
                extractor.config, "joint_attention_dim", getattr(extractor.config, "text_embed_dim", 1024)
            ),
            device=device,
            dtype=dtype,
        )
        from lerobot.policies.vam.cosmos3_features import checkpoint_digest

        extractor.cache_feature_identity = {
            "preprocessing_version": "native-history-v2",
            "checkpoint_sha256": checkpoint_digest(resolved_checkpoint),
            "vae_sha256": checkpoint_digest(resolved_vae),
            "lora_sha256": sha256_file(args.lora_weights) if args.lora_weights else None,
            "text_sha256": sha256_file(args.text_embedding_path),
            "prompt": args.prompt,
            "fps": CUBE_OUT_OF_BOX_CONTRACT.fps,
        }
        extractor.config.extra_kwargs["text_conditioning"] = text_provenance
    # 4. Extract Train Split
    train_dir = output_dir / "train"
    extract_split(
        extractor=extractor,
        dataset=dataset,
        episodes=train_eps,
        stride=args.train_stride,
        output_dir=train_dir,
        device=device,
        dtype=dtype,
        split_name="train",
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    # 5. Extract Validation Split (Eval-Set 1 Historical Benchmark)
    val_dir = output_dir / "val"
    extract_split(
        extractor=extractor,
        dataset=dataset,
        episodes=val_eps,
        stride=args.val_stride,
        output_dir=val_dir,
        device=device,
        dtype=dtype,
        split_name="val",
        batch_size=args.batch_size,
        max_samples=args.max_val_samples,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    eval1_symlink = output_dir / "eval1"
    if not eval1_symlink.exists() and val_dir.is_dir():
        with contextlib.suppress(OSError):
            eval1_symlink.symlink_to("val")

    # 6. Extract Eval-Set 2 Split (New Benchmark) if requested
    if eval2_eps:
        eval2_dir = output_dir / "eval2"
        extract_split(
            extractor=extractor,
            dataset=dataset,
            episodes=eval2_eps,
            stride=args.eval2_stride,
            output_dir=eval2_dir,
            device=device,
            dtype=dtype,
            split_name="eval2",
            batch_size=args.batch_size,
            max_samples=args.max_val_samples,
            overwrite=args.overwrite,
            resume=args.resume,
            dry_run=args.dry_run,
        )

    print("=" * 70)
    print("Feature Extraction Completed Successfully!")
    print(f"Train manifest:  {train_dir / 'manifest.json'}")
    print(f"Eval-1 manifest: {val_dir / 'manifest.json'}")
    if eval2_eps:
        print(f"Eval-2 manifest: {output_dir / 'eval2' / 'manifest.json'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
