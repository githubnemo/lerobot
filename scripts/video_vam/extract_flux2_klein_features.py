#!/usr/bin/env python3
"""Extract representations from FLUX.2 [klein] (9B) using multi-reference conditioning.

Supports:
- Multi-reference conditioning: observation history (e.g. t-2, t-1, t) + optional goal frames.
- Dual tapped locations:
    a) Junction between double-stream and single-stream transformer blocks (--tap-location junction).
    b) Intermediate/later step of the rectified flow generative trajectory (--tap-location rectified_flow_trajectory).
- Video-LoRA weight loading.
- Saving features and manifests for SmolExpert downstream policy training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.vam.flux2_klein_extractor import (
    Flux2KleinExtractionOutput,
    Flux2KleinExtractor,
    Flux2KleinExtractorConfig,
    build_flux2_klein_dummy_model,
    prepare_multi_reference_conditioning,
)
from lerobot.policies.vam.flux2_klein_lora import (
    Flux2KleinLoRAConfig,
    inject_flux2_klein_lora,
    load_flux2_klein_lora,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract features from FLUX.2 [klein] (9B) with multi-reference conditioning."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path or HuggingFace ID for FLUX.2 [klein] (9B) checkpoint.",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help="Optional path to fine-tuned FLUX.2 [klein] LoRA safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/flux2-klein-features"),
        help="Directory to save extracted feature artifacts and manifest.json.",
    )
    parser.add_argument(
        "--tap-location",
        type=str,
        choices=["junction", "rectified_flow_trajectory"],
        default="junction",
        help="Feature tap point: 'junction' (double/single stream boundary) or 'rectified_flow_trajectory'.",
    )
    parser.add_argument(
        "--rf-num-steps",
        type=int,
        default=4,
        help="Total steps in rectified flow trajectory when using trajectory tap point (default: 4).",
    )
    parser.add_argument(
        "--rf-tap-step",
        type=int,
        default=2,
        help="Step index along rectified flow trajectory to tap (e.g. 2 for intermediate/later step).",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=3,
        help="Number of past observation history frames to condition on (default: 3 for t-2, t-1, t).",
    )
    parser.add_argument(
        "--use-goal-frame",
        action="store_true",
        default=True,
        help="Include goal frame conditioning alongside observation history.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda).",
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
    device = torch.device(args.device)

    print(f"[FLUX.2 klein Extractor] Device: {device}")
    print(f"[FLUX.2 klein Extractor] Tap location: {args.tap_location}")
    if args.tap_location == "rectified_flow_trajectory":
        print(f"[FLUX.2 klein Extractor] RF Steps: {args.rf_num_steps}, Tap step: {args.rf_tap_step}")
    print(
        f"[FLUX.2 klein Extractor] History frames: {args.history_frames}, Goal conditioning: {args.use_goal_frame}"
    )

    if args.dry_run:
        print("[FLUX.2 klein Extractor] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        dummy = build_flux2_klein_dummy_model(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=128,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 4, 4),
            device=device,
        )
        extractor_cfg = Flux2KleinExtractorConfig(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            hidden_dim=32,
            in_channels=128,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 4, 4),
            tap_location=args.tap_location,
            rf_num_steps=args.rf_num_steps,
            rf_tap_step=args.rf_tap_step,
            device=args.device,
            dtype="float32",
        )
        extractor = Flux2KleinExtractor(config=extractor_cfg, model=dummy)
    else:
        extractor_cfg = Flux2KleinExtractorConfig(
            checkpoint_path=args.checkpoint_path,
            tap_location=args.tap_location,
            rf_num_steps=args.rf_num_steps,
            rf_tap_step=args.rf_tap_step,
            device=args.device,
            dtype=args.dtype,
            num_layers=8,
            num_single_layers=0,
        )
        extractor = Flux2KleinExtractor(config=extractor_cfg)

    # Optional LoRA injection & loading
    if args.lora_path is not None:
        print(f"[FLUX.2 klein Extractor] Loading LoRA from {args.lora_path}...")
        lora_cfg = Flux2KleinLoRAConfig()
        inject_flux2_klein_lora(extractor.transformer, lora_cfg)
        load_flux2_klein_lora(extractor.transformer, args.lora_path)
        print("[FLUX.2 klein Extractor] LoRA loaded successfully.")

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
                f"[FLUX.2 klein Extractor] Loaded robot dataset with {len(dataset)} frames from {args.dataset_root}"
            )
        except Exception as e:
            print(f"[FLUX.2 klein Extractor] Note: Dataset loading skipped ({e})")

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

    print(f"[FLUX.2 klein Extractor] Extracting {len(sample_indices)} samples (stride={args.stride})...")
    t0 = time.time()
    for count, idx in enumerate(sample_indices):
        state = None
        action = None
        target_latent = None
        hist_latents = None
        goal_latent = None

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
                rgb = row["observation.images.front"]  # [5, 3, 480, 640] or [3, 480, 640]
                if rgb.ndim == 3:
                    rgb = rgb.unsqueeze(0).expand(5, -1, -1, -1)
                elif rgb.shape[0] < 5:
                    rgb = torch.nn.functional.pad(rgb, (0, 0, 0, 0, 0, 0, 0, 5 - rgb.shape[0]))
                rgb_norm = (rgb[:5].float() / 255.0 - 0.5) * 2.0
                down = torch.nn.functional.interpolate(
                    rgb_norm, size=(32, 32), mode="bilinear", align_corners=False
                )
                p48 = down.view(5, 3, 8, 4, 8, 4).permute(0, 1, 3, 5, 2, 4).reshape(5, 48, 8, 8)
                p128 = (
                    torch.nn.functional.pad(p48, (0, 0, 0, 0, 0, 80))
                    .unsqueeze(1)
                    .to(device=device, dtype=extractor.dtype)
                )
                hist_latents = [p128[i] for i in range(min(args.history_frames, 3))]
                target_latent = p128[3]
                goal_latent = p128[4] if args.use_goal_frame else None

        if target_latent is None:
            c_in = extractor_cfg.in_channels
            h_lat, w_lat = (4, 4) if args.dry_run else (8, 8)
            target_latent = torch.randn(1, c_in, h_lat, w_lat, device=device, dtype=extractor.dtype)
            hist_latents = [
                torch.randn(1, c_in, h_lat, w_lat, device=device, dtype=extractor.dtype)
                for _ in range(args.history_frames)
            ]
            goal_latent = (
                torch.randn(1, c_in, h_lat, w_lat, device=device, dtype=extractor.dtype)
                if args.use_goal_frame
                else None
            )

        cond = prepare_multi_reference_conditioning(
            target_latent=target_latent,
            observation_history=hist_latents,
            goal_latent=goal_latent,
            joint_attention_dim=extractor_cfg.joint_attention_dim,
            device=device,
            dtype=extractor.dtype,
        )

        out: Flux2KleinExtractionOutput = extractor.extract(cond)

        artifact_name = f"flux2_klein_sample_{count:05d}.safetensors"
        artifact_path = args.output_dir / artifact_name

        if state is None:
            state = torch.randn(6, dtype=torch.float32)
        if action is None:
            action = torch.randn(30, 6, dtype=torch.float32)

        tensors_to_save = {
            "features": out.features.detach().cpu().clone(),
            "visual_tokens": out.visual_tokens.detach().cpu().clone(),
            "state": state,
            "action": action,
        }
        if out.text_tokens is not None:
            tensors_to_save["text_tokens"] = out.text_tokens.detach().cpu().clone()
        if out.junction_features is not None:
            tensors_to_save["junction_features"] = out.junction_features.detach().cpu().clone()

        metadata = {
            "sample_index": str(count),
            "tap_location": out.tap_location,
            "trajectory_step": str(out.trajectory_step),
            "num_tokens": str(out.num_tokens),
            "feature_dim": str(out.feature_dim),
            "token_slices": json.dumps(out.token_slices),
        }
        save_file(tensors_to_save, str(artifact_path), metadata=metadata)

        manifest_entries.append(
            {
                "sample_index": count,
                "dataset_index": idx,
                "artifact_file": artifact_name,
                "tap_location": out.tap_location,
                "trajectory_step": out.trajectory_step,
                "num_tokens": out.num_tokens,
                "feature_dim": out.feature_dim,
                "token_slices": out.token_slices,
            }
        )

    manifest = {
        "model": "FLUX.2-klein-9B",
        "tap_location": args.tap_location,
        "rf_num_steps": args.rf_num_steps if args.tap_location == "rectified_flow_trajectory" else None,
        "rf_tap_step": args.rf_tap_step if args.tap_location == "rectified_flow_trajectory" else None,
        "history_frames": args.history_frames,
        "use_goal_frame": args.use_goal_frame,
        "total_samples": len(manifest_entries),
        "entries": manifest_entries,
    }
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"[FLUX.2 klein Extractor] Successfully extracted {len(manifest_entries)} samples in {elapsed:.2f}s "
        f"-> saved to {args.output_dir}"
    )
    print(f"[FLUX.2 klein Extractor] Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
