#!/usr/bin/env python3
"""Train Video LoRA on Cosmos 3 Edge using Flow-Matching on robot demonstration clips."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from diffusers import Cosmos3OmniPipeline

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos3_lora import (
    inject_cosmos3_lora,
    save_cosmos3_lora,
)
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index

DEFAULT_CHECKPOINT_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")
DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos3-edge-video-lora-20260903")
DEFAULT_CACHE_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge-latents-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--val-episodes", type=int, nargs="+", default=list(range(32, 40)))
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--val-stride", type=int, default=20)
    parser.add_argument(
        "--clip-frames", type=int, default=17, help="17 RGB frames -> 5 latents (2 cond + 3 gen)"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def get_clip_indices(
    dataset: LeRobotDataset, episodes: list[int], stride: int, clip_frames: int
) -> list[tuple[int, int]]:
    """Return list of (episode, start_frame) tuples."""
    samples = []
    for ep in episodes:
        row = dataset.meta.episodes[ep]
        from_idx = int(row["dataset_from_index"])
        to_idx = int(row["dataset_to_index"])
        # ensure full clip fits within episode bounds
        max_start = to_idx - clip_frames
        for f in range(from_idx, max_start + 1, stride):
            samples.append((ep, f))
    return samples


def preencode_latents(
    pipe: Cosmos3OmniPipeline,
    dataset: LeRobotDataset,
    samples: list[tuple[int, int]],
    cache_path: Path,
    clip_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Precompute and cache VAE latents [N, 48, T_lat, 30, 40]."""
    if cache_path.is_file():
        print(f"Loading pre-encoded latents from {cache_path}...")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(data)} latent clips. Shape: {data.shape}")
        return data

    print(f"Pre-encoding {len(samples)} clips using Wan2.2 VAE to {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_latents = []
    t0 = time.time()
    for idx, (_ep, start_f) in enumerate(samples):
        # Load clip_frames RGB frames
        frames = []
        for f in range(start_f, start_f + clip_frames):
            item = dataset[_relative_index(dataset, f)]
            img = item.get("observation.images.front", item.get("observation.images.phone"))  # [3, 480, 640]
            frames.append(img)
        # [1, 3, T, 480, 640] in [0, 1] normalized
        rgb_clip = torch.stack(frames, dim=1).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        if rgb_clip.max() > 1.0:
            rgb_clip = rgb_clip / 255.0
        # Normalize to [-1, 1] for VAE
        rgb_clip = (rgb_clip * 2.0) - 1.0

        with torch.no_grad():
            lat = pipe.vae.encode(rgb_clip).latent_dist.sample().cpu().squeeze(0)  # [48, T_lat, 30, 40]
        all_latents.append(lat)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"Encoded {idx + 1}/{len(samples)} clips ({rate:.1f} clips/s)...")

    stacked = torch.stack(all_latents, dim=0)  # [N, 48, T_lat, 30, 40]
    torch.save(stacked, cache_path)
    print(
        f"Saved {len(stacked)} latent clips to {cache_path} ({stacked.element_size() * stacked.numel() / 1e6:.1f} MB)"
    )
    return stacked


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    train_samples = get_clip_indices(dataset, args.train_episodes, args.train_stride, args.clip_frames)
    val_samples = get_clip_indices(dataset, args.val_episodes, args.val_stride, args.clip_frames)
    print(f"Found {len(train_samples)} training clips and {len(val_samples)} validation clips.")

    print("Loading Cosmos3OmniPipeline...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.checkpoint_dir),
        dtype=torch.bfloat16,
    ).to(device)

    # 1. Precompute or load VAE latents
    train_cache_file = cache_dir / f"train_latents_c{args.clip_frames}_s{args.train_stride}.pt"
    val_cache_file = cache_dir / f"val_latents_c{args.clip_frames}_s{args.val_stride}.pt"
    train_latents = preencode_latents(
        pipe, dataset, train_samples, train_cache_file, args.clip_frames, device
    )
    val_latents = preencode_latents(pipe, dataset, val_samples, val_cache_file, args.clip_frames, device)

    # Free VAE to reclaim memory for training
    pipe.vae.cpu()
    torch.cuda.empty_cache()

    # 2. Inject LoRA into Cosmos 3 Transformer
    print(f"Injecting LoRA (rank={args.rank}, alpha={args.alpha}) into all 28 transformer blocks...")
    wrapped_layers = inject_cosmos3_lora(
        pipe.transformer, rank=args.rank, alpha=args.alpha, block_indices=range(28)
    )
    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"LoRA injected into {len(wrapped_layers)} linear modules ({num_trainable / 1e6:.2f}M trainable parameters)."
    )

    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.train()

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=1e-6)

    # Pre-pack text segment (static prompt: 'take cube out of box')
    prompt_text = "take cube out of box"
    prompt_tokens = pipe.text_tokenizer(prompt_text).input_ids
    text_seg = pipe._prepare_text_segment(input_ids=prompt_tokens, device=device)

    # Pre-pack vision segment layout
    t_lat = train_latents.shape[2]  # 5
    sample_latent = torch.zeros(1, 48, t_lat, 30, 40, device=device, dtype=torch.bfloat16)
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=sample_latent,
        has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=24.0,
        curr=text_seg["und_len"],
        device=device,
        condition_frame_indexes=[0, 1],
    )
    num_noisy_tokens = vision_seg["num_noisy_vision_tokens"]
    position_ids = torch.cat([text_seg["text_mrope_ids"], vision_seg["vision_mrope_ids"]], dim=1)
    seq_len = text_seg["und_len"] + vision_seg["num_vision_tokens"]

    print(f"Starting training: {args.max_steps} steps, val every {args.val_every} steps...")
    best_val_loss = float("inf")
    patience_count = 0
    t_start = time.time()

    val_latents_gpu = val_latents.to(device=device, dtype=torch.bfloat16)
    fixed_val_timesteps = torch.tensor([250.0, 500.0, 750.0], device=device)

    for step in range(1, args.max_steps + 1):
        # Sample random training clip
        idx = random.randint(0, len(train_latents) - 1)
        x_clean = (
            train_latents[idx].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        )  # [1, 48, 5, 30, 40]

        # Sample diffusion timestep t in [0, 1000] and corresponding sigma
        t_val = random.uniform(10.0, 990.0)
        sigma = t_val / 1000.0
        epsilon = torch.randn_like(x_clean)

        # Flow matching perturbation on non-conditioning frames (2, 3, 4)
        x_noisy = x_clean.clone()
        x_noisy[:, :, 2:] = (1.0 - sigma) * x_clean[:, :, 2:] + sigma * epsilon[:, :, 2:]
        v_target = epsilon[:, :, 2:] - x_clean[:, :, 2:]

        vision_timesteps = torch.full((num_noisy_tokens,), t_val, device=device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds_vision, _, _ = pipe.transformer(
                input_ids=text_seg["input_ids"],
                text_indexes=text_seg["text_indexes"],
                position_ids=position_ids,
                und_len=text_seg["und_len"],
                sequence_length=seq_len,
                vision_tokens=[x_noisy],
                vision_token_shapes=vision_seg["vision_token_shapes"],
                vision_sequence_indexes=vision_seg["vision_sequence_indexes"],
                vision_mse_loss_indexes=vision_seg["vision_mse_loss_indexes"],
                vision_timesteps=vision_timesteps,
                vision_noisy_frame_indexes=vision_seg["vision_noisy_frame_indexes"],
                return_dict=False,
            )
            v_pred = preds_vision[0][:, :, 2:]
            loss = (v_pred - v_target).square().mean()

        torch.autograd.backward(loss, inputs=trainable_params)
        torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            elapsed = time.time() - t_start
            steps_per_sec = step / elapsed
            lr_curr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:05d}/{args.max_steps:05d} | Train Loss: {loss.item():.4f} | LR: {lr_curr:.2e} | Speed: {steps_per_sec:.2f} step/s"
            )

        # Validation
        if step % args.val_every == 0 or step == args.max_steps:
            pipe.transformer.eval()
            val_losses = []
            with torch.no_grad():
                # Evaluate subset of validation set across fixed timesteps
                val_indices = random.sample(range(len(val_latents_gpu)), min(20, len(val_latents_gpu)))
                for v_idx in val_indices:
                    v_clean = val_latents_gpu[v_idx].unsqueeze(0)
                    for t_eval in fixed_val_timesteps:
                        sig = (t_eval / 1000.0).item()
                        eps = torch.randn_like(v_clean)
                        v_noisy = v_clean.clone()
                        v_noisy[:, :, 2:] = (1.0 - sig) * v_clean[:, :, 2:] + sig * eps[:, :, 2:]
                        v_tgt = eps[:, :, 2:] - v_clean[:, :, 2:]
                        v_timesteps = torch.full((num_noisy_tokens,), t_eval.item(), device=device)

                        preds, _, _ = pipe.transformer(
                            input_ids=text_seg["input_ids"],
                            text_indexes=text_seg["text_indexes"],
                            position_ids=position_ids,
                            und_len=text_seg["und_len"],
                            sequence_length=seq_len,
                            vision_tokens=[v_noisy],
                            vision_token_shapes=vision_seg["vision_token_shapes"],
                            vision_sequence_indexes=vision_seg["vision_sequence_indexes"],
                            vision_mse_loss_indexes=vision_seg["vision_mse_loss_indexes"],
                            vision_timesteps=v_timesteps,
                            vision_noisy_frame_indexes=vision_seg["vision_noisy_frame_indexes"],
                            return_dict=False,
                        )
                        v_loss = (preds[0][:, :, 2:] - v_tgt).square().mean().item()
                        val_losses.append(v_loss)

            mean_val_loss = sum(val_losses) / len(val_losses)
            pipe.transformer.train()
            print(
                f"*** Step {step:05d} VALIDATION: Mean Loss = {mean_val_loss:.4f} (best: {best_val_loss:.4f}) ***"
            )

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_count = 0
                best_lora_path = output_dir / "best_lora.safetensors"
                save_cosmos3_lora(pipe.transformer, best_lora_path)
                with open(output_dir / "best.json", "w") as f:
                    json.dump({"step": step, "val_loss": best_val_loss, "time": time.time()}, f, indent=2)
                print(f"Saved new best LoRA to {best_lora_path}!")
            else:
                patience_count += 1
                if patience_count >= args.patience:
                    print(
                        f"Early stopping triggered after {patience_count} validation checks without improvement."
                    )
                    break

    print(f"Video LoRA training complete! Best Val Loss: {best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
