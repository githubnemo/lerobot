#!/usr/bin/env python3
"""Train Video LoRA on Cosmos 3 Edge using Flow-Matching on robot demonstration clips."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from diffusers import Cosmos3OmniPipeline

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.base.split_guard import enforce_protocol_1_0_split
from lerobot.policies.vam.cosmos3_features import (
    checkpoint_digest,
    compute_cosmos3_cache_key,
    encode_cosmos3_video,
)
from lerobot.policies.vam.cosmos3_lora import (
    DUAL_PATHWAY_TARGET_MODULES,
    GEN_TARGET_MODULES,
    UND_TARGET_MODULES,
    inject_cosmos3_lora,
    save_cosmos3_lora,
)

try:
    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index
except ModuleNotFoundError:
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index

DEFAULT_CHECKPOINT_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")
DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos3-edge-video-lora-20260903")
DEFAULT_CACHE_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge-latents-cache")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-repo-id", type=str, default="hubnemo/cube_out_of_box_dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--protocol", choices=["protocol1", "scale100"], default="protocol1")
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--val-episodes", type=int, nargs="+", default=list(range(32, 40)))
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--val-stride", type=int, default=20)
    parser.add_argument(
        "--clip-frames", type=int, default=17, help="17 RGB frames -> 5 latents (2 cond + 3 gen)"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument(
        "--pathway",
        type=str,
        default="dual",
        choices=["dual", "gen_only", "und_only"],
        help="LoRA pathway targets: dual (gen_seq + und_seq), gen_only, or und_only",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10000)
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
    cache_meta: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Precompute and cache VAE latents [N, 48, T_lat, 30, 40] with native posterior mode and mean/std normalization."""
    meta_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and meta_path.is_file():
        try:
            with open(meta_path) as mf:
                saved_meta = json.load(mf)
            if cache_meta is not None and saved_meta == json.loads(json.dumps(cache_meta)):
                print(f"Loading validated pre-encoded latents from {cache_path}...")
                data = torch.load(cache_path, map_location="cpu", weights_only=True)
                if data.ndim != 5 or len(data) != len(samples) or not torch.isfinite(data).all():
                    raise ValueError("Cached latent shape/count/values are invalid")
                print(f"Loaded {len(data)} latent clips. Shape: {data.shape}")
                return data
            else:
                print(f"Stale cache detected at {cache_path} (cache_key mismatch); rebuilding...")
        except Exception as exc:
            print(f"Could not validate cache metadata ({exc}); rebuilding...")

    print(f"Pre-encoding {len(samples)} clips using Wan2.2 VAE to {cache_path}...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_latents = []
    t0 = time.time()
    for idx, (_ep, start_f) in enumerate(samples):
        frames = []
        for f in range(start_f, start_f + clip_frames):
            item = dataset[_relative_index(dataset, f)]
            img = item.get("observation.images.front", item.get("observation.images.phone"))
            frames.append(img)
        rgb_clip = torch.stack(frames, dim=1).unsqueeze(0).to(device=device)

        with torch.no_grad():
            lat = (
                encode_cosmos3_video(
                    pipe.vae,
                    rgb_clip,
                    sample_mode="argmax",
                )
                .cpu()
                .squeeze(0)
            )  # [48, T_lat, 30, 40]
        all_latents.append(lat)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(samples):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"Encoded {idx + 1}/{len(samples)} clips ({rate:.1f} clips/s)...")

    stacked = torch.stack(all_latents, dim=0)  # [N, 48, T_lat, 30, 40]
    torch.save(stacked, cache_path)
    if cache_meta is not None:
        with open(meta_path, "w") as mf:
            json.dump(cache_meta, mf, indent=2)
    print(
        f"Saved {len(stacked)} latent clips to {cache_path} ({stacked.element_size() * stacked.numel() / 1e6:.1f} MB)"
    )
    return stacked


def main() -> int:
    args = parse_args()

    # Enforce strict Protocol 1.0 train/val episode disjointness
    enforce_protocol_1_0_split(
        args.train_episodes, args.val_episodes, allow_subset=True, protocol=args.protocol
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh training output directory: {output_dir}")
    if args.clip_frames < 9 or (args.clip_frames - 1) % 4:
        raise ValueError("clip-frames must be 4k+1 with at least one future latent")
    if min(args.train_stride, args.val_stride, args.max_steps, args.val_every, args.patience) <= 0:
        raise ValueError("Strides, steps and patience must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Loading LeRobot dataset...")
    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=str(args.dataset_root),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    train_samples = get_clip_indices(dataset, args.train_episodes, args.train_stride, args.clip_frames)
    val_samples = get_clip_indices(dataset, args.val_episodes, args.val_stride, args.clip_frames)
    if not train_samples or not val_samples:
        raise ValueError("Both train and validation selections must contain complete clips")
    print(f"Found {len(train_samples)} training clips and {len(val_samples)} validation clips.")

    print("Loading Cosmos3OmniPipeline...")
    pipe = Cosmos3OmniPipeline.from_pretrained(
        str(args.checkpoint_dir),
        torch_dtype=torch.bfloat16,
    ).to(device)

    chk_sha = checkpoint_digest(args.checkpoint_dir)

    # 1. Precompute or load VAE latents with honest cache key verification
    train_cache_key = compute_cosmos3_cache_key(
        dataset_repo=args.dataset_repo_id,
        dataset_revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        episodes=args.train_episodes,
        stride=args.train_stride,
        clip_frames=args.clip_frames,
        hidden_layer=20,
        fps=10.0,
        checkpoint_sha256=chk_sha,
        preprocessing_version="cosmos3_native_v2",
    )
    val_cache_key = compute_cosmos3_cache_key(
        dataset_repo=args.dataset_repo_id,
        dataset_revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        episodes=args.val_episodes,
        stride=args.val_stride,
        clip_frames=args.clip_frames,
        hidden_layer=20,
        fps=10.0,
        checkpoint_sha256=chk_sha,
        preprocessing_version="cosmos3_native_v2",
    )

    train_cache_file = (
        cache_dir / f"train_latents_c{args.clip_frames}_s{args.train_stride}_{train_cache_key[:8]}.pt"
    )
    val_cache_file = cache_dir / f"val_latents_c{args.clip_frames}_s{args.val_stride}_{val_cache_key[:8]}.pt"

    train_latents = preencode_latents(
        pipe,
        dataset,
        train_samples,
        train_cache_file,
        args.clip_frames,
        device,
        cache_meta={
            "cache_key": train_cache_key,
            "episodes": args.train_episodes,
            "stride": args.train_stride,
            "samples": train_samples,
        },
    )
    val_latents = preencode_latents(
        pipe,
        dataset,
        val_samples,
        val_cache_file,
        args.clip_frames,
        device,
        cache_meta={
            "cache_key": val_cache_key,
            "episodes": args.val_episodes,
            "stride": args.val_stride,
            "samples": val_samples,
        },
    )

    # Free VAE to reclaim memory for training
    pipe.vae.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Inject LoRA into Cosmos 3 Transformer
    if args.pathway == "dual":
        targets = DUAL_PATHWAY_TARGET_MODULES
        print(
            f"Injecting LoRA (rank={args.rank}, alpha={args.alpha}) into DUAL PATHWAY (und_seq + gen_seq)..."
        )
    elif args.pathway == "gen_only":
        targets = GEN_TARGET_MODULES
        print(f"Injecting LoRA (rank={args.rank}, alpha={args.alpha}) into generation pathway (gen_seq)...")
    elif args.pathway == "und_only":
        targets = UND_TARGET_MODULES
        print(
            f"Injecting LoRA (rank={args.rank}, alpha={args.alpha}) into understanding pathway (und_seq)..."
        )
    else:
        raise ValueError(f"Unknown pathway: {args.pathway}")

    wrapped_layers = inject_cosmos3_lora(
        pipe.transformer,
        rank=args.rank,
        alpha=args.alpha,
        block_indices=range(len(pipe.transformer.layers)),
        target_modules=targets,
    )
    trainable_params = [p for p in pipe.transformer.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(
        f"LoRA injected into {len(wrapped_layers)} linear modules ({num_trainable / 1e6:.2f}M trainable parameters)."
    )

    pipe.transformer.enable_gradient_checkpointing()
    pipe.transformer.train()

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(1e-3, float(step) / float(max(1, args.warmup_steps)))
        progress = float(step - args.warmup_steps) / float(max(1, args.max_steps - args.warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = 1e-5 / args.lr
        return max(min_factor, cosine_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Pre-pack text segment (static prompt: 'take cube out of box')
    prompt_text = "take cube out of box"
    prompt_tokens = pipe.text_tokenizer(prompt_text).input_ids
    text_seg = pipe._prepare_text_segment(input_ids=prompt_tokens, device=device)

    # Pre-pack vision segment layout with real dataset FPS 10.0
    sample_latent = torch.zeros_like(train_latents[:1], device=device)
    vision_seg = pipe._prepare_vision_segment(
        input_vision_tokens=sample_latent,
        has_image_condition=True,
        mrope_offset=text_seg["vision_start_temporal_offset"],
        vision_fps=10.0,
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
    val_generator = torch.Generator(device=device).manual_seed(args.seed + 999)

    for step in range(1, args.max_steps + 1):
        # Sample random training clip
        idx = random.randint(0, len(train_latents) - 1)
        x_clean = train_latents[idx].unsqueeze(0).to(device=device)

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
        with (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        ):
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
            loss = (v_pred.float() - v_target.float()).square().mean()

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
            val_rng = random.Random(args.seed + 999)
            val_generator.manual_seed(args.seed + 999)
            with torch.no_grad():
                val_indices = val_rng.sample(range(len(val_latents_gpu)), min(20, len(val_latents_gpu)))
                for v_idx in val_indices:
                    v_clean = val_latents_gpu[v_idx].unsqueeze(0)
                    for t_eval in fixed_val_timesteps:
                        sig = (t_eval / 1000.0).item()
                        eps = torch.randn(
                            v_clean.shape, generator=val_generator, device=device, dtype=v_clean.dtype
                        )
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
                        v_loss = (preds[0][:, :, 2:].float() - v_tgt.float()).square().mean().item()
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
                save_cosmos3_lora(
                    pipe.transformer,
                    best_lora_path,
                    rank=args.rank,
                    alpha=args.alpha,
                    target_modules=targets,
                )
                provenance_info = {
                    "step": step,
                    "val_loss": best_val_loss,
                    "checkpoint_sha256": chk_sha,
                    "train_cache_key": train_cache_key,
                    "val_cache_key": val_cache_key,
                    "preprocessing_version": "cosmos3_native_v2",
                    "train_episodes": args.train_episodes,
                    "val_episodes": args.val_episodes,
                    "time": time.time(),
                    "lora": {
                        "rank": args.rank,
                        "alpha": args.alpha,
                        "pathway": args.pathway,
                        "num_trainable": num_trainable,
                        "lr": args.lr,
                        "max_steps": args.max_steps,
                    },
                }
                with open(output_dir / "best.json", "w") as f:
                    json.dump(provenance_info, f, indent=2)
                with open(output_dir / "best_lora.json", "w") as f:
                    json.dump(provenance_info, f, indent=2)
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
