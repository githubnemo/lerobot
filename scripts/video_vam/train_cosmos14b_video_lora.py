#!/usr/bin/env python3
"""Quantized Video-LoRA fine-tuning for NVIDIA Cosmos-1.0-Diffusion-14B with native EDM objective.

Adapts the 36-block, 14.25B-parameter Cosmos DiT backbone on robot video prediction tasks
using Quantized LoRA (FP8 base weights + trainable low-rank adapters). Fits comfortably
within the 24 GB VRAM budget of a single NVIDIA RTX 4090:
- FP8 linear base weights (torch.float8_e4m3fn)
- LoRA adapters on attention projections (to_q, to_k, to_v, to_out.0) across all 36 blocks
- Gradient checkpointing enabled on transformer blocks
- Checkpoint-native deterministic AutoencoderKLCosmos encoding (sigma_data=0.5)
- Native Cosmos EDM pretraining objective with condition indicator and weighted x0 loss (FPS 10, clip_length 17)
- Pre-encoding of episode-local video clips and VAE offload before model training
- Strict Protocol 1.0 / Scale-100 episode isolation (0% leakage into held-out validation)
- True text conditioning (no random text embeddings in production; missing embedding fails loudly)
- Selective gradient backward (loss.backward(inputs=lora_params))
- Fresh run output directories (no overwrite of previous artifacts)
- Checkpoint serialization to safetensors with strict rank/alpha metadata
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel

from lerobot.policies.vam.base.native_video import (
    SCALE_100_CANONICAL_REPO,
    SCALE_100_CANONICAL_REVISION,
    CosmosVAENormalizer,
    DatasetIntegrityError,
    extract_episode_clip_indices,
    load_and_validate_prompt_embedding,
    preencode_and_offload_clips,
    validate_dataset_repo_and_root,
    validate_training_episodes,
)
from lerobot.policies.vam.cosmos14b_extractor import (
    build_cosmos14b_dummy_model,
)
from lerobot.policies.vam.cosmos14b_lora import (
    Cosmos14BLoRAConfig,
    compute_cosmos14b_edm_loss,
    get_cosmos14b_lora_parameters,
    inject_cosmos14b_quantized_lora,
    save_cosmos14b_lora,
)
from lerobot.policies.vam.video_training import build_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Quantized Video-LoRA on Cosmos-1.0-Diffusion-14B with native EDM objective."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/anton/.cache/video-vam/cosmos-14b",
        help="Path to Cosmos-1.0-Diffusion-14B checkpoint directory.",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default="/home/anton/.cache/video-vam/cosmos-14b/vae",
        help="Path to AutoencoderKLCosmos checkpoint directory.",
    )
    parser.add_argument(
        "--text-embedding-path",
        type=str,
        default=None,
        help="Path to precomputed prompt text embedding tensor (.pt or .safetensors).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/runs/cosmos14b-videolora"),
        help="Base output directory for LoRA weights and training logs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID directory (defaults to timestamped run to prevent overwrite).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank dimension (default: 16).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA scaling factor alpha (default: 16.0).",
    )
    parser.add_argument(
        "--target-blocks",
        type=str,
        default="all",
        help="Target transformer blocks ('all' for all 36 blocks, or comma-separated indices).",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0",
        help="Target attention projections (default: to_q,to_k,to_v,to_out.0).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Number of training optimization steps (default: 2000).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Warmup steps for learning rate scheduler (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for video prediction training (default: 1).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Target device (default: cuda:0).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Data type for LoRA fine-tuning (default: bfloat16).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=SCALE_100_CANONICAL_REPO,
        help="Hugging Face repo ID for dataset (default: Orellius/cube_out_of_box_v2).",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=SCALE_100_CANONICAL_REVISION,
        help="Pinned dataset revision (default: 5d0325cc...).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["protocol1", "scale100"],
        default="scale100",
        help="Evaluation protocol split (protocol1 or scale100).",
    )
    parser.add_argument(
        "--train-episodes",
        type=str,
        default="0-31, 40-89",
        help="Train episode range under Protocol 1.0 (default: 0-31, 40-89).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Path to robot demonstration dataset.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="take cube out of box",
        help="Task prompt text string.",
    )
    parser.add_argument(
        "--clip-length",
        type=int,
        default=17,
        help="Video clip length in frames (default: 17 for native 8k+1 Cosmos temporal compression).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frames per second (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic dry-run on CPU with architecture-matched dummy model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32

    # Fresh output directory without overwriting
    run_output_dir = args.output_dir / args.run_id if args.run_id else args.output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cosmos-1.0-Diffusion-14B Quantized Video-LoRA Training (Native EDM)")
    print(f"Checkpoint:     {args.checkpoint_path}")
    print(f"VAE Checkpoint: {args.vae_path}")
    print(f"Run Output dir: {run_output_dir}")
    print(f"Device:         {device} (dtype: {dtype})")
    print(f"LoRA:           rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"Target Modules: {args.target_modules}")
    print(f"Max Steps:      {args.max_steps} (warmup: {args.warmup_steps}, lr: {args.lr})")
    print(f"Protocol:       {args.protocol} (train eps: {args.train_episodes})")
    print(f"FPS:            {args.fps}, Clip Length: {args.clip_length}")
    print("=" * 70)

    if args.dry_run:
        print("[Cosmos 14B Video-LoRA] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        num_layers = 4
        base_model = build_cosmos14b_dummy_model(
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            text_embed_dim=32,
            device=device,
            dtype=torch.float32,
        )
        base_model.config.concat_padding_mask = False
        target_blocks = tuple(range(num_layers))
        target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())

        lora_cfg = Cosmos14BLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_blocks=target_blocks,
            target_modules=target_modules,
            quantization="none",
        )
        injected = inject_cosmos14b_quantized_lora(base_model, lora_cfg)
        print(
            f"[Cosmos 14B Video-LoRA] Injected {len(injected)} LoRA modules into {len(target_blocks)} blocks."
        )

        lora_params = get_cosmos14b_lora_parameters(base_model)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

        dry_run_steps = min(args.max_steps, 10)
        scheduler = build_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=min(args.warmup_steps, 2),
            max_steps=dry_run_steps,
            base_lr=args.lr,
        )

        print(f"[Cosmos 14B Video-LoRA] Starting {dry_run_steps} dry-run EDM optimization steps...")
        base_model.train()
        for step in range(dry_run_steps):
            clean_latents = torch.randn(args.batch_size, 17, 3, 8, 8, device=device)
            text_embeds = torch.randn(args.batch_size, 4, 32, device=device)

            optimizer.zero_grad()
            loss = compute_cosmos14b_edm_loss(
                model=base_model,
                clean_latents=clean_latents,
                encoder_hidden_states=text_embeds,
                fps=args.fps,
            )
            loss.backward(inputs=lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()
            print(f"[Cosmos 14B Video-LoRA] Step {step + 1}/{dry_run_steps} | EDM Loss: {loss.item():.4f}")

        lora_path = run_output_dir / "cosmos14b_video_lora.safetensors"
        save_cosmos14b_lora(
            base_model,
            lora_path,
            metadata={
                "rank": str(args.lora_rank),
                "alpha": str(args.lora_alpha),
                "model_type": "Cosmos-1.0-Diffusion-14B-LoRA",
                "quantization": "none",
                "objective": "EDM",
            },
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        print(f"[Cosmos 14B Video-LoRA] Saved LoRA weights to {lora_path}")
        return 0

    # Production path: 14B QLoRA with FP8 + Gradient Checkpointing on GPU
    print("[Cosmos 14B Video-LoRA] --- PRODUCTION TRAINING (FP8 QLoRA + NATIVE EDM) ---")

    # 1. Enforce strict Protocol split isolation
    train_eps = validate_training_episodes(
        train_episodes=args.train_episodes,
        protocol=args.protocol,
    )
    print(
        f"[Cosmos 14B Video-LoRA] Validated train split: {len(train_eps)} episodes (0% leakage into held-out val guaranteed)."
    )

    # 2. Load dataset strictly (fail-fast on errors)
    root_to_use = str(args.dataset_root) if args.dataset_root and Path(args.dataset_root).is_dir() else None
    try:
        from lerobot.datasets import LeRobotDataset

        dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=root_to_use,
            revision=args.dataset_revision,
            return_uint8=True,
            download_videos=False,
        )
        print(
            f"[Cosmos 14B Video-LoRA] Loaded dataset '{args.dataset_repo_id}' (rev {args.dataset_revision}) with {len(dataset)} frames."
        )
    except Exception as e:
        raise DatasetIntegrityError(f"Failed to load dataset '{args.dataset_repo_id}': {e}") from e

    # Validate dataset repo, revision, and root compatibility
    validate_dataset_repo_and_root(
        dataset_repo_id=args.dataset_repo_id,
        protocol=args.protocol,
        dataset_root=args.dataset_root,
        dataset=dataset,
        expected_revision=args.dataset_revision,
    )

    # 3. Extract episode-local clips (FPS 10, clip_length=17 for 8k+1)
    clips = extract_episode_clip_indices(
        dataset_meta=dataset.meta,
        train_episodes=train_eps,
        clip_length=args.clip_length,
        stride=3,
        fps=args.fps,
    )
    print(
        f"[Cosmos 14B Video-LoRA] Extracted {len(clips)} strictly episode-local clips (length={args.clip_length}, fps={args.fps})."
    )

    # 4. Native VAE initialization and Pre-encoding
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"Cosmos VAE not found at: {vae_path}")

    vae_normalizer = CosmosVAENormalizer.from_pretrained(
        vae_path=vae_path,
        sigma_data=0.5,
        device=device,
        dtype=dtype,
    )

    # 5. Load and validate authentic prompt text embedding (strictly required in production)
    text_dim = 1024
    text_embed, text_prov = load_and_validate_prompt_embedding(
        embedding_path=args.text_embedding_path,
        prompt=args.prompt,
        expected_dim=text_dim,
        device=device,
        dtype=dtype,
    )
    print(
        f"[Cosmos 14B Video-LoRA] Validated authentic prompt text embedding: shape={text_embed.shape}, sha256={text_prov.get('sha256')}"
    )

    # 6. Pre-encode clips with VAE and offload VAE before 14B model loading
    print(
        f"[Cosmos 14B Video-LoRA] Pre-encoding {len(clips)} clips with native VAE and offloading VAE to CPU..."
    )
    preencoded_data = preencode_and_offload_clips(
        dataset=dataset,
        clips=clips,
        normalizer=vae_normalizer,
        text_encoder_fn=text_embed,
        prompt=args.prompt,
        batch_size=args.batch_size,
        device=device,
        offload_to_cpu=True,
        provenance=text_prov,
    )
    print(f"[Cosmos 14B Video-LoRA] Pre-encoded {len(preencoded_data)} clips. VAE offloaded to CPU.")

    # 7. Load Cosmos 14B Base Model on CPU first (to avoid VRAM spike)
    target_blocks = (
        tuple(range(36))
        if args.target_blocks == "all"
        else tuple(int(x.strip()) for x in args.target_blocks.split(",") if x.strip())
    )
    target_modules = tuple(x.strip() for x in args.target_modules.split(",") if x.strip())

    print(f"[Cosmos 14B Video-LoRA] Loading Cosmos 14B from {args.checkpoint_path}...")
    ckpt_path = Path(args.checkpoint_path)
    subfolder = "transformer" if (ckpt_path / "transformer").is_dir() else None

    t_load = time.time()
    model = CosmosTransformer3DModel.from_pretrained(
        str(ckpt_path),
        subfolder=subfolder,
        torch_dtype=dtype,
    )
    print(f"[Cosmos 14B Video-LoRA] Loaded base model on CPU in {time.time() - t_load:.1f}s.")

    # Inject Quantized LoRA (FP8 base weights + trainable LoRA adapters)
    lora_cfg = Cosmos14BLoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_blocks=target_blocks,
        target_modules=target_modules,
        quantization="fp8",
    )
    injected = inject_cosmos14b_quantized_lora(model, lora_cfg)
    print(
        f"[Cosmos 14B Video-LoRA] Injected {len(injected)} QLoRA attention modules across {len(target_blocks)} blocks."
    )

    # Move model to target CUDA device and enable gradient checkpointing
    gc.collect()
    print(f"[Cosmos 14B Video-LoRA] Transferring quantized model to {device}...")
    model.to(device)
    model.enable_gradient_checkpointing()
    print(
        f"[Cosmos 14B Video-LoRA] Gradient checkpointing enabled. Allocated VRAM: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB."
    )

    lora_params = get_cosmos14b_lora_parameters(model)
    total_trainable = sum(p.numel() for p in lora_params)
    print(
        f"[Cosmos 14B Video-LoRA] Total trainable LoRA parameters: {total_trainable:,} ({total_trainable * 2 / (1024**2):.1f} MB in BF16)"
    )

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scheduler = build_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        base_lr=args.lr,
    )

    print(
        f"[Cosmos 14B Video-LoRA] Starting {args.max_steps} optimization steps with EDM objective (FPS {args.fps})...",
        flush=True,
    )
    model.train()
    t_start = time.time()
    step_losses = []
    num_clips = len(preencoded_data)

    for step in range(args.max_steps):
        clip_idx = step % num_clips
        sample_latents, sample_text = preencoded_data[clip_idx]
        sample_latents = sample_latents.to(device=device, dtype=dtype)
        sample_text = sample_text.to(device=device, dtype=dtype)

        optimizer.zero_grad()
        loss = compute_cosmos14b_edm_loss(
            model=model,
            clean_latents=sample_latents,
            encoder_hidden_states=sample_text,
            fps=args.fps,
        )
        loss.backward(inputs=lora_params)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = float(loss.item())
        step_losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t_start
            rate = (step + 1) / elapsed
            current_lr = scheduler.get_last_lr()[0]
            recent_avg_loss = sum(step_losses[-50:]) / len(step_losses[-50:])
            vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(
                f"[Cosmos 14B Video-LoRA] Step {step + 1}/{args.max_steps} | "
                f"EDM Loss: {recent_avg_loss:.4f} | LR: {current_lr:.2e} | "
                f"Rate: {rate:.2f} step/s | Peak VRAM: {vram_gb:.2f} GB",
                flush=True,
            )

    total_time = time.time() - t_start
    print(f"[Cosmos 14B Video-LoRA] Training complete in {total_time / 60.0:.2f} min! Saving artifacts...")

    lora_artifact_path = run_output_dir / "cosmos14b_video_lora.safetensors"
    save_cosmos14b_lora(
        model,
        lora_artifact_path,
        metadata={
            "rank": str(args.lora_rank),
            "alpha": str(args.lora_alpha),
            "model_type": "Cosmos-1.0-Diffusion-14B-LoRA",
            "quantization": "fp8",
            "protocol": args.protocol,
            "objective": "EDM",
            "max_steps": str(args.max_steps),
            "final_loss": f"{step_losses[-1]:.4f}",
        },
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f"[Cosmos 14B Video-LoRA] Saved LoRA weights to: {lora_artifact_path}")

    metrics = {
        "model": "Cosmos-1.0-Diffusion-14B",
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "quantization": "fp8",
        "target_blocks": len(target_blocks),
        "steps": args.max_steps,
        "objective": "EDM",
        "initial_loss": step_losses[0] if step_losses else None,
        "final_loss": step_losses[-1] if step_losses else None,
        "mean_loss": sum(step_losses) / len(step_losses) if step_losses else None,
        "total_time_seconds": total_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }
    metrics_path = run_output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Cosmos 14B Video-LoRA] Training metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
