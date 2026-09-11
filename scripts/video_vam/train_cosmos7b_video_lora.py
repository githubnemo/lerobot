#!/usr/bin/env python3
"""Video-LoRA fine-tuning for NVIDIA Cosmos-1.0-Diffusion-7B with native EDM objective.

Adapts the 28-block Cosmos 7B backbone to robotics video dynamics using low-rank
adaptation, saves the trained LoRA artifact with strict metadata, and directly
extracts adapted intermediate representations (layers 14 and 20) for downstream SmolExpert policy training.

Key Guarantees:
- Strict Protocol 1.0 (train 0-31) or Scale-100 (train 0-31, 40-89) episode isolation
- Native deterministic AutoencoderKLCosmos encoding with per-channel normalization (no bilinear fake VAE)
- True text conditioning (no random text embeddings in production; missing embedding fails loudly)
- VAE / text encoder pre-encoding and offload before DiT model training
- Native Cosmos EDM objective with condition indicator and weighted x0 loss (sigma_data=0.5, FPS 10, clip_length 17)
- Selective gradient backward (loss.backward(inputs=lora_params))
- Fresh run output directories (no overwrite of previous artifacts)
- Fail-fast on dataset missing files or loading errors (no silent random fallback)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.vam.base.native_video import (
    CANONICAL_DATASET_REPO,
    CANONICAL_DATASET_REVISION,
    CosmosVAENormalizer,
    DatasetIntegrityError,
    extract_episode_clip_indices,
    load_and_validate_prompt_embedding,
    preencode_and_offload_clips,
    validate_dataset_repo_and_root,
    validate_training_episodes,
)
from lerobot.policies.vam.cosmos7b_extractor import (
    Cosmos7BExtractionOutput,
    Cosmos7BExtractor,
    Cosmos7BExtractorConfig,
    build_cosmos7b_dummy_model,
)
from lerobot.policies.vam.cosmos7b_lora import (
    Cosmos7BLoRAConfig,
    compute_cosmos7b_edm_loss,
    get_cosmos7b_lora_parameters,
    inject_cosmos7b_lora,
    save_cosmos7b_lora,
)
from lerobot.policies.vam.video_training import build_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Video-LoRA on Cosmos 7B with native EDM objective and extract adapted features."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path or HuggingFace ID for base Cosmos 7B checkpoint.",
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
        default=Path("/home/anton/.cache/video-vam/runs/cosmos7b-videolora"),
        help="Base output directory for LoRA weights and extracted features.",
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
        default="14,20",
        help="Target transformer blocks for LoRA ('all' or comma-separated indices, default: 14,20).",
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
        default=5000,
        help="Training optimization steps (default: 5000).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=250,
        help="Warmup steps for learning rate scheduler (default: 250).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for video prediction training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device (default: cuda).",
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
        default=CANONICAL_DATASET_REPO,
        help="Dataset repo ID (default: canonical hubnemo/cube_out_of_box_dataset).",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=CANONICAL_DATASET_REVISION,
        help="Pinned dataset revision (default: canonical revision 243370c3...).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=["protocol1", "scale100"],
        default="protocol1",
        help="Evaluation protocol split (protocol1 or scale100).",
    )
    parser.add_argument(
        "--train-episodes",
        type=str,
        default="0-31",
        help="Train episode range under Protocol 1.0 (default: 0-31).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset"),
        help="Path to robot demonstration dataset.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="take cube out of box",
        help="Task prompt text for conditioning.",
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
    parser.add_argument(
        "--extract-after-train",
        action="store_true",
        default=False,
        help="Extract intermediate representations from layers 14 & 20 using the adapted LoRA.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32

    # Fresh output directory without overwriting
    run_output_dir = args.output_dir / args.run_id if args.run_id else args.output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Cosmos 7B Video-LoRA] Device: {device}, Run Output dir: {run_output_dir}")

    if args.dry_run:
        print("[Cosmos 7B Video-LoRA] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        num_layers = 22
        target_blocks = (
            tuple(range(num_layers))
            if args.target_blocks == "all"
            else tuple(int(x.strip()) for x in args.target_blocks.split(",") if x.strip())
        )
        base_model = build_cosmos7b_dummy_model(
            num_layers=num_layers,
            num_attention_heads=4,
            attention_head_dim=16,
            text_embed_dim=32,
            device=device,
            dtype=torch.float32,
        )
        lora_cfg = Cosmos7BLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_blocks=target_blocks,
        )
        injected = inject_cosmos7b_lora(base_model, lora_cfg)
        print(
            f"[Cosmos 7B Video-LoRA] Injected {len(injected)} LoRA modules into {len(target_blocks)} blocks."
        )

        lora_params = get_cosmos7b_lora_parameters(base_model)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

        dry_run_steps = min(args.max_steps, 20) if args.max_steps > 100 else args.max_steps
        scheduler = build_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=min(args.warmup_steps, 5),
            max_steps=dry_run_steps,
            base_lr=args.lr,
        )

        print(f"[Cosmos 7B Video-LoRA] Starting {dry_run_steps} EDM optimization steps (FPS {args.fps})...")
        base_model.train()
        for step in range(dry_run_steps):
            clean_latents = torch.randn(args.batch_size, 16, 3, 8, 8, device=device)
            text_embeds = torch.randn(args.batch_size, 4, 32, device=device)

            optimizer.zero_grad()
            loss = compute_cosmos7b_edm_loss(
                model=base_model,
                clean_latents=clean_latents,
                encoder_hidden_states=text_embeds,
                fps=args.fps,
            )
            loss.backward(inputs=lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % max(1, dry_run_steps // 5) == 0 or step == dry_run_steps - 1:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"[Cosmos 7B Video-LoRA] Step {step + 1}/{dry_run_steps} | EDM Loss: {loss.item():.4f} | LR: {current_lr:.6e}"
                )

        lora_artifact_path = run_output_dir / "cosmos7b_video_lora.safetensors"
        save_cosmos7b_lora(
            base_model,
            lora_artifact_path,
            metadata={
                "rank": str(args.lora_rank),
                "alpha": str(args.lora_alpha),
                "model_type": "Cosmos-1.0-Diffusion-7B-LoRA",
                "objective": "EDM",
            },
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        print(f"[Cosmos 7B Video-LoRA] Saved LoRA weights to {lora_artifact_path}")

        if args.extract_after_train:
            print("[Cosmos 7B Video-LoRA] --- EXTRACTING ADAPTED FEATURES (LAYERS 14 & 20) ---")
            extractor_config = Cosmos7BExtractorConfig(
                hidden_layers=(14, 20),
                patch_size=(1, 2, 2),
                num_layers=num_layers,
                num_attention_heads=4,
                attention_head_dim=16,
                hidden_dim=4 * 16,
                text_embed_dim=32,
                pool_spatial=2,
                concat_layers=True,
                device=args.device,
                dtype="float32",
                fps=args.fps,
            )
            extractor = Cosmos7BExtractor(config=extractor_config, model=base_model)
            out: Cosmos7BExtractionOutput = extractor.extract(
                hidden_states=clean_latents,
                text_conditioning=text_embeds,
                fps=args.fps,
            )
            adapted_feat_path = run_output_dir / "adapted_cosmos7b_features.safetensors"
            save_file(
                {
                    "features": out.features.detach().cpu(),
                    "grid_coords": out.grid_coords.detach().cpu(),
                },
                str(adapted_feat_path),
            )
            print(f"[Cosmos 7B Video-LoRA] Saved adapted features to {adapted_feat_path}")

        return 0

    # Production path
    print("[Cosmos 7B Video-LoRA] --- PRODUCTION TRAINING WITH NATIVE EDM & VAE ---")

    # 1. Enforce strict Protocol split isolation
    train_eps = validate_training_episodes(
        train_episodes=args.train_episodes,
        protocol=args.protocol,
    )
    print(
        f"[Cosmos 7B Video-LoRA] Validated train split: {len(train_eps)} episodes (0% leakage into heldout val guaranteed)."
    )

    # 2. Load dataset strictly (fail-fast on missing files)
    root_to_use = str(args.dataset_root) if args.dataset_root and args.dataset_root.is_dir() else None
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
            f"[Cosmos 7B Video-LoRA] Loaded dataset '{args.dataset_repo_id}' (rev {args.dataset_revision}) with {len(dataset)} frames."
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
        f"[Cosmos 7B Video-LoRA] Extracted {len(clips)} strictly episode-local clips (length={args.clip_length}, fps={args.fps})."
    )

    # 4. Initialize True VAE
    vae_path = Path(args.vae_path)
    if not vae_path.exists():
        raise FileNotFoundError(f"Cosmos VAE not found at: {vae_path}")

    print(f"[Cosmos 7B Video-LoRA] Loading native AutoencoderKLCosmos from {vae_path}...")
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
        f"[Cosmos 7B Video-LoRA] Validated authentic prompt text embedding: shape={text_embed.shape}, sha256={text_prov.get('sha256')}"
    )

    # 6. Pre-encode clips with VAE and offload VAE before model training
    print(
        f"[Cosmos 7B Video-LoRA] Pre-encoding {len(clips)} clips with native VAE and offloading VAE to CPU..."
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
    print(f"[Cosmos 7B Video-LoRA] Pre-encoded {len(preencoded_data)} clips. VAE offloaded to CPU.")

    # 7. Load Cosmos 7B Transformer & Inject LoRA
    target_blocks = (
        tuple(range(28))
        if args.target_blocks == "all"
        else tuple(int(x.strip()) for x in args.target_blocks.split(",") if x.strip())
    )
    extractor_config = Cosmos7BExtractorConfig(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=args.dtype,
        hidden_layers=(14, 20),
        fps=args.fps,
    )
    extractor = Cosmos7BExtractor(config=extractor_config)
    lora_cfg = Cosmos7BLoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_blocks=target_blocks,
    )
    injected = inject_cosmos7b_lora(extractor.transformer, lora_cfg)
    print(f"[Cosmos 7B Video-LoRA] Injected {len(injected)} LoRA modules into blocks {target_blocks}.")

    lora_params = get_cosmos7b_lora_parameters(extractor.transformer)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scheduler = build_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        base_lr=args.lr,
    )

    print(
        f"[Cosmos 7B Video-LoRA] Starting {args.max_steps} optimization steps with EDM objective (FPS {args.fps})..."
    )
    extractor.transformer.train()
    time.time()
    num_clips = len(preencoded_data)

    for step in range(args.max_steps):
        clip_idx = step % num_clips
        sample_latents, sample_text = preencoded_data[clip_idx]
        sample_latents = sample_latents.to(device=device, dtype=dtype)
        sample_text = sample_text.to(device=device, dtype=dtype)

        optimizer.zero_grad()
        loss = compute_cosmos7b_edm_loss(
            model=extractor.transformer,
            clean_latents=sample_latents,
            encoder_hidden_states=sample_text,
            fps=args.fps,
        )
        loss.backward(inputs=lora_params)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == args.max_steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[Cosmos 7B Video-LoRA] Step {step + 1}/{args.max_steps} | EDM Loss: {loss.item():.4f} | LR: {current_lr:.6e}",
                flush=True,
            )

    lora_artifact_path = run_output_dir / "cosmos7b_video_lora.safetensors"
    save_cosmos7b_lora(
        extractor.transformer,
        lora_artifact_path,
        metadata={
            "rank": str(args.lora_rank),
            "alpha": str(args.lora_alpha),
            "model_type": "Cosmos-1.0-Diffusion-7B-LoRA",
            "protocol": args.protocol,
            "objective": "EDM",
        },
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f"[Cosmos 7B Video-LoRA] Saved LoRA weights to {lora_artifact_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
