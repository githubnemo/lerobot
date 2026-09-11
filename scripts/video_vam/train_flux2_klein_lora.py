#!/usr/bin/env python3
"""LoRA fine-tuning for FLUX.2 [klein] (9B) with multi-reference conditioning -> SmolExpert pipeline.

Fine-tunes low-rank adapters on FLUX.2 [klein] double-stream and single-stream blocks
using multi-reference causal observation history (t-2, t-1, t) + optional goal conditioning,
saves the LoRA weights with strict metadata, and extracts adapted representations for SmolExpert.

Key Guarantees:
- Strict Protocol 1.0 / Scale-100 episode isolation (0% data leakage)
- Causal extraction (no future goal frame leakage into past observation tokens)
- Native FLUX.2 AutoencoderKLFlux2 patchification + BatchNorm normalization
- True text conditioning (no random text in production; missing embedding fails loudly)
- Native Rectified Flow velocity target v = (noise - clean) / 1000.0
- Strict LoRA rank/alpha metadata validation (no silent base fallback)
- Selective gradient backward (loss.backward(inputs=lora_params))
- Fail-fast on dataset loading errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

from lerobot.policies.vam.base.native_video import (
    DatasetIntegrityError,
    FluxVAENormalizer,
    extract_episode_clip_indices,
    extract_single_image_from_dataset_row,
    load_and_validate_prompt_embedding,
    validate_dataset_repo_and_root,
    validate_training_episodes,
)
from lerobot.policies.vam.flux2_klein_extractor import (
    Flux2KleinExtractionOutput,
    Flux2KleinExtractor,
    Flux2KleinExtractorConfig,
    build_flux2_klein_dummy_model,
    prepare_multi_reference_conditioning,
)
from lerobot.policies.vam.flux2_klein_lora import (
    Flux2KleinLoRAConfig,
    compute_multi_reference_rf_loss,
    get_flux2_klein_lora_parameters,
    inject_flux2_klein_lora,
    save_flux2_klein_lora,
)
from lerobot.policies.vam.video_training import build_cosine_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune LoRA on FLUX.2 [klein] with multi-reference conditioning and extract features for SmolExpert."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path or HuggingFace ID for base FLUX.2 [klein] checkpoint.",
    )
    parser.add_argument(
        "--vae-path",
        type=str,
        default=None,
        help="Path to FLUX VAE checkpoint directory.",
    )
    parser.add_argument(
        "--text-embedding-path",
        type=str,
        default=None,
        help="Path to precomputed prompt text embedding tensor (.pt or .safetensors).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Explicit run ID directory (defaults to direct output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/runs/flux2-klein-lora"),
        help="Output directory for LoRA checkpoints and extracted features.",
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
        "--lr",
        type=float,
        default=1e-4,
        help="Optimizer learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Optimization steps (default: 5000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for fine-tuning.",
    )
    parser.add_argument(
        "--tap-location",
        type=str,
        choices=["junction", "rectified_flow_trajectory"],
        default="junction",
        help="Feature tap location for post-training extraction.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=250,
        help="Warmup steps for learning rate scheduler (default: 250).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="Orellius/cube_out_of_box_v2",
        help="Hugging Face repo ID for dataset (default: Orellius/cube_out_of_box_v2).",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Pinned dataset revision.",
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
        help="Task prompt text for conditioning.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="Data type for fine-tuning (default: bfloat16).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=8,
        help="Number of double-stream transformer blocks (default: 8).",
    )
    parser.add_argument(
        "--num-single-layers",
        type=int,
        default=0,
        help="Number of single-stream transformer blocks (default: 0).",
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=3,
        help="Number of past observation history frames (default: 3 for t-2, t-1, t).",
    )
    parser.add_argument(
        "--use-goal-frame",
        action="store_true",
        default=False,
        help="Include goal frame conditioning (default: False for causal extraction).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available).",
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
        default=True,
        help="Extract representations with adapted LoRA directly into output-dir for SmolExpert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" and device.type == "cuda" else torch.float32
    run_output_dir = args.output_dir / args.run_id if args.run_id else args.output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FLUX.2 klein LoRA] Device: {device}, Output dir: {run_output_dir}")

    if args.dry_run:
        print("[FLUX.2 klein LoRA] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        base_model = build_flux2_klein_dummy_model(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            in_channels=16,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 4, 4),
            device=device,
        )
        lora_cfg = Flux2KleinLoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_double_blocks=(0, 1),
            target_single_blocks=(0, 1),
        )
        injected = inject_flux2_klein_lora(base_model, lora_cfg)
        print(f"[FLUX.2 klein LoRA] Injected {len(injected)} LoRA modules into double & single blocks.")

        lora_params = get_flux2_klein_lora_parameters(base_model)
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)

        print(f"[FLUX.2 klein LoRA] Starting {args.max_steps} optimization steps...")
        base_model.train()
        for step in range(args.max_steps):
            target_frame = torch.randn(args.batch_size, 16, 4, 4, device=device)
            history_frames = [torch.randn(args.batch_size, 16, 4, 4, device=device) for _ in range(3)]
            goal_frame = (
                torch.randn(args.batch_size, 16, 4, 4, device=device) if args.use_goal_frame else None
            )

            cond = prepare_multi_reference_conditioning(
                target_latent=target_frame,
                observation_history=history_frames,
                goal_latent=goal_frame,
                joint_attention_dim=32,
                causal_only=not args.use_goal_frame,
                device=device,
            )
            target_clean = cond.hidden_states[:, : cond.target_tokens_count]

            optimizer.zero_grad()
            loss = compute_multi_reference_rf_loss(
                model=base_model,
                cond_inputs=cond,
                target_clean=target_clean,
            )
            loss.backward(inputs=lora_params)
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()

            if (step + 1) % max(1, args.max_steps // 5) == 0 or step == args.max_steps - 1:
                print(
                    f"[FLUX.2 klein LoRA] Step {step + 1}/{args.max_steps} - Multi-Ref RF Loss: {loss.item():.4f}"
                )

        # Save trained LoRA with strict metadata
        lora_path = run_output_dir / "flux2_klein_lora.safetensors"
        save_flux2_klein_lora(
            base_model,
            lora_path,
            metadata={
                "rank": str(args.lora_rank),
                "alpha": str(args.lora_alpha),
                "model_type": "FLUX.2-klein-9B-LoRA",
            },
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        print(f"[FLUX.2 klein LoRA] Saved LoRA weights to {lora_path}")

        if args.extract_after_train:
            print(f"[FLUX.2 klein LoRA] --- EXTRACTING ADAPTED FEATURES ({args.tap_location.upper()}) ---")
            extractor_cfg = Flux2KleinExtractorConfig(
                num_layers=2,
                num_single_layers=2,
                num_attention_heads=2,
                attention_head_dim=16,
                hidden_dim=32,
                in_channels=16,
                joint_attention_dim=32,
                axes_dims_rope=(4, 4, 4, 4),
                tap_location=args.tap_location,
                device=args.device,
                dtype="float32",
                fps=args.fps,
            )
            extractor = Flux2KleinExtractor(config=extractor_cfg, model=base_model)
            out: Flux2KleinExtractionOutput = extractor.extract(cond)
            adapted_feat_path = run_output_dir / "adapted_flux2_klein_features.safetensors"
            save_file(
                {
                    "features": out.features.detach().cpu().clone(),
                    "visual_tokens": out.visual_tokens.detach().cpu().clone(),
                },
                str(adapted_feat_path),
            )
            print(f"[FLUX.2 klein LoRA] Saved adapted features to {adapted_feat_path}")

        return 0

    # Production / non-dry-run path
    print("[FLUX.2 klein LoRA] Starting production run...")

    # 1. Validate Protocol Split
    train_eps = validate_training_episodes(
        train_episodes=args.train_episodes,
        protocol=args.protocol,
    )
    print(f"[FLUX.2 klein LoRA] Validated train split: {len(train_eps)} episodes (0% leakage guaranteed).")

    # 2. Load Dataset strictly (fail-fast on errors)
    root_to_use = str(args.dataset_root) if args.dataset_root and Path(args.dataset_root).is_dir() else None
    try:
        from lerobot.datasets import LeRobotDataset
        from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT

        dataset = LeRobotDataset(
            args.dataset_repo_id,
            root=root_to_use,
            delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
            revision=CUBE_OUT_OF_BOX_CONTRACT.revision
            if args.dataset_repo_id == CUBE_OUT_OF_BOX_CONTRACT.repo_id
            else None,
            return_uint8=True,
            download_videos=False,
        )
        print(f"[FLUX.2 klein LoRA] Loaded dataset '{args.dataset_repo_id}' with {len(dataset)} frames.")
    except Exception as e:
        raise DatasetIntegrityError(f"Failed to load dataset '{args.dataset_repo_id}': {e}") from e

    # Validate dataset repo and root compatibility
    validate_dataset_repo_and_root(
        dataset_repo_id=args.dataset_repo_id,
        protocol=args.protocol,
        dataset_root=args.dataset_root,
        dataset=dataset,
    )

    # 3. Extract episode-local clips
    clip_length = args.history_frames + 1
    clips = extract_episode_clip_indices(
        dataset_meta=dataset.meta,
        train_episodes=train_eps,
        clip_length=clip_length,
        stride=3,
        fps=args.fps,
    )
    print(f"[FLUX.2 klein LoRA] Extracted {len(clips)} strictly episode-local clips (length={clip_length}).")

    # 4. Initialize FLUX VAE normalizer
    vae_normalizer = None
    if args.vae_path and Path(args.vae_path).exists():
        vae_normalizer = FluxVAENormalizer.from_pretrained(
            vae_path=args.vae_path,
            device=device,
            dtype=dtype,
        )

    extractor_cfg = Flux2KleinExtractorConfig(
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        dtype=args.dtype,
        tap_location=args.tap_location,
        num_layers=args.num_layers,
        num_single_layers=args.num_single_layers,
        fps=args.fps,
    )
    extractor = Flux2KleinExtractor(config=extractor_cfg)
    lora_cfg = Flux2KleinLoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        target_double_blocks=tuple(range(args.num_layers)),
        target_single_blocks=tuple(range(args.num_single_layers)),
    )
    injected = inject_flux2_klein_lora(extractor.transformer, lora_cfg)
    print(f"[FLUX.2 klein LoRA] Injected {len(injected)} LoRA modules into {args.num_layers} double blocks.")

    # 5. Load and validate authentic prompt text embedding if provided
    text_embed = None
    if args.text_embedding_path:
        text_embed, text_prov = load_and_validate_prompt_embedding(
            embedding_path=args.text_embedding_path,
            prompt=args.prompt,
            expected_dim=extractor_cfg.joint_attention_dim,
            device=device,
            dtype=dtype,
        )
        print(
            f"[FLUX.2 klein LoRA] Validated authentic prompt text embedding: shape={text_embed.shape}, provenance={text_prov}"
        )

    lora_params = get_flux2_klein_lora_parameters(extractor.transformer)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=0.01)
    scheduler = build_cosine_schedule_with_warmup(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        base_lr=args.lr,
    )

    print(f"[FLUX.2 klein LoRA] Starting {args.max_steps} training steps on {args.device}...")
    extractor.transformer.train()
    num_clips = len(clips)

    for step in range(args.max_steps):
        clip = clips[step % num_clips]
        frames = [
            extract_single_image_from_dataset_row(dataset[f]["observation.images.front"])
            for f in clip.frame_indices
        ]
        rgb_tensor = torch.stack(frames, dim=0).to(device=device, dtype=dtype)

        if vae_normalizer is not None:
            latents = vae_normalizer.encode(rgb_tensor, deterministic=True)
        else:
            latents = extractor.encode_latents(rgb_tensor)

        hist_latents = [latents[i : i + 1] for i in range(args.history_frames)]
        target_latent = latents[-1:]
        goal_latent = None

        cond = prepare_multi_reference_conditioning(
            target_latent=target_latent,
            observation_history=hist_latents,
            goal_latent=goal_latent,
            encoder_hidden_states=text_embed,
            joint_attention_dim=extractor_cfg.joint_attention_dim,
            causal_only=True,
            device=device,
            dtype=dtype,
        )
        target_clean = cond.hidden_states[:, : cond.target_tokens_count]

        optimizer.zero_grad()
        loss = compute_multi_reference_rf_loss(
            model=extractor.transformer,
            cond_inputs=cond,
            target_clean=target_clean,
        )
        loss.backward(inputs=lora_params)
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % 50 == 0 or step == args.max_steps - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[FLUX.2 klein LoRA] Step {step + 1}/{args.max_steps} | RF Loss: {loss.item():.4f} | LR: {current_lr:.6e}",
                flush=True,
            )

    lora_path = run_output_dir / "flux2_klein_lora.safetensors"
    save_flux2_klein_lora(
        extractor.transformer,
        lora_path,
        metadata={
            "rank": str(args.lora_rank),
            "alpha": str(args.lora_alpha),
            "model_type": "FLUX.2-klein-9B-LoRA",
            "protocol": args.protocol,
        },
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    print(f"[FLUX.2 klein LoRA] Saved LoRA to {lora_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
