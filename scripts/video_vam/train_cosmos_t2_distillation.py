#!/usr/bin/env python3
"""Distill Cosmos Predict2 T=16 representations directly into a fast T=2 LoRA."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import torch
from safetensors.torch import load_file

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_cache_dataset import (
    CacheManifest,
    CacheManifestEntry,
    load_cache_manifest,
)
from lerobot.policies.vam.cosmos_lora import (
    freeze_base_parameters,
    inject_lora,
    lora_parameters,
    merge_lora_file_into_base,
    save_lora_state_dict,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample
from scripts.video_vam.train_cosmos_world2action import (
    SafeWandbLogger,
    autocast_context,
)


class EarlyStoppingState:
    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.bad_evaluations = 0

    def update(self, metric: float) -> tuple[bool, bool]:
        improved = metric < self.best_metric - self.min_delta
        if improved:
            self.best_metric = metric
            self.bad_evaluations = 0
            return True, False
        self.bad_evaluations += 1
        should_stop = self.bad_evaluations >= self.patience
        return False, should_stop


DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_PROMPT = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
DEFAULT_MERGE_LORA = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
)
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos-t2-direct-distillation-aligned")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True, help="Teacher train cache manifest.")
    parser.add_argument("--val-manifest", type=Path, required=True, help="Teacher val cache manifest.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--merge-lora-weights", type=Path, default=DEFAULT_MERGE_LORA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--wandb-project", type=str, default="video-vam-world2action")
    parser.add_argument("--run-name", type=str, default="cosmos-t2-direct-distillation-aligned")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _load_sample_clip(
    dataset: LeRobotDataset, entry: CacheManifestEntry, device: torch.device
) -> torch.Tensor:
    sample = dataset[_relative_index(dataset, entry.frame_index)]
    prepared = prepare_sample(sample, frame_index=entry.frame_index, config=CUBE_OUT_OF_BOX_CONTRACT)
    return prepared.rgb_history.to(device=device)


def _load_teacher_target(
    manifest_root: Path, entry: CacheManifestEntry, device: torch.device
) -> torch.Tensor:
    target_path = manifest_root / entry.safetensors
    target_dict = load_file(target_path)
    ctx = target_dict["context"].to(device=device, dtype=torch.bfloat16)
    # Slices the 2 observation frames (2,400 tokens) if given an unpooled 16-frame cache
    if ctx.shape[1] == 19200:
        return ctx[:, :2400]
    return ctx


def _enable_trunk_checkpointing(backbone: torch.nn.Module) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 28:
        raise ValueError("Cosmos backbone must expose exactly 28 blocks")
    for index in range(19):
        block = blocks[index]
        if not getattr(block, "_lerobot_t2_distill_checkpoint", False):
            wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
            wrapped._lerobot_t2_distill_checkpoint = True
            blocks[index] = wrapped


@torch.no_grad()
def evaluate_distillation(
    student: CosmosPredict2Extractor,
    dataset: LeRobotDataset,
    manifest: CacheManifest,
    prompt: torch.Tensor,
    *,
    device: torch.device,
    cosine_weight: float = 1.0,
) -> dict[str, float]:
    student.backbone.eval()
    total_mse = 0.0
    total_cos_sim = 0.0
    total_norm = 0.0
    total_mean = 0.0
    total_std = 0.0
    count = 0

    with autocast_context(device):
        for entry in manifest.entries:
            clip_5 = _load_sample_clip(dataset, entry, device)
            teacher_target = _load_teacher_target(manifest.root, entry, device).float()

            extraction = student.forward_features(
                clip_5,
                prompt.expand(clip_5.shape[0], -1, -1),
                noise_seed=entry.noise_seed,
                sigma=torch.full((clip_5.shape[0],), 80.0, device=device),
            )
            student_tokens = extraction.tokens.float()  # [1, 2400, 2048]

            mse = torch.nn.functional.mse_loss(student_tokens, teacher_target)
            cos_sim = torch.nn.functional.cosine_similarity(student_tokens, teacher_target, dim=-1).mean()

            total_mse += mse.item()
            total_cos_sim += cos_sim.item()
            total_norm += student_tokens.norm(dim=-1).mean().item()
            total_mean += student_tokens.mean().item()
            total_std += student_tokens.std().item()
            count += 1

    val_mse = total_mse / count
    val_cos_sim = total_cos_sim / count
    val_loss = val_mse + cosine_weight * (1.0 - val_cos_sim)

    return {
        "val_loss": val_loss,
        "val_mse": val_mse,
        "val_cos_sim": val_cos_sim,
        "val_token_norm": total_norm / count,
        "val_token_mean": total_mean / count,
        "val_token_std": total_std / count,
        "val_samples": count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"Loading teacher manifests: train={args.train_manifest}, val={args.val_manifest}")
    train_manifest = load_cache_manifest(args.train_manifest)
    val_manifest = load_cache_manifest(args.val_manifest)

    print(f"Loading dataset with full canonical contract from {args.dataset_root}")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=str(args.dataset_root.expanduser().resolve()),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )

    print(f"Loading prompt embedding from {args.prompt}")
    prompt = load_file(str(args.prompt.expanduser().resolve()))["prompt_embedding"].to(
        device=device, dtype=torch.bfloat16
    )

    print("Initializing student Cosmos Predict2 Extractor (T=2)...")
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=args.backbone_checkpoint.expanduser().resolve(),
        tokenizer_path=args.tokenizer.expanduser().resolve(),
        device=str(device),
        dtype="bfloat16",
        hidden_layer=20,
        high_noise_sigma=80.0,
        seed=args.seed,
        state_t=2,
        vae_input_mode="observed_prefix",
        input_frames=5,
    )
    student = CosmosPredict2Extractor(config)

    print(f"Merging base Video LoRA: {args.merge_lora_weights}")
    merge_lora_file_into_base(student.backbone, args.merge_lora_weights.expanduser().resolve())

    print("Injecting new trainable LoRA into student blocks 0-19...")
    adapter_names = inject_lora(
        student.backbone,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        block_indices=range(20),
    )
    freeze_base_parameters(student.backbone)
    _enable_trunk_checkpointing(student.backbone)
    student.backbone.train()

    trainable_params = tuple(lora_parameters(student.backbone))
    print(f"Trainable LoRA parameters in blocks 0-19: {len(trainable_params)}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / max(args.warmup_steps, 1)
        progress = min(
            1.0,
            (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1),
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    early_stopping = EarlyStoppingState(patience=args.patience, min_delta=args.min_delta)
    wandb_logger = SafeWandbLogger(
        output_dir=output_dir,
        project=args.wandb_project,
        run_name=args.run_name,
        tags=["cosmos", "t2", "direct-distillation", "aligned-contract"],
        config=vars(args),
        resume=args.resume,
        disabled=args.no_wandb,
    )

    metrics_log_path = output_dir / "metrics.jsonl"
    best_json_path = output_dir / "best.json"
    best_lora_path = output_dir / "best_lora.safetensors"
    last_json_path = output_dir / "last.json"
    last_lora_path = output_dir / "last_lora.safetensors"

    start_step = 1
    best_metric = float("inf")
    best_step = 0
    start_time = time.time()
    train_entries = list(train_manifest.entries)

    print("Running initial baseline validation before distillation...", flush=True)
    init_val = evaluate_distillation(
        student, dataset, val_manifest, prompt, device=device, cosine_weight=args.cosine_weight
    )
    print(
        f"[Step 0 Baseline] Val Loss: {init_val['val_loss']:.4f} | "
        f"MSE: {init_val['val_mse']:.4f} | CosSim: {init_val['val_cos_sim']:.4f} | "
        f"Token Mean: {init_val['val_token_mean']:.4f} | Norm: {init_val['val_token_norm']:.1f}",
        flush=True,
    )

    print("\nStarting distillation training...", flush=True)
    step = start_step
    while step <= args.max_steps:
        elapsed_hours = (time.time() - start_time) / 3600.0
        if args.max_hours > 0 and elapsed_hours >= args.max_hours:
            print(f"Reached max hours limit ({args.max_hours} h). Stopping.")
            break

        step_start = time.perf_counter()
        student.backbone.train()
        optimizer.zero_grad(set_to_none=True)

        batch_entries = [
            train_entries[((step - 1) * args.batch_size + i) % len(train_entries)]
            for i in range(args.batch_size)
        ]

        batch_clips = torch.cat([_load_sample_clip(dataset, e, device) for e in batch_entries], dim=0)
        batch_targets = torch.cat(
            [_load_teacher_target(train_manifest.root, e, device) for e in batch_entries], dim=0
        ).float()

        with autocast_context(device):
            extraction = student.forward_features(
                batch_clips,
                prompt.expand(batch_clips.shape[0], -1, -1),
                noise_seed=batch_entries[0].noise_seed,
                sigma=torch.full((batch_clips.shape[0],), 80.0, device=device),
            )
            student_tokens = extraction.tokens.float()  # [B, 2400, 2048]

            mse_loss = torch.nn.functional.mse_loss(student_tokens, batch_targets)
            cos_sim = torch.nn.functional.cosine_similarity(student_tokens, batch_targets, dim=-1).mean()
            loss = mse_loss + args.cosine_weight * (1.0 - cos_sim)

        torch.autograd.backward(loss, inputs=trainable_params)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        optimizer.step()
        scheduler.step()

        elapsed = time.perf_counter() - step_start

        if step % 100 == 0:
            print(
                f"step={step:5d} loss={loss.item():.4f} mse={mse_loss.item():.4f} "
                f"cos_sim={cos_sim.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"sec={elapsed:.3f}",
                flush=True,
            )

        if step % args.val_every == 0:
            val_results = evaluate_distillation(
                student, dataset, val_manifest, prompt, device=device, cosine_weight=args.cosine_weight
            )
            candidate_metric = val_results["val_loss"]
            is_best, should_stop = early_stopping.update(candidate_metric)

            print(
                f"\n>>> [Step {step}] Val Loss: {candidate_metric:.4f} (MSE: {val_results['val_mse']:.4f}, "
                f"CosSim: {val_results['val_cos_sim']:.4f}) | "
                f"Token Mean: {val_results['val_token_mean']:.4f} | Norm: {val_results['val_token_norm']:.1f} | "
                f"Best: {is_best} (patience={early_stopping.bad_evaluations}/{args.patience})\n",
                flush=True,
            )

            record = {
                "step": step,
                "train_loss": loss.item(),
                "train_mse": mse_loss.item(),
                "train_cos_sim": cos_sim.item(),
                "learning_rate": optimizer.param_groups[0]["lr"],
                **val_results,
            }
            with open(metrics_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            wandb_logger.log_metrics(record, step=step)

            meta_payload = {
                "step": step,
                "best_step": best_step if not is_best else step,
                "best_val_loss": best_metric if not is_best else candidate_metric,
                "val_mse": val_results["val_mse"],
                "val_cos_sim": val_results["val_cos_sim"],
                "lora": {
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "block_indices": list(range(20)),
                    "adapter_targets": list(adapter_names),
                },
                "hyperparameters": vars(args),
                "timestamp": datetime.now().isoformat(),
            }

            best_lora_json_path = output_dir / "best_lora.json"
            last_lora_json_path = output_dir / "last_lora.json"

            if is_best:
                best_metric = candidate_metric
                best_step = step
                save_lora_state_dict(student.backbone, best_lora_path)
                with open(best_json_path, "w") as f:
                    json.dump(meta_payload, f, indent=2, default=str)
                with open(best_lora_json_path, "w") as f:
                    json.dump(meta_payload, f, indent=2, default=str)

            save_lora_state_dict(student.backbone, last_lora_path)
            with open(last_json_path, "w") as f:
                json.dump(meta_payload, f, indent=2, default=str)
            with open(last_lora_json_path, "w") as f:
                json.dump(meta_payload, f, indent=2, default=str)

            if should_stop:
                print(f"Early stopping triggered at step {step} on validation plateau.")
                break

        step += 1

    print(f"\nDistillation training finished! Best step: {best_step} (Best Val Loss: {best_metric:.4f})")
    wandb_logger.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
