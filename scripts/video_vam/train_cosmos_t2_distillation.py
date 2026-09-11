#!/usr/bin/env python3
"""Distill Cosmos Predict2 T=16 representations directly into a fast T=2 LoRA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.base.lora import extract_lora_state_dict
from lerobot.policies.vam.base.split_guard import (
    validate_manifests_for_protocol,
)
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
    parser.add_argument(
        "--protocol",
        type=str,
        choices=("protocol1", "scale100"),
        default="protocol1",
        help="Benchmark protocol (protocol1 or scale100).",
    )
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
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=None,
        help="Explicit interruption step cap for resume testing without changing the immutable schedule.",
    )
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--wandb-project", type=str, default="video-vam-world2action")
    parser.add_argument("--run-name", type=str, default="cosmos-t2-direct-distillation-aligned")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite own artifacts in output directory."
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint in output directory.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run lightweight dry-run without full backbone."
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_identity_hashes(args: argparse.Namespace) -> dict[str, str]:
    """Compute cryptographic hashes of all input manifests and static artifacts."""
    hashes = {}
    paths_to_hash = {
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "prompt": args.prompt,
        "backbone_checkpoint": args.backbone_checkpoint,
        "tokenizer": args.tokenizer,
        "merge_lora_weights": args.merge_lora_weights,
    }
    for name, path in paths_to_hash.items():
        if path is not None:
            p = Path(path).expanduser().resolve()
            if p.is_file():
                hashes[name] = sha256_file(p)
            else:
                hashes[name] = "absent"
        else:
            hashes[name] = "none"
    return hashes


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
    if not target_path.is_file():
        raise FileNotFoundError(f"Teacher target file not found: {target_path}")
    target_dict = load_file(str(target_path))
    ctx = target_dict.get("context") if "context" in target_dict else target_dict.get("features")
    if ctx is None:
        raise KeyError(f"Teacher target artifact {target_path} missing 'context' or 'features' key")
    ctx = ctx.to(device=device, dtype=torch.float32)
    if ctx.ndim == 2:
        ctx = ctx.unsqueeze(0)
    # Slices the 2 observation frames (2,400 tokens) if given an unpooled 16-frame cache
    if ctx.shape[1] == 19200:
        ctx = ctx[:, :2400]
    if ctx.shape[1:] != (2400, 2048):
        raise ValueError(
            f"Teacher target {target_path} shape mismatch: expected [B, 2400, 2048], got {tuple(ctx.shape)}"
        )
    if not torch.isfinite(ctx).all().item():
        raise ValueError(f"Teacher target {target_path} contains non-finite values (NaN / Inf)")
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


class DummyExtractionOutput:
    def __init__(self, tokens: torch.Tensor) -> None:
        self.tokens = tokens


class DummyDistillationBackbone(nn.Module):
    def __init__(self, in_features: int = 2048) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        # Explicit small adapters for deterministic resume testing
        self.lora_A = nn.Parameter(torch.randn(16, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(in_features, 16))
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(20)])


class DummyDistillationStudent(nn.Module):
    """Lightweight dummy student model for dry-runs and CPU unit tests."""

    def __init__(self, in_features: int = 2048) -> None:
        super().__init__()
        self.backbone = DummyDistillationBackbone(in_features)

    def forward_features(
        self,
        clip: torch.Tensor,
        prompt: torch.Tensor,
        noise_seed: int = 0,
        sigma: torch.Tensor | None = None,
    ) -> DummyExtractionOutput:
        b = clip.shape[0]
        dtype = self.backbone.linear.weight.dtype
        # Base representation derived from clip to provide realistic non-constant forward dynamics
        tokens = torch.zeros((b, 2400, 2048), dtype=dtype, device=clip.device)
        if clip.numel() > 0:
            flat_clip = clip.view(b, -1)[:, :2048].to(dtype=dtype)
            tokens = tokens + flat_clip.unsqueeze(1)
        base_out = self.backbone.linear(tokens)
        lora_out = (tokens @ self.backbone.lora_A.t()) @ self.backbone.lora_B.t()
        return DummyExtractionOutput(tokens=base_out + lora_out)


def _restore_lora_state_dict(backbone: nn.Module, lora_sd: dict[str, torch.Tensor]) -> int:
    """Restore LoRA state dict with canonical key matching, stripping wrapper prefixes."""
    model_params = dict(backbone.named_parameters())
    norm_to_actual: dict[str, str] = {}
    for actual_name in model_params:
        norm_name = actual_name.replace("._checkpoint_wrapped_module.", ".").replace(
            "_checkpoint_wrapped_module.", ""
        )
        norm_to_actual[norm_name] = actual_name
        norm_to_actual[actual_name] = actual_name

    restored_count = 0
    for key, tensor in lora_sd.items():
        norm_key = key.replace("._checkpoint_wrapped_module.", ".").replace("_checkpoint_wrapped_module.", "")
        target_name = norm_to_actual.get(norm_key) or norm_to_actual.get(key)
        if target_name and target_name in model_params:
            param = model_params[target_name]
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            restored_count += 1
        elif key in model_params:
            param = model_params[key]
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
            restored_count += 1

    return restored_count


@torch.no_grad()
def evaluate_distillation(
    student: Any,
    dataset: Any,
    manifest: CacheManifest,
    prompt: torch.Tensor,
    *,
    device: torch.device,
    cosine_weight: float = 1.0,
    dry_run: bool = False,
) -> dict[str, float]:
    if hasattr(student, "backbone"):
        student.backbone.eval()
    elif hasattr(student, "eval"):
        student.eval()

    total_mse = 0.0
    total_cos_sim = 0.0
    total_norm = 0.0
    total_mean = 0.0
    total_std = 0.0
    count = 0

    with autocast_context(device):
        for entry in manifest.entries:
            if dry_run:
                clip_5 = torch.full((1, 3, 5, 64, 64), float(entry.episode_index + 1), device=device)
            else:
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

    val_mse = total_mse / max(count, 1)
    val_cos_sim = total_cos_sim / max(count, 1)
    val_loss = val_mse + cosine_weight * (1.0 - val_cos_sim)

    return {
        "val_loss": val_loss,
        "val_mse": val_mse,
        "val_cos_sim": val_cos_sim,
        "val_token_norm": total_norm / max(count, 1),
        "val_token_mean": total_mean / max(count, 1),
        "val_token_std": total_std / max(count, 1),
        "val_samples": count,
    }


def clean_own_artifacts(output_dir: Path) -> None:
    """Clean only own distillation artifacts when --overwrite is set."""
    own_filenames = {
        "best.json",
        "best_lora.safetensors",
        "best_lora.json",
        "last.json",
        "last_lora.safetensors",
        "last_lora.json",
        "init.json",
        "init_lora.safetensors",
        "init_lora.json",
        "final_distill_summary.json",
        "metrics.jsonl",
        "checkpoint.pt",
    }
    for file_name in own_filenames:
        target = output_dir / file_name
        if target.is_file():
            target.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Output directory safety check
    if output_dir.exists():
        existing_files = [p for p in output_dir.iterdir() if p.is_file()]
        if existing_files:
            if not args.overwrite and not args.resume:
                raise FileExistsError(
                    f"Output directory {output_dir} is not empty (contains {len(existing_files)} files). "
                    f"Pass --resume to resume training or --overwrite to clean own artifacts."
                )
            if args.overwrite and not args.resume:
                clean_own_artifacts(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))

    print(f"Loading teacher manifests: train={args.train_manifest}, val={args.val_manifest}")
    train_manifest = load_cache_manifest(args.train_manifest)
    val_manifest = load_cache_manifest(args.val_manifest)

    # 1. Enforce Split Protocol and Data Leakage Prevention
    with open(args.train_manifest, encoding="utf-8") as f:
        train_m_dict = json.load(f)
    with open(args.val_manifest, encoding="utf-8") as f:
        val_m_dict = json.load(f)

    train_eps, val_eps = validate_manifests_for_protocol(
        train_m_dict,
        val_m_dict,
        protocol=args.protocol,
        allow_subset=True,
    )
    print(
        f"Split Protocol ({args.protocol}) verified: Train episodes {list(train_eps)}, Val episodes {list(val_eps)}"
    )

    dataset = None
    if not args.dry_run:
        print(f"Loading dataset with full canonical contract from {args.dataset_root}")
        dataset = LeRobotDataset(
            CUBE_OUT_OF_BOX_CONTRACT.repo_id,
            root=str(args.dataset_root.expanduser().resolve()),
            delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
            revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
            return_uint8=True,
            download_videos=False,
        )

    if args.dry_run:
        prompt = torch.zeros((1, 512, 4096), device=device, dtype=torch.float32)
        student = DummyDistillationStudent().to(device=device)
        adapter_names = ["backbone.lora_A", "backbone.lora_B"]
        trainable_params = tuple(p for p in student.parameters() if p.requires_grad)
    else:
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
    init_json_path = output_dir / "init.json"
    init_lora_path = output_dir / "init_lora.safetensors"
    ckpt_pt_path = output_dir / "checkpoint.pt"

    start_step = 1
    best_metric = float("inf")
    best_step = 0
    start_time = time.time()
    train_entries = list(train_manifest.entries)
    train_indices = list(range(len(train_entries)))
    random.shuffle(train_indices)

    # 2. Resuming State Truly Restoring All Contexts
    if args.resume:
        if not ckpt_pt_path.is_file():
            raise FileNotFoundError(f"Cannot resume: {ckpt_pt_path} not found in {output_dir}")
        print(f"Resuming distillation training from {ckpt_pt_path}...")
        ckpt = torch.load(ckpt_pt_path, map_location="cpu")

        # Verify configuration & cache identity match
        saved_config = ckpt.get("config", {})
        for param in (
            "lora_rank",
            "lora_alpha",
            "lora_lr",
            "weight_decay",
            "cosine_weight",
            "warmup_steps",
            "max_steps",
            "grad_clip",
            "batch_size",
        ):
            if saved_config.get(param) != getattr(args, param):
                raise ValueError(
                    f"Resume config mismatch for {param}: saved {saved_config.get(param)} != current {getattr(args, param)}"
                )
        if ckpt.get("protocol") != args.protocol:
            raise ValueError(
                f"Resume protocol mismatch: saved {ckpt.get('protocol')} != current {args.protocol}"
            )

        saved_hashes = ckpt.get("identity_hashes", {})
        current_hashes = _compute_identity_hashes(args)
        for h_key, current_h in current_hashes.items():
            saved_h = saved_hashes.get(h_key)
            if saved_h is not None and saved_h != current_h:
                raise ValueError(
                    f"Resume identity mismatch for {h_key}: saved hash {saved_h} != current {current_h}"
                )

        start_step = ckpt["step"] + 1
        best_step = ckpt["best_step"]
        best_metric = ckpt["best_metric"]
        early_stopping.best_metric = ckpt["early_stopping"]["best_metric"]
        early_stopping.bad_evaluations = ckpt["early_stopping"]["bad_evaluations"]
        early_stopping.patience = ckpt["early_stopping"]["patience"]

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu())
        if "cuda_rng" in ckpt and ckpt["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in ckpt["cuda_rng"]])
        if "python_rng" in ckpt:
            random.setstate(tuple(ckpt["python_rng"]))
        if "numpy_rng_keys" in ckpt:
            np_keys = ckpt["numpy_rng_keys"].cpu().numpy()
            np.random.set_state(
                (
                    ckpt["numpy_rng_name"],
                    np_keys,
                    ckpt["numpy_rng_pos"],
                    ckpt["numpy_rng_has_gauss"],
                    ckpt["numpy_rng_cached_gaussian"],
                )
            )
        if "train_indices" in ckpt:
            train_indices = ckpt["train_indices"]

        if "lora_state_dict" in ckpt:
            restored = _restore_lora_state_dict(student.backbone, ckpt["lora_state_dict"])
            print(f"Restored {restored} LoRA tensor weights from checkpoint.")

        print(
            f"Successfully resumed at step {start_step} (Best step: {best_step}, Best Loss: {best_metric:.4f})"
        )
    else:
        # Initial validation & baseline checkpoint save at step 0
        print("Running initial baseline validation before distillation...", flush=True)
        init_val = evaluate_distillation(
            student,
            dataset,
            val_manifest,
            prompt,
            device=device,
            cosine_weight=args.cosine_weight,
            dry_run=args.dry_run,
        )
        print(
            f"[Step 0 Baseline] Val Loss: {init_val['val_loss']:.4f} | "
            f"MSE: {init_val['val_mse']:.4f} | CosSim: {init_val['val_cos_sim']:.4f} | "
            f"Token Mean: {init_val['val_token_mean']:.4f} | Norm: {init_val['val_token_norm']:.1f}",
            flush=True,
        )
        init_payload = {
            "step": 0,
            "baseline_val_loss": init_val["val_loss"],
            "val_mse": init_val["val_mse"],
            "val_cos_sim": init_val["val_cos_sim"],
            "protocol": args.protocol,
            "timestamp": datetime.now().isoformat(),
        }
        init_json_path.write_text(json.dumps(init_payload, indent=2) + "\n")
        if not args.dry_run:
            save_lora_state_dict(student.backbone, init_lora_path)

    print("\nStarting distillation training...", flush=True)
    step = start_step
    interrupted_early = False
    while step <= args.max_steps:
        elapsed_hours = (time.time() - start_time) / 3600.0
        if args.max_hours > 0 and elapsed_hours >= args.max_hours:
            print(f"Reached max hours limit ({args.max_hours} h). Stopping.")
            break

        step_start = time.perf_counter()
        if hasattr(student, "backbone"):
            student.backbone.train()
        elif hasattr(student, "train"):
            student.train()
        optimizer.zero_grad(set_to_none=True)

        batch_entries = [
            train_entries[train_indices[((step - 1) * args.batch_size + i) % len(train_entries)]]
            for i in range(args.batch_size)
        ]

        if args.dry_run:
            batch_clips = torch.full((args.batch_size, 3, 5, 64, 64), float(step), device=device)
        else:
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

        if step % 100 == 0 or step == args.max_steps:
            print(
                f"step={step:5d} loss={loss.item():.4f} mse={mse_loss.item():.4f} "
                f"cos_sim={cos_sim.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"sec={elapsed:.3f}",
                flush=True,
            )

        if (
            step % args.val_every == 0
            or step == args.max_steps
            or (args.stop_after_steps is not None and step == args.stop_after_steps)
        ):
            val_results = evaluate_distillation(
                student,
                dataset,
                val_manifest,
                prompt,
                device=device,
                cosine_weight=args.cosine_weight,
                dry_run=args.dry_run,
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
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            wandb_logger.log_metrics(record, step=step)

            meta_payload = {
                "step": step,
                "best_step": best_step if not is_best else step,
                "best_val_loss": best_metric if not is_best else candidate_metric,
                "val_mse": val_results["val_mse"],
                "val_cos_sim": val_results["val_cos_sim"],
                "protocol": args.protocol,
                "lora": {
                    "rank": int(args.lora_rank),
                    "alpha": float(args.lora_alpha),
                    "block_indices": list(range(20)),
                    "adapter_targets": list(adapter_names),
                },
                "hyperparameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "timestamp": datetime.now().isoformat(),
            }

            best_lora_json_path = output_dir / "best_lora.json"
            last_lora_json_path = output_dir / "last_lora.json"

            if is_best:
                best_metric = candidate_metric
                best_step = step
                if not args.dry_run:
                    save_lora_state_dict(student.backbone, best_lora_path)
                best_json_path.write_text(json.dumps(meta_payload, indent=2, default=str) + "\n")
                best_lora_json_path.write_text(json.dumps(meta_payload, indent=2, default=str) + "\n")

            if not args.dry_run:
                save_lora_state_dict(student.backbone, last_lora_path)
            last_json_path.write_text(json.dumps(meta_payload, indent=2, default=str) + "\n")
            last_lora_json_path.write_text(json.dumps(meta_payload, indent=2, default=str) + "\n")

            # Checkpoint for full resumability (clean primitive & tensor types for safe weights_only load)
            np_state = np.random.get_state()
            clean_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            ckpt_dict = {
                "step": step,
                "best_step": best_step,
                "best_metric": best_metric,
                "early_stopping": {
                    "best_metric": early_stopping.best_metric,
                    "bad_evaluations": early_stopping.bad_evaluations,
                    "patience": early_stopping.patience,
                    "min_delta": early_stopping.min_delta,
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state().cpu(),
                "cuda_rng": [t.cpu() for t in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None,
                "python_rng": list(random.getstate()),
                "numpy_rng_name": str(np_state[0]),
                "numpy_rng_keys": torch.from_numpy(np_state[1].copy()),
                "numpy_rng_pos": int(np_state[2]),
                "numpy_rng_has_gauss": int(np_state[3]),
                "numpy_rng_cached_gaussian": float(np_state[4]),
                "train_indices": list(train_indices),
                "config": clean_config,
                "identity_hashes": _compute_identity_hashes(args),
                "protocol": args.protocol,
                "lora_state_dict": (
                    dict(student.backbone.named_parameters())
                    if args.dry_run
                    else extract_lora_state_dict(student.backbone)
                ),
            }
            torch.save(ckpt_dict, ckpt_pt_path)

            if args.stop_after_steps is not None and step >= args.stop_after_steps:
                print(
                    f"Interruption cap reached at step {step} (--stop-after-steps={args.stop_after_steps}). Checkpointed."
                )
                interrupted_early = True
                break

            if should_stop:
                print(f"Early stopping triggered at step {step} on validation plateau.")
                break

        step += 1

    summary_payload = {
        "status": "interrupted" if interrupted_early else "completed",
        "best_step": best_step,
        "best_val_loss": best_metric,
        "total_steps": step if interrupted_early else (step - 1),
        "protocol": args.protocol,
        "elapsed_seconds": time.time() - start_time,
    }
    (output_dir / "final_distill_summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    print(
        f"\nDistillation training {'interrupted' if interrupted_early else 'finished'}! Best step: {best_step} (Best Val Loss: {best_metric:.4f})"
    )
    wandb_logger.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
