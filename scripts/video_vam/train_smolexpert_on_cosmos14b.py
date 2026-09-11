#!/usr/bin/env python3
"""Train SmolExpert downstream action policy on Cosmos 14B extracted representations.

Injects Cosmos-1.0-Diffusion-14B intermediate representations (tapped layers 18 & 30)
at the prefix K/V cache boundary of SmolVLA's pretrained action expert.
Trains until convergence with early stopping and logs RMSE, H1, and First-5 horizon metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLVLA_CHECKPOINT,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    base_lr: float,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build a learning rate scheduler with linear warmup and cosine decay."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-3, float(current_step) / float(max(1, warmup_steps)))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / base_lr if base_lr > min_lr else 0.01
        return max(min_factor, cosine_decay)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_synthetic_tiny_decoder(
    input_channels: int,
    device: str | torch.device = "cpu",
) -> SmolExpertActionDecoder:
    """Build a lightweight synthetic SmolExpertActionDecoder for testing and dry-runs."""

    class _TinyAttention(nn.Module):
        def __init__(self, hidden: int = 16, kv: int = 8) -> None:
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv, bias=False)
            self.v_proj = nn.Linear(hidden, kv, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

    class _TinyLayer(nn.Module):
        def __init__(self, cross: bool) -> None:
            super().__init__()
            self.input_layernorm = nn.LayerNorm(16)
            self.self_attn = _TinyAttention()
            if cross:
                self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
                self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
            self.post_attention_layernorm = nn.LayerNorm(16)
            self.mlp = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))

    class _TinyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_TinyLayer(cross=i % 2 == 1) for i in range(2)])
            self.norm = nn.LayerNorm(16)

    normalizer = SmolVLANormalizer(
        state_mean=torch.zeros(ACTION_DIM),
        state_std=torch.ones(ACTION_DIM),
        action_mean=torch.zeros(ACTION_DIM),
        action_std=torch.ones(ACTION_DIM),
    )

    decoder = SmolExpertActionDecoder(
        normalizer,
        expert=_TinyExpert(),
        prefix_hidden_size=16,
        expert_hidden_size=16,
        kv_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_steps=2,
        input_channels=input_channels,
        max_action_dim=16,
        max_state_dim=16,
    )
    return decoder.to(device=device)


def compute_evaluation_metrics(pred_actions: Tensor, target_actions: Tensor) -> dict[str, float]:
    """Compute standard action prediction metrics: aggregate RMSE, H1 RMSE, and First-5 RMSE."""
    delta = (pred_actions - target_actions).float()
    squared = delta.square()
    aggregate_rmse = float(squared.mean().sqrt().item())

    # H1: Horizon 1 RMSE
    h1_rmse = float(squared[:, 0].mean().sqrt().item()) if squared.shape[1] >= 1 else aggregate_rmse

    # First-5: Mean RMSE across the first 5 horizons
    first5_len = min(5, squared.shape[1])
    first5_rmse = float(squared[:, :first5_len].mean().sqrt().item())

    return {
        "rmse": aggregate_rmse,
        "h1": h1_rmse,
        "first5": first5_rmse,
    }


class Cosmos14BFeatureDataset(Dataset):
    """Dataset loading cached Cosmos 14B feature artifacts."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found in {self.cache_dir}")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.entries = self.manifest["entries"]
        self.feature_dim = self.manifest.get("entries", [{}])[0].get("feature_dim", 10240)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self.entries[idx]
        file_name = entry.get("artifact_file", entry.get("file"))
        file_path = self.cache_dir / file_name
        data = load_file(str(file_path))
        features = data["features"]
        if features.ndim == 3:
            features = features.squeeze(0)

        state = data.get("state", torch.randn(ACTION_DIM, dtype=torch.float32))
        action = data.get("action", torch.randn(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32))

        return {
            "context": features,
            "state": state,
            "action": action,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SmolExpert downstream action policy on Cosmos 14B features."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/cosmos14b-features"),
        help="Directory containing Cosmos 14B extracted features and manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/runs/cosmos14b-smolexpert"),
        help="Directory to save checkpoints, logs, and evaluation metrics.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=SMOLVLA_CHECKPOINT,
        help="Pretrained SmolVLA checkpoint for action decoder initialization.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for trainable action head parameters (default: 3e-5).",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["cosine", "constant"],
        default="cosine",
        help="Learning rate schedule type (default: cosine).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Linear warmup steps for learning rate scheduler (default: 1000).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Minimum training steps before early stopping can trigger (default: 0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60000,
        help="Maximum training steps (default: 60000).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Evaluation interval in steps (default: 500).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=60,
        help="Patience (eval intervals without val improvement) before early stopping (default: 60).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda:0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic dry-run on CPU with lightweight decoder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SmolExpert Downstream Policy Training on Cosmos-1.0-Diffusion-14B")
    print(f"Features Cache: {args.cache_dir}")
    print(f"Output Dir:     {args.output_dir}")
    print(f"Device:         {device}")
    print(f"LR:             {args.lr} (scheduler: {args.lr_scheduler}, warmup: {args.warmup_steps})")
    print(f"Min Steps:      {args.min_steps}")
    print(f"Max Steps:      {args.max_steps}")
    print(f"Eval Interval:  {args.eval_interval}")
    print(f"Patience:       {args.patience}")
    print("=" * 70)

    if args.dry_run:
        print("[SmolExpert Cosmos 14B] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        input_channels = 64  # Matches synthetic test cache dimension
        decoder = build_synthetic_tiny_decoder(input_channels=input_channels, device=device)

        b = args.batch_size
        s = 16
        context_tensor = torch.randn(b, s, input_channels, device=device)
        state_tensor = torch.randn(b, ACTION_DIM, device=device)
        action_tensor = torch.randn(b, ACTION_HORIZON, ACTION_DIM, device=device)

        optimizer = torch.optim.AdamW(
            [p for p in decoder.parameters() if p.requires_grad],
            lr=args.lr,
        )
        scheduler = build_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=min(args.warmup_steps, 5),
            max_steps=min(args.max_steps, 20),
            base_lr=args.lr,
        )

        dry_run_steps = min(args.max_steps, 20)
        print(f"[SmolExpert Cosmos 14B] Starting {dry_run_steps} dry-run optimization steps...")
        decoder.train()
        for step in range(dry_run_steps):
            optimizer.zero_grad()
            loss = decoder.flow_matching_loss(
                state=state_tensor,
                action=action_tensor,
                context=context_tensor,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (step + 1) % max(1, dry_run_steps // 5) == 0 or step == dry_run_steps - 1:
                print(f"[SmolExpert Cosmos 14B] Step {step + 1}/{dry_run_steps} - Loss: {loss.item():.4f}")

        decoder.eval()
        with torch.no_grad():
            pred_action = decoder.sample_actions(
                state=state_tensor,
                context=context_tensor,
                num_steps=2,
            )
            metrics = compute_evaluation_metrics(pred_action, action_tensor)
            print(
                f"[SmolExpert Cosmos 14B Evaluation Metrics] RMSE: {metrics['rmse']:.4f}° | "
                f"H1: {metrics['h1']:.4f}° | First-5: {metrics['first5']:.4f}°"
            )
            metrics_path = args.output_dir / "eval_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[SmolExpert Cosmos 14B] Evaluation metrics saved to {metrics_path}")

        ckpt_path = args.output_dir / "smolexpert_cosmos14b_dryrun.safetensors"
        save_file({f"model.{k}": v.cpu() for k, v in decoder.state_dict().items()}, str(ckpt_path))
        print(f"[SmolExpert Cosmos 14B] Checkpoint saved to {ckpt_path}")
        return 0

    dataset = Cosmos14BFeatureDataset(args.cache_dir)
    input_channels = dataset.feature_dim
    print(f"[SmolExpert Cosmos 14B] Loaded {len(dataset)} feature artifacts, feature_dim: {input_channels}")

    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"[SmolExpert Cosmos 14B] Split: {train_size} train, {val_size} val samples.")

    # Initialize normalizer from training subset
    train_states = torch.stack(
        [
            dataset[i]["state"].squeeze(0) if dataset[i]["state"].ndim > 1 else dataset[i]["state"]
            for i in train_dataset.indices
        ]
    )
    train_actions = torch.stack([dataset[i]["action"] for i in train_dataset.indices])
    normalizer = SmolVLANormalizer.from_training_tensors(train_states, train_actions)

    print("[SmolExpert Cosmos 14B] Loading SmolVLA pretrained action decoder...")
    decoder = SmolExpertActionDecoder.from_pretrained(
        args.checkpoint_path,
        normalizer=normalizer,
        input_channels=input_channels,
        device=device,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(
        [p for p in decoder.parameters() if p.requires_grad],
        lr=args.lr,
    )

    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = build_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            base_lr=args.lr,
        )

    best_val_loss = float("inf")
    best_step = 0
    patience_counter = 0
    step = 0
    best_weights_path = args.output_dir / "best_model.pt"
    loss_trajectory: list[dict[str, float | int]] = []

    t_start = time.time()
    print("[SmolExpert Cosmos 14B] Starting training until TRUE convergence...", flush=True)

    decoder.train()
    converged = False

    while step < args.max_steps and not converged:
        for batch in train_loader:
            optimizer.zero_grad()
            b_state = batch["state"].to(device)
            if b_state.ndim == 3:
                b_state = b_state.squeeze(1)
            b_action = batch["action"].to(device)
            b_context = batch["context"].to(device)

            loss = decoder.flow_matching_loss(
                state=b_state,
                action=b_action,
                context=b_context,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            step += 1

            if step % 50 == 0 and step % args.eval_interval != 0:
                elapsed = time.time() - t_start
                rate = step / elapsed
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.lr
                print(
                    f"[SmolExpert Cosmos 14B] Step {step}/{args.max_steps} | Train Loss: {loss.item():.4f} | LR: {current_lr:.2e} | {rate:.1f} step/s",
                    flush=True,
                )

            if step % args.eval_interval == 0:
                decoder.eval()
                val_losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vs = vbatch["state"].to(device)
                        if vs.ndim == 3:
                            vs = vs.squeeze(1)
                        vl = decoder.flow_matching_loss(
                            state=vs,
                            action=vbatch["action"].to(device),
                            context=vbatch["context"].to(device),
                        )
                        val_losses.append(vl.item())

                mean_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
                current_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else float(args.lr)

                loss_trajectory.append(
                    {
                        "step": step,
                        "train_loss": round(float(loss.item()), 6),
                        "val_loss": round(mean_val_loss, 6),
                        "lr": current_lr,
                    }
                )

                print(
                    f"[SmolExpert Cosmos 14B] *** Step {step}/{args.max_steps} | Val Loss: {mean_val_loss:.4f} "
                    f"(Best: {best_val_loss:.4f} at {best_step}, Patience: {patience_counter}/{args.patience}, LR: {current_lr:.2e}) ***",
                    flush=True,
                )

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_step = step
                    patience_counter = 0
                    torch.save(decoder.state_dict(), best_weights_path)
                    print(f"  --> Saved new best model checkpoint to {best_weights_path}")
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience and step >= args.min_steps:
                        print(
                            f"\n[SmolExpert Cosmos 14B] Early stopping triggered! TRUE convergence after {step} steps (best step: {best_step}, best val loss: {best_val_loss:.4f})."
                        )
                        converged = True
                        break

                decoder.train()

            if step >= args.max_steps:
                break

    # Save loss trajectory
    trajectory_path = args.output_dir / "loss_trajectory.json"
    with open(trajectory_path, "w", encoding="utf-8") as f:
        json.dump(loss_trajectory, f, indent=2)
    print(
        f"[SmolExpert Cosmos 14B] Saved full loss trajectory ({len(loss_trajectory)} evals) to {trajectory_path}"
    )

    # Load best weights for final evaluation
    if best_weights_path.is_file():
        decoder.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(
            f"[SmolExpert Cosmos 14B] Loaded best weights from step {best_step} (val loss: {best_val_loss:.4f}) for final evaluation."
        )

    decoder.eval()
    print("\n[SmolExpert Cosmos 14B] Computing final action evaluation metrics on validation set...")
    all_pred_actions = []
    all_tgt_actions = []

    with torch.no_grad():
        for vbatch in val_loader:
            vs = vbatch["state"].to(device)
            if vs.ndim == 3:
                vs = vs.squeeze(1)
            vc = vbatch["context"].to(device)
            preds = decoder.sample_actions(state=vs, context=vc, num_steps=2)
            all_pred_actions.append(preds.cpu())
            all_tgt_actions.append(vbatch["action"])

    pred_cat = torch.cat(all_pred_actions, dim=0)
    tgt_cat = torch.cat(all_tgt_actions, dim=0)
    metrics = compute_evaluation_metrics(pred_cat, tgt_cat)
    wall_time_min = (time.time() - t_start) / 60.0

    metrics.update(
        {
            "converged_step": step,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "wall_time_minutes": wall_time_min,
            "loss_trajectory_file": str(trajectory_path),
        }
    )

    print("=" * 60)
    print("COSMOS 14B SMOL_EXPERT EVALUATION METRICS:")
    print(f"  Val RMSE:    {metrics['rmse']:.4f}°")
    print(f"  H1 RMSE:     {metrics['h1']:.4f}°")
    print(f"  First-5 RMSE:{metrics['first5']:.4f}°")
    print(f"  Best Step:   {best_step} (Val Loss: {best_val_loss:.4f})")
    print(f"  Total Steps: {step}")
    print(f"  Wall time:   {wall_time_min:.2f} min")
    print("=" * 60)

    # Save metrics JSON
    metrics_path = args.output_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    # Save final safetensors
    final_safetensors = args.output_dir / "smolexpert_cosmos14b_final.safetensors"
    save_file({f"model.{k}": v.cpu() for k, v in decoder.state_dict().items()}, str(final_safetensors))
    print(f"Saved final safetensors to: {final_safetensors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
