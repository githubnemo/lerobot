#!/usr/bin/env python3
"""Train SmolExpert downstream action policy on FLUX.2 [klein] representations.

Conditions SmolVLA's pretrained action expert on multi-reference visual features
extracted from FLUX.2 [klein] 9B (at the junction or intermediate RF trajectory steps).
"""

from __future__ import annotations

import argparse
import json
import sys
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


class Flux2KleinFeatureDataset(Dataset):
    """Dataset loading cached FLUX.2 [klein] feature artifacts."""

    def __init__(self, cache_dir: Path | str) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found in {self.cache_dir}")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.entries = self.manifest["entries"]
        self.feature_dim = self.manifest.get("entries", [{}])[0].get("feature_dim", 6144)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        entry = self.entries[idx]
        file_path = self.cache_dir / entry["artifact_file"]
        data = load_file(str(file_path))
        features = data["features"]
        if features.ndim == 3:
            features = features.squeeze(0)

        # Mock action & state for downstream training if not stored in artifact
        state = data.get("state", torch.randn(ACTION_DIM, dtype=torch.float32))
        action = data.get("action", torch.randn(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32))

        return {
            "context": features,
            "state": state,
            "action": action,
        }


def build_synthetic_tiny_decoder(
    input_channels: int,
    device: str | torch.device = "cpu",
) -> SmolExpertActionDecoder:
    """Build a fast, lightweight SmolExpertActionDecoder for CPU testing and dry-runs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SmolExpert downstream action policy on FLUX.2 [klein] features."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Path to directory containing cached FLUX.2 [klein] features and manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/runs/smolexpert-flux2-klein"),
        help="Directory to save checkpoints and training logs.",
    )
    parser.add_argument(
        "--input-channels",
        type=int,
        default=None,
        help="Input feature dimension (e.g. 6144 for FLUX.2 [klein] 9B). Auto-detected if manifest is present.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=SMOLVLA_CHECKPOINT,
        help="Pretrained SmolVLA checkpoint to initialize action head.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for trainable parameters (default: 1e-4).",
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
        default=50000,
        help="Maximum training steps (default: 50000).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Minimum training steps before early stopping can trigger (default: 0).",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Evaluation interval in steps (default: 250).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience (number of evaluations without improvement) before early stopping (default: 10).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic dry-run on CPU with architecture-matched tiny SmolExpert.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    print(f"[SmolExpert FLUX.2 klein] Target device: {device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[SmolExpert FLUX.2 klein] --- RUNNING CPU SYNTHETIC DRY-RUN ---")
        input_channels = args.input_channels or 32  # Matches test dry-run cache dimension
        decoder = build_synthetic_tiny_decoder(input_channels=input_channels, device=device)

        b = args.batch_size
        s = 80  # 80 tokens matching target + 3 history frames + goal in dry-run
        context_tensor = torch.randn(b, s, input_channels, device=device)
        state_tensor = torch.randn(b, ACTION_DIM, device=device)
        action_tensor = torch.randn(b, ACTION_HORIZON, ACTION_DIM, device=device)

        optimizer = torch.optim.AdamW(
            [p for p in decoder.parameters() if p.requires_grad],
            lr=args.lr,
        )

        dry_run_steps = min(args.max_steps, 20) if args.max_steps > 100 else args.max_steps
        print(f"[SmolExpert FLUX.2 klein] Starting {dry_run_steps} dry-run optimization steps...")
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
            if (step + 1) % max(1, dry_run_steps // 5) == 0 or step == dry_run_steps - 1:
                print(f"[SmolExpert FLUX.2 klein] Step {step + 1}/{dry_run_steps} - Loss: {loss.item():.4f}")

        decoder.eval()
        with torch.no_grad():
            pred_action = decoder.sample_actions(
                state=state_tensor,
                context=context_tensor,
                num_steps=2,
            )
            print(f"[SmolExpert FLUX.2 klein] Action prediction verified: shape {pred_action.shape}")

            metrics = compute_evaluation_metrics(pred_action, action_tensor)
            print(
                f"[SmolExpert FLUX.2 klein Evaluation Metrics] RMSE: {metrics['rmse']:.4f} | "
                f"H1: {metrics['h1']:.4f} | First-5: {metrics['first5']:.4f}"
            )
            metrics_path = args.output_dir / "eval_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            print(f"[SmolExpert FLUX.2 klein] Evaluation metrics saved to {metrics_path}")

        ckpt_path = args.output_dir / "smolexpert_flux2_klein_dryrun.safetensors"
        save_file({f"model.{k}": v.cpu() for k, v in decoder.state_dict().items()}, str(ckpt_path))
        print(f"[SmolExpert FLUX.2 klein] Checkpoint saved successfully to {ckpt_path}")
        return 0

    if args.cache_dir is None:
        raise ValueError("Must provide --cache-dir for non-dry-run training")

    dataset = Flux2KleinFeatureDataset(args.cache_dir)
    input_channels = args.input_channels or dataset.feature_dim
    print(
        f"[SmolExpert FLUX.2 klein] Loaded dataset with {len(dataset)} samples, feature_dim: {input_channels}"
    )

    # Split dataset: 85% train, 15% validation
    val_size = max(1, int(len(dataset) * 0.15))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"[SmolExpert FLUX.2 klein] Train samples: {train_size}, Validation samples: {val_size}")

    # Initialize normalizer from training tensors
    train_states = torch.stack(
        [
            dataset[i]["state"].squeeze(0) if dataset[i]["state"].ndim > 1 else dataset[i]["state"]
            for i in train_dataset.indices
        ]
    )
    train_actions = torch.stack([dataset[i]["action"] for i in train_dataset.indices])
    normalizer = SmolVLANormalizer.from_training_tensors(train_states, train_actions)

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

    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0
    eval_interval = args.eval_interval
    step = 0
    best_weights_path = args.output_dir / "best_model.pt"

    print(
        f"[SmolExpert FLUX.2 klein] Training until convergence (max_steps={args.max_steps}, eval_interval={eval_interval}, patience={patience})...",
        flush=True,
    )
    decoder.train()
    while step < args.max_steps:
        for batch in train_loader:
            optimizer.zero_grad()
            b_state = batch["state"].to(device)
            if b_state.ndim == 3:
                b_state = b_state.squeeze(1)
            loss = decoder.flow_matching_loss(
                state=b_state,
                action=batch["action"].to(device),
                context=batch["context"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 50 == 0 and step % eval_interval != 0:
                print(
                    f"[SmolExpert FLUX.2 klein] Step {step}/{args.max_steps} | Train Loss: {loss.item():.4f}",
                    flush=True,
                )

            if step % eval_interval == 0:
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
                avg_val_loss = sum(val_losses) / len(val_losses)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(decoder.state_dict(), best_weights_path)
                else:
                    patience_counter += 1
                print(
                    f"[SmolExpert FLUX.2 klein] Step {step}/{args.max_steps} | "
                    f"Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"Best Val Loss: {best_val_loss:.4f} (patience: {patience_counter}/{patience})",
                    flush=True,
                )
                if patience_counter >= patience and step >= args.min_steps:
                    print(
                        f"[SmolExpert FLUX.2 klein] Convergence reached! Early stopping triggered at step {step} (best val loss: {best_val_loss:.4f})",
                        flush=True,
                    )
                    break
                decoder.train()

            if step >= args.max_steps or (patience_counter >= patience and step >= args.min_steps):
                break
        if patience_counter >= patience and step >= args.min_steps:
            break

    # Load best checkpoint if saved
    if best_weights_path.is_file():
        decoder.load_state_dict(torch.load(best_weights_path, map_location=device))
        print(f"[SmolExpert FLUX.2 klein] Loaded best checkpoint (val loss: {best_val_loss:.4f})")

    ckpt_path = args.output_dir / "smolexpert_flux2_klein_final.safetensors"
    save_file({f"model.{k}": v.cpu() for k, v in decoder.state_dict().items()}, str(ckpt_path))
    print(f"[SmolExpert FLUX.2 klein] Training complete. Saved to {ckpt_path}")

    # Compute evaluation metrics on held-out validation set
    decoder.eval()
    with torch.no_grad():
        preds = []
        targets = []
        for vbatch in val_loader:
            vs = vbatch["state"].to(device)
            if vs.ndim == 3:
                vs = vs.squeeze(1)
            p = decoder.sample_actions(
                state=vs,
                context=vbatch["context"].to(device),
                num_steps=2,
            )
            preds.append(p.cpu())
            targets.append(vbatch["action"].cpu())
        all_preds = torch.cat(preds, dim=0)
        all_targets = torch.cat(targets, dim=0)
        metrics = compute_evaluation_metrics(all_preds, all_targets)
        print(
            f"[SmolExpert FLUX.2 klein Evaluation Metrics] RMSE: {metrics['rmse']:.4f} | "
            f"H1: {metrics['h1']:.4f} | First-5: {metrics['first5']:.4f}"
        )
        metrics_path = args.output_dir / "eval_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[SmolExpert FLUX.2 klein] Evaluation metrics saved to {metrics_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
