#!/usr/bin/env python3
"""Consolidated Video-LoRA Trainer and Dual-Episode Rollout Harness.

Trains task-specific Video-LoRA across Physical AI world models:
- cosmos3_edge: Cosmos 3 Edge
- cosmos2b: Cosmos-Predict2-2B
- cosmos7b: Cosmos 1.0 7B
- cosmos14b: Cosmos 1.0 14B
- ltx: LTX-Video 2.5

Upon completing training, automatically runs full-episode autoregressive rollouts
on two validation episodes (default: Episodes 32 and 35), renders 3-panel synchronized
MP4 comparison videos (Base vs. LoRA vs. Ground Truth), and logs metrics.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path("/home/anton/lerobot-video-vam")
PYTHON = REPO_ROOT / ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["cosmos3_edge", "cosmos2b", "cosmos7b", "cosmos14b", "ltx"],
        help="Target world model backbone architecture",
    )
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--val-episodes", type=int, nargs="+", default=list(range(32, 40)))
    parser.add_argument(
        "--test-episodes",
        type=int,
        nargs="+",
        default=[32, 35],
        help="Validation episodes for full-length autoregressive rollout comparison",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-rollout", action="store_true", help="Skip post-training full episode rollout")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CONSOLIDATED VIDEO-LORA TRAINING HARNESS")
    print("=" * 80)
    print(f"Backbone:        {args.backbone.upper()}")
    print(f"Train Episodes:  {len(args.train_episodes)} episodes")
    print(f"Val Episodes:    {len(args.val_episodes)} episodes")
    print(f"Test Episodes:   {args.test_episodes} (full-length rollouts)")
    print(f"Output Dir:      {output_dir}")
    print(f"Max Steps:       {args.max_steps} | Rank: {args.rank} | LR: {args.lr}")
    print("=" * 80, flush=True)

    # 1. Dispatch Backbone Training
    if args.backbone == "cosmos3_edge":
        cmd = [
            str(PYTHON),
            "scripts/video_vam/train_cosmos3_edge_video_lora.py",
            "--train-episodes",
            *map(str, args.train_episodes),
            "--val-episodes",
            *map(str, args.val_episodes),
            "--rank",
            str(args.rank),
            "--alpha",
            str(args.alpha),
            "--lr",
            str(args.lr),
            "--max-steps",
            str(args.max_steps),
            "--val-every",
            str(args.val_every),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
        ]
    elif args.backbone == "cosmos2b":
        cmd = [
            str(PYTHON),
            "scripts/video_vam/train_cosmos_video_lora.py",
            "--train-episodes",
            *map(str, args.train_episodes),
            "--val-episodes",
            *map(str, args.val_episodes),
            "--rank",
            str(args.rank),
            "--alpha",
            str(args.alpha),
            "--lr",
            str(args.lr),
            "--max-steps",
            str(args.max_steps),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(output_dir),
        ]
    elif args.backbone == "cosmos7b":
        cmd = [
            str(PYTHON),
            "scripts/video_vam/train_cosmos7b_video_lora.py",
            "--train-episodes",
            *map(str, args.train_episodes),
            "--val-episodes",
            *map(str, args.val_episodes),
            "--output-dir",
            str(output_dir),
        ]
    elif args.backbone == "cosmos14b":
        cmd = [
            str(PYTHON),
            "scripts/video_vam/train_cosmos14b_video_lora.py",
            "--train-episodes",
            *map(str, args.train_episodes),
            "--val-episodes",
            *map(str, args.val_episodes),
            "--output-dir",
            str(output_dir),
        ]
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not supported yet")

    print(f"\n[Phase 1] Executing Video-LoRA training for {args.backbone}...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:src"
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)

    best_lora = output_dir / "best_lora.safetensors"
    if not best_lora.is_file():
        candidates = list(output_dir.glob("*lora*.safetensors"))
        if candidates:
            best_lora = candidates[0]
        else:
            raise FileNotFoundError(f"No LoRA weights found in {output_dir}")

    print(f"\nTraining completed! Saved best LoRA to: {best_lora}", flush=True)

    # 2. Automatically Run Full-Episode Autoregressive Rollouts
    if not args.skip_rollout and args.backbone in ["cosmos3_edge", "cosmos2b"]:
        print("\n" + "=" * 80)
        print("[Phase 2] Executing Automatic Dual Full-Episode Autoregressive Rollouts")
        print(f"Target Episodes: {args.test_episodes}")
        print("=" * 80, flush=True)

        from lerobot.policies.vam.rollout_harness import run_full_episode_rollout

        eval_dir = output_dir / "full_episode_evaluations"
        results = run_full_episode_rollout(
            backbone=args.backbone,
            lora_checkpoint=best_lora,
            episodes=args.test_episodes,
            output_dir=eval_dir,
            steps=25,
            seed=args.seed,
        )
        print(f"\nDual full-episode rollouts successfully generated in: {eval_dir}")

    print("\nConsolidated Video-LoRA training and evaluation finished successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
