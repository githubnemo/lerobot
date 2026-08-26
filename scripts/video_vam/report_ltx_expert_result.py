#!/usr/bin/env python3
"""Append one completed LTX+SmolExpert result to the research records."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--confound", default="none")
    parser.add_argument(
        "--leaderboard",
        type=Path,
        default=Path("docs/video_vam_cube_out_of_box_leaderboard.md"),
    )
    parser.add_argument(
        "--diary",
        type=Path,
        default=Path("docs/video_vam_research_diary.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = json.loads(args.result.read_text())
    metrics = result["best_validation"]
    required = {
        "val_aggregate_rmse_deg",
        "val_prefix_h1_rmse_deg",
        "val_prefix_first5_mean_rmse_deg",
    }
    if not required.issubset(metrics):
        raise ValueError(f"result is missing metrics: {sorted(required - set(metrics))}")
    if result["stop_reason"] != "validation_plateau":
        raise ValueError(f"refusing to report a non-plateau run: {result['stop_reason']!r}")
    date = datetime.now().astimezone().date().isoformat()
    wall_hours = float(result["wall_clock_seconds"]) / 3600
    url = result.get("wandb_url")
    if not url:
        raise ValueError("completed result has no W&B URL")
    run_id = url.rstrip("/").rsplit("/", 1)[-1]
    full = float(metrics["val_aggregate_rmse_deg"])
    h1 = float(metrics["val_prefix_h1_rmse_deg"])
    first5 = float(metrics["val_prefix_first5_mean_rmse_deg"])
    context = result["context"]
    leaderboard_line = (
        f"| {date} | {args.arm} | trainer fixed-probe val | **{full:.2f}** "
        f"(h1 {h1:.2f}; first-5 mean {first5:.2f}) | best step "
        f"{result['best_step']:,}; plateau step {result['final_step']:,} / {wall_hours:.2f} h | "
        f"[{run_id}]({url}) | tokens={context['tokens']:,}, train stride={context['train_stride']}; "
        f"confound: {args.confound}. |\n"
    )
    diary_entry = (
        f"\n## {date} — {args.arm}\n\n"
        f"Trainer fixed-probe validation plateaued at step {result['final_step']:,} "
        f"({wall_hours:.2f} h; patience {result['patience']}, min-delta {result['min_delta']}). "
        f"Best step {result['best_step']:,}: full-30 masked RMSE {full:.4f}°, "
        f"executed h=1 {h1:.4f}°, first-five per-step mean {first5:.4f}°. "
        f"Context was {context['transform']} with {context['tokens']:,}x{context['channels']:,}; "
        f"train/val stride {context['train_stride']}/{context['val_stride']}. "
        f"Confound: {args.confound}. W&B: {url}\n"
    )
    with args.leaderboard.open("a") as stream:
        stream.write(leaderboard_line)
    with args.diary.open("a") as stream:
        stream.write(diary_entry)
    print(leaderboard_line, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
