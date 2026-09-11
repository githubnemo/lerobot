#!/usr/bin/env python3
"""Synchronize 100-Episode Demonstration Scaling results to docs."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LEADERBOARD = REPO_ROOT / "docs" / "video_vam_cube_out_of_box_leaderboard.md"
DEFAULT_OVERVIEW = REPO_ROOT / "docs" / "EXPERIMENTS_OVERVIEW.md"
DEFAULT_COSMOS_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos14b-scale100-smolexpert")
DEFAULT_FLUX_DIR = Path("/home/anton/.cache/video-vam/runs/flux2-klein-scale100-smolexpert")
DEFAULT_COSMOS_LORA_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos14b-scale100-videolora-smolexpert")
DEFAULT_FLUX_LORA_DIR = Path("/home/anton/.cache/video-vam/runs/flux2-klein-scale100-videolora-smolexpert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-dir", type=Path, default=DEFAULT_COSMOS_DIR)
    parser.add_argument("--flux-dir", type=Path, default=DEFAULT_FLUX_DIR)
    parser.add_argument("--cosmos-lora-dir", type=Path, default=DEFAULT_COSMOS_LORA_DIR)
    parser.add_argument("--flux-lora-dir", type=Path, default=DEFAULT_FLUX_LORA_DIR)
    parser.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--init", action="store_true", help="Initialize queued table in docs if not present")
    return parser.parse_args()


def load_metrics(run_dir: Path) -> dict | None:
    dual_file = run_dir / "final_dual_eval_metrics.json"
    if dual_file.is_file():
        try:
            with open(dual_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    best_file = run_dir / "best_metrics.json"
    if best_file.is_file():
        try:
            with open(best_file, encoding="utf-8") as f:
                data = json.load(f)
                return {
                    "backbone": data.get("backbone"),
                    "best_step": data.get("step"),
                    "eval1_historical": {
                        "val_rmse": data.get("eval1_rmse", data.get("val_rmse")),
                        "val_h1": data.get("eval1_h1", data.get("val_h1")),
                        "val_first5": data.get("eval1_first5", data.get("val_first5")),
                        "val_flow_loss": data.get("eval1_flow_loss", data.get("val_flow_loss")),
                    },
                    "eval2_new_benchmark": {
                        "val_rmse": data.get("eval2_rmse"),
                        "val_h1": data.get("eval2_h1"),
                        "val_first5": data.get("eval2_first5"),
                        "val_flow_loss": data.get("eval2_flow_loss"),
                    }
                    if "eval2_rmse" in data and data["eval2_rmse"] is not None
                    else None,
                }
        except Exception:
            pass
    return None


def format_row(model_name: str, metrics: dict | None, budget_str: str, notes: str) -> str:
    today = datetime.date.today().isoformat()
    if metrics is None:
        return f"| {today} | {model_name} | — | — | — | — | {budget_str} | **QUEUED** | {notes} |"

    e1 = metrics.get("eval1_historical", {})
    e2 = metrics.get("eval2_new_benchmark") or {}
    step = metrics.get("best_step", "?")

    e1_rmse = f"{e1.get('val_rmse', 0.0):.2f}" if e1.get("val_rmse") is not None else "—"
    e1_h1 = f"{e1.get('val_h1', 0.0):.2f}" if e1.get("val_h1") is not None else "—"
    e2_rmse = f"{e2.get('val_rmse', 0.0):.2f}" if e2.get("val_rmse") is not None else "—"
    e2_h1 = f"{e2.get('val_h1', 0.0):.2f}" if e2.get("val_h1") is not None else "—"

    return f"| {today} | {model_name} | **{e1_rmse}°** (h1 {e1_h1}°) | **{e2_rmse}°** (h1 {e2_h1}°) | step {step:,} | {budget_str} | **COMPLETED** | {notes} |"


def main() -> int:
    args = parse_args()
    cosmos_metrics = load_metrics(args.cosmos_dir)
    flux_metrics = load_metrics(args.flux_dir)
    cosmos_lora_metrics = load_metrics(args.cosmos_lora_dir)
    flux_lora_metrics = load_metrics(args.flux_lora_dir)

    print(f"Cosmos 14B Base loaded: {cosmos_metrics is not None}")
    print(f"FLUX.2 klein Base loaded: {flux_metrics is not None}")
    print(f"Cosmos 14B Video-LoRA loaded: {cosmos_lora_metrics is not None}")
    print(f"FLUX.2 klein Video-LoRA loaded: {flux_lora_metrics is not None}")

    # Build Section Markdown
    today = datetime.date.today().isoformat()
    cosmos_row = format_row(
        "Cosmos 14B Base (Layers 18 & 30, FP8 Block-Streaming) + SmolExpert",
        cosmos_metrics,
        "min 20k / max 60k (patience 25)",
        "100 eps (82 train, 8 historical val, 10 new val). True VAE causal latents.",
    )
    flux_row = format_row(
        "FLUX.2 [klein] Base (Multi-Reference Junction Tap) + SmolExpert",
        flux_metrics,
        "min 15k / max 50k (patience 25)",
        "100 eps (82 train, 8 historical val, 10 new val). Junction tap 256 tokens.",
    )
    flux_lora_row = format_row(
        "FLUX.2 [klein] Video-LoRA (Multi-Ref 5k steps) + SmolExpert",
        flux_lora_metrics,
        "min 15k / max 50k (patience 25)",
        "100 eps (82 train, 8 historical val, 10 new val). Multi-Ref LoRA adaptation.",
    )
    cosmos_lora_row = format_row(
        "Cosmos 14B Video-LoRA (FP8 QLoRA 4k steps) + SmolExpert",
        cosmos_lora_metrics,
        "min 20k / max 60k (patience 25)",
        "100 eps (82 train, 8 historical val, 10 new val). Quantized LoRA on all 36 blocks.",
    )

    section_header = (
        "## Phase 5: Expanded 100-Episode Demonstration Scaling & Dual-Evaluation Benchmark (2026-09-07)"
    )
    section_content = f"""{section_header}

Evaluates the exact scaling impact of expanding demonstration data from 40 to 100 episodes (40 original + 60 new samples from `Orellius/cube_out_of_box_v2` / 12,163 frames).
Compares Base feature representations against domain-adapted Video-LoRA representations under strict Protocol 1.0.

**Dual-Evaluation Benchmark Protocol:**
- **Train Set:** 82 episodes (Original 0–31 + 50 new demonstration episodes 40–89; stride 3, 3,065 samples).
- **Eval-Set 1 (Historical Benchmark):** 8 original held-out episodes 32–39 (strict Protocol 1.0, stride 20, exactly 88 anchors). Directly isolates and measures the generalization benefit of 100 vs 40 demonstration trajectories against historical benchmarks.
- **Eval-Set 2 (New Benchmark):** 10 held-out episodes from the new demonstrations (episodes 90–99; stride 20, 51 anchors). Measures in-distribution generalization on newly collected demonstration dynamics.
- **Action Masking:** `action_is_pad` strictly respected across normalizer statistics, flow loss, and trajectory evaluation.
- **Queue:** Sequentially running in detached tmux session `scale100_videolora_queue`.

| date | arm | Eval-Set 1 RMSE (Hist) | Eval-Set 2 RMSE (New) | best step | budget | status | notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{cosmos_row}
{flux_row}
{flux_lora_row}
{cosmos_lora_row}
"""

    # Update Leaderboard
    if args.leaderboard.is_file():
        text = args.leaderboard.read_text(encoding="utf-8")
        if section_header in text:
            # Replace existing section
            parts = text.split(section_header)
            before = parts[0]
            # Find next header if any or end of file
            after_lines = parts[1].split("\n## ")
            after = ("\n## " + "\n## ".join(after_lines[1:])) if len(after_lines) > 1 else ""
            new_text = before.rstrip() + "\n\n" + section_content.strip() + after
        else:
            new_text = text.rstrip() + "\n\n" + section_content

        args.leaderboard.write_text(new_text, encoding="utf-8")
        print(f"Updated {args.leaderboard}")

    # Update Experiments Overview
    if args.overview.is_file():
        ov_text = args.overview.read_text(encoding="utf-8")
        ov_header = "## 4. Phase 5: Expanded 100-Episode Demonstration Scaling"
        ov_content = f"""{ov_header}

Evaluates the scaling impact of increasing demonstration data from 40 to 100 episodes using `Orellius/cube_out_of_box_v2` (100 episodes, 12,163 frames).

### Dual-Evaluation Protocol
- **Train Split**: 82 episodes (0–31 and 40–89; stride 3, 3,065 samples)
- **Eval-Set 1 (Historical Benchmark)**: 8 episodes (32–39; stride 20, 88 anchors). Directly compares 40 vs 100 demonstration scaling under strict Protocol 1.0.
- **Eval-Set 2 (New Benchmark)**: 10 episodes (90–99; stride 20, 51 anchors). Validates generalization on the newly collected demonstration distribution.

### Models & Convergence Configuration
1. **Cosmos 14B Base**:
   - Layers 18 & 30 concatenation (10,240-dim), FP8 linear quantization + block streaming (<2.7 GB VRAM).
   - SmolExpert convergence budget: `min_steps=20000`, `max_steps=60000`, `patience=25`.
2. **FLUX.2 [klein] Base**:
   - Multi-reference junction tap (256 tokens, 6,144-dim).
   - SmolExpert convergence budget: `min_steps=15000`, `max_steps=50000`, `patience=25`.

### Results Matrix
| Model | Eval-Set 1 RMSE (Hist) | Eval-Set 2 RMSE (New) | Status |
| :--- | :--- | :--- | :--- |
| **Cosmos 14B Base** | {"**" + f"{cosmos_metrics['eval1_historical']['val_rmse']:.2f}°" + "**" if cosmos_metrics else "—"} | {"**" + f"{cosmos_metrics['eval2_new_benchmark']['val_rmse']:.2f}°" + "**" if (cosmos_metrics and cosmos_metrics.get("eval2_new_benchmark")) else "—"} | {"COMPLETED" if cosmos_metrics else "QUEUED"} |
| **FLUX.2 [klein] Base** | {"**" + f"{flux_metrics['eval1_historical']['val_rmse']:.2f}°" + "**" if flux_metrics else "—"} | {"**" + f"{flux_metrics['eval2_new_benchmark']['val_rmse']:.2f}°" + "**" if (flux_metrics and flux_metrics.get("eval2_new_benchmark")) else "—"} | {"COMPLETED" if flux_metrics else "QUEUED"} |
| **FLUX.2 [klein] Video-LoRA** | {"**" + f"{flux_lora_metrics['eval1_historical']['val_rmse']:.2f}°" + "**" if flux_lora_metrics else "—"} | {"**" + f"{flux_lora_metrics['eval2_new_benchmark']['val_rmse']:.2f}°" + "**" if (flux_lora_metrics and flux_lora_metrics.get("eval2_new_benchmark")) else "—"} | {"COMPLETED" if flux_lora_metrics else "QUEUED"} |
| **Cosmos 14B Video-LoRA** | {"**" + f"{cosmos_lora_metrics['eval1_historical']['val_rmse']:.2f}°" + "**" if cosmos_lora_metrics else "—"} | {"**" + f"{cosmos_lora_metrics['eval2_new_benchmark']['val_rmse']:.2f}°" + "**" if (cosmos_lora_metrics and cosmos_lora_metrics.get("eval2_new_benchmark")) else "—"} | {"COMPLETED" if cosmos_lora_metrics else "QUEUED"} |
"""
        if ov_header in ov_text:
            parts = ov_text.split(ov_header)
            before = parts[0]
            after_lines = parts[1].split("\n## ")
            after = ("\n## " + "\n## ".join(after_lines[1:])) if len(after_lines) > 1 else ""
            new_ov = before.rstrip() + "\n\n" + ov_content.strip() + after
        else:
            new_ov = ov_text.rstrip() + "\n\n" + ov_content

        args.overview.write_text(new_ov, encoding="utf-8")
        print(f"Updated {args.overview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
