#!/usr/bin/env python3
"""Append the completed LTX layer-probe results to research documentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARKER = "<!-- ltx-layer-probe-20260826 -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/home/anton/.cache/video-vam/runs/ltx25-layer-probe-20260826"),
    )
    parser.add_argument("--repo", type=Path, default=Path("/home/anton/lerobot-video-vam"))
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def link(result: dict[str, Any]) -> str:
    url = result.get("wandb_url")
    return f"[run]({url})" if url else "—"


def metric(result: dict[str, Any], key: str) -> float:
    value = result.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"missing numeric {key}")
    return float(value)


def per_joint(result: dict[str, Any]) -> str:
    values = result.get("best_val_action_rmse_per_joint_degrees")
    if not isinstance(values, list) or len(values) != 6:
        raise ValueError("missing six-joint best validation RMSE")
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    return ", ".join(f"{name}={float(value):.3f}" for name, value in zip(names, values, strict=True))


def append_once(path: Path, section: str) -> None:
    text = path.read_text()
    if MARKER in text:
        raise RuntimeError(f"probe marker already exists in {path}")
    with path.open("a") as stream:
        if text and not text.endswith("\n\n"):
            stream.write("\n")
        stream.write(section.rstrip() + "\n")


def main() -> int:
    args = parse_args()
    root = args.run_root.expanduser().resolve()
    repo = args.repo.expanduser().resolve()
    mix = load(root / "mix" / "result.json")
    mix_config = load(root / "mix" / "config.json")
    top1 = load(root / "top1" / "result.json")
    top2 = load(root / "top2" / "result.json")
    selection = load(root / "selected_layers.json")
    cost = load(root / "extraction-cost.json")
    diagnostics = mix.get("best_mix_diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("mix result is missing best_mix_diagnostics")
    weights = diagnostics.get("weights")
    norms = diagnostics.get("mean_contribution_norms")
    if not isinstance(weights, dict) or not isinstance(norms, dict):
        raise ValueError("mix diagnostics are incomplete")
    layers = [8, 14, 20, 26, 34, 40]
    table = "\n".join(
        f"| {layer} | {float(weights[str(layer)]):.6f} | {float(norms[str(layer)]):.6f} |" for layer in layers
    )
    top1_layer = int(selection["top1"])
    top2_layers = [int(value) for value in selection["top2"]]
    transform = str(mix_config["layer_mix"]["context_transform"])
    tokens = int(mix_config["layer_mix"]["token_count_after_mix"])

    leaderboard = f"""
{MARKER}
| 2026-08-26 | LTX multi-depth learned scalar mix, blocks 8/14/20/26/34/40 | trainer fixed-probe val | **{metric(mix, "best_val_action_rmse_degrees"):.2f}** | best {mix["best_step"]:,} / stopped {mix["step"]:,} | {link(mix)} | Full-30 global masked mixed-unit RMSE; h=1 {metric(mix, "best_val_action_rmse_h1_degrees"):.2f}, first-5 {metric(mix, "best_val_action_rmse_first5_degrees"):.2f}; per-joint: {per_joint(mix)}. {transform} ({tokens} tokens) + World2Action; weights are a hypothesis. |
| 2026-08-26 | LTX top-1 confirmation, block {top1_layer} only | trainer fixed-probe val | **{metric(top1, "best_val_action_rmse_degrees"):.2f}** | best {top1["best_step"]:,} / stopped {top1["step"]:,} | {link(top1)} | Same {transform} + World2Action; h=1 {metric(top1, "best_val_action_rmse_h1_degrees"):.2f}, first-5 {metric(top1, "best_val_action_rmse_first5_degrees"):.2f}; per-joint: {per_joint(top1)}. |
| 2026-08-26 | LTX top-2 confirmation, blocks {top2_layers[0]}/{top2_layers[1]} | trainer fixed-probe val | **{metric(top2, "best_val_action_rmse_degrees"):.2f}** | best {top2["best_step"]:,} / stopped {top2["step"]:,} | {link(top2)} | Same {transform} + World2Action; h=1 {metric(top2, "best_val_action_rmse_h1_degrees"):.2f}, first-5 {metric(top2, "best_val_action_rmse_first5_degrees"):.2f}; per-joint: {per_joint(top2)}. |
"""
    append_once(repo / "docs/video_vam_cube_out_of_box_leaderboard.md", leaderboard)

    extra = cost["extra_cost"]
    diary = f"""
{MARKER}
## 2026-08-26 — LTX depth-selection probe

Captured blocks 8/14/20/26/34/40 in one frozen LTX pass through block 40,
{transform}-reduced each aligned grid to {tokens} tokens, and trained a non-attention scalar
mix with separate non-affine LayerNorms and one global gain ahead of the native
World2Action head. The best full mix reached {metric(mix, "best_val_action_rmse_degrees"):.3f}°
full-30 RMSE, {metric(mix, "best_val_action_rmse_h1_degrees"):.3f}° at h=1, and
{metric(mix, "best_val_action_rmse_first5_degrees"):.3f}° over the first five actions.
Block {top1_layer} had the largest converged scalar weight; the top-two set was
{top2_layers}. The isolated top-1/top-2 confirmations scored
{metric(top1, "best_val_action_rmse_degrees"):.3f}° and
{metric(top2, "best_val_action_rmse_degrees"):.3f}° full-30, respectively.

On the same real prompt/window and persistent FP8-cast CPU-streaming path, tapping
to block 40 cost {float(extra["total_percent"]):.1f}% more total extraction time
and {float(extra["transformer_percent"]):.1f}% more transformer time than the
single block-34 default. This is a {transform} + World2Action ranking experiment, not
evidence that the learned scalar weights are causal importance scores.
"""
    append_once(repo / "docs/video_vam_research_diary.md", diary)

    ltx = f"""
{MARKER}
## Multi-depth layer-selection result (2026-08-26)

This probe used the canonical five-frame input, nine-frame causal VAE prefix,
two clean plus six sigma-1 noise latent frames, real Gemma-4 prompt artifact,
persistent FP8-cast CPU streaming, and one transformer prefix through block 40.
All six tapped grids were independently {transform}-reduced to `[1,{tokens},4096]`. The
consumer applied a separate non-affine LayerNorm per layer, a learned scalar
softmax, and one global learned gain before the unchanged 4096-to-2048 adapter
and native World2Action decoder.

| block | best softmax weight | mean `||weight * LN(h)||` |
|---:|---:|---:|
{table}

Best mix validation: {metric(mix, "best_val_action_rmse_degrees"):.3f}° full-30,
{metric(mix, "best_val_action_rmse_h1_degrees"):.3f}° h=1, and
{metric(mix, "best_val_action_rmse_first5_degrees"):.3f}° first-5. Confirmation
with only block {top1_layer} scored {metric(top1, "best_val_action_rmse_degrees"):.3f}°;
the top-two blocks {top2_layers} scored {metric(top2, "best_val_action_rmse_degrees"):.3f}°.
The confirmation runs, not attention mass or scalar weights alone, determine
whether the ranking holds.

Block-40 multi-tapping added {float(extra["total_percent"]):.1f}% total extraction
time ({float(extra["transformer_percent"]):.1f}% transformer-only) relative to the
established block-34 path. Layer-34 outputs agreed with the single-layer path at
cosine {float(cost["layer34_equivalence"]["cosine"]):.7f}, max absolute
{float(cost["layer34_equivalence"]["max_abs"]):.3g}.

Exact commands:

```bash
.venv/bin/python -m scripts.video_vam.benchmark_ltx_layer_mix \\
  --output /home/anton/.cache/video-vam/runs/ltx25-layer-probe-20260826/extraction-cost.json
.venv/bin/python -m scripts.video_vam.build_ltx_layer_mix_cache \\
  --train-output-dir /home/anton/.cache/video-vam/ltx25-layer-probe-train0-31-stride3-{transform} \\
  --val-output-dir /home/anton/.cache/video-vam/ltx25-layer-probe-val32-39-stride20-{transform} \\
  --train-stride 3 --val-stride 20 --context-transform {transform} --min-free-gib 20 --seed 0 --resume
.venv/bin/python -m scripts.video_vam.train_ltx_layer_mix \\
  --context-transform {transform} --layers 8,14,20,26,34,40 --output-dir {root}/mix \\
  --max-steps 200000 --max-hours 72 --val-every 300 --patience 20
.venv/bin/python -m scripts.video_vam.train_ltx_layer_mix \\
  --context-transform {transform} --layers {top1_layer} --output-dir {root}/top1 --max-steps 30000 --patience 10
.venv/bin/python -m scripts.video_vam.train_ltx_layer_mix \\
  --context-transform {transform} --layers {",".join(map(str, top2_layers))} --output-dir {root}/top2 \\
  --max-steps 30000 --patience 10
```
"""
    append_once(repo / "docs/video_vam_ltx_extractor.md", ltx)

    report = {
        "mix": mix,
        "top1": top1,
        "top2": top2,
        "selection": selection,
        "extraction_cost": cost,
        "documentation_updated": [
            "docs/video_vam_cube_out_of_box_leaderboard.md",
            "docs/video_vam_research_diary.md",
            "docs/video_vam_ltx_extractor.md",
        ],
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("LTX_LAYER_PROBE_REPORT=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
