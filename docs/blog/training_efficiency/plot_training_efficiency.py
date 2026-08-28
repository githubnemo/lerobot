#!/usr/bin/env python3
"""Reproduce the cube-out-of-box decoder efficiency figure.

VAM series are trainer fixed-probe val RMSE every 1k steps (W&B).
SmolVLA had eval_steps=0; points are frozen-protocol RMSE at saved checkpoints.

    python plot_training_efficiency.py
    python plot_training_efficiency.py --out /tmp/efficiency.png

Requires matplotlib.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _load(name: str) -> list[dict]:
    return json.loads((DATA / f"eff_{name}.json").read_text())


def main() -> int:
    """Render the SmolExpert training-efficiency figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "vam-smolexpert-training-efficiency.png",
    )
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )

    lora, cosmos, ltx, smol = (_load(n) for n in ("lora", "cosmos", "ltx", "smolvla"))
    colors = {
        "lora": "#1b7f4e",
        "cosmos": "#2f6fbf",
        "ltx": "#5c6570",
        "smol": "#c47b16",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.7))

    ax = axes[0]
    ax.plot(
        [r["step"] / 1000 for r in lora],
        [r["rmse"] for r in lora],
        color=colors["lora"],
        lw=2.1,
        label="Cosmos LoRA + SmolExpert",
    )
    ax.plot(
        [r["step"] / 1000 for r in cosmos],
        [r["rmse"] for r in cosmos],
        color=colors["cosmos"],
        lw=2.1,
        label="Cosmos generic pool2 + SmolExpert",
    )
    ax.plot(
        [r["step"] / 1000 for r in ltx],
        [r["rmse"] for r in ltx],
        color=colors["ltx"],
        lw=2.1,
        label="LTX unpooled + SmolExpert",
    )
    ax.plot(
        [r["step"] / 1000 for r in smol],
        [r["rmse"] for r in smol],
        color=colors["smol"],
        lw=2.1,
        marker="o",
        ms=5,
        label="SmolVLA train-only",
    )
    ax.scatter([38], [13.06], color=colors["lora"], s=32, zorder=5)
    ax.scatter([36], [13.81], color=colors["cosmos"], s=32, zorder=5)
    ax.scatter([45], [13.84], color=colors["ltx"], s=32, zorder=5)
    ax.scatter([29.2], [14.93], color=colors["smol"], s=32, zorder=5)
    ax.set_xlabel("Optimizer steps (thousands)")
    ax.set_ylabel("Val RMSE (degrees)")
    ax.set_title("RMSE vs steps")
    ax.set_xlim(0, 56)
    ax.set_ylim(12.6, 23)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(
        [r["hours"] for r in lora],
        [r["rmse"] for r in lora],
        color=colors["lora"],
        lw=2.1,
        label="Cosmos LoRA + SmolExpert",
    )
    ax.plot(
        [r["hours"] for r in cosmos],
        [r["rmse"] for r in cosmos],
        color=colors["cosmos"],
        lw=2.1,
        label="Cosmos generic pool2 + SmolExpert",
    )
    ax.plot(
        [r["hours"] for r in ltx],
        [r["rmse"] for r in ltx],
        color=colors["ltx"],
        lw=2.1,
        label="LTX unpooled + SmolExpert",
    )
    ax.plot(
        [r["hours"] for r in smol],
        [r["rmse"] for r in smol],
        color=colors["smol"],
        lw=2.1,
        marker="o",
        ms=5,
        label="SmolVLA train-only",
    )
    ax.scatter([1.91], [13.06], color=colors["lora"], s=32, zorder=5)
    ax.scatter([1.67], [13.81], color=colors["cosmos"], s=32, zorder=5)
    ax.scatter([2.99], [13.84], color=colors["ltx"], s=32, zorder=5)
    ax.scatter([1.45], [14.93], color=colors["smol"], s=32, zorder=5)
    ax.set_xlabel("Wall time (hours)")
    ax.set_ylabel("Val RMSE (degrees)")
    ax.set_title("RMSE vs wall clock")
    ax.set_xlim(0, 3.85)
    ax.set_ylim(12.6, 23)
    ax.legend(loc="upper right", frameon=False)

    fig.suptitle("Cube-out-of-box decoder training efficiency (RTX 4090)", fontsize=13, y=1.03)
    fig.text(
        0.0,
        -0.06,
        "VAM: trainer fixed-probe val every 1k. SmolVLA: frozen-protocol RMSE at checkpoints 7/14/21/28/29.2k. "
        "Filled dots mark each run's best. Decoder-only hours; excludes cache extract and video-LoRA pretrain.",
        fontsize=8,
        color="#444444",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
