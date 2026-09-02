#!/usr/bin/env python3
"""Head comparison on identical frozen Cosmos pool2 features:
World2Action (499M, from scratch, kn787l8b) vs SmolExpert (pretrained ~100M, oy0an3ag).
Val RMSE vs training steps and vs wall-clock hours.

    python plot_head_comparison.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def main() -> int:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "smolexpert-vs-world2action.png"
    )
    args = parser.parse_args()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )

    series = {
        "World2Action head (499M, from scratch)": (
            "#2f6fbf",
            json.loads((DATA / "head_world2action.json").read_text()),
        ),
        "SmolExpert head (pretrained, ~100M)": (
            "#1b7f4e",
            json.loads((DATA / "head_smolexpert.json").read_text()),
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for label, (color, rows) in series.items():
        steps = [r["step"] for r in rows]
        hours = [r["hours"] for r in rows]
        rmse = [r["rmse"] for r in rows]
        best_i = min(range(len(rmse)), key=rmse.__getitem__)
        for ax, x in ((axes[0], steps), (axes[1], hours)):
            ax.plot(x, rmse, color=color, lw=1.8, label=label)
            ax.scatter([x[best_i]], [rmse[best_i]], color=color, zorder=5, s=28)
        axes[0].annotate(
            f"{rmse[best_i]:.2f}",
            (steps[best_i], rmse[best_i]),
            textcoords="offset points",
            xytext=(6, -12),
            color=color,
            fontsize=9,
        )
        axes[1].annotate(
            f"{rmse[best_i]:.2f}",
            (hours[best_i], rmse[best_i]),
            textcoords="offset points",
            xytext=(6, -12),
            color=color,
            fontsize=9,
        )

    axes[0].set_xlabel("training steps")
    axes[1].set_xlabel("wall-clock hours")
    axes[0].set_ylabel("val aggregate RMSE (deg)")
    axes[0].set_ylim(13, 26)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Same frozen Cosmos features, two action heads", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
