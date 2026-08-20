#!/usr/bin/env python3
"""Create a strict episode-level VAM split and persisted fixed validation probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.policies.vam.cosmos_cache_dataset import load_cache_manifest
from lerobot.policies.vam.vam_split import (
    DEFAULT_TRAIN_EPISODES,
    DEFAULT_VAL_EPISODES,
    create_vam_split,
)


def _episodes(value: str) -> list[int]:
    episodes = [int(item) for item in value.split(",") if item]
    if not episodes or any(episode < 0 for episode in episodes):
        raise argparse.ArgumentTypeError("episodes must be a comma-separated non-negative list")
    return episodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-episodes", type=_episodes)
    parser.add_argument("--val-episodes", type=_episodes)
    parser.add_argument("--split-name", default="cube_out_of_box_episode_32_val")
    parser.add_argument("--probe-seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_cache_manifest(args.manifest)
    output = create_vam_split(
        manifest,
        args.output,
        train_episodes=DEFAULT_TRAIN_EPISODES if args.train_episodes is None else args.train_episodes,
        val_episodes=DEFAULT_VAL_EPISODES if args.val_episodes is None else args.val_episodes,
        split_name=args.split_name,
        probe_seed=args.probe_seed,
        overwrite=args.overwrite,
    )
    print(f"split: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
