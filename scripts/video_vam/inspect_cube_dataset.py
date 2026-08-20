#!/usr/bin/env python3
"""Audit the pinned cube dataset without writing or pushing dataset data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import (
    CUBE_OUT_OF_BOX_CONTRACT,
    ValidationReport,
    validate_metadata,
    validate_samples,
)

SUBSETS = {
    "1": [0],
    "3": [0, 20, 39],
    "5": [0, 9, 20, 33, 39],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse audit options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=["1", "3", "5", "all"], default="1")
    parser.add_argument("--root", type=Path, help="Explicit local LeRobot dataset root")
    return parser.parse_args(argv)


def selected_episodes(subset: str) -> list[int]:
    """Resolve a deterministic episode subset."""
    if subset == "all":
        return list(range(CUBE_OUT_OF_BOX_CONTRACT.total_episodes))
    return list(SUBSETS[subset])


def episode_row(episodes: Any, episode_index: int) -> Any:
    """Return one episode metadata row from a Hugging Face or list-like object."""
    return episodes[episode_index]


def sample_indices(dataset: LeRobotDataset, episode_indices: list[int]) -> list[tuple[int, int]]:
    """Return relative positions for the first and last frame of each episode."""
    mapping = dataset.absolute_to_relative_idx
    result = []
    for episode_index in episode_indices:
        row = episode_row(dataset.meta.episodes, episode_index)
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        positions = [start, max(start, end - 1)]
        for absolute_index in positions:
            relative_index = absolute_index if mapping is None else mapping[absolute_index]
            result.append((episode_index, int(relative_index)))
    return result


def print_report(
    dataset: LeRobotDataset, subset: str, episode_indices: list[int], report: ValidationReport
) -> None:
    """Print a compact, human-readable audit summary."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    print("Video VAM cube dataset audit")
    print(f"  repo:     {dataset.repo_id}")
    print(f"  revision: {dataset.revision}")
    print(f"  root:     {dataset.root}")
    print(f"  contract: LeRobotDataset {config.codebase_version}, {config.fps} Hz, task={config.task!r}")
    print(f"  dataset:  {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"  subset:   {subset} -> {episode_indices}")
    print(
        f"  sample:   camera={config.sample_camera_shape}, state={config.sample_state_shape}, action={config.sample_action_shape}"
    )
    print(
        f"  offsets:  image={config.history_offsets}, state={config.state_offsets}, action=0..{config.action_chunk_size - 1}"
    )
    print(f"  checked:  {report.samples_checked} boundary samples")
    if report.is_valid:
        print("  result:   PASS")
    else:
        print("  result:   FAIL")
        for violation in report.violations:
            print(f"  - {violation}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Load and audit the pinned dataset."""
    args = parse_args(argv)
    config = CUBE_OUT_OF_BOX_CONTRACT
    episodes = selected_episodes(args.subset)
    delta_timestamps = config.delta_timestamps()
    try:
        dataset = LeRobotDataset(
            config.repo_id,
            root=args.root,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            revision=config.revision,
            return_uint8=True,
            download_videos=True,
        )
        report = validate_metadata(dataset.meta, config)
        boundary_samples = [
            dataset[relative_index] for _, relative_index in sample_indices(dataset, episodes)
        ]
        report.extend(validate_samples(boundary_samples, config))
    except Exception as error:
        print(f"AUDIT ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    print_report(dataset, args.subset, episodes, report)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
