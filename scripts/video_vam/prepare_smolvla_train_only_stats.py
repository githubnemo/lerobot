#!/usr/bin/env python3
"""Build an opt-in LeRobot dataset overlay with train-episode normalization stats."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.compute_stats import get_feature_stats
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ACTION = "action"
STATE = "observation.state"
DEFAULT_EPISODES = tuple(range(32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="hubnemo/cube_out_of_box_dataset")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(DEFAULT_EPISODES))
    return parser.parse_args()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _link_payload(source_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        if source.name == "meta":
            continue
        target = output_root / source.name
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError(f"overlay symlink points at the wrong source: {target}")
            continue
        if target.exists():
            raise FileExistsError(f"refusing to replace existing overlay payload: {target}")
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    shutil.copytree(source_root / "meta", output_root / "meta", dirs_exist_ok=True)


def _selected_array(raw_dataset: Any, key: str) -> np.ndarray:
    return np.stack(
        [np.asarray(raw_dataset[index][key], dtype=np.float64) for index in range(len(raw_dataset))],
        axis=0,
    )


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    episodes = tuple(args.episodes)
    if source_root == output_root:
        raise ValueError("output root must differ from the source dataset")
    if episodes != DEFAULT_EPISODES:
        raise ValueError(f"this reproducibility tool requires episodes {list(DEFAULT_EPISODES)}")
    for required in ("meta", "data", "videos"):
        if not (source_root / required).exists():
            raise FileNotFoundError(f"source dataset is missing {required}: {source_root}")

    dataset = LeRobotDataset(
        args.repo_id,
        root=source_root,
        episodes=list(episodes),
        revision=args.revision,
        download_videos=False,
    )
    raw_dataset = dataset.reader.hf_dataset
    if raw_dataset is None:
        raise RuntimeError("selected LeRobot dataset reader did not activate")
    selected_frame_count = len(raw_dataset)
    if selected_frame_count != 4816:
        raise RuntimeError(f"expected 4,816 selected train frames, got {selected_frame_count}")

    global_stats = deepcopy(dataset.meta.stats)
    if global_stats is None:
        raise RuntimeError("source dataset has no meta/stats.json")
    global_action_count = int(np.asarray(global_stats[ACTION]["count"]).reshape(-1)[0])
    global_state_count = int(np.asarray(global_stats[STATE]["count"]).reshape(-1)[0])
    if (global_action_count, global_state_count) != (6536, 6536):
        raise RuntimeError(
            "expected source dataset-global action/state counts of 6,536, "
            f"got {(global_action_count, global_state_count)}"
        )

    corrected_stats = deepcopy(global_stats)
    for key in (ACTION, STATE):
        corrected_stats[key] = get_feature_stats(
            _selected_array(raw_dataset, key),
            axis=0,
            keepdims=False,
        )
        count = int(np.asarray(corrected_stats[key]["count"]).reshape(-1)[0])
        if count != selected_frame_count:
            raise RuntimeError(f"{key} corrected count mismatch: {count} != {selected_frame_count}")

    _link_payload(source_root, output_root)
    write_stats(corrected_stats, output_root)

    delta: dict[str, Any] = {
        "artifact": "smolvla_train_only_stats_delta",
        "source_root": str(source_root),
        "overlay_root": str(output_root),
        "episodes": list(episodes),
        "global_frame_count": global_action_count,
        "corrected_frame_count": selected_frame_count,
        "features": {},
    }
    for key in (ACTION, STATE):
        global_feature = global_stats[key]
        corrected_feature = corrected_stats[key]
        delta["features"][key] = {
            "global_mean": global_feature["mean"],
            "corrected_mean": corrected_feature["mean"],
            "mean_delta_corrected_minus_global": corrected_feature["mean"] - global_feature["mean"],
            "global_std": global_feature["std"],
            "corrected_std": corrected_feature["std"],
            "std_delta_corrected_minus_global": corrected_feature["std"] - global_feature["std"],
        }

    provenance = {
        "artifact": "smolvla_train_only_lerobot_dataset_overlay",
        "source_dataset": {
            "repo_id": args.repo_id,
            "revision": args.revision,
            "root": str(source_root),
        },
        "episodes": list(episodes),
        "selected_frame_count": selected_frame_count,
        "padded_actions_excluded": True,
        "derivation": (
            "one native action and state per selected raw LeRobot frame; delta-timestamp padding is "
            "not part of meta statistics"
        ),
        "normalization": "(value - mean) / (std + 1e-8), population std",
        "comparison_note": (
            "Matches LeRobot's native-frame meta-stat convention while applying the same population-std "
            "and no-padded-action invariant as the VAM normalizer. It intentionally does not import the "
            "VAM stride-3 anchor/30-step weighting, which would confound the leakage correction."
        ),
    }
    (output_root / "meta" / "train_only_stats_provenance.json").write_text(
        json.dumps(_json_value(provenance), indent=2) + "\n"
    )
    (output_root / "meta" / "train_only_stats_delta.json").write_text(
        json.dumps(_json_value(delta), indent=2) + "\n"
    )

    verification = LeRobotDataset(
        args.repo_id,
        root=output_root,
        episodes=list(episodes),
        revision=args.revision,
        download_videos=False,
    )
    for key in (ACTION, STATE):
        count = int(np.asarray(verification.meta.stats[key]["count"]).reshape(-1)[0])
        if count != selected_frame_count:
            raise RuntimeError(f"overlay verification failed for {key}: count={count}")
    print(json.dumps(_json_value(delta), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
