#!/usr/bin/env python
"""
Post-processing script to add reward labels to a LeRobot dataset.

This script takes a dataset collected with `lerobot-record` (which doesn't include
reward/done columns) and adds:
- `reward`: 1.0 for the last N frames of each episode, 0.0 otherwise
- `next.done`: True for the last frame of each episode, False otherwise

Usage:
    python add_reward_labels.py --repo-id <dataset_repo_id> [--n-success-frames 1] [--push-to-hub]

Example:
    python add_reward_labels.py --repo-id nemo/cube_out_of_box_demos --n-success-frames 3 --push-to-hub
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME, REWARD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def add_reward_labels_to_dataset(
    repo_id: str,
    root: Path | None = None,
    n_success_frames: int = 1,
    push_to_hub: bool = False,
) -> None:
    """
    Add reward and done labels to a LeRobot dataset.
    
    Args:
        repo_id: The dataset repository ID (e.g., "nemo/cube_out_of_box_demos")
        root: Local root directory. If None, uses default HF_LEROBOT_HOME.
        n_success_frames: Number of frames at end of each episode to mark as success (reward=1.0)
        push_to_hub: Whether to push the modified dataset to HuggingFace Hub
    """
    logging.info(f"Loading dataset: {repo_id}")
    
    # Determine root path
    if root is None:
        root = HF_LEROBOT_HOME / repo_id
    root = Path(root)
    
    # Load the dataset
    dataset = LeRobotDataset(repo_id, root=root)
    
    logging.info(f"Dataset has {dataset.num_frames} frames across {dataset.num_episodes} episodes")
    
    # Get episode boundaries from the HF dataset
    hf_dataset = dataset.hf_dataset
    
    # Create reward and done arrays
    num_frames = len(hf_dataset)
    rewards = np.zeros(num_frames, dtype=np.float32)
    dones = np.zeros(num_frames, dtype=bool)
    
    # Get episode indices as plain Python ints (not tensors!)
    episode_indices = hf_dataset["episode_index"]
    if hasattr(episode_indices, "to_pylist"):
        episode_indices = episode_indices.to_pylist()
    elif hasattr(episode_indices, "tolist"):
        episode_indices = episode_indices.tolist()
    # Ensure all are plain ints
    episode_indices = [int(ep) for ep in episode_indices]
    
    # Build a mapping of episode -> frame indices (O(N) instead of O(N*M))
    from collections import defaultdict
    episode_to_frames = defaultdict(list)
    for i, ep in enumerate(episode_indices):
        episode_to_frames[ep].append(i)
    
    unique_episodes = sorted(episode_to_frames.keys())
    logging.info(f"Processing {len(unique_episodes)} episodes...")
    
    for ep_idx in unique_episodes:
        frame_indices = episode_to_frames[ep_idx]
        
        if not frame_indices:
            continue
            
        # Mark the last N frames as success
        success_start = max(0, len(frame_indices) - n_success_frames)
        for i, frame_idx in enumerate(frame_indices):
            if i >= success_start:
                rewards[frame_idx] = 1.0
        
        # Mark the last frame as done
        last_frame_idx = frame_indices[-1]
        dones[last_frame_idx] = True
        
        logging.debug(f"Episode {ep_idx}: {len(frame_indices)} frames, "
                     f"success frames: {frame_indices[success_start:]}")
    
    logging.info(f"Marked {int(rewards.sum())} success frames and {int(dones.sum())} done frames")
    
    # Now we need to modify the parquet files
    # First, find all parquet files
    data_path = root / "data"
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_path}")
    
    logging.info(f"Found {len(parquet_files)} parquet files to modify")
    
    # Track current position in our reward/done arrays
    current_idx = 0
    
    for parquet_file in parquet_files:
        logging.info(f"Processing: {parquet_file}")
        
        # Read the parquet file
        table = pq.read_table(parquet_file)
        num_rows = table.num_rows
        
        # Get the slice of rewards/dones for this file
        file_rewards = rewards[current_idx:current_idx + num_rows]
        file_dones = dones[current_idx:current_idx + num_rows]
        
        # Check if columns already exist and remove them
        existing_columns = table.column_names
        columns_to_remove = []
        if REWARD in existing_columns:
            columns_to_remove.append(REWARD)
            logging.info(f"  Replacing existing '{REWARD}' column")
        if "next.done" in existing_columns:
            columns_to_remove.append("next.done")
            logging.info(f"  Replacing existing 'next.done' column")
        
        # Remove existing columns if present
        for col in columns_to_remove:
            col_idx = table.column_names.index(col)
            table = table.remove_column(col_idx)
        
        # Add new columns
        table = table.append_column(REWARD, pa.array(file_rewards))
        table = table.append_column("next.done", pa.array(file_dones))
        
        # Write back to the same file
        pq.write_table(table, parquet_file)
        
        current_idx += num_rows
        logging.info(f"  Updated {num_rows} rows")
    
    # Update info.json to include the new features
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        
        # Add reward and done to features if not present
        if "features" in info:
            if REWARD not in info["features"]:
                info["features"][REWARD] = {
                    "dtype": "float32",
                    "shape": [1],
                    "names": None,
                }
                logging.info(f"Added '{REWARD}' to features")
            
            if "next.done" not in info["features"]:
                info["features"]["next.done"] = {
                    "dtype": "bool",
                    "shape": [1],
                    "names": None,
                }
                logging.info("Added 'next.done' to features")
        
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logging.info(f"Updated {info_path}")
    
    # Clean up any cached arrow files (force reload on next use)
    cache_dir = root / ".cache"
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        logging.info(f"Cleared cache directory: {cache_dir}")
    
    logging.info("✓ Successfully added reward labels to dataset!")
    
    # Optionally push to hub
    if push_to_hub:
        logging.info("Pushing to HuggingFace Hub...")
        # Reload the dataset to pick up changes
        dataset = LeRobotDataset(repo_id, root=root, force_cache_sync=True)
        dataset.push_to_hub()
        logging.info("✓ Pushed to hub!")


def main():
    parser = argparse.ArgumentParser(
        description="Add reward labels to a LeRobot dataset for reward classifier training"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., 'nemo/cube_out_of_box_demos')",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local root directory for the dataset. Defaults to HF_LEROBOT_HOME/repo_id",
    )
    parser.add_argument(
        "--n-success-frames",
        type=int,
        default=1,
        help="Number of frames at end of each episode to mark as success (default: 1)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the modified dataset to HuggingFace Hub",
    )
    
    args = parser.parse_args()
    
    add_reward_labels_to_dataset(
        repo_id=args.repo_id,
        root=Path(args.root) if args.root else None,
        n_success_frames=args.n_success_frames,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()

