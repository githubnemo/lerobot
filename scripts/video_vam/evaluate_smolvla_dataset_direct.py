"""Offline dataset-direct RMSE with stable sample seeds and strict cache validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.action_rmse import (
    EvaluationBatch,
    SmolVLABackend,
    StateRepeatBackend,
    evaluate_backend,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stable_seed(seed, sample_id):
    return int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


class StableBackend:
    """Ignore the legacy aggregator batch-index seed; seed singleton calls by identity."""

    def __init__(self, backend, seed, progress=None):
        self.backend, self.seed, self.progress = backend, seed, progress
        self.name = backend.name
        self.count = 0

    def predict_actions(self, batch, seed):
        predictions = []
        for i, sample_id in enumerate(batch.sample_ids):
            one = EvaluationBatch(
                sample_ids=(sample_id,),
                target_actions=batch.target_actions[i : i + 1],
                action_is_pad=batch.action_is_pad[i : i + 1],
                current_state=batch.current_state[i : i + 1],
                observations=None if batch.observations is None else (batch.observations[i],),
            )
            predictions.append(self.backend.predict_actions(one, stable_seed(self.seed, sample_id)))
            self.count += 1
            if self.progress:
                self.progress(self.count, sample_id)
        return torch.cat(predictions)


def validate_tables(root, info):
    data = pd.concat(
        [pq.read_table(p).to_pandas() for p in sorted((root / "data").rglob("*.parquet"))], ignore_index=True
    )
    episodes = pd.concat(
        [pq.read_table(p).to_pandas() for p in sorted((root / "meta/episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    if len(data) != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("cache row/episode counts disagree with info.json")
    if (
        data["index"].duplicated().any()
        or data.duplicated(["episode_index", "frame_index"]).any()
        or episodes.episode_index.duplicated().any()
    ):
        raise ValueError("duplicate frame or episode identities")
    data = data.sort_values("index").reset_index(drop=True)
    if not np.array_equal(data["index"], np.arange(len(data))):
        raise ValueError("non-contiguous global indices")
    for ep in episodes.to_dict("records"):
        rows = data[data.episode_index == ep["episode_index"]]
        if len(rows) != ep["length"] or not np.array_equal(rows.frame_index, np.arange(len(rows))):
            raise ValueError("episode length/frame identity mismatch")
        if (
            rows["index"].iloc[0] != ep["dataset_from_index"]
            or rows["index"].iloc[-1] + 1 != ep["dataset_to_index"]
        ):
            raise ValueError("episode boundary mismatch")
    return data


def make_batches(job, preflight=False):
    root = Path(job["dataset_root"])
    if root.name != job["dataset_revision"] or root.parent.name != "snapshots":
        raise ValueError("dataset root is not the pinned snapshot")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 10:
        raise ValueError("protocol requires 10 FPS")
    data = validate_tables(root, info)
    checkpoint = Path(job["checkpoint"])
    train = json.loads((checkpoint / "train_config.json").read_text())
    config = json.loads((checkpoint / "config.json").read_text())
    if (
        train["dataset"]["repo_id"] != job["dataset_repo_id"]
        or train["dataset"]["revision"] != job["dataset_revision"]
    ):
        raise ValueError("checkpoint dataset identity/revision mismatch or unpinned")
    if train["dataset"]["episodes"] != job["train_episodes"] or set(job["train_episodes"]) & set(
        job["eval_episodes"]
    ):
        raise ValueError("training split mismatch or overlap")
    if (
        config["type"] != "smolvla"
        or config["chunk_size"] < 30
        or config["output_features"]["action"]["shape"] != [6]
    ):
        raise ValueError("checkpoint action interface mismatch")
    dataset = LeRobotDataset(
        job["dataset_repo_id"],
        root=root,
        revision=job["dataset_revision"],
        episodes=job["eval_episodes"],
        delta_timestamps={"action": [i / 10 for i in range(30)]},
        download_videos=False,
        video_backend="torchcodec",
    )
    raw = dataset.reader.hf_dataset
    lookup = {int(raw[i]["index"]): i for i in range(len(raw))}
    expected = data[data.episode_index.isin(job["eval_episodes"])]
    if len(lookup) != len(expected) or set(lookup) != set(expected["index"]):
        raise ValueError("reader frame identities differ from verified parquet")
    anchors = expected[(expected.frame_index >= 4) & ((expected.frame_index - 4) % 20 == 0)]
    if len(anchors) != job["expected_anchors"]:
        raise ValueError("anchor count mismatch")
    if preflight:
        anchors = anchors.iloc[[0, -1]]
    batches = []
    for anchor in anchors.to_dict("records"):
        ep, frame, absolute = int(anchor["episode_index"]), int(anchor["frame_index"]), int(anchor["index"])
        row = dataset[lookup[absolute]]
        ep_rows = data[data.episode_index == ep]
        indices = np.minimum(np.arange(frame, frame + 30), len(ep_rows) - 1)
        target = torch.tensor(np.stack(ep_rows.iloc[indices].action), dtype=torch.float32)
        padding = torch.arange(frame, frame + 30) >= len(ep_rows)
        if not torch.equal(row["action_is_pad"], padding) or not torch.allclose(
            row["action"], target, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("physical target/padding disagrees with episode-local parquet")
        if not torch.equal(row["observation.state"], torch.tensor(anchor["observation.state"])):
            raise ValueError("state identity mismatch")
        sample_id = (
            f"{job['dataset_repo_id']}@{job['dataset_revision']}:episode-{ep:04d}-frame-{absolute:06d}"
        )
        observation = {k: v for k, v in row.items() if k.startswith("observation.")}
        observation["task"] = row["task"]
        batches.append(
            EvaluationBatch(
                (sample_id,),
                target[None],
                padding[None],
                row["observation.state"][None],
                observations=(observation,),
            )
        )
    return batches


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    job = json.loads(args.job.read_text())
    status_path = args.output.with_suffix(".status.json")

    def progress(count, sample_id):
        write_json(
            status_path,
            {
                "state": "running",
                "samples_completed": count,
                "sample_id": sample_id,
                "updated_unix": time.time(),
            },
        )
        print(f"PROGRESS {count} {sample_id}", flush=True)

    try:
        progress(0, "validating")
        for path, digest in job["input_sha256"].items():
            if sha256(path) != digest:
                raise ValueError(f"input changed since manifest: {path}")
        batches = make_batches(job, args.preflight)
        baseline = evaluate_backend(
            StableBackend(StateRepeatBackend(), job["seed"]), batches, seed=job["seed"]
        )
        backend = SmolVLABackend.from_pretrained(job["checkpoint"], device=torch.device("cuda"))
        if args.preflight:
            stable = StableBackend(backend, job["seed"])
            first = stable.predict_actions(batches[0], 0)
            second = stable.predict_actions(batches[0], 999)
            if not torch.equal(first, second):
                raise ValueError("fixed sample seed is not reproducible")
        result = evaluate_backend(StableBackend(backend, job["seed"], progress), batches, seed=job["seed"])
        write_json(
            args.output,
            {
                "manifest": job,
                "preflight": args.preflight,
                "backends": {"state_repeat": baseline, "smolvla": result},
            },
        )
        write_json(
            status_path, {"state": "complete", "samples_completed": len(batches), "updated_unix": time.time()}
        )
        print(
            json.dumps(
                {"smolvla": result["aggregate_rmse_deg"], "state_repeat": baseline["aggregate_rmse_deg"]}
            ),
            flush=True,
        )
    except BaseException as error:
        write_json(status_path, {"state": "failed", "error": repr(error), "updated_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
