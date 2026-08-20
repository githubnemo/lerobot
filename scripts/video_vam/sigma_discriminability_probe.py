"""Measure Cosmos context discriminability across extraction noise levels."""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_feature_cache import derive_window_seed
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample

MANIFEST_PATH = Path("/home/anton/.cache/video-vam/cosmos-rehearsal-stride20/manifest.json")
DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
PROMPT_PATH = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
CHECKPOINT_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
TOKENIZER_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
OUTPUT_DIR = Path("/home/anton/.cache/video-vam/sigma-discriminability")
SIGMAS = (0.1, 0.5, 1.0, 2.0, 4.27, 8.0, 10.0)
EPISODES = (0, 5, 10, 15, 20, 25, 30, 35)
WINDOWS_PER_EPISODE = 4
RIDGE_ALPHA = 10.0


def select_windows() -> list[dict[str, int | str]]:
    """Select four manifest windows from each spaced episode."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_episode: dict[int, list[dict[str, int | str]]] = {episode: [] for episode in EPISODES}
    for entry in manifest["entries"]:
        episode = int(entry["episode_index"])
        if episode in by_episode:
            by_episode[episode].append(entry)
    selected: list[dict[str, int | str]] = []
    for episode in EPISODES:
        candidates = by_episode[episode]
        if len(candidates) < WINDOWS_PER_EPISODE:
            raise RuntimeError(f"episode {episode} has only {len(candidates)} candidate windows")
        positions = np.linspace(0, len(candidates) - 1, WINDOWS_PER_EPISODE, dtype=int)
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def centered_cosines(features: np.ndarray, episodes: np.ndarray) -> tuple[float, float]:
    """Return same-episode and different-episode centered cosine means."""
    centered = features - features.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    values: dict[str, list[float]] = {"same": [], "different": []}
    for left, right in combinations(range(len(centered)), 2):
        cosine = float(centered[left] @ centered[right] / max(norms[left] * norms[right], 1e-12))
        key = "same" if episodes[left] == episodes[right] else "different"
        values[key].append(cosine)
    return float(np.mean(values["same"])), float(np.mean(values["different"]))


def ridge_r2(features: np.ndarray, target: np.ndarray, episodes: np.ndarray) -> float:
    """Fit a grouped ridge probe and report R2 against a train-mean baseline."""
    train_episodes = set(EPISODES[:6])
    train = np.array([episode in train_episodes for episode in episodes])
    test = ~train
    x_mean = features[train].mean(axis=0)
    x_scale = features[train].std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    x_train = (features[train] - x_mean) / x_scale
    x_test = (features[test] - x_mean) / x_scale
    y_train = target[train]
    y_mean = y_train.mean(axis=0)
    gram = x_train @ x_train.T + RIDGE_ALPHA * np.eye(x_train.shape[0], dtype=np.float32)
    dual = np.linalg.solve(gram, y_train - y_mean)
    prediction = (x_test @ x_train.T) @ dual + y_mean
    baseline = y_mean
    residual = float(np.square(prediction - target[test]).sum())
    baseline_error = float(np.square(target[test] - baseline).sum())
    return float(1.0 - residual / max(baseline_error, 1e-12))


def main() -> None:
    """Extract all sigma contexts and write metrics/artifacts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the extraction sweep")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_windows()
    episodes = np.asarray([int(entry["episode_index"]) for entry in selected], dtype=np.int64)
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=DATASET_ROOT,
        episodes=list(EPISODES),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    prompt = load_prompt_embedding(PROMPT_PATH).embedding
    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=CHECKPOINT_PATH,
            tokenizer_path=TOKENIZER_PATH,
            device="cuda",
            dtype="bfloat16",
            high_noise_sigma=10.0,
            seed=0,
            hidden_layer=20,
            stop_after_step=0,
        )
    )
    features = np.empty((len(SIGMAS), len(selected), 2048), dtype=np.float32)
    states = np.empty((len(selected), 6), dtype=np.float32)
    actions = np.empty((len(selected), 6), dtype=np.float32)
    for index, entry in enumerate(selected):
        frame = int(entry["frame_index"])
        sample = dataset[_relative_index(dataset, frame)]
        prepared = prepare_sample(sample, frame_index=frame, config=CUBE_OUT_OF_BOX_CONTRACT)
        states[index] = prepared.state[0, 0].float().numpy()
        actions[index] = prepared.target_action[0, 0].float().numpy()
        noise_generator = torch.Generator(device="cuda").manual_seed(
            derive_window_seed(dataset.revision, int(entry["episode_index"]), frame, 0)
        )
        noise = torch.randn(
            (1, 16, 16, 60, 80), device="cuda", dtype=torch.bfloat16, generator=noise_generator
        )
        for sigma_index, sigma in enumerate(SIGMAS):
            with torch.inference_mode():
                extraction = extractor.extract(
                    prepared.rgb_history,
                    prompt,
                    sigma=torch.tensor([sigma], device="cuda", dtype=torch.float32),
                    noise=noise,
                )
            features[sigma_index, index] = extraction.tokens.float().mean(dim=1).cpu().numpy()[0]
        print(
            f"window {index + 1}/{len(selected)} episode={entry['episode_index']} frame={frame}", flush=True
        )
    np.savez_compressed(OUTPUT_DIR / "pooled_contexts.npz", features=features, episodes=episodes)
    np.savez_compressed(OUTPUT_DIR / "labels.npz", state=states, action=actions, episodes=episodes)

    rows: list[dict[str, float]] = []
    for sigma_index, sigma in enumerate(SIGMAS):
        feature = features[sigma_index]
        same_cosine, different_cosine = centered_cosines(feature, episodes)
        mean_norm = np.linalg.norm(feature.mean(axis=0))
        std_norm = np.linalg.norm(feature.std(axis=0))
        rows.append(
            {
                "sigma": sigma,
                "same_episode_centered_cosine": same_cosine,
                "different_episode_centered_cosine": different_cosine,
                "std_norm_over_mean_norm": float(std_norm / max(mean_norm, 1e-12)),
                "state_r2": ridge_r2(feature, states, episodes),
                "near_term_action_r2": ridge_r2(feature, actions, episodes),
            }
        )
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("sigma same_cos different_cos std_over_mean state_r2 action_r2")
    for row in rows:
        print(
            f"{row['sigma']:>5g} {row['same_episode_centered_cosine']:>+.6f} "
            f"{row['different_episode_centered_cosine']:>+.6f} "
            f"{row['std_norm_over_mean_norm']:.6f} {row['state_r2']:+.6f} "
            f"{row['near_term_action_r2']:+.6f}"
        )
    print(f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
