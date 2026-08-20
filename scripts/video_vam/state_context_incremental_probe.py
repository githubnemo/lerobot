"""Measure incremental action information from cached state and context."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.policies.vam.cosmos_cache_dataset import CosmosFeatureCacheDataset, load_cache_manifest

FEATURES_PATH = Path("/home/anton/.cache/video-vam/sigma-discriminability/pooled_contexts.npz")
LABELS_PATH = Path("/home/anton/.cache/video-vam/sigma-discriminability/labels.npz")
MANIFEST_PATH = Path("/home/anton/.cache/video-vam/cosmos-rehearsal-stride20/manifest.json")
OUTPUT_DIR = Path("/home/anton/.cache/video-vam/sigma-discriminability")
EPISODES = (0, 5, 10, 15, 20, 25, 30, 35)
WINDOWS_PER_EPISODE = 4
SIGMAS = (0.1, 0.5, 1.0, 2.0, 4.27, 8.0, 10.0)
RIDGE_ALPHA = 10.0
CONDITIONING_TOKENS = 2 * 30 * 40


def select_windows() -> list[dict[str, int | str]]:
    """Reproduce the window ordering used by the GPU sweep."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_episode: dict[int, list[dict[str, int | str]]] = {episode: [] for episode in EPISODES}
    for entry in manifest["entries"]:
        episode = int(entry["episode_index"])
        if episode in by_episode:
            by_episode[episode].append(entry)
    selected: list[dict[str, int | str]] = []
    for episode in EPISODES:
        candidates = by_episode[episode]
        positions = np.linspace(0, len(candidates) - 1, WINDOWS_PER_EPISODE, dtype=int)
        selected.extend(candidates[int(position)] for position in positions)
    return selected


def ridge_fit_predict(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    *,
    alpha: float = RIDGE_ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit standardized dual-form ridge and return train/test predictions."""
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_features - mean) / scale
    test_x = (test_features - mean) / scale
    target_shape = train_target.shape[1:]
    train_y = train_target.reshape(train_target.shape[0], -1)
    target_mean = train_y.mean(axis=0)
    gram = train_x @ train_x.T + alpha * np.eye(train_x.shape[0], dtype=np.float64)
    dual = np.linalg.solve(gram, train_y - target_mean)
    train_prediction = (train_x @ train_x.T) @ dual + target_mean
    test_prediction = (test_x @ train_x.T) @ dual + target_mean
    return train_prediction.reshape((-1, *target_shape)), test_prediction.reshape((-1, *target_shape))


def r2_against_train_mean(prediction: np.ndarray, target: np.ndarray, train_target: np.ndarray) -> float:
    """Compute R² against the training-set mean predictor."""
    baseline = train_target.mean(axis=0)
    residual = float(np.square(prediction - target).sum())
    baseline_error = float(np.square(target - baseline).sum())
    return float(1.0 - residual / max(baseline_error, 1e-12))


def per_timestep_r2(prediction: np.ndarray, target: np.ndarray, train_target: np.ndarray) -> list[float]:
    """Compute one joint-aggregated R² value for each action timestep."""
    return [
        r2_against_train_mean(prediction[:, timestep], target[:, timestep], train_target[:, timestep])
        for timestep in range(target.shape[1])
    ]


def probe_action(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_action: np.ndarray,
    test_action: np.ndarray,
) -> dict[str, Any]:
    """Return state/context/joint and residual action probe metrics."""
    state_train, state_test = ridge_fit_predict(train_features[:, :6], test_features[:, :6], train_action)
    context_train, context_test = ridge_fit_predict(train_features[:, 6:], test_features[:, 6:], train_action)
    joint_train, joint_test = ridge_fit_predict(train_features, test_features, train_action)
    state_r2 = r2_against_train_mean(state_test, test_action, train_action)
    context_r2 = r2_against_train_mean(context_test, test_action, train_action)
    joint_r2 = r2_against_train_mean(joint_test, test_action, train_action)
    state_residual_train = train_action - state_train
    state_residual_test = test_action - state_test
    _, residual_prediction = ridge_fit_predict(
        train_features[:, 6:], test_features[:, 6:], state_residual_train
    )
    residual_r2 = float(
        1.0
        - np.square(residual_prediction - state_residual_test).sum()
        / max(np.square(state_residual_test).sum(), 1e-12)
    )
    return {
        "state_r2": state_r2,
        "context_r2": context_r2,
        "joint_r2": joint_r2,
        "incremental_r2": joint_r2 - state_r2,
        "residual_r2_after_state": residual_r2,
        "state_per_timestep_r2": per_timestep_r2(state_test, test_action, train_action),
        "context_per_timestep_r2": per_timestep_r2(context_test, test_action, train_action),
        "joint_per_timestep_r2": per_timestep_r2(joint_test, test_action, train_action),
        "incremental_per_timestep_r2": [
            joint - state
            for joint, state in zip(
                per_timestep_r2(joint_test, test_action, train_action),
                per_timestep_r2(state_test, test_action, train_action),
                strict=True,
            )
        ],
    }


def main() -> None:
    """Run incremental probes and write JSON/CSV reports."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_windows()
    feature_data = np.load(FEATURES_PATH)
    label_data = np.load(LABELS_PATH)
    pooled = feature_data["features"].astype(np.float64)
    episodes = feature_data["episodes"].astype(np.int64)
    saved_state = label_data["state"].astype(np.float64)
    if len(selected) != len(episodes) or not np.array_equal(episodes, label_data["episodes"]):
        raise RuntimeError("saved sweep ordering does not match labels or selected windows")
    manifest = load_cache_manifest(MANIFEST_PATH)
    dataset = CosmosFeatureCacheDataset(manifest, shuffle_seed=0)
    entry_by_id = {entry.sample_id: entry for entry in manifest.entries}
    full_action = np.empty((len(selected), 30, 6), dtype=np.float64)
    cache_state = np.empty_like(saved_state)
    conditioning = np.empty((len(selected), 2048), dtype=np.float64)
    generated = np.empty((len(selected), 2048), dtype=np.float64)
    for index, entry in enumerate(selected):
        item = dataset.load_entry(entry_by_id[str(entry["sample_id"])])
        full_action[index] = item.target_action[0].float().numpy()
        cache_state[index] = item.state[0, 0].float().numpy()
        context = item.context[0].float().numpy()
        conditioning[index] = context[:CONDITIONING_TOKENS].mean(axis=0)
        generated[index] = context[CONDITIONING_TOKENS:].mean(axis=0)
    if not np.allclose(cache_state, saved_state):
        raise RuntimeError("cached state labels do not match sweep labels")
    train = np.isin(episodes, EPISODES[:6])
    test = ~train
    results: dict[str, Any] = {
        "alpha": RIDGE_ALPHA,
        "train_episodes": list(EPISODES[:6]),
        "test_episodes": list(EPISODES[6:]),
    }
    rows: list[dict[str, float | str]] = []
    for sigma_index, sigma in enumerate(SIGMAS):
        feature = np.concatenate((cache_state, pooled[sigma_index]), axis=1)
        metrics = probe_action(feature[train], feature[test], full_action[train], full_action[test])
        results[str(sigma)] = metrics
        rows.append(
            {
                "sigma": sigma,
                "state_r2": float(metrics["state_r2"]),
                "context_r2": float(metrics["context_r2"]),
                "joint_r2": float(metrics["joint_r2"]),
                "incremental_r2": float(metrics["incremental_r2"]),
                "residual_r2_after_state": float(metrics["residual_r2_after_state"]),
            }
        )
    token_metrics: dict[str, Any] = {}
    for name, token_feature in (("conditioning_tokens", conditioning), ("generated_tokens", generated)):
        feature = np.concatenate((cache_state, token_feature), axis=1)
        token_metrics[name] = probe_action(
            feature[train], feature[test], full_action[train], full_action[test]
        )
    results["sigma_10_token_groups"] = token_metrics
    (OUTPUT_DIR / "incremental_probe.json").write_text(json.dumps(results, indent=2) + "\n")
    with (OUTPUT_DIR / "incremental_probe.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("sigma state_r2 context_r2 joint_r2 incremental residual_r2")
    for row in rows:
        print(
            f"{row['sigma']:>5g} {row['state_r2']:+.6f} {row['context_r2']:+.6f} "
            f"{row['joint_r2']:+.6f} {row['incremental_r2']:+.6f} {row['residual_r2_after_state']:+.6f}"
        )
    for name, metrics in token_metrics.items():
        print(
            f"{name} state={metrics['state_r2']:+.6f} context={metrics['context_r2']:+.6f} "
            f"joint={metrics['joint_r2']:+.6f} incremental={metrics['incremental_r2']:+.6f}"
        )
    print("sigma_10_incremental_per_timestep", json.dumps(results["10.0"]["incremental_per_timestep_r2"]))
    print("sigma_4.27_incremental_per_timestep", json.dumps(results["4.27"]["incremental_per_timestep_r2"]))


if __name__ == "__main__":
    main()
