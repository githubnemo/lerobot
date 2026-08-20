"""Extract resumable pooled Cosmos context and run episode-grouped probes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

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

DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
PROMPT_PATH = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")
CHECKPOINT_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
TOKENIZER_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
OUTPUT_DIR = Path("/home/anton/.cache/video-vam/context-probe-sigma427-v4")
SIGMA = 4.27
PROJECTION_SEED = 20260818
MAX_PROJECTION_DIM = 512
PROJECTION_DIMS = (128, 256, 512)
COARSE_HEIGHT = 5
COARSE_WIDTH = 5
FEATURE_PROJECTION_DIM = 32
LATENT_FRAMES = 16
LATENT_HEIGHT = 30
LATENT_WIDTH = 40
CONDITIONING_FRAMES = 2
ACTION_HORIZON = 30
CHECKPOINT_EVERY = 8
PROGRESSIVE_COUNT = 1000
EPISODES = tuple(range(40))
TRAIN_EPISODES = tuple(range(32))
TEST_EPISODES = tuple(range(32, 40))
ALPHAS = tuple(np.logspace(-6, 18, 33))


class Artifact:
    """Memory-mapped resumable extraction arrays."""

    def __init__(self, count: int) -> None:
        self.features = self._open("features.npy", np.float32, (count, MAX_PROJECTION_DIM))
        self.episodes = self._open("episodes.npy", np.int16, (count,))
        self.frames = self._open("frames.npy", np.int32, (count,))
        self.states = self._open("states.npy", np.float32, (count, 6))
        self.actions = self._open("actions.npy", np.float32, (count, ACTION_HORIZON, 6))
        self.action_pad = self._open("action_pad.npy", np.bool_, (count, ACTION_HORIZON))
        self.complete = self._open("complete.npy", np.bool_, (count,))

    @staticmethod
    def _open(name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
        path = OUTPUT_DIR / name
        if path.exists():
            array = np.load(path, mmap_mode="r+")
            if array.shape != shape or array.dtype != dtype:
                raise RuntimeError(f"incompatible existing artifact {path}: {array.shape} {array.dtype}")
            return array
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)

    def flush(self) -> None:
        for array in (
            self.features,
            self.episodes,
            self.frames,
            self.states,
            self.actions,
            self.action_pad,
            self.complete,
        ):
            array.flush()


def save_complete(array: np.memmap) -> None:
    """Atomically refresh the small completion bitmap."""
    temporary = OUTPUT_DIR / "complete.npy.tmp"
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(array))
    temporary.replace(OUTPUT_DIR / "complete.npy")


def episode_schedule(lengths: dict[int, int]) -> list[tuple[int, int, int]]:
    """Interleave valid local frames across every episode."""
    offsets: dict[int, int] = {}
    offset = 0
    for episode in EPISODES:
        offsets[episode] = offset
        offset += lengths[episode]
    schedule = []
    for local_frame in range(max(lengths.values())):
        for episode in EPISODES:
            if local_frame < lengths[episode]:
                schedule.append((offsets[episode] + local_frame, episode, local_frame))
    return schedule


def build_projection() -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed two-stage coarse-spatial random projection."""
    path = OUTPUT_DIR / "projection.npz"
    if path.exists():
        data = np.load(path)
        return data["feature_projection"], data["output_projection"]
    rng = np.random.default_rng(PROJECTION_SEED)
    feature_projection = rng.standard_normal((2048, FEATURE_PROJECTION_DIM), dtype=np.float32) / np.sqrt(
        FEATURE_PROJECTION_DIM
    )
    structured_dim = LATENT_FRAMES * COARSE_HEIGHT * COARSE_WIDTH * FEATURE_PROJECTION_DIM
    output_projection = rng.standard_normal((structured_dim, MAX_PROJECTION_DIM), dtype=np.float32) / np.sqrt(
        MAX_PROJECTION_DIM
    )
    np.savez_compressed(
        path,
        feature_projection=feature_projection,
        output_projection=output_projection,
        seed=np.asarray(PROJECTION_SEED),
    )
    return feature_projection, output_projection


def pooled_projection(
    tokens: torch.Tensor, feature_projection: np.ndarray, output_projection: np.ndarray
) -> np.ndarray:
    """Retain coarse spatial and latent-frame structure before projection."""
    if tuple(tokens.shape) != (1, LATENT_FRAMES * LATENT_HEIGHT * LATENT_WIDTH, 2048):
        raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
    grid = tokens[0].float().reshape(LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH, 2048)
    grid = (
        grid.reshape(
            LATENT_FRAMES,
            COARSE_HEIGHT,
            LATENT_HEIGHT // COARSE_HEIGHT,
            COARSE_WIDTH,
            LATENT_WIDTH // COARSE_WIDTH,
            2048,
        )
        .mean(dim=(2, 4))
        .cpu()
        .numpy()
    )
    structured = grid.reshape(-1, 2048) @ feature_projection
    return (structured.reshape(-1) @ output_projection).astype(np.float32)


def r2_score(prediction: np.ndarray, target: np.ndarray, baseline: np.ndarray) -> float:
    """Compute scalar R² against a supplied baseline prediction."""
    error = np.square(prediction - target).sum()
    baseline_error = np.square(baseline - target).sum()
    return float(1.0 - error / max(float(baseline_error), 1e-12))


def fit_ridge(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardized primal ridge and return train/test prediction plus target mean."""
    x_mean = train_x.mean(axis=0)
    x_scale = train_x.std(axis=0)
    x_scale[x_scale < 1e-6] = 1.0
    x_train = (train_x - x_mean) / x_scale
    x_test = (test_x - x_mean) / x_scale
    y_shape = train_y.shape[1:]
    y_train = train_y.reshape(train_y.shape[0], -1)
    y_mean = y_train.mean(axis=0)
    if x_train.shape[0] <= x_train.shape[1]:
        gram = x_train @ x_train.T + alpha * np.eye(x_train.shape[0], dtype=np.float64)
        dual = np.linalg.solve(gram, y_train - y_mean)
        train_prediction = (x_train @ x_train.T @ dual + y_mean).reshape((-1, *y_shape))
        test_prediction = (x_test @ x_train.T @ dual + y_mean).reshape((-1, *y_shape))
    else:
        gram = x_train.T @ x_train + alpha * np.eye(x_train.shape[1], dtype=np.float64)
        coefficients = np.linalg.solve(gram, x_train.T @ (y_train - y_mean))
        train_prediction = (x_train @ coefficients + y_mean).reshape((-1, *y_shape))
        test_prediction = (x_test @ coefficients + y_mean).reshape((-1, *y_shape))
    return train_prediction, test_prediction, y_mean.reshape(y_shape)


def choose_alpha(
    features: np.ndarray,
    target: np.ndarray,
    episodes: np.ndarray,
    train_episodes: tuple[int, ...],
    model: str,
    states: np.ndarray | None = None,
) -> float:
    """Select ridge alpha using episode-level four-fold validation."""
    fold_episodes = [tuple(e for e in train_episodes if e % 4 == fold) for fold in range(4)]
    scores: list[float] = []
    for alpha in ALPHAS:
        fold_scores = []
        for validation_episodes in fold_episodes:
            fit_episodes = tuple(e for e in train_episodes if e not in validation_episodes)
            fit_mask = np.isin(episodes, fit_episodes)
            validation_mask = np.isin(episodes, validation_episodes)
            x_fit = model_features(features, model, fit_mask, states)
            x_validation = model_features(features, model, validation_mask, states)
            _, prediction, baseline = fit_ridge(x_fit, x_validation, target[fit_mask], alpha)
            fold_scores.append(
                r2_score(
                    prediction,
                    target[validation_mask],
                    np.broadcast_to(baseline, target[validation_mask].shape),
                )
            )
        scores.append(float(np.mean(fold_scores)))
    return float(ALPHAS[int(np.argmax(scores))])


def model_features(
    features: np.ndarray, model: str, mask: np.ndarray, states: np.ndarray | None = None
) -> np.ndarray:
    """Select state, context, or joint features."""
    context = features[mask]
    state = (CURRENT_STATES if states is None else states)[mask]
    if model == "state":
        return state
    if model == "context":
        return context
    if model == "joint":
        return np.concatenate((state, context), axis=1)
    raise ValueError(f"unknown model {model}")


CURRENT_STATES: np.ndarray


def probe(
    features: np.ndarray,
    episodes: np.ndarray,
    actions: np.ndarray,
    *,
    train_episodes: tuple[int, ...],
    test_episodes: tuple[int, ...],
    dimensions: tuple[int, ...],
) -> dict[str, Any]:
    """Run cross-validated action and residual probes for one episode split."""
    global CURRENT_STATES
    train_mask = np.isin(episodes, train_episodes)
    test_mask = np.isin(episodes, test_episodes)
    result: dict[str, Any] = {"train_episodes": list(train_episodes), "test_episodes": list(test_episodes)}
    for dimension in dimensions:
        dimension_result: dict[str, Any] = {}
        reduced = features[:, :dimension]
        for model in ("state", "context", "joint"):
            model_input = reduced if model != "joint" else reduced
            CURRENT_STATES = STATES_FOR_PROBE
            alpha = choose_alpha(model_input, actions, episodes, train_episodes, model)
            train_x = model_features(model_input, model, train_mask)
            test_x = model_features(model_input, model, test_mask)
            train_prediction, test_prediction, baseline = fit_ridge(
                train_x, test_x, actions[train_mask], alpha
            )
            baseline_test = np.broadcast_to(baseline, actions[test_mask].shape)
            dimension_result[model] = {
                "alpha": alpha,
                "r2": r2_score(test_prediction, actions[test_mask], baseline_test),
                "per_timestep_r2": [
                    r2_score(test_prediction[:, step], actions[test_mask][:, step], baseline_test[:, step])
                    for step in range(ACTION_HORIZON)
                ],
            }
            if model == "state":
                state_train_prediction = train_prediction
                state_test_prediction = test_prediction
        state_result = dimension_result["state"]
        joint_result = dimension_result["joint"]
        state_per_step = state_result["per_timestep_r2"]
        joint_per_step = joint_result["per_timestep_r2"]
        dimension_result["incremental"] = {
            "r2": float(joint_result["r2"] - state_result["r2"]),
            "joint_ge_state_sanity": bool(joint_result["r2"] >= state_result["r2"] - 0.02),
            "per_timestep_r2": [
                joint - state for joint, state in zip(joint_per_step, state_per_step, strict=True)
            ],
        }
        residual_train = actions[train_mask] - state_train_prediction
        residual_test = actions[test_mask] - state_test_prediction
        context_alpha = choose_alpha(
            reduced[train_mask],
            residual_train,
            episodes[train_mask],
            train_episodes,
            "context",
            states=STATES_FOR_PROBE[train_mask],
        )
        _, residual_prediction, _ = fit_ridge(
            reduced[train_mask], reduced[test_mask], residual_train, context_alpha
        )
        residual_baseline = np.zeros_like(residual_test)
        dimension_result["residual"] = {
            "alpha": context_alpha,
            "r2": r2_score(residual_prediction, residual_test, residual_baseline),
            "per_timestep_r2": [
                r2_score(residual_prediction[:, step], residual_test[:, step], residual_baseline[:, step])
                for step in range(ACTION_HORIZON)
            ],
        }
        result[str(dimension)] = dimension_result
    return result


def uncertainty(
    features: np.ndarray,
    episodes: np.ndarray,
    actions: np.ndarray,
    dimensions: tuple[int, ...],
) -> dict[str, Any]:
    """Estimate split variability using five random episode-level test sets."""
    rng = np.random.default_rng(20260819)
    splits = []
    for _split in range(5):
        test = tuple(sorted(int(e) for e in rng.choice(EPISODES, size=8, replace=False)))
        train = tuple(e for e in EPISODES if e not in test)
        splits.append(
            probe(
                features, episodes, actions, train_episodes=train, test_episodes=test, dimensions=dimensions
            )
        )
    summary: dict[str, Any] = {"splits": splits}
    for dimension in dimensions:
        values = [float(split[str(dimension)]["incremental"]["r2"]) for split in splits]
        summary[str(dimension)] = {
            "incremental_r2_mean": float(np.mean(values)),
            "incremental_r2_std": float(np.std(values, ddof=1)),
            "values": values,
        }
    return summary


def write_probe(count: int, tag: str) -> None:
    """Run and persist probes on the completed prefix."""
    global STATES_FOR_PROBE
    artifact = Artifact(TOTAL_COUNT)
    indices = np.flatnonzero(np.asarray(artifact.complete)[:count])
    if len(indices) < count:
        raise RuntimeError("progressive probe requires a complete prefix")
    features = np.asarray(artifact.features[indices], dtype=np.float64)
    episodes = np.asarray(artifact.episodes[indices], dtype=np.int64)
    actions = np.asarray(artifact.actions[indices], dtype=np.float64)
    STATES_FOR_PROBE = np.asarray(artifact.states[indices], dtype=np.float64)
    result = probe(
        features,
        episodes,
        actions,
        train_episodes=TRAIN_EPISODES,
        test_episodes=TEST_EPISODES,
        dimensions=PROJECTION_DIMS,
    )
    if tag == "final":
        result["uncertainty"] = uncertainty(features, episodes, actions, (MAX_PROJECTION_DIM,))
    result["count"] = int(count)
    result["projection_dims"] = list(PROJECTION_DIMS)
    (OUTPUT_DIR / f"probe_{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = result[str(MAX_PROJECTION_DIM)]
    print(
        f"PROBE_{tag.upper()} count={count} "
        f"state_r2={summary['state']['r2']:.4f} "
        f"context_r2={summary['context']['r2']:.4f} "
        f"joint_r2={summary['joint']['r2']:.4f} "
        f"incremental_r2={summary['incremental']['r2']:.4f} "
        f"residual_r2={summary['residual']['r2']:.4f}",
        flush=True,
    )


def main() -> None:
    """Extract all valid windows with resumable checkpoints."""
    global TOTAL_COUNT
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for extraction")
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=DATASET_ROOT,
        episodes=list(EPISODES),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    lengths = {
        int(row["episode_index"]): int(row["length"]) - ACTION_HORIZON for row in dataset.meta.episodes
    }
    schedule = episode_schedule(lengths)
    TOTAL_COUNT = len(schedule)
    metadata = {
        "count": TOTAL_COUNT,
        "sigma": SIGMA,
        "projection_seed": PROJECTION_SEED,
        "projection_dims": list(PROJECTION_DIMS),
        "coarse_grid": [LATENT_FRAMES, COARSE_HEIGHT, COARSE_WIDTH],
        "feature_projection_dim": FEATURE_PROJECTION_DIM,
        "train_episodes": list(TRAIN_EPISODES),
        "test_episodes": list(TEST_EPISODES),
        "action_horizon": ACTION_HORIZON,
    }
    metadata_path = OUTPUT_DIR / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("existing artifact metadata does not match this run")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    artifact = Artifact(TOTAL_COUNT)
    feature_projection, output_projection = build_projection()
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
    completed = np.asarray(artifact.complete)
    start_time = time.perf_counter()
    initial = int(completed.sum())
    print(f"START total={TOTAL_COUNT} already_complete={initial}", flush=True)
    for index, (absolute_frame, episode, local_frame) in enumerate(schedule):
        if completed[index]:
            continue
        sample = dataset[_relative_index(dataset, absolute_frame)]
        prepared = prepare_sample(sample, frame_index=local_frame, config=CUBE_OUT_OF_BOX_CONTRACT)
        noise_seed = derive_window_seed(dataset.revision, episode, local_frame, 0)
        noise = torch.randn(
            (1, 16, 16, 60, 80),
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(noise_seed),
        )
        with torch.inference_mode():
            extraction = extractor.extract(
                prepared.rgb_history,
                prompt,
                sigma=torch.tensor([SIGMA], device="cuda", dtype=torch.float32),
                noise=noise,
            )
        artifact.features[index] = pooled_projection(extraction.tokens, feature_projection, output_projection)
        artifact.episodes[index] = episode
        artifact.frames[index] = local_frame
        artifact.states[index] = prepared.state[0, 0].float().cpu().numpy()
        artifact.actions[index] = prepared.target_action[0].float().cpu().numpy()
        artifact.action_pad[index] = prepared.action_is_pad[0].bool().cpu().numpy()
        artifact.complete[index] = True
        completed[index] = True
        if (index + 1) % CHECKPOINT_EVERY == 0 or index + 1 == TOTAL_COUNT:
            artifact.flush()
            save_complete(artifact.complete)
        del extraction, noise, prepared, sample
        if (index + 1) % 10 == 0:
            elapsed = time.perf_counter() - start_time
            done = int(completed.sum())
            rate = (done - initial) / max(elapsed, 1e-6)
            eta = (TOTAL_COUNT - done) / max(rate, 1e-6)
            print(
                f"PROGRESS completed={done}/{TOTAL_COUNT} rate={rate:.3f}/s eta_hours={eta / 3600:.2f} episode={episode} frame={local_frame}",
                flush=True,
            )
    artifact.flush()
    save_complete(artifact.complete)
    print("EXTRACTION_DONE", TOTAL_COUNT, flush=True)


TOTAL_COUNT = 0
STATES_FOR_PROBE = np.empty((0, 6), dtype=np.float64)


if __name__ == "__main__":
    main()
