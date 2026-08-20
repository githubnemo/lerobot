"""Run isolated probes over the resumable sigma-4.27 artifact."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

EXTRACTOR_SCRIPT = Path("/home/anton/lerobot-video-vam/scripts/video_vam/resumable_sigma427_probe.py")
OUTPUT_DIR = Path("/home/anton/.cache/video-vam/context-probe-sigma427-v4")
sys.path.insert(0, str(EXTRACTOR_SCRIPT.parent.parent.parent))


def load_extractor_module():
    spec = importlib.util.spec_from_file_location("resumable_sigma427_probe", EXTRACTOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load shared probe implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK_ALPHAS = tuple(np.logspace(-3, 4, 8))


def fit_block_ridge(context_train, context_test, state_train, state_test, target, alpha_state, alpha_context):
    """Fit a joint ridge model with independent block penalties."""
    sm, ss = state_train.mean(0), state_train.std(0)
    cm, cs = context_train.mean(0), context_train.std(0)
    ss[ss < 1e-6] = 1.0
    cs[cs < 1e-6] = 1.0
    x_train = np.concatenate(((state_train - sm) / ss, (context_train - cm) / cs), axis=1)
    x_test = np.concatenate(((state_test - sm) / ss, (context_test - cm) / cs), axis=1)
    shape = target.shape[1:]
    y = target.reshape(target.shape[0], -1)
    ym = y.mean(0)
    penalty = np.diag([alpha_state] * state_train.shape[1] + [alpha_context] * context_train.shape[1])
    coefficients = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ (y - ym))
    return (
        (x_train @ coefficients + ym).reshape((-1, *shape)),
        (x_test @ coefficients + ym).reshape((-1, *shape)),
        ym.reshape(shape),
    )


def module_r2(prediction, target, baseline):
    return float(
        1.0 - np.square(prediction - target).sum() / max(float(np.square(baseline - target).sum()), 1e-12)
    )


def choose_block_alphas(context, states, target, episodes, train_episodes):
    """Select block penalties by episode-level inner CV."""
    folds = [tuple(e for e in train_episodes if e % 4 == fold) for fold in range(4)]
    best = (-np.inf, BLOCK_ALPHAS[0], BLOCK_ALPHAS[0])
    for a_state in BLOCK_ALPHAS:
        for a_context in BLOCK_ALPHAS:
            scores = []
            for validation in folds:
                fitting = tuple(e for e in train_episodes if e not in validation)
                fit_mask, val_mask = np.isin(episodes, fitting), np.isin(episodes, validation)
                _, prediction, baseline = fit_block_ridge(
                    context[fit_mask],
                    context[val_mask],
                    states[fit_mask],
                    states[val_mask],
                    target[fit_mask],
                    a_state,
                    a_context,
                )
                scores.append(
                    module_r2(prediction, target[val_mask], np.broadcast_to(baseline, target[val_mask].shape))
                )
            score = float(np.mean(scores))
            if score > best[0]:
                best = (score, a_state, a_context)
    return best[1], best[2]


def block_result(module, features, states, actions, episodes):
    train, test = np.isin(episodes, module.TRAIN_EPISODES), np.isin(episodes, module.TEST_EPISODES)
    a_state, a_context = choose_block_alphas(features, states, actions, episodes, module.TRAIN_EPISODES)
    _, prediction, baseline = fit_block_ridge(
        features[train], features[test], states[train], states[test], actions[train], a_state, a_context
    )
    base = np.broadcast_to(baseline, actions[test].shape)
    score = module_r2(prediction, actions[test], base)
    module.CURRENT_STATES = states
    scalar_alpha = module.choose_alpha(states, actions, episodes, module.TRAIN_EPISODES, "state")
    _, state_prediction, state_baseline = module.fit_ridge(
        states[train], states[test], actions[train], scalar_alpha
    )
    state_score = module_r2(
        state_prediction, actions[test], np.broadcast_to(state_baseline, actions[test].shape)
    )
    return {
        "state_alpha": a_state,
        "context_alpha": a_context,
        "state_alpha_at_boundary": a_state in (min(BLOCK_ALPHAS), max(BLOCK_ALPHAS)),
        "context_alpha_at_boundary": a_context in (min(BLOCK_ALPHAS), max(BLOCK_ALPHAS)),
        "r2": score,
        "state_scalar_r2": state_score,
        "incremental_r2": score - state_score,
        "joint_ge_state_sanity": bool(score >= state_score - 0.02),
        "per_timestep_r2": [
            module_r2(prediction[:, i], actions[test][:, i], base[:, i]) for i in range(module.ACTION_HORIZON)
        ],
    }


def uncertainty_with_residual(module, features, states, actions, episodes):
    """Estimate incremental and residual uncertainty across episode splits."""
    rng = np.random.default_rng(20260819)
    splits = []
    for _ in range(5):
        test_episodes = tuple(sorted(int(e) for e in rng.choice(module.EPISODES, size=8, replace=False)))
        train_episodes = tuple(e for e in module.EPISODES if e not in test_episodes)
        split = module.probe(
            features,
            episodes,
            actions,
            train_episodes=train_episodes,
            test_episodes=test_episodes,
            dimensions=(module.MAX_PROJECTION_DIM,),
        )
        splits.append(split["512"])
    incremental = [float(split["incremental"]["r2"]) for split in splits]
    residual = [float(split["residual"]["r2"]) for split in splits]
    return {
        "incremental_r2_mean": float(np.mean(incremental)),
        "incremental_r2_std": float(np.std(incremental, ddof=1)),
        "incremental_values": incremental,
        "residual_r2_mean": float(np.mean(residual)),
        "residual_r2_std": float(np.std(residual, ddof=1)),
        "residual_values": residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=0, help="number of completed rows; zero means all")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    module = load_extractor_module()
    metadata = json.loads((OUTPUT_DIR / "metadata.json").read_text())
    total = int(metadata["count"])
    module.TOTAL_COUNT = total
    artifact = module.Artifact(total)
    completed = np.asarray(artifact.complete, dtype=bool)
    available = np.flatnonzero(completed)
    count = len(available) if args.count == 0 else args.count
    if count > len(available):
        raise RuntimeError(f"requested {count} rows but only {len(available)} are complete")
    indices = available[:count]
    arrays = {
        "features": np.asarray(artifact.features[indices], dtype=np.float64),
        "episodes": np.asarray(artifact.episodes[indices], dtype=np.int64),
        "states": np.asarray(artifact.states[indices], dtype=np.float64),
        "actions": np.asarray(artifact.actions[indices], dtype=np.float64),
    }
    lengths = {name: value.shape[0] for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"artifact row lengths disagree: {lengths}")
    module.STATES_FOR_PROBE = arrays["states"]
    result = module.probe(
        arrays["features"],
        arrays["episodes"],
        arrays["actions"],
        train_episodes=module.TRAIN_EPISODES,
        test_episodes=module.TEST_EPISODES,
        dimensions=module.PROJECTION_DIMS,
    )
    result["512"]["block_joint"] = block_result(
        module,
        arrays["features"][:, : module.MAX_PROJECTION_DIM],
        arrays["states"],
        arrays["actions"],
        arrays["episodes"],
    )
    if args.tag in {"1000", "final"}:
        result["uncertainty"] = uncertainty_with_residual(
            module, arrays["features"], arrays["states"], arrays["actions"], arrays["episodes"]
        )
    result["count"] = count
    result["available_rows"] = int(len(available))
    result["projection_dims"] = list(module.PROJECTION_DIMS)
    result["alpha_grid"] = list(module.ALPHAS)
    result["block_alpha_grid"] = list(BLOCK_ALPHAS)
    path = OUTPUT_DIR / f"probe_{args.tag}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    summary = result[str(module.MAX_PROJECTION_DIM)]
    print(
        f"PROBE_{args.tag.upper()} count={count} "
        f"state_r2={summary['state']['r2']:.6f} "
        f"context_r2={summary['context']['r2']:.6f} "
        f"joint_r2={summary['joint']['r2']:.6f} "
        f"incremental_r2={summary['incremental']['r2']:.6f} "
        f"residual_r2={summary['residual']['r2']:.6f}",
        flush=True,
    )
    print("incremental_per_timestep", json.dumps(summary["incremental"]["per_timestep_r2"]), flush=True)
    if "uncertainty" in result:
        print("uncertainty", json.dumps(result["uncertainty"]), flush=True)


if __name__ == "__main__":
    main()
