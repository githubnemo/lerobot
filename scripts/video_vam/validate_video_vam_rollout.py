#!/usr/bin/env python3
"""Numerically compare the rollout wrapper with the frozen offline backend."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.modeling_video_vam import HISTORY_FRAME_INDEX_KEY, VideoVAMPolicy
from scripts.video_vam.dry_run_rollout import (
    DEFAULT_COSMOS_CHECKPOINT,
    DEFAULT_COSMOS_PROMPT,
    DEFAULT_COSMOS_RUN,
    DEFAULT_COSMOS_TOKENIZER,
    DEFAULT_DATASET,
    DEFAULT_LTX_PROMPT,
    DEFAULT_LTX_RUN,
    DEFAULT_LTX_TRANSFORMER,
    DEFAULT_LTX_VAE,
    CosmosSmolExpertBackend,
    EpisodeStream,
    LTXSmolExpertBackend,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _release(value: Any) -> None:
    closer = getattr(value, "close", None)
    if closer is not None:
        closer()
    del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _first_anchor(dataset_root: Path, episode: int):
    stream = EpisodeStream(dataset_root, episode)
    for local_index in range(len(stream)):
        observation = stream.read(local_index)
        if observation is not None:
            return observation
    raise RuntimeError("episode has fewer than five frames")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    observation = _first_anchor(args.dataset_root, args.episode)
    if args.backend == "cosmos":
        offline = CosmosSmolExpertBackend(
            args.run_dir,
            device,
            args.cosmos_checkpoint,
            args.cosmos_tokenizer,
            args.cosmos_prompt,
        )
    else:
        offline = LTXSmolExpertBackend(
            args.run_dir,
            device,
            args.ltx_transformer,
            args.ltx_vae,
            args.ltx_prompt,
        )
    try:
        reference = offline.predict_chunk(observation, args.action_seed)
    finally:
        _release(offline)

    frame_offset = observation.absolute_frame_index - observation.local_frame_index
    config = VideoVAMConfig(
        backend=args.backend,
        device=str(device),
        pretrained_path=args.run_dir,
        action_seed=args.action_seed,
        feature_seed_episode_index=observation.episode_index,
        feature_seed_frame_offset=frame_offset,
        joint_limits_min=[-10_000.0] * 6,
        joint_limits_max=[10_000.0] * 6,
        cosmos_checkpoint=args.cosmos_checkpoint,
        cosmos_tokenizer=args.cosmos_tokenizer,
        cosmos_prompt=args.cosmos_prompt,
        ltx_transformer=args.ltx_transformer,
        ltx_vae=args.ltx_vae,
        ltx_prompt=args.ltx_prompt,
    )
    policy = VideoVAMPolicy.from_pretrained(args.run_dir, config=config)
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=str(args.run_dir))

    def wrapper_forward() -> tuple[torch.Tensor, float]:
        _sync(device)
        started = time.perf_counter()
        batch = preprocessor(
            {
                "observation.state": observation.state.unsqueeze(0),
                "task": [observation.task],
            }
        )
        batch[f"{config.camera_key}.history"] = observation.rgb_history.float().div(255.0).to(device)
        batch[HISTORY_FRAME_INDEX_KEY] = torch.tensor(
            [observation.local_frame_index], device=device, dtype=torch.long
        )
        prediction = postprocessor(policy.predict_action_chunk(batch)).squeeze(0)
        _sync(device)
        return prediction.float(), time.perf_counter() - started

    try:
        wrapper_prediction, first_latency = wrapper_forward()
        steady_prediction, steady_latency = wrapper_forward()
    finally:
        policy.close()
        _release(policy)

    delta = wrapper_prediction - reference
    steady_delta = steady_prediction - reference
    result = {
        "backend": args.backend,
        "episode": observation.episode_index,
        "local_frame_index": observation.local_frame_index,
        "absolute_frame_index": observation.absolute_frame_index,
        "action_seed": args.action_seed,
        "max_abs_deviation": float(delta.abs().max()),
        "mean_abs_deviation": float(delta.abs().mean()),
        "rmse_deviation": float(delta.square().mean().sqrt()),
        "steady_repeat_max_abs_deviation": float((steady_prediction - wrapper_prediction).abs().max()),
        "steady_vs_offline_max_abs_deviation": float(steady_delta.abs().max()),
        "first_wrapper_latency_s": first_latency,
        "steady_wrapper_latency_s": steady_latency,
        "chunk_budget_s": 3.0,
        "steady_fits_budget": steady_latency < 3.0,
        "allclose_atol": args.atol,
        "allclose": bool(torch.allclose(wrapper_prediction, reference, rtol=0.0, atol=args.atol)),
    }
    if not result["allclose"]:
        raise RuntimeError(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cosmos", "ltx"), required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=32)
    parser.add_argument("--action-seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cosmos-checkpoint", type=Path, default=DEFAULT_COSMOS_CHECKPOINT)
    parser.add_argument("--cosmos-tokenizer", type=Path, default=DEFAULT_COSMOS_TOKENIZER)
    parser.add_argument("--cosmos-prompt", type=Path, default=DEFAULT_COSMOS_PROMPT)
    parser.add_argument("--ltx-transformer", type=Path, default=DEFAULT_LTX_TRANSFORMER)
    parser.add_argument("--ltx-vae", type=Path, default=DEFAULT_LTX_VAE)
    parser.add_argument("--ltx-prompt", type=Path, default=DEFAULT_LTX_PROMPT)
    args = parser.parse_args(argv)
    if args.run_dir is None:
        args.run_dir = DEFAULT_COSMOS_RUN if args.backend == "cosmos" else DEFAULT_LTX_RUN
    return args


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
