#!/usr/bin/env python3
"""Measure action-flow uncertainty, conditioning stability, and RTC seams offline."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
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
    DEFAULT_SMOLVLA,
    FPS,
    HORIZON,
    JOINT_NAMES,
    JOINT_UNITS,
    CosmosSmolExpertBackend,
    EpisodeStream,
    LTXSmolExpertBackend,
    ReplayObservation,
    SmolExpertRolloutBackend,
    SmolVLABackend,
    SmolVLARolloutBackend,
    _finite_chunk,
    _train_stats,
)

POLICIES = ("smolvla", "cosmos-smolexpert", "ltx-smolexpert")
DEFAULT_MANIFESTS = {
    "cosmos-smolexpert": Path(
        "/home/anton/.cache/video-vam/cosmos-train0-31-stride3-sigma80-prefix-pool2/manifest.json"
    ),
    "ltx-smolexpert": Path("/home/anton/.cache/video-vam/ltx25-train0-31-stride3-pool2/manifest.json"),
}
MEASURED_INFERENCE_SECONDS = {
    "smolvla": 0.265,
    "cosmos-smolexpert": 1.288,
    "ltx-smolexpert": 2.384,
}


@dataclass(frozen=True, slots=True)
class Anchor:
    sample_id: str
    episode_index: int
    frame_index: int
    state: Tensor
    target_actions: Tensor
    action_is_pad: Tensor
    context: Tensor | None = None
    observation: ReplayObservation | None = None


class CachedExpertBackend(SmolExpertRolloutBackend):
    """Action-only backend whose frozen contexts are loaded from cache."""

    def __init__(self, run_dir: Path, device: torch.device, name: str, input_channels: int) -> None:
        self.name = name
        self.input_channels = input_channels
        super().__init__(run_dir, device)

    def extract_context(self, observation: ReplayObservation) -> Tensor:
        del observation
        raise RuntimeError("cached analysis never runs a video extractor")

    def predict(
        self,
        anchor: Anchor,
        seed: int,
        *,
        previous_actions: Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if anchor.context is None:
            raise ValueError("cached expert anchor has no context")
        rtc_processor = None
        if previous_actions is not None:
            if inference_delay is None or execution_horizon is None:
                raise ValueError("RTC requires delay and execution horizon")
            rtc_processor = RTCProcessor(RTCConfig(execution_horizon=execution_horizon))
        actions = self.decoder.sample_actions(
            anchor.state[None, None].to(self.device, dtype=torch.float32),
            anchor.context.to(self.device, dtype=torch.bfloat16),
            seed=seed,
            rtc_processor=rtc_processor,
            inference_delay=inference_delay,
            prev_chunk_left_over=previous_actions,
            execution_horizon=execution_horizon,
            use_cuda_graph=False,
        )
        return _finite_chunk(actions[0], self.name)


def load_cached_anchors(manifest_path: Path, episode: int, count: int) -> tuple[list[Anchor], int, dict]:
    manifest = json.loads(manifest_path.read_text())
    entries = [entry for entry in manifest["entries"] if int(entry["episode_index"]) == episode]
    if len(entries) < count:
        raise ValueError(f"only {len(entries)} cache entries for episode {episode}; requested {count}")
    entries = entries[:count]
    frames = [int(entry["frame_index"]) for entry in entries]
    strides = {right - left for left, right in zip(frames, frames[1:], strict=False)}
    if len(strides) != 1:
        raise ValueError(f"selected anchors have non-constant strides: {sorted(strides)}")
    anchors = []
    for entry in entries:
        tensors = load_file(str(manifest_path.parent / entry["safetensors"]), device="cpu")
        anchors.append(
            Anchor(
                sample_id=str(entry["sample_id"]),
                episode_index=int(entry["episode_index"]),
                frame_index=int(entry["frame_index"]),
                state=tensors["state"].reshape(-1, len(JOINT_NAMES))[-1].float(),
                target_actions=tensors["target_action"][0].float(),
                action_is_pad=tensors["action_is_pad"][0].bool(),
                context=tensors["context"],
            )
        )
    provenance = {
        "manifest": str(manifest_path.resolve()),
        "subset": manifest.get("subset"),
        "provenance": manifest.get("provenance"),
    }
    return anchors, strides.pop(), provenance


def load_smolvla_anchors(dataset_root: Path, episode: int, count: int, stride: int) -> list[Anchor]:
    stream = EpisodeStream(dataset_root, episode)
    first = 4
    wanted = {first + index * stride for index in range(count)}
    anchors = []
    for local_index in range(len(stream)):
        observation = stream.read(local_index)
        if observation is None or local_index not in wanted:
            continue
        anchors.append(
            Anchor(
                sample_id=f"episode-{episode:04d}-frame-{observation.absolute_frame_index:06d}",
                episode_index=episode,
                frame_index=observation.absolute_frame_index,
                state=observation.state,
                target_actions=observation.target_actions,
                action_is_pad=observation.action_is_pad,
                observation=observation,
            )
        )
        if len(anchors) == count:
            break
    if len(anchors) != count:
        raise ValueError(f"episode yielded {len(anchors)} anchors; requested {count}")
    return anchors


def action_delta(left: Tensor, right: Tensor) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 2 or left.shape[-1] != len(JOINT_NAMES):
        raise ValueError(f"expected equal [T,6] actions, got {left.shape} and {right.shape}")
    delta = left.float() - right.float()
    return {
        "aggregate_rmse": float(delta.square().mean().sqrt()),
        "per_joint_rmse": delta.square().mean(dim=0).sqrt().tolist(),
        "max_abs": float(delta.abs().max()),
    }


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": float(tensor.quantile(0.5)),
        "p90": float(tensor.quantile(0.9)),
        "max": max(values),
    }


def aggregate_deltas(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot aggregate an empty metric list")
    aggregate = [float(item["aggregate_rmse"]) for item in items]
    return {
        "aggregate_rmse_mean": statistics.fmean(aggregate),
        "aggregate_rmse_distribution": distribution(aggregate),
        "per_joint_rmse_mean": [
            statistics.fmean(item["per_joint_rmse"][joint] for item in items)
            for joint in range(len(JOINT_NAMES))
        ],
        "per_joint_rmse_distribution": [
            distribution([float(item["per_joint_rmse"][joint]) for item in items])
            for joint in range(len(JOINT_NAMES))
        ],
        "max_abs_max": max(item["max_abs"] for item in items),
    }


def draw_spread(draws: Tensor) -> dict[str, Any]:
    std = draws.float().std(dim=0, unbiased=False)
    pairwise = []
    pairwise_joint = []
    for left, right in combinations(range(draws.shape[0]), 2):
        delta = draws[left].float() - draws[right].float()
        pairwise.append(float(delta.square().mean().sqrt()))
        pairwise_joint.append(delta.square().mean(dim=0).sqrt())
    return {
        "std_per_timestep_per_joint": std.tolist(),
        "std_mean": float(std.mean()),
        "std_per_joint_mean": std.mean(dim=0).tolist(),
        "std_by_horizon_mean_over_joints": std.mean(dim=1).tolist(),
        "std_first_action_mean_over_joints": float(std[0].mean()),
        "std_step30_mean_over_joints": float(std[-1].mean()),
        "mean_pairwise_rmse": statistics.fmean(pairwise),
        "mean_pairwise_rmse_per_joint": torch.stack(pairwise_joint).mean(dim=0).tolist(),
    }


def context_delta(left: Tensor, right: Tensor) -> dict[str, float]:
    left_f = left.float().reshape(-1)
    right_f = right.float().reshape(-1)
    rms = float((left_f - right_f).square().mean().sqrt())
    reference = float(((left_f.square().mean() + right_f.square().mean()) / 2).sqrt())
    return {
        "rms_delta": rms,
        "relative_rms_delta": rms / max(reference, 1e-12),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(left_f, right_f, dim=0)),
    }


def smolvla_rtc_predict(
    backend: SmolVLARolloutBackend,
    anchor: Anchor,
    seed: int,
    checkpoint: Path,
    previous_actions: Tensor,
    inference_delay: int,
    execution_horizon: int,
) -> Tensor:
    if anchor.observation is None:
        raise ValueError("SmolVLA anchor has no observation")
    wrapped: SmolVLABackend = backend.backend
    policy = wrapped.policy
    policy.config.rtc_config = RTCConfig(execution_horizon=execution_horizon)
    policy.init_rtc_processor()
    observation = anchor.observation
    raw = {
        "observation.images.front": observation.rgb_history[0, :, -1].float().div(255),
        "observation.state": observation.state,
        "task": observation.task,
    }
    processed = wrapped.preprocessor(raw)
    device, dtype = wrapped._device_and_dtype()
    processed = {
        key: value.to(device) if isinstance(value, Tensor) else value for key, value in processed.items()
    }
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    stats = load_file(
        str(checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"), device="cpu"
    )
    previous_model = (previous_actions.float() - stats["action.mean"].float()) / stats[
        "action.std"
    ].float().clamp_min(1e-8)
    action = policy.predict_action_chunk(
        processed,
        noise=noise,
        inference_delay=inference_delay,
        prev_chunk_left_over=previous_model.to(device),
        execution_horizon=execution_horizon,
    )
    action = wrapped.postprocessor(action).cpu()
    return _finite_chunk(action[0, :HORIZON, : len(JOINT_NAMES)], backend.name)


def load_live_expert_anchors(
    args: argparse.Namespace, device: torch.device
) -> tuple[list[Anchor], dict[str, Any]]:
    """Extract a few exact held-out contexts once, then release the backbone."""
    if args.policy == "cosmos-smolexpert":
        live = CosmosSmolExpertBackend(
            args.cosmos_run,
            device,
            args.cosmos_checkpoint,
            args.cosmos_tokenizer,
            args.cosmos_prompt,
        )
    else:
        live = LTXSmolExpertBackend(
            args.ltx_run,
            device,
            args.ltx_transformer,
            args.ltx_vae,
            args.ltx_prompt,
        )
    stream = EpisodeStream(args.dataset_root, args.episode)
    first = 4
    wanted = {first + index * args.analysis_stride for index in range(args.anchors)}
    anchors: list[Anchor] = []
    try:
        for local_index in range(len(stream)):
            observation = stream.read(local_index)
            if observation is None or local_index not in wanted:
                continue
            anchors.append(
                Anchor(
                    sample_id=f"episode-{args.episode:04d}-frame-{observation.absolute_frame_index:06d}",
                    episode_index=args.episode,
                    frame_index=observation.absolute_frame_index,
                    state=observation.state,
                    target_actions=observation.target_actions,
                    action_is_pad=observation.action_is_pad,
                    context=live.extract_context(observation).cpu(),
                )
            )
            if len(anchors) == args.anchors:
                break
    finally:
        live.close()
    if len(anchors) != args.anchors:
        raise ValueError(f"episode yielded {len(anchors)} live anchors; requested {args.anchors}")
    return anchors, {
        "mode": "live_in_memory_exact_contexts",
        "backbone_forwards": len(anchors),
        "artifacts_written": False,
    }


def make_backend(
    args: argparse.Namespace, device: torch.device
) -> tuple[list[Anchor], int, dict | None, Any, Callable[[Anchor, int], Tensor], Callable[..., Tensor]]:
    if args.policy == "smolvla":
        anchors = load_smolvla_anchors(args.dataset_root, args.episode, args.anchors, args.analysis_stride)
        backend = SmolVLARolloutBackend(args.smolvla_checkpoint, _train_stats(args.dataset_root), device)

        def predict(anchor: Anchor, seed: int) -> Tensor:
            assert anchor.observation is not None
            return backend.predict_chunk(anchor.observation, seed)

        def predict_rtc(
            anchor: Anchor,
            seed: int,
            previous_actions: Tensor,
            inference_delay: int,
            execution_horizon: int,
        ) -> Tensor:
            return smolvla_rtc_predict(
                backend,
                anchor,
                seed,
                args.smolvla_checkpoint,
                previous_actions,
                inference_delay,
                execution_horizon,
            )

        return anchors, args.analysis_stride, None, backend, predict, predict_rtc

    if args.live_features:
        anchors, provenance = load_live_expert_anchors(args, device)
        stride = args.analysis_stride
    else:
        manifest = args.manifest or DEFAULT_MANIFESTS[args.policy]
        anchors, stride, provenance = load_cached_anchors(manifest, args.episode, args.anchors)
    run_dir = args.cosmos_run if args.policy == "cosmos-smolexpert" else args.ltx_run
    channels = 2048 if args.policy == "cosmos-smolexpert" else 4096
    backend = CachedExpertBackend(run_dir, device, args.policy, channels)
    return anchors, stride, provenance, backend, backend.predict, backend.predict


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.draws < 2 or args.anchors < 2:
        raise ValueError("draws and anchors must both be at least 2")
    device = torch.device(args.device)
    anchors, stride, provenance, backend, predict, predict_rtc = make_backend(args, device)
    started = time.perf_counter()
    try:
        draws_by_anchor = [
            torch.stack([predict(anchor, args.seed + draw) for draw in range(args.draws)])
            for anchor in anchors
        ]
        per_anchor = [
            {
                "sample_id": anchor.sample_id,
                "episode_index": anchor.episode_index,
                "frame_index": anchor.frame_index,
                **draw_spread(draws),
            }
            for anchor, draws in zip(anchors, draws_by_anchor, strict=True)
        ]

        adjacent = []
        for index, (left, right) in enumerate(zip(anchors, anchors[1:], strict=False)):
            left_draws = draws_by_anchor[index]
            right_draws = draws_by_anchor[index + 1]
            left_fixed = left_draws[0]
            right_fixed = right_draws[0]
            item: dict[str, Any] = {
                "from_sample_id": left.sample_id,
                "to_sample_id": right.sample_id,
                "frame_delta": right.frame_index - left.frame_index,
                "seconds_delta": (right.frame_index - left.frame_index) / FPS,
                "fixed_noise_same_horizon_action_delta": action_delta(left_fixed, right_fixed),
                "fixed_noise_time_aligned_action_delta": action_delta(
                    left_fixed[stride:], right_fixed[:-stride]
                ),
                "draw_mean_same_horizon_action_delta": action_delta(
                    left_draws.mean(dim=0), right_draws.mean(dim=0)
                ),
                "draw_mean_time_aligned_action_delta": action_delta(
                    left_draws.mean(dim=0)[stride:], right_draws.mean(dim=0)[:-stride]
                ),
            }
            if left.context is not None and right.context is not None:
                item["context_delta"] = context_delta(left.context, right.context)
                feature_only = predict(
                    Anchor(
                        sample_id=right.sample_id,
                        episode_index=right.episode_index,
                        frame_index=right.frame_index,
                        state=left.state,
                        target_actions=right.target_actions,
                        action_is_pad=right.action_is_pad,
                        context=right.context,
                    ),
                    args.seed,
                )
                state_only = predict(
                    Anchor(
                        sample_id=left.sample_id,
                        episode_index=left.episode_index,
                        frame_index=left.frame_index,
                        state=right.state,
                        target_actions=left.target_actions,
                        action_is_pad=left.action_is_pad,
                        context=left.context,
                    ),
                    args.seed,
                )
                item["feature_only_fixed_noise_action_delta"] = action_delta(left_fixed, feature_only)
                item["state_only_fixed_noise_action_delta"] = action_delta(left_fixed, state_only)
            adjacent.append(item)

        delay = math.ceil(MEASURED_INFERENCE_SECONDS[args.policy] * FPS)
        rtc = {}
        for minimum_horizon in (10, HORIZON):
            execution_horizon = max(delay, minimum_horizon)
            pairs = []
            for index, (left, right) in enumerate(zip(anchors, anchors[1:], strict=False)):
                old_draw_index = index % args.draws
                new_draw_index = (index + 1) % args.draws
                old = draws_by_anchor[index][old_draw_index]
                new = draws_by_anchor[index + 1][new_draw_index]
                old_last = stride + delay - 1
                if old_last >= HORIZON or delay >= HORIZON:
                    continue
                guided = predict_rtc(
                    right,
                    args.seed + new_draw_index,
                    previous_actions=old[stride:],
                    inference_delay=delay,
                    execution_horizon=execution_horizon,
                )
                prefix_count = min(5, HORIZON - delay)
                target = right.target_actions[delay : delay + prefix_count]
                pairs.append(
                    {
                        "from_sample_id": left.sample_id,
                        "to_sample_id": right.sample_id,
                        "old_flow_seed": args.seed + old_draw_index,
                        "new_flow_seed": args.seed + new_draw_index,
                        "old_last_executed_index": old_last,
                        "new_first_executed_index": delay,
                        "naive_unskipped_replacement_seam": action_delta(
                            old[old_last : old_last + 1], new[:1]
                        ),
                        "delay_aligned_baseline_seam": action_delta(
                            old[old_last : old_last + 1], new[delay : delay + 1]
                        ),
                        "rtc_guided_seam": action_delta(
                            old[old_last : old_last + 1], guided[delay : delay + 1]
                        ),
                        "executed_prefix_baseline_accuracy": action_delta(
                            new[delay : delay + prefix_count], target
                        ),
                        "executed_prefix_rtc_accuracy": action_delta(
                            guided[delay : delay + prefix_count], target
                        ),
                    }
                )
            for pair in pairs:
                pair["rtc_seam_change"] = (
                    pair["rtc_guided_seam"]["aggregate_rmse"]
                    - pair["delay_aligned_baseline_seam"]["aggregate_rmse"]
                )
                pair["executed_prefix_accuracy_cost"] = (
                    pair["executed_prefix_rtc_accuracy"]["aggregate_rmse"]
                    - pair["executed_prefix_baseline_accuracy"]["aggregate_rmse"]
                )
            baseline_seams = [item["delay_aligned_baseline_seam"] for item in pairs]
            guided_seams = [item["rtc_guided_seam"] for item in pairs]
            baseline_accuracy = [item["executed_prefix_baseline_accuracy"] for item in pairs]
            guided_accuracy = [item["executed_prefix_rtc_accuracy"] for item in pairs]
            baseline_seam_mean = statistics.fmean(item["aggregate_rmse"] for item in baseline_seams)
            guided_seam_mean = statistics.fmean(item["aggregate_rmse"] for item in guided_seams)
            baseline_accuracy_mean = statistics.fmean(item["aggregate_rmse"] for item in baseline_accuracy)
            guided_accuracy_mean = statistics.fmean(item["aggregate_rmse"] for item in guided_accuracy)
            rtc[f"s_min_{minimum_horizon}"] = {
                "minimum_execution_horizon": minimum_horizon,
                "execution_horizon": execution_horizon,
                "pairs": pairs,
                "delay_aligned_baseline_seam": aggregate_deltas(baseline_seams),
                "rtc_guided_seam": aggregate_deltas(guided_seams),
                "seam_reduction_fraction": (
                    (baseline_seam_mean - guided_seam_mean) / baseline_seam_mean
                    if baseline_seam_mean
                    else 0.0
                ),
                "executed_prefix_baseline_accuracy": aggregate_deltas(baseline_accuracy),
                "executed_prefix_rtc_accuracy": aggregate_deltas(guided_accuracy),
                "executed_prefix_accuracy_cost": guided_accuracy_mean - baseline_accuracy_mean,
                "rtc_seam_change_distribution": distribution(
                    [float(item["rtc_seam_change"]) for item in pairs]
                ),
                "executed_prefix_accuracy_cost_distribution": distribution(
                    [float(item["executed_prefix_accuracy_cost"]) for item in pairs]
                ),
                "worst_rtc_seam_degradation": max(pairs, key=lambda item: item["rtc_seam_change"]),
                "worst_executed_prefix_accuracy_cost": max(
                    pairs, key=lambda item: item["executed_prefix_accuracy_cost"]
                ),
            }

        std = torch.stack([torch.tensor(item["std_per_timestep_per_joint"]) for item in per_anchor])
        pairwise_joint = torch.tensor([item["mean_pairwise_rmse_per_joint"] for item in per_anchor])
        uncertainty = {
            "per_anchor": per_anchor,
            "aggregated": {
                "std_per_timestep_per_joint_mean_over_anchors": std.mean(dim=0).tolist(),
                "std_mean": float(std.mean()),
                "std_per_joint_mean": std.mean(dim=(0, 1)).tolist(),
                "std_by_horizon_mean_over_anchors_and_joints": std.mean(dim=(0, 2)).tolist(),
                "std_first_action_mean": float(std[:, 0].mean()),
                "std_step30_mean": float(std[:, -1].mean()),
                "mean_pairwise_rmse": statistics.fmean(item["mean_pairwise_rmse"] for item in per_anchor),
                "mean_pairwise_rmse_per_joint": pairwise_joint.mean(dim=0).tolist(),
            },
        }
        adjacent_aggregate: dict[str, Any] = {
            key: aggregate_deltas([item[key] for item in adjacent])
            for key in (
                "fixed_noise_same_horizon_action_delta",
                "fixed_noise_time_aligned_action_delta",
                "draw_mean_same_horizon_action_delta",
                "draw_mean_time_aligned_action_delta",
            )
        }
        if anchors[0].context is not None:
            adjacent_aggregate["context_delta"] = {
                key: statistics.fmean(item["context_delta"][key] for item in adjacent)
                for key in ("rms_delta", "relative_rms_delta", "cosine_similarity")
            }
            for key in (
                "feature_only_fixed_noise_action_delta",
                "state_only_fixed_noise_action_delta",
            ):
                adjacent_aggregate[key] = aggregate_deltas([item[key] for item in adjacent])

        result = {
            "schema_version": 1,
            "protocol": {
                "policy": args.policy,
                "metric": "independent action-flow draws with fixed context and state",
                "draws_per_anchor": args.draws,
                "anchor_count": len(anchors),
                "episode": args.episode,
                "episode_split": "train" if args.episode < 32 else "validation",
                "frame_stride": stride,
                "seconds_between_adjacent_anchors": stride / FPS,
                "flow_seeds": [args.seed + draw for draw in range(args.draws)],
                "action_names": list(JOINT_NAMES),
                "action_units": list(JOINT_UNITS),
                "measured_latency_seconds": MEASURED_INFERENCE_SECONDS[args.policy],
                "inference_delay_steps": delay,
                "latency_note": "supplied prior measurement; not remeasured under current contention",
                "cache": provenance,
            },
            "sampling_uncertainty": uncertainty,
            "adjacent_conditioning": {"pairs": adjacent, "aggregated": adjacent_aggregate},
            "rtc": rtc,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        backend.close()

    output = args.output or Path(
        f"/home/anton/.cache/video-vam/dry-runs/{args.policy}-consistency-episode-{args.episode}.json"
    )
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    spread: Any = uncertainty["aggregated"]
    print("TEMPORAL CONSISTENCY ANALYSIS — offline only; no motor commands")
    print(
        f"policy={args.policy} draws={args.draws} anchors={len(anchors)} stride={stride}; "
        f"pairwise={spread['mean_pairwise_rmse']:.3f}; "
        f"std h1/h30={spread['std_first_action_mean']:.3f}/{spread['std_step30_mean']:.3f}"
    )
    print(
        "adjacent fixed-noise time-aligned RMSE="
        f"{adjacent_aggregate['fixed_noise_time_aligned_action_delta']['aggregate_rmse_mean']:.3f}"
    )
    for name, profile in rtc.items():
        print(
            f"RTC {name}: seam {profile['delay_aligned_baseline_seam']['aggregate_rmse_mean']:.3f} -> "
            f"{profile['rtc_guided_seam']['aggregate_rmse_mean']:.3f}; "
            f"prefix RMSE cost={profile['executed_prefix_accuracy_cost']:+.3f}"
        )
    print(f"JSON_SUMMARY={output.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--analysis-stride", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--live-features", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smolvla-checkpoint", type=Path, default=DEFAULT_SMOLVLA)
    parser.add_argument("--cosmos-run", type=Path, default=DEFAULT_COSMOS_RUN)
    parser.add_argument("--cosmos-checkpoint", type=Path, default=DEFAULT_COSMOS_CHECKPOINT)
    parser.add_argument("--cosmos-tokenizer", type=Path, default=DEFAULT_COSMOS_TOKENIZER)
    parser.add_argument("--cosmos-prompt", type=Path, default=DEFAULT_COSMOS_PROMPT)
    parser.add_argument("--ltx-run", type=Path, default=DEFAULT_LTX_RUN)
    parser.add_argument("--ltx-transformer", type=Path, default=DEFAULT_LTX_TRANSFORMER)
    parser.add_argument("--ltx-vae", type=Path, default=DEFAULT_LTX_VAE)
    parser.add_argument("--ltx-prompt", type=Path, default=DEFAULT_LTX_PROMPT)
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
