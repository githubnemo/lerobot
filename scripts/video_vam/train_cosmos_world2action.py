#!/usr/bin/env python3
"""Train the frozen-Cosmos World2Action decoder with an episode-held-out split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from lerobot.common.wandb_utils import get_wandb_run_id_from_filesystem
from lerobot.datasets import LeRobotDataset
from lerobot.policies.vam.context_transform import (
    CONTEXT_TRANSFORMS,
    apply_context_transform,
)
from lerobot.policies.vam.cosmos_cache_dataset import (
    CacheDatasetItem,
    CacheManifest,
    CacheManifestEntry,
    CosmosFeatureCacheDataset,
    load_cache_manifest,
    manifest_sha256,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    VAE_INPUT_MODE_OBSERVED_PREFIX,
    VAE_INPUT_MODES,
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.vam_split import VAMSplit, get_train_entries, get_val_entries, load_vam_split
from lerobot.policies.vam.world2action import ActionStateNormalizer, World2ActionConfig, World2ActionDecoder
from scripts.video_vam.build_cosmos_feature_cache import load_prompt_embedding
from scripts.video_vam.smoke_test_cosmos_extractor import (
    CUBE_OUT_OF_BOX_CONTRACT,
    PreparedSample,
    _relative_index,
    prepare_sample,
)

UPSTREAM_COMMIT = "e3355dbc93132b576c02f920a59b4fc18a4f5906"
EXPECTED_NATIVE_DECODER_PARAMETERS = 499_171_958
DEFAULT_MAX_STEPS = 500_000
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_VAL_EVERY = 1_000
DEFAULT_SAVE_EVERY = 1_000
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 0.0
DEFAULT_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_BETAS = (0.9, 0.99)
DEFAULT_EPS = 1.0e-8
DEFAULT_LOSS_SCALE = 10.0
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_EVAL_SIGMA = 4.0
ONLINE_SIGMA_MEDIAN = 4.0
ONLINE_SIGMA_TAIL_PROBABILITY = 0.05
ONLINE_SIGMA_TAIL_RANGE = (200.0, 100_000.0)
DEFAULT_F_MAX = 1.0
DEFAULT_F_MIN = 0.2
DEFAULT_F_START = 1.0e-6
DEFAULT_CYCLE_LENGTH = 500_000
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WANDB_PROJECT = "video-vam-world2action"
CACHED_CONTEXT_MODES = ("cached", "online")
RANDOM_ANCHOR_MODES = ("online-random", "state-only")
CONTEXT_MODES = CACHED_CONTEXT_MODES + RANDOM_ANCHOR_MODES

_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "checkpoint_kind",
        "status",
        "rollout_readiness",
        "robot_ready",
        "step",
        "best_step",
        "best_val_fixed_flow_loss",
        "manifest",
        "manifest_sha256",
        "split",
        "split_sha256",
        "normalizer",
        "normalizer_metadata",
        "backbone",
        "decoder_config",
        "parameter_count",
        "hyperparameters",
        "optimizer",
        "scheduler",
        "autocast",
        "frozen_context_dtype",
        "action_padding_semantics",
        "provenance",
        "resume_state",
    }
)
# Emitted only by the random-anchor modes, so older checkpoints stay loadable.
_OPTIONAL_METADATA_KEYS = frozenset({"normalizer_source"})


class SafeWandbLogger:
    # Best-effort W&B logger; local JSONL remains authoritative.
    def __init__(
        self,
        *,
        output_dir: Path,
        project: str,
        run_name: str,
        tags: list[str],
        config: dict[str, Any],
        resume: bool,
        disabled: bool,
    ) -> None:
        self.enabled = False
        self.run = None
        self._warned = False
        if disabled:
            self._disable("W&B logging disabled by --no-wandb")
            return
        if os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            self._disable("W&B logging disabled by WANDB_DISABLED")
            return
        try:
            import wandb

            mode = os.environ.get("WANDB_MODE", "online")
            if mode not in {"online", "offline", "disabled"}:
                mode = "online"
            run_id = None
            run_id_path = output_dir / "wandb_run_id.txt"
            if resume:
                if run_id_path.is_file():
                    run_id = run_id_path.read_text().strip() or None
                if run_id is None:
                    try:
                        run_id = get_wandb_run_id_from_filesystem(output_dir)
                    except Exception:
                        run_id = None
            kwargs: dict[str, Any] = {
                "project": project,
                "name": run_name,
                "tags": tags,
                "config": config,
                "dir": str(output_dir),
                "mode": mode,
                "job_type": "train",
            }
            if hasattr(wandb, "Settings"):
                kwargs["settings"] = wandb.Settings(init_timeout=10)
            if run_id is not None:
                kwargs.update(id=run_id, resume="must")
            self.run = wandb.init(**kwargs)
            if self.run is None:
                raise RuntimeError("wandb.init returned no run")
            self.enabled = True
            run_id_path.write_text(str(self.run.id) + "\n")
            print(f"W&B logging enabled: {getattr(self.run, 'url', None) or mode}", flush=True)
        except Exception as exc:
            self._disable(f"W&B unavailable; continuing without remote logging: {exc}")

    def _disable(self, message: str) -> None:
        self.enabled = False
        if not self._warned:
            print(message, file=sys.stderr, flush=True)
            self._warned = True

    def log_metrics(self, record: Mapping[str, Any], step: int) -> None:
        if not self.enabled or self.run is None:
            return
        payload: dict[str, float | int] = {}
        for key, value in record.items():
            if value is None or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                payload[key] = value
            elif key == "val_per_joint_rmse" and isinstance(value, list):
                payload.update({f"val_rmse_joint_{i}": v for i, v in enumerate(value)})
            elif key == "val_horizon_rmse" and isinstance(value, list):
                payload.update({f"val_rmse_horizon_{i:02d}": v for i, v in enumerate(value)})
        try:
            self.run.log(payload, step=step)
        except Exception as exc:
            self._disable(f"W&B metrics upload failed; continuing locally: {exc}")

    def update_summary(self, *, best_metric: float | None, best_step: int) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.summary["best_val_fixed_flow_loss"] = best_metric
            self.run.summary["best_val_step"] = best_step
        except Exception as exc:
            self._disable(f"W&B summary update failed; continuing locally: {exc}")

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                self._disable(f"W&B finalization failed; continuing locally: {exc}")


@dataclass(frozen=True, slots=True)
class EarlyStopping:
    """State machine for minimization-based validation early stopping."""

    best: float = math.inf
    bad_evaluations: int = 0
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA

    def update(self, metric: float) -> tuple[EarlyStopping, bool, bool]:
        """Return updated state, whether it improved, and whether training stops."""
        if not math.isfinite(metric):
            raise ValueError("early-stopping metric must be finite")
        if self.patience <= 0 or self.min_delta < 0:
            raise ValueError("patience must be positive and min_delta non-negative")
        improved = metric < self.best - self.min_delta
        if improved:
            updated = EarlyStopping(metric, 0, self.patience, self.min_delta)
            return updated, True, False
        bad = self.bad_evaluations + 1
        updated = EarlyStopping(self.best, bad, self.patience, self.min_delta)
        return updated, False, bad >= self.patience


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument(
        "--loss-scale",
        type=float,
        default=DEFAULT_LOSS_SCALE,
        help=(
            "Constant multiplying the flow-matching loss before backward. AdamW is scale "
            "invariant, so this only matters through --grad-clip, which it makes that many "
            "times more aggressive."
        ),
    )
    parser.add_argument(
        "--flow-draws",
        type=int,
        default=1,
        help=(
            "Independent (flow-time, noise) pairs drawn per extracted context, averaged into one "
            "gradient. The frozen backbone dominates cost, so K>1 multiplies decoder supervision "
            "per hour almost for free."
        ),
    )
    parser.add_argument(
        "--batched-flow-draws",
        action="store_true",
        help=(
            "Fold the K flow draws into the batch dimension (one forward/backward over K*B "
            "samples) instead of K sequential passes. Identical gradient in expectation; only "
            "use with compact contexts (context transforms) since activations grow K-fold."
        ),
    )
    parser.add_argument(
        "--flow-draw-chunk",
        type=int,
        default=4,
        help="Max draws folded into one batched pass when --batched-flow-draws is set.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backbone-identity", default="cosmos-predict2-2B-hidden-layer-20")
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument(
        "--vae-input-mode",
        choices=VAE_INPUT_MODES,
        default=VAE_INPUT_MODE_OBSERVED_PREFIX,
        help=(
            "VAE input contract for online extraction: observed_prefix encodes only real "
            "observed pixels (default); legacy_padded_vae restores 5->61 padding."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume optimizer, scheduler, RNG, and weights from last.safetensors.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-save-checkpoints",
        action="store_true",
        help="Run diagnostics without writing decoder checkpoint or optimizer-state files.",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--context-control", choices=("real", "shuffled"), default="real")
    parser.add_argument(
        "--context-transform",
        choices=CONTEXT_TRANSFORMS,
        default="none",
        help=(
            "Reduce the cached (B, 19200, 2048) context before the decoder: "
            "cond_frames keeps the 2 conditioning latent frames (2,400 tokens); "
            "gen_frames_pool2 keeps the 14 generated frames 2x2-pooled (4,200); "
            "pool2/pool4 spatially pool all frames (4,800/1,280); frame_mean "
            "keeps one token per latent frame (16); global_mean keeps 1 token. "
            "Applied identically to training and validation batches."
        ),
    )
    parser.add_argument(
        "--context-mode",
        choices=CONTEXT_MODES,
        default="cached",
        help=(
            "cached/online train from the precomputed window manifest; online-random samples a "
            "uniformly random training episode and valid frame per example and extracts its context "
            "on the fly; state-only reuses that sampler but replaces the context with zeros."
        ),
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--online-sigma-seed", type=int, default=0)
    parser.add_argument(
        "--train-sigma",
        type=float,
        help=(
            "Pin training extraction and decoder conditioning to one video sigma instead of "
            "drawing it per sample. Set it to --eval-sigma to train and validate at the same point."
        ),
    )
    parser.add_argument(
        "--eval-sigma",
        type=float,
        default=DEFAULT_EVAL_SIGMA,
        help=(
            "Fixed video sigma used for every validation. Contexts must have been extracted at this "
            "sigma too (see --val-manifest), because the decoder is conditioned on it."
        ),
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        help=(
            "Optional separate cache manifest whose contexts were extracted at --eval-sigma. "
            "Required whenever --eval-sigma differs from the sigma of --manifest."
        ),
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=DEFAULT_WARMUP_STEPS,
        help="Linear warm-up length in optimizer steps; scale down with the total step budget.",
    )
    parser.add_argument(
        "--select-metric",
        choices=("flow", "rmse"),
        default="flow",
        help="Validation metric that selects the best checkpoint and drives early stopping.",
    )
    parser.add_argument("--context-permutation-seed", type=int, default=0)
    parser.add_argument(
        "--wandb-log-horizon-rmse",
        action="store_true",
        help="Log optional per-action-horizon RMSE diagnostics.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("--max-hours must be positive")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("batch-size and grad-accum-steps must be positive")
    if args.val_every <= 0 or args.save_every <= 0 or args.patience <= 0:
        raise ValueError("val-every, save-every, and patience must be positive")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("lr must be positive; weight-decay and grad-clip must be non-negative")
    if args.flow_draws <= 0:
        raise ValueError("--flow-draws must be positive")
    if not math.isfinite(args.loss_scale) or args.loss_scale <= 0:
        raise ValueError("--loss-scale must be finite and positive")
    if args.min_delta < 0 or not math.isfinite(args.min_delta):
        raise ValueError("min-delta must be finite and non-negative")
    if args.device != "cuda":
        raise ValueError("the full native decoder trainer requires --device cuda")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.resume and getattr(args, "no_save_checkpoints", False):
        raise ValueError("--resume cannot be combined with --no-save-checkpoints")
    if not math.isfinite(args.eval_sigma) or args.eval_sigma <= 0:
        raise ValueError("--eval-sigma must be finite and positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("--warmup-steps must be non-negative and smaller than --max-steps")
    if args.context_mode in RANDOM_ANCHOR_MODES:
        if args.dataset_root is None:
            raise ValueError(f"--dataset-root is required for --context-mode {args.context_mode}")
        if args.context_control != "real":
            raise ValueError("random-anchor modes do not support --context-control shuffled")
    if args.context_mode == "online-random" and (args.tokenizer is None or args.prompt is None):
        raise ValueError("--context-mode online-random requires --tokenizer and --prompt")
    if args.train_sigma is not None and (not math.isfinite(args.train_sigma) or args.train_sigma <= 0):
        raise ValueError("--train-sigma must be finite and positive")


def resolve_context_transform(requested: str, manifest: CacheManifest, *, label: str = "manifest") -> str:
    """Resolve runtime reduction while rejecting double-transforming a cache."""
    stored = manifest.context_transform
    if stored != "none" and requested != "none":
        raise ValueError(
            f"{label} already stores context_transform={stored!r}; "
            "pass --context-transform none to avoid applying it twice"
        )
    return "none" if stored != "none" else requested


def should_save_checkpoint(args: argparse.Namespace, step: int, should_validate: bool, stop: bool) -> bool:
    """Return whether this step should write a checkpoint artifact."""
    if getattr(args, "no_save_checkpoints", False):
        return False
    return (
        step % args.save_every == 0 or should_validate or step == args.max_steps or (should_validate and stop)
    )


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is required")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _strict_state_load(module: torch.nn.Module, path: Path) -> None:
    tensors = load_file(str(path), device="cpu")
    expected = module.state_dict()
    if set(tensors) != set(expected):
        raise ValueError(
            f"checkpoint tensor keys mismatch: missing={sorted(set(expected) - set(tensors))}, "
            f"unexpected={sorted(set(tensors) - set(expected))}"
        )
    module.load_state_dict(tensors, strict=True)


def convert_decoder_parameters_to_fp32(decoder: World2ActionDecoder) -> None:
    decoder.denoiser.float()
    for parameter in decoder.denoiser.parameters():
        parameter.requires_grad_(True)
        if parameter.dtype != torch.float32:
            raise TypeError("all trainable decoder parameters must be float32")


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], *, lr: float, weight_decay: float, device: torch.device
):
    """Build upstream fused AdamW, with a recorded fallback for unsupported runtimes."""
    kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": DEFAULT_BETAS,
        "eps": DEFAULT_EPS,
        "capturable": device.type == "cuda",
    }
    settings: dict[str, Any] = {
        "name": "fused AdamW",
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": list(DEFAULT_BETAS),
        "eps": DEFAULT_EPS,
        "capturable": kwargs["capturable"],
        "fused_requested": True,
        "fused": False,
        "fused_fallback": False,
        "master_weights": "upstream_config_only; native torch AdamW stores FP32 parameters",
    }
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if supports_fused:
        try:
            optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
            settings["fused"] = True
            return optimizer, settings
        except (RuntimeError, TypeError) as exc:
            settings["fused_fallback"] = True
            settings["fused_fallback_reason"] = str(exc)
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    settings["fused_fallback"] = True
    return optimizer, settings


def lambda_linear_factor(
    step: int,
    *,
    max_steps: int,
    warm_up_steps: int = DEFAULT_WARMUP_STEPS,
    f_start: float = DEFAULT_F_START,
    f_max: float = DEFAULT_F_MAX,
    f_min: float = DEFAULT_F_MIN,
) -> float:
    """Match mimic-video LambdaLinear while scaling cycle length to max_steps."""
    if step < warm_up_steps:
        return f_start + (f_max - f_start) * step / max(warm_up_steps, 1)
    cycle_length = max(max_steps, 1)
    progress = min(max((step - warm_up_steps) / max(cycle_length - warm_up_steps, 1), 0.0), 1.0)
    return f_max + (f_min - f_max) * progress


def build_scheduler(
    optimizer: torch.optim.Optimizer, max_steps: int, warm_up_steps: int = DEFAULT_WARMUP_STEPS
):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lambda_linear_factor(step, max_steps=max_steps, warm_up_steps=warm_up_steps),
    )
    settings = {
        "name": "lambdalinear",
        "warm_up_steps": warm_up_steps,
        "f_start": DEFAULT_F_START,
        "f_max": DEFAULT_F_MAX,
        "f_min": DEFAULT_F_MIN,
        "cycle_lengths": max_steps,
        "upstream_cycle_length": DEFAULT_CYCLE_LENGTH,
        "cycle_length_deviation": "scaled to configured max_steps",
    }
    return scheduler, settings


def _subset_manifest(manifest: CacheManifest, entries: Sequence[CacheManifestEntry]) -> CacheManifest:
    return CacheManifest(manifest.path, manifest.payload, tuple(entries))


def _load_items(
    manifest: CacheManifest | CosmosFeatureCacheDataset | ContextControlDataset,
    entries: Sequence[CacheManifestEntry],
) -> list[CacheDatasetItem]:
    # Retain the dataset-process verification cache across batches.
    if not entries:
        raise ValueError("cannot load an empty cache batch")
    if isinstance(manifest, (CosmosFeatureCacheDataset, ContextControlDataset)):
        return [manifest.load_entry(entry) for entry in entries]
    dataset = CosmosFeatureCacheDataset(_subset_manifest(manifest, entries), shuffle_seed=0)
    dataset._order = list(range(len(entries)))
    return [dataset[index] for index in range(len(entries))]


def compute_training_normalizer(items: Iterable[Any]) -> ActionStateNormalizer:
    """Compute statistics from train items only and exclude padded action tokens."""
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for item in items:
        states.append(item.state)
        actions.append(item.target_action)
        masks.append(item.action_is_pad)
    if not states:
        raise ValueError("cannot compute a normalizer from an empty train split")
    return ActionStateNormalizer.from_training_tensors(
        torch.cat(states, dim=0),
        torch.cat(actions, dim=0),
        action_is_pad=torch.cat(masks, dim=0),
        source_split="train",
    )


def compute_training_normalizer_from_entries(
    manifest: CacheManifest, entries: Sequence[CacheManifestEntry]
) -> ActionStateNormalizer:
    """Load train entries one at a time so contexts never accumulate in memory."""
    return compute_training_normalizer(_load_items(manifest, [entry])[0] for entry in entries)


def _batch_entries(
    entries: Sequence[CacheManifestEntry], batch_size: int, microbatch_index: int, seed: int
) -> tuple[CacheManifestEntry, ...]:
    if not entries:
        raise ValueError("training split contains no entries")
    batch_count = max(math.ceil(len(entries) / batch_size), 1)
    epoch = microbatch_index // batch_count
    offset = (microbatch_index % batch_count) * batch_size
    order = list(range(len(entries)))
    random.Random(seed + epoch).shuffle(order)
    return tuple(entries[order[(offset + position) % len(order)]] for position in range(batch_size))


@dataclass(frozen=True, slots=True)
class Anchor:
    """One training example location: an episode and a global frame index."""

    episode_index: int
    frame_index: int


def episode_anchor_bounds(
    dataset: LeRobotDataset, episodes: Sequence[int], *, config=CUBE_OUT_OF_BOX_CONTRACT
) -> dict[int, tuple[int, int]]:
    """Return the inclusive global frame range that can anchor a training example.

    A frame is valid when the five-frame causal history lies inside its own episode,
    which is exactly the rule the stride-20 window manifest was built with. The action
    chunk needs no upper bound because the dataset pads it and ``action_is_pad`` marks
    the padded tokens.
    """
    history_frames = -min(config.history_offsets)
    metadata = dataset.meta.episodes
    position_by_episode = {
        int(episode): position for position, episode in enumerate(metadata["episode_index"])
    }
    from_index = metadata["dataset_from_index"]
    to_index = metadata["dataset_to_index"]
    bounds: dict[int, tuple[int, int]] = {}
    for episode in sorted({int(episode) for episode in episodes}):
        position = position_by_episode.get(episode)
        if position is None:
            raise ValueError(f"episode {episode} is absent from the dataset metadata")
        first = int(from_index[position]) + history_frames
        last = int(to_index[position]) - 1
        if last >= first:
            bounds[episode] = (first, last)
    if not bounds:
        raise ValueError("no training episode is long enough to provide a valid anchor")
    return bounds


class RandomAnchorSampler:
    """Draw uniform (episode, valid frame) anchors reproducibly per microbatch.

    Seeding from the microbatch index rather than from a mutable generator keeps the
    anchor stream identical after a mid-run resume.
    """

    def __init__(self, bounds: Mapping[int, tuple[int, int]], *, seed: int) -> None:
        if not bounds:
            raise ValueError("random-anchor sampling requires at least one usable episode")
        self.seed = int(seed)
        self.episodes = tuple(sorted(bounds))
        self.bounds = {episode: tuple(bounds[episode]) for episode in self.episodes}

    @property
    def anchor_count(self) -> int:
        """Return how many distinct anchors the sampler can draw from."""
        return sum(last - first + 1 for first, last in self.bounds.values())

    def batch(self, microbatch_index: int, batch_size: int) -> tuple[Anchor, ...]:
        if microbatch_index < 0 or batch_size <= 0:
            raise ValueError("microbatch index must be non-negative and batch size positive")
        generator = random.Random(f"{self.seed}:{microbatch_index}")
        anchors = []
        for _ in range(batch_size):
            episode = generator.choice(self.episodes)
            first, last = self.bounds[episode]
            anchors.append(Anchor(episode, generator.randint(first, last)))
        return tuple(anchors)

    def drawn_anchors(self, microbatches: int, batch_size: int) -> set[tuple[int, int]]:
        """Replay the anchor stream so resumed runs report an exact unique-anchor count."""
        seen: set[tuple[int, int]] = set()
        for microbatch in range(max(microbatches, 0)):
            seen.update(
                (anchor.episode_index, anchor.frame_index) for anchor in self.batch(microbatch, batch_size)
            )
        return seen


class RandomAnchorDataset:
    """Read camera history, state, action, and padding for an arbitrary anchor."""

    def __init__(self, dataset: LeRobotDataset, *, config=CUBE_OUT_OF_BOX_CONTRACT) -> None:
        self.dataset = dataset
        self.config = config

    def load(self, anchor: Anchor) -> PreparedSample:
        sample = cast(dict[str, Any], self.dataset[_relative_index(self.dataset, anchor.frame_index)])
        episode_index = int(sample["episode_index"])
        if episode_index != anchor.episode_index:
            raise ValueError(
                f"anchor frame {anchor.frame_index} belongs to episode {episode_index}, "
                f"not {anchor.episode_index}"
            )
        return prepare_sample(sample, frame_index=anchor.frame_index, config=self.config)


def build_anchor_dataset(manifest: CacheManifest, args: argparse.Namespace) -> LeRobotDataset:
    """Open the pinned dataset with the contract the extractor and cache both use."""
    dataset_payload = manifest.payload["dataset"]
    return LeRobotDataset(
        dataset_payload["repo_id"],
        root=args.dataset_root.expanduser(),
        episodes=list(range(CUBE_OUT_OF_BOX_CONTRACT.total_episodes)),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=dataset_payload["revision"],
        return_uint8=True,
        download_videos=False,
    )


def compute_training_normalizer_from_anchors(
    anchor_dataset: RandomAnchorDataset, bounds: Mapping[int, tuple[int, int]]
) -> tuple[ActionStateNormalizer, dict[str, Any]]:
    """Derive statistics from every valid training anchor instead of manifest windows."""
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for episode in sorted(bounds):
        first, last = bounds[episode]
        for frame in range(first, last + 1):
            prepared = anchor_dataset.load(Anchor(episode, frame))
            states.append(prepared.state)
            actions.append(prepared.target_action)
            masks.append(prepared.action_is_pad)
    normalizer = ActionStateNormalizer.from_training_tensors(
        torch.cat(states, dim=0),
        torch.cat(actions, dim=0),
        action_is_pad=torch.cat(masks, dim=0),
        source_split="train",
    )
    provenance = {
        "derivation": "all_valid_training_episode_anchors",
        "episodes": sorted(bounds),
        "anchor_count": len(states),
        "padded_actions_excluded": True,
    }
    return normalizer, provenance


class ContextControlDataset:
    # Apply a deterministic true derangement while preserving labels.

    def __init__(self, base: CosmosFeatureCacheDataset, *, seed: int, shuffled: bool) -> None:
        self.base = base
        self.manifest = base.manifest
        self._order = base._order
        self.shuffled = shuffled
        self.seed = seed
        entries = self.manifest.entries
        permutation = list(range(len(entries)))
        if shuffled:
            if len(permutation) < 2:
                raise ValueError("shuffled context control requires at least two cache entries")
            random.Random(seed).shuffle(permutation)
            for _ in range(len(permutation)):
                if all(index != source for index, source in enumerate(permutation)):
                    break
                permutation = permutation[1:] + permutation[:1]
            if any(index == source for index, source in enumerate(permutation)):
                raise RuntimeError("failed to construct a true context derangement")
        self._source_by_sample_id = {
            entry.sample_id: entries[source] for entry, source in zip(entries, permutation, strict=True)
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> CacheDatasetItem:
        return self.load_entry(self.manifest.entries[self.base._order[index]])

    def load_entry(self, entry: CacheManifestEntry) -> CacheDatasetItem:
        target = self.base.load_entry(entry)
        if not self.shuffled:
            return target
        source = self.base.load_entry(self._source_by_sample_id[entry.sample_id])
        artifact = replace(target.artifact, context=source.context)
        return replace(target, artifact=artifact)


class _DeterministicBatchSampler:
    # Yield reproducible manifest-index batches, including after resume.

    def __init__(
        self,
        entries: Sequence[CacheManifestEntry],
        manifest_entries: Sequence[CacheManifestEntry],
        *,
        batch_size: int,
        start_microbatch: int,
        batches: int,
        seed: int,
    ) -> None:
        self.entries = entries
        self.index_by_sample_id = {entry.sample_id: index for index, entry in enumerate(manifest_entries)}
        self.batch_size = batch_size
        self.start_microbatch = start_microbatch
        self.batches = batches
        self.seed = seed

    def __iter__(self):
        for microbatch in range(self.start_microbatch, self.start_microbatch + self.batches):
            entries = _batch_entries(self.entries, self.batch_size, microbatch, self.seed)
            yield [self.index_by_sample_id[entry.sample_id] for entry in entries]

    def __len__(self) -> int:
        return self.batches


def _collate_cache_items(items: Sequence[CacheDatasetItem]) -> dict[str, torch.Tensor]:
    return {
        "state": torch.cat([item.state for item in items], dim=0),
        "action": torch.cat([item.target_action for item in items], dim=0),
        "context": torch.cat([item.context for item in items], dim=0),
        "padding": torch.cat([item.action_is_pad for item in items], dim=0),
    }


def build_training_loader(
    dataset: CosmosFeatureCacheDataset | ContextControlDataset,
    entries: Sequence[CacheManifestEntry],
    *,
    batch_size: int,
    start_microbatch: int,
    batches: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    # Build a deterministic, optionally prefetched loader for training microbatches.
    dataset._order = list(range(len(dataset.manifest.entries)))
    if isinstance(dataset, ContextControlDataset):
        dataset.base._order = dataset._order
    sampler = _DeterministicBatchSampler(
        entries,
        dataset.manifest.entries,
        batch_size=batch_size,
        start_microbatch=start_microbatch,
        batches=batches,
        seed=seed,
    )
    kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": _collate_cache_items,
    }
    if num_workers:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True)
    return DataLoader(dataset, **kwargs)


class OnlineSigmaSampler:
    """Draw upstream's per-sample video sigma: 4*exp(N(0,1)) with a 5% loguniform tail."""

    def __init__(self, device: torch.device, *, seed: int, fixed: float | None = None) -> None:
        self.device = device
        self.seed = int(seed)
        self.fixed = None if fixed is None else float(fixed)
        if self.fixed is not None and (not math.isfinite(self.fixed) or self.fixed <= 0):
            raise ValueError("a fixed training sigma must be finite and positive")
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @torch.no_grad()
    def draw_for(self, microbatch_index: int, batch_size: int) -> torch.Tensor:
        """Reseed from the microbatch index so the sigma stream survives a resume."""
        if microbatch_index < 0:
            raise ValueError("microbatch index must be non-negative")
        if self.fixed is not None:
            return torch.full((batch_size,), self.fixed, device=self.device, dtype=torch.float32)
        self.generator.manual_seed(((self.seed + 1) * 2_654_435_761 + microbatch_index) % (2**63))
        return self.draw(batch_size)

    @torch.no_grad()
    def draw(self, batch_size: int) -> torch.Tensor:
        if self.fixed is not None:
            return torch.full((batch_size,), self.fixed, device=self.device, dtype=torch.float32)
        sigma = ONLINE_SIGMA_MEDIAN * torch.exp(
            torch.randn(batch_size, device=self.device, generator=self.generator)
        )
        tail = torch.rand((batch_size,), device=self.device, generator=self.generator)
        low, high = ONLINE_SIGMA_TAIL_RANGE
        tail_sigma = torch.exp(
            torch.rand((batch_size,), device=self.device, generator=self.generator)
            * (math.log(high) - math.log(low))
            + math.log(low)
        )
        return torch.where(tail < ONLINE_SIGMA_TAIL_PROBABILITY, tail_sigma, sigma)


class ZeroContextProvider:
    """Supply a constant context so only its information content differs from a real run.

    Shapes, dtype, and the per-sample sigma the decoder is conditioned on are drawn
    exactly as in ``online-random``; the Cosmos forward pass is the only thing skipped.
    """

    def __init__(
        self,
        *,
        tokens: int,
        channels: int,
        device: torch.device,
        seed: int,
        fixed_sigma: float | None = None,
    ) -> None:
        if tokens <= 0 or channels <= 0:
            raise ValueError("constant context needs a positive token and channel count")
        self.tokens = tokens
        self.channels = channels
        self.device = device
        self.sigma_sampler = OnlineSigmaSampler(device, seed=seed, fixed=fixed_sigma)

    @torch.no_grad()
    def extract(self, batch_size: int, microbatch_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = self.sigma_sampler.draw_for(microbatch_index, batch_size)
        context = torch.zeros(
            (batch_size, self.tokens, self.channels), device=self.device, dtype=torch.bfloat16
        )
        return context, sigma[:, None]


class OnlineCosmosContext:
    def __init__(
        self,
        manifest: CacheManifest,
        args: argparse.Namespace,
        device: torch.device,
        context_dataset: ContextControlDataset | None,
        dataset: LeRobotDataset | None = None,
    ) -> None:
        self.context_dataset = context_dataset
        self.dataset = dataset if dataset is not None else build_anchor_dataset(manifest, args)
        self.prompt = load_prompt_embedding(args.prompt.expanduser()).embedding
        if args.backbone_checkpoint is None:
            raise ValueError("--backbone-checkpoint is required for online extraction")
        config = CosmosPredict2ExtractorConfig(
            checkpoint_path=args.backbone_checkpoint.expanduser(),
            tokenizer_path=args.tokenizer.expanduser(),
            device=str(device),
            dtype="bfloat16",
            # Only a fallback: every extract() call passes an explicit per-sample sigma.
            high_noise_sigma=ONLINE_SIGMA_MEDIAN,
            seed=args.seed,
            hidden_layer=20,
            stop_after_step=0,
            vae_input_mode=args.vae_input_mode,
        )
        self.extractor = CosmosPredict2Extractor(config)
        self.device = device
        self.sigma_sampler = OnlineSigmaSampler(
            device, seed=args.online_sigma_seed, fixed=getattr(args, "train_sigma", None)
        )
        self.noise_generator = torch.Generator(device=device)

    @torch.no_grad()
    def draw_sigma(self, batch_size: int) -> torch.Tensor:
        return self.sigma_sampler.draw(batch_size)

    @torch.no_grad()
    def extract(
        self, entries: Sequence[CacheManifestEntry], microbatch_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.context_dataset is None:
            raise RuntimeError("manifest-entry extraction requires a cache-backed context dataset")
        samples = []
        context_entries = [
            self.context_dataset._source_by_sample_id[entry.sample_id]
            if self.context_dataset.shuffled
            else entry
            for entry in entries
        ]
        for entry in context_entries:
            sample = self.dataset[_relative_index(self.dataset, entry.frame_index)]
            prepared = prepare_sample(sample, frame_index=entry.frame_index, config=CUBE_OUT_OF_BOX_CONTRACT)
            samples.append(prepared.rgb_history)
        return self.extract_images(torch.cat(samples, dim=0), microbatch_index, resumable_sigma=False)

    @torch.no_grad()
    def extract_images(
        self, images: torch.Tensor, microbatch_index: int, *, resumable_sigma: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(images.shape[0])
        sigma = (
            self.sigma_sampler.draw_for(microbatch_index, batch_size)
            if resumable_sigma
            else self.draw_sigma(batch_size)
        )
        # Every microbatch needs its own latent noise; reusing one seed per
        # optimizer step would repeat the same noise across accumulation.
        noise_seed = microbatch_index + 1_000_003
        self.noise_generator.manual_seed(noise_seed)
        noise = torch.randn(
            (batch_size, 16, 16, 60, 80),
            device=self.device,
            dtype=torch.bfloat16,
            generator=self.noise_generator,
        )
        prompt = self.prompt.expand(batch_size, -1, -1)
        output = self.extractor.extract(images, prompt, noise_seed=noise_seed, sigma=sigma, noise=noise)
        # The returned sigma is the one the features were generated at, and it
        # is what the decoder must be conditioned on.
        return output.tokens, output.sigma[:, None]


# Set once from --context-transform in main(); read by _batch_tensors so both
# the training and validation paths apply the identical reduction.
ACTIVE_CONTEXT_TRANSFORM = "none"


def _batch_tensors(items: Sequence[CacheDatasetItem] | Mapping[str, torch.Tensor], device: torch.device):
    if isinstance(items, Mapping):
        state = items["state"].to(device=device, dtype=torch.float32, non_blocking=True)
        action = items["action"].to(device=device, dtype=torch.float32, non_blocking=True)
        context = items["context"].to(device=device, dtype=torch.bfloat16, non_blocking=True)
        padding = items["padding"].to(device=device, non_blocking=True)
        context = apply_context_transform(context, ACTIVE_CONTEXT_TRANSFORM)
        return state, action, context, padding
    state = torch.cat([item.state for item in items], dim=0).to(device=device, dtype=torch.float32)
    action = torch.cat([item.target_action for item in items], dim=0).to(device=device, dtype=torch.float32)
    context = torch.cat([item.context for item in items], dim=0).to(device=device, dtype=torch.bfloat16)
    padding = torch.cat([item.action_is_pad for item in items], dim=0).to(device=device)
    context = apply_context_transform(context, ACTIVE_CONTEXT_TRANSFORM)
    return state, action, context, padding


def fixed_probe_loss(
    decoder: World2ActionDecoder,
    state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    padding: torch.Tensor,
    tau_a: torch.Tensor,
    epsilon: torch.Tensor,
    eval_sigma: float = DEFAULT_EVAL_SIGMA,
) -> torch.Tensor:
    """Compute one fixed-probe loss without any fresh stochastic draws."""
    decoder.eval()
    with torch.no_grad(), autocast_context(state.device):
        return decoder.flow_matching_loss(
            state,
            action,
            context,
            t=tau_a,
            epsilon=epsilon,
            action_is_pad=padding,
            context_timestep=torch.full((state.shape[0], 1), eval_sigma, device=state.device),
            obs_dropout=0.0,
        )


def _masked_sums(prediction: torch.Tensor, target: torch.Tensor, padding: torch.Tensor):
    valid = (~padding).unsqueeze(-1).expand_as(prediction)
    squared = (prediction.float() - target.float()).square()
    per_joint_sum = (squared * valid).sum(dim=(0, 1))
    per_joint_count = valid.sum(dim=(0, 1)).to(torch.float32)
    return per_joint_sum, per_joint_count


@torch.no_grad()
def evaluate_validation(
    decoder: World2ActionDecoder,
    manifest: CacheManifest | CosmosFeatureCacheDataset | ContextControlDataset,
    entries: Sequence[CacheManifestEntry],
    split: VAMSplit,
    *,
    device: torch.device,
    batch_size: int,
    log_horizon_rmse: bool = False,
    eval_sigma: float = DEFAULT_EVAL_SIGMA,
    zero_context: bool = False,
) -> dict[str, Any]:
    """Evaluate fixed flow inputs and deterministic ten-step physical action errors.

    ``eval_sigma`` must be the sigma the supplied contexts were extracted at;
    the decoder is conditioned on it through ``context_timestep``.
    """
    if not entries:
        raise ValueError("validation split contains no entries")
    decoder.eval()
    flow_total = 0.0
    flow_count = 0.0
    squared_sum = torch.zeros(6, dtype=torch.float64, device=device)
    scalar_count = torch.zeros(6, dtype=torch.float64, device=device)
    horizon_squared_sum = torch.zeros(30, dtype=torch.float64, device=device) if log_horizon_rmse else None
    horizon_count = torch.zeros(30, dtype=torch.float64, device=device) if log_horizon_rmse else None
    for start in range(0, len(entries), batch_size):
        batch_entries = tuple(entries[start : start + batch_size])
        items = _load_items(manifest, batch_entries)
        state, action, context, padding = _batch_tensors(items, device)
        if zero_context:
            context = torch.zeros_like(context)
        probes = [split.probe_for(entry.sample_id) for entry in batch_entries]
        tau = torch.tensor([probe.tau_a for probe in probes], device=device, dtype=torch.float32)
        epsilon = torch.stack([probe.epsilon_tensor() for probe in probes]).to(
            device=device, dtype=torch.float32
        )
        flow_loss = fixed_probe_loss(
            decoder, state, action, context, padding, tau, epsilon, eval_sigma=eval_sigma
        )
        valid_count = float((~padding).sum().item() * action.shape[-1])
        flow_total += float(flow_loss.float().item()) * valid_count
        flow_count += valid_count
        for index, probe in enumerate(probes):
            with autocast_context(device):
                sampled = decoder.sample_actions(
                    state[index : index + 1],
                    context[index : index + 1],
                    seed=probe.sample_seed,
                    context_timestep=torch.full((1, 1), eval_sigma, device=device),
                )
            joint_sum, joint_count = _masked_sums(
                sampled, action[index : index + 1], padding[index : index + 1]
            )
            squared_sum += joint_sum.to(torch.float64)
            scalar_count += joint_count.to(torch.float64)
            if log_horizon_rmse:
                valid = ~padding[index]
                assert horizon_squared_sum is not None and horizon_count is not None
                horizon_squared_sum += (
                    (sampled[0].float() - action[index].float()).square().mean(dim=-1) * valid
                ).to(torch.float64)
                horizon_count += valid.to(torch.float64)
    if flow_count <= 0 or bool((scalar_count <= 0).any().item()):
        raise ValueError("validation padding leaves no valid action positions")
    per_joint_rmse = torch.sqrt(squared_sum / scalar_count).cpu().tolist()
    aggregate_rmse = math.sqrt(float(squared_sum.sum().item() / scalar_count.sum().item()))
    result: dict[str, Any] = {
        "val_fixed_flow_loss": flow_total / flow_count,
        "val_per_joint_rmse": [float(value) for value in per_joint_rmse],
        "val_aggregate_rmse": aggregate_rmse,
        "val_samples": len(entries),
        "val_sigma": float(eval_sigma),
        "val_context_zeroed": bool(zero_context),
    }
    if log_horizon_rmse:
        assert horizon_squared_sum is not None and horizon_count is not None
        result["val_horizon_rmse"] = torch.sqrt(horizon_squared_sum / horizon_count).cpu().tolist()
    return result


def _json_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def _decoder_config_payload(decoder: World2ActionDecoder) -> dict[str, Any]:
    config = decoder.config
    return {
        "state_shape": list(config.state_shape),
        "action_shape": list(config.action_shape),
        "context_channels": config.context_channels,
        "context_layer": config.context_layer,
        "model_channels": config.model_channels,
        "num_blocks": config.num_blocks,
        "num_heads": config.num_heads,
        "attention_backend": config.attention_backend,
        "num_denoising_steps": config.num_denoising_steps,
    }


def _hyperparameters(args: argparse.Namespace) -> dict[str, Any]:
    mode = getattr(args, "context_mode", "cached")
    fixed_sigma = getattr(args, "train_sigma", None)
    online = mode in {"online", "online-random", "state-only"} and fixed_sigma is None
    eval_sigma = float(getattr(args, "eval_sigma", DEFAULT_EVAL_SIGMA))
    payload: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    payload.update(
        {
            "loss_reduce": "mean",
            "loss_scale": float(getattr(args, "loss_scale", DEFAULT_LOSS_SCALE)),
            "flow_draws_per_context": int(getattr(args, "flow_draws", 1)),
            "cross_attention_layer": 20,
            "video_sigma": fixed_sigma if fixed_sigma is not None else (None if online else 10.0),
            "video_sigma_distribution": (
                f"{ONLINE_SIGMA_MEDIAN}*exp(N(0,1)) with probability "
                f"{1.0 - ONLINE_SIGMA_TAIL_PROBABILITY} else loguniform"
                f"[{ONLINE_SIGMA_TAIL_RANGE[0]}, {ONLINE_SIGMA_TAIL_RANGE[1]}]"
                if online
                else "fixed"
            ),
            "video_sigma_per_sample": online,
            "anchor_sampling": (
                "uniform_random_train_episode_then_uniform_random_valid_frame"
                if mode in RANDOM_ANCHOR_MODES
                else "precomputed_stride_window_manifest"
            ),
            "context_information": "constant_zero" if mode == "state-only" else "cosmos_layer20_hidden",
            "evaluation_sigma": eval_sigma,
            "best_selection_metric": getattr(args, "select_metric", "flow"),
            "action_horizon": 30,
            "state_dim": 6,
            "observation_dropout_train": 0.2,
            "observation_dropout_validation": 0.0,
            "observation_dropout_inference": 0.0,
        }
    )
    return payload


def checkpoint_metadata(
    *,
    kind: str,
    step: int,
    best_step: int,
    best_metric: float | None,
    manifest: Path,
    split: Path,
    normalizer: Path,
    decoder: World2ActionDecoder,
    args: argparse.Namespace,
    optimizer_settings: dict[str, Any],
    scheduler_settings: dict[str, Any],
    backbone: dict[str, Any],
    resume_state: Path,
    normalizer_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = decoder.parameter_count_report()
    optional = {} if normalizer_source is None else {"normalizer_source": normalizer_source}
    return {
        **optional,
        "schema_version": 1,
        "artifact": "world2action_training_checkpoint",
        "checkpoint_kind": kind,
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": step,
        "best_step": best_step,
        "best_val_fixed_flow_loss": None if best_metric is None or math.isinf(best_metric) else best_metric,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": manifest_sha256(manifest),
        "split": str(split.resolve()),
        "split_sha256": _json_hash(split),
        "normalizer": str(normalizer.resolve()),
        "normalizer_metadata": str(normalizer.with_suffix(".json").resolve()),
        "backbone": backbone,
        "decoder_config": _decoder_config_payload(decoder),
        "parameter_count": {
            "decoder": report.decoder_parameters,
            "trainable_decoder": report.trainable_decoder_parameters,
            "expected_native": EXPECTED_NATIVE_DECODER_PARAMETERS,
        },
        "hyperparameters": _hyperparameters(args),
        "optimizer": optimizer_settings,
        "scheduler": scheduler_settings,
        "autocast": "cuda_bfloat16",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "action_is_pad=true excluded from train loss, validation loss, and normalization statistics",
        "provenance": {
            "upstream_mimic_video_commit": UPSTREAM_COMMIT,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(0),
            "python_version": platform.python_version(),
        },
        "resume_state": str(resume_state.resolve()),
    }


def validate_checkpoint_metadata(payload: Mapping[str, Any]) -> None:
    keys = set(payload)
    if not keys >= _METADATA_KEYS or not keys <= _METADATA_KEYS | _OPTIONAL_METADATA_KEYS:
        raise ValueError(
            f"checkpoint metadata keys are not strict; missing={sorted(_METADATA_KEYS - keys)}, "
            f"extra={sorted(keys - _METADATA_KEYS - _OPTIONAL_METADATA_KEYS)}"
        )
    if payload["schema_version"] != 1 or payload["artifact"] != "world2action_training_checkpoint":
        raise ValueError("checkpoint metadata schema/artifact is invalid")
    if payload["status"] != "validation_trained_non_rollout" or payload["rollout_readiness"] is not False:
        raise ValueError("training checkpoint must explicitly remain non-rollout-ready")
    if payload["robot_ready"] is not False:
        raise ValueError("training checkpoint robot_ready must be false")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    best_metric = payload["best_val_fixed_flow_loss"]
    if best_metric is not None and (
        not isinstance(best_metric, (int, float)) or not math.isfinite(float(best_metric))
    ):
        raise ValueError("checkpoint best validation metric must be finite or null before first validation")


def save_training_checkpoint(
    decoder: World2ActionDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    *,
    path: Path,
    metadata: dict[str, Any],
    rng_state: dict[str, Any],
    overwrite: bool = True,
) -> None:
    """Save safetensors weights, strict JSON metadata, and resumable optimizer state."""
    validate_checkpoint_metadata(metadata)
    state_path = Path(metadata["resume_state"])
    if not overwrite and any(item.exists() for item in (path, path.with_suffix(".json"), state_path)):
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    tensors = {
        name: value.detach().cpu().contiguous() for name, value in decoder.denoiser.state_dict().items()
    }
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    _atomic_json(path.with_suffix(".json"), metadata)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": rng_state,
        },
        state_path,
    )


def load_resume_checkpoint(
    decoder: World2ActionDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore a last checkpoint and return metadata plus saved RNG state."""
    metadata_path = path.with_suffix(".json")
    state_path = Path(json.loads(metadata_path.read_text())["resume_state"])
    metadata = json.loads(metadata_path.read_text())
    validate_checkpoint_metadata(metadata)
    _strict_state_load(decoder.denoiser, path)
    # Resume state is written by this same script; optimizer/scheduler dicts need full unpickling.
    state = torch.load(state_path, map_location="cpu", weights_only=False)  # nosec B614
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return metadata, state["rng_state"]


def _capture_rng(train_generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator": train_generator.get_state(),
    }


def _restore_rng(payload: Mapping[str, Any], train_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload["cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    train_generator.set_state(payload["train_generator"])


def _backbone_payload(args: argparse.Namespace, first_item: CacheDatasetItem) -> dict[str, Any]:
    mode = getattr(args, "context_mode", "cached")
    # state-only skips the backbone but still randomizes the sigma the decoder sees.
    fixed_train_sigma = getattr(args, "train_sigma", None)
    randomized_sigma = mode in {"online", "online-random", "state-only"} and fixed_train_sigma is None
    weights = first_item.artifact.provenance.payload["weights"]
    checkpoint = (
        args.backbone_checkpoint.resolve() if args.backbone_checkpoint else Path(weights["checkpoint_path"])
    )
    checkpoint_sha256 = _json_hash(checkpoint) if args.backbone_checkpoint else weights["checkpoint_sha256"]
    stored_output = first_item.artifact.provenance.payload["output"]
    stored_transform = stored_output.get("context_transform", "none")
    context = apply_context_transform(first_item.context, ACTIVE_CONTEXT_TRANSFORM)
    return {
        "identity": "constant-zero-context" if mode == "state-only" else args.backbone_identity,
        "context_mode": mode,
        "context_is_constant": mode == "state-only",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "context_layer": 20,
        "context_tokens": int(context.shape[-2]),
        "context_channels": int(context.shape[-1]),
        "context_transform": (stored_transform if stored_transform != "none" else ACTIVE_CONTEXT_TRANSFORM),
        "stored_context_transform": stored_transform,
        "stored_context_grid": stored_output.get("context_grid"),
        "context_adapter_used": False,
        "high_noise_sigma": (
            None
            if randomized_sigma
            else (float(fixed_train_sigma) if fixed_train_sigma is not None else 10.0)
        ),
        "training_sigma_mode": "randomized_per_sample" if randomized_sigma else "fixed",
        "evaluation_sigma": float(getattr(args, "eval_sigma", DEFAULT_EVAL_SIGMA)),
    }


def build_wandb_config(
    args: argparse.Namespace,
    manifest: CacheManifest,
    split: VAMSplit,
    backbone: dict[str, Any],
    optimizer_settings: dict[str, Any],
    scheduler_settings: dict[str, Any],
) -> dict[str, Any]:
    subset = manifest.payload["subset"]
    return {
        "backbone": backbone,
        "dataset": dict(manifest.payload["dataset"]),
        "manifest": {
            "path": str(manifest.path.resolve()),
            "sha256": manifest_sha256(manifest.path),
            "cache_stride": int(subset.get("stride", 1)),
        },
        "split": {
            "name": split.split_name,
            "path": str(split.path.resolve()),
            "sha256": _json_hash(split.path),
            "train_episodes": list(split.train_episodes),
            "val_episodes": list(split.val_episodes),
            "validation_probe_seed": split.probe_seed,
        },
        "optimizer": optimizer_settings,
        "scheduler": scheduler_settings,
        "training": _hyperparameters(args),
        "context_control": {
            "mode": getattr(args, "context_mode", "cached"),
            "shuffle_mode": getattr(args, "context_control", "real"),
            "permutation_seed": getattr(args, "context_permutation_seed", 0),
            "validation_context_shuffled": getattr(args, "context_control", "real") == "shuffled",
        },
    }


def _write_log(stream, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    global ACTIVE_CONTEXT_TRANSFORM
    if args.context_transform != "none" and args.context_mode != "cached":
        raise ValueError("--context-transform requires --context-mode cached")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no model or workload is run on CPU")
    device = torch.device(args.device)
    manifest_path = args.manifest.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_cache_manifest(manifest_path)
    val_manifest: CacheManifest | None = None
    if args.context_mode == "cached" and args.train_sigma is not None:
        cache_sigma = manifest.payload["provenance"].get("high_noise_sigma")
        if cache_sigma is None or abs(float(cache_sigma) - args.train_sigma) > 1e-9:
            raise ValueError(
                f"--train-sigma {args.train_sigma} does not match the manifest extraction "
                f"sigma {cache_sigma!r}; cached contexts must be conditioned at their own sigma"
            )
    val_manifest_path: Path | None = None
    if args.val_manifest is not None:
        # The split and its fixed validation probes are defined against the validation
        # manifest; the training manifest then only has to cover train episodes (it may
        # be a denser, train-only cache built at the same sigma).
        val_manifest_path = args.val_manifest.expanduser().resolve()
        val_manifest = load_cache_manifest(val_manifest_path)
        if val_manifest.context_transform != manifest.context_transform:
            raise ValueError(
                "training and validation manifests must declare the same stored context transform"
            )
        val_cache_sigma = val_manifest.payload["provenance"].get("high_noise_sigma")
        if val_cache_sigma is None or abs(float(val_cache_sigma) - args.eval_sigma) > 1e-9:
            raise ValueError(
                f"--eval-sigma {args.eval_sigma} does not match the --val-manifest extraction "
                f"sigma {val_cache_sigma!r}"
            )
        split = load_vam_split(split_path, val_manifest, allow_partial=True)
        train_dataset_payload = manifest.payload["dataset"]
        if (
            train_dataset_payload["repo_id"] != split.dataset_repo_id
            or train_dataset_payload["revision"] != split.dataset_revision
        ):
            raise ValueError("training manifest dataset identity does not match the split")
        extra_episodes = sorted(
            {entry.episode_index for entry in manifest.entries} - set(split.train_episodes)
        )
        if extra_episodes:
            raise ValueError(f"training manifest contains episodes outside the train split: {extra_episodes}")
        train_entries = tuple(
            entry for entry in manifest.entries if entry.episode_index in split.train_episodes
        )
        # ``val_manifest`` is intentionally a held-out-only partial manifest;
        # load_vam_split(..., allow_partial=True) already validated its probes.
        val_entries = tuple(
            entry for entry in val_manifest.entries if entry.episode_index in split.val_episodes
        )
    else:
        split = load_vam_split(split_path, manifest)
        train_entries = get_train_entries(manifest, split)
        val_entries = get_val_entries(manifest, split)
    ACTIVE_CONTEXT_TRANSFORM = resolve_context_transform(args.context_transform, manifest)
    if not train_entries or not val_entries:
        raise ValueError("both train and validation splits must contain cache entries")
    cache_dataset = CosmosFeatureCacheDataset(manifest, shuffle_seed=0)
    context_dataset = ContextControlDataset(
        cache_dataset,
        seed=args.context_permutation_seed,
        shuffled=args.context_control == "shuffled",
    )
    val_dataset: CosmosFeatureCacheDataset | ContextControlDataset = context_dataset
    if args.val_manifest is not None:
        assert val_manifest is not None
        if not val_entries:
            raise ValueError("--val-manifest contains no validation entries")
        val_dataset = CosmosFeatureCacheDataset(val_manifest, shuffle_seed=0)
    use_random_anchors = args.context_mode in RANDOM_ANCHOR_MODES
    anchor_dataset: RandomAnchorDataset | None = None
    anchor_sampler: RandomAnchorSampler | None = None
    anchor_bounds: dict[int, tuple[int, int]] = {}
    if use_random_anchors:
        lerobot_dataset = build_anchor_dataset(manifest, args)
        anchor_dataset = RandomAnchorDataset(lerobot_dataset)
        anchor_bounds = episode_anchor_bounds(lerobot_dataset, split.train_episodes)
        anchor_sampler = RandomAnchorSampler(anchor_bounds, seed=args.seed)
        print(
            f"random-anchor pool: {anchor_sampler.anchor_count} anchors across "
            f"{len(anchor_bounds)} training episodes",
            flush=True,
        )
    online_context = (
        OnlineCosmosContext(
            manifest,
            args,
            device,
            context_dataset if args.context_mode == "online" else None,
            dataset=anchor_dataset.dataset if anchor_dataset is not None else None,
        )
        if args.context_mode in {"online", "online-random"}
        else None
    )
    zero_context = args.context_mode == "state-only"
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.safetensors"
    if any(output_dir.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output directory; pass --overwrite: {output_dir}")
    set_reproducible_seeds(args.seed)
    normalizer_path = output_dir / "normalizer.safetensors"
    normalizer_source_path = output_dir / "normalizer_source.json"
    if args.resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"--resume requires {last_path}")
        normalizer = ActionStateNormalizer.load(normalizer_path)
        normalizer_source = (
            json.loads(normalizer_source_path.read_text()) if normalizer_source_path.is_file() else None
        )
    elif use_random_anchors:
        assert anchor_dataset is not None
        assert anchor_sampler is not None
        print(
            f"deriving normalization statistics from {anchor_sampler.anchor_count} training anchors",
            flush=True,
        )
        normalizer, normalizer_source = compute_training_normalizer_from_anchors(
            anchor_dataset, anchor_bounds
        )
        normalizer.save(normalizer_path, overwrite=args.overwrite)
        _atomic_json(normalizer_source_path, normalizer_source)
    else:
        train_items = (_load_items(cache_dataset, [entry])[0] for entry in train_entries)
        normalizer = compute_training_normalizer(train_items)
        normalizer.save(normalizer_path, overwrite=args.overwrite)
        normalizer_source = {
            "derivation": "cache_manifest_train_windows",
            "episodes": sorted(split.train_episodes),
            "anchor_count": len(train_entries),
            "padded_actions_excluded": True,
        }
        _atomic_json(normalizer_source_path, normalizer_source)
    decoder = World2ActionDecoder(
        World2ActionConfig(device=str(device), dtype=torch.bfloat16), normalizer=normalizer
    )
    convert_decoder_parameters_to_fp32(decoder)
    report = decoder.parameter_count_report()
    if report.decoder_parameters != EXPECTED_NATIVE_DECODER_PARAMETERS:
        raise RuntimeError(
            f"native decoder parameter count is {report.decoder_parameters}, expected {EXPECTED_NATIVE_DECODER_PARAMETERS}"
        )
    parameters = [parameter for parameter in decoder.denoiser.parameters() if parameter.requires_grad]
    optimizer, optimizer_settings = build_optimizer(
        parameters, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    scheduler, scheduler_settings = build_scheduler(optimizer, args.max_steps, args.warmup_steps)
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(args.seed)
    step = 0
    best_step = 0
    best_metric: float | None = None
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    if args.resume:
        metadata, saved_rng = load_resume_checkpoint(decoder, optimizer, scheduler, last_path)
        if metadata["manifest"] != str(manifest_path) or metadata["manifest_sha256"] != manifest_sha256(
            manifest_path
        ):
            raise ValueError("resume checkpoint manifest provenance does not match requested manifest")
        if metadata["split"] != str(split_path) or metadata["split_sha256"] != _json_hash(split_path):
            raise ValueError("resume checkpoint split provenance does not match requested split")
        step = int(metadata["step"])
        best_step = int(metadata["best_step"])
        best_metric = (
            None
            if metadata["best_val_fixed_flow_loss"] is None
            else float(metadata["best_val_fixed_flow_loss"])
        )
        early_stopping = EarlyStopping(
            math.inf if best_metric is None else best_metric,
            int(metadata["hyperparameters"].get("early_stopping_bad_evaluations", 0)),
            args.patience,
            args.min_delta,
        )
        _restore_rng(saved_rng, train_generator)
    first_item = _load_items(context_dataset, [train_entries[0]])[0]
    backbone = _backbone_payload(args, first_item)
    constant_context = (
        ZeroContextProvider(
            tokens=int(first_item.context.shape[-2]),
            channels=int(first_item.context.shape[-1]),
            device=device,
            seed=args.online_sigma_seed,
            fixed_sigma=args.train_sigma,
        )
        if zero_context
        else None
    )
    seen_anchors: set[tuple[int, int]] = (
        anchor_sampler.drawn_anchors(step * args.grad_accum_steps, args.batch_size)
        if anchor_sampler is not None
        else set()
    )
    wandb_config = build_wandb_config(args, manifest, split, backbone, optimizer_settings, scheduler_settings)
    wandb_run_name = f"{args.backbone_identity}-{manifest_sha256(manifest_path)[:8]}-s{args.seed}"
    wandb_logger = SafeWandbLogger(
        output_dir=output_dir,
        project=args.wandb_project,
        run_name=wandb_run_name,
        tags=[args.backbone_identity[:64], f"split:{split.split_name}"[:64]],
        config=wandb_config,
        resume=args.resume,
        disabled=args.no_wandb,
    )
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Refusing existing metrics file: {metrics_path}")
    metrics = metrics_path.open("a" if args.resume else "w")
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_loader = (
        None
        if online_context is not None or use_random_anchors
        else build_training_loader(
            context_dataset,
            train_entries,
            batch_size=args.batch_size,
            start_microbatch=step * args.grad_accum_steps,
            batches=(args.max_steps - step) * args.grad_accum_steps,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
    )
    train_iterator = iter(train_loader) if train_loader is not None else None
    try:
        while step < args.max_steps:
            if args.max_hours is not None and time.perf_counter() - started >= args.max_hours * 3600:
                break
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            raw_losses: list[float] = []
            samples_this_step = 0
            for _accumulation in range(args.grad_accum_steps):
                microbatch_index = step * args.grad_accum_steps + _accumulation
                if use_random_anchors:
                    assert anchor_sampler is not None and anchor_dataset is not None
                    anchors = anchor_sampler.batch(microbatch_index, args.batch_size)
                    seen_anchors.update((anchor.episode_index, anchor.frame_index) for anchor in anchors)
                    items = [anchor_dataset.load(anchor) for anchor in anchors]
                    state = torch.cat([item.state for item in items], dim=0).to(
                        device=device, dtype=torch.float32
                    )
                    action = torch.cat([item.target_action for item in items], dim=0).to(
                        device=device, dtype=torch.float32
                    )
                    padding = torch.cat([item.action_is_pad for item in items], dim=0).to(device=device)
                    if online_context is not None:
                        context, context_timestep = online_context.extract_images(
                            torch.cat([item.rgb_history for item in items], dim=0), microbatch_index
                        )
                    else:
                        assert constant_context is not None
                        context, context_timestep = constant_context.extract(
                            args.batch_size, microbatch_index
                        )
                elif online_context is None:
                    assert train_iterator is not None
                    items = next(train_iterator)
                    state, action, context, padding = _batch_tensors(items, device)
                    # Cached contexts were extracted at a fixed sigma; condition the decoder
                    # on that value (historical sigma-10 caches predate --train-sigma).
                    cached_sigma = args.train_sigma if args.train_sigma is not None else 10.0
                    context_timestep = torch.full((state.shape[0], 1), cached_sigma, device=device)
                else:
                    batch_entries = _batch_entries(
                        train_entries, args.batch_size, microbatch_index, args.seed
                    )
                    items = _load_items(cache_dataset, batch_entries)
                    state, action, _, padding = _batch_tensors(items, device)
                    context, context_timestep = online_context.extract(batch_entries, microbatch_index)
                samples_this_step += int(state.shape[0])
                draw_losses: list[float] = []
                if args.batched_flow_draws and args.flow_draws > 1:
                    # Fold the K independent flow draws into the batch dimension in
                    # chunks: fewer, larger forward/backward passes instead of K
                    # small sequential ones. Mathematically identical gradient (t
                    # and epsilon are drawn per batch element; padding is replicated
                    # so per-replica valid counts match). Profiling on 2026-08-20
                    # showed the sequential loop is launch-latency bound (~2.0s per
                    # optimizer step regardless of context size). Chunking caps the
                    # K-fold activation growth: all 8 draws at once OOMs a 24GB card
                    # with pool2's 4,800-token contexts.
                    remaining = args.flow_draws
                    while remaining > 0:
                        k = min(remaining, args.flow_draw_chunk)
                        remaining -= k

                        def _rep(t: torch.Tensor, k: int = k) -> torch.Tensor:
                            return t.repeat(k, *([1] * (t.dim() - 1)))

                        with autocast_context(device):
                            loss = decoder.flow_matching_loss(
                                _rep(state),
                                _rep(action),
                                _rep(context),
                                action_is_pad=_rep(padding),
                                context_timestep=_rep(context_timestep),
                                generator=train_generator,
                            )
                            scaled = loss * args.loss_scale * k / (args.grad_accum_steps * args.flow_draws)
                        if not torch.isfinite(scaled).item():
                            raise FloatingPointError("training loss is non-finite")
                        scaled.backward()
                        draw_losses.append(float(loss.detach().float().item()))
                        del loss, scaled
                else:
                    for _draw in range(args.flow_draws):
                        with autocast_context(device):
                            loss = decoder.flow_matching_loss(
                                state,
                                action,
                                context,
                                action_is_pad=padding,
                                context_timestep=context_timestep,
                                generator=train_generator,
                            )
                            scaled = loss * args.loss_scale / (args.grad_accum_steps * args.flow_draws)
                        if not torch.isfinite(scaled).item():
                            raise FloatingPointError("training loss is non-finite")
                        scaled.backward()
                        draw_losses.append(float(loss.detach().float().item()))
                        del loss, scaled
                raw_losses.append(sum(draw_losses) / len(draw_losses))
                del items, state, action, context, padding
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed = time.perf_counter() - step_started
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            record: dict[str, Any] = {
                "step": step,
                "train_loss": float(sum(raw_losses) / len(raw_losses)),
                "val_fixed_flow_loss": None,
                "val_per_joint_rmse": None,
                "val_aggregate_rmse": None,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "wall_clock_seconds": time.perf_counter() - started,
                "samples_per_sec": samples_this_step / max(elapsed, 1e-9),
                "peak_vram_bytes": int(peak),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "grad_clip_factor": min(
                    1.0, args.grad_clip / max(float(torch.as_tensor(grad_norm).item()), 1e-12)
                ),
                "flow_draws": args.flow_draws,
            }
            if anchor_sampler is not None:
                record["unique_anchors_seen"] = len(seen_anchors)
                record["anchor_pool_size"] = anchor_sampler.anchor_count
            anchor_suffix = (
                f" unique_anchors={len(seen_anchors)}/{anchor_sampler.anchor_count}"
                if anchor_sampler is not None
                else ""
            )
            should_validate = step % args.val_every == 0 or step == args.max_steps
            stop = False
            if should_validate:
                validation = evaluate_validation(
                    decoder,
                    val_dataset,
                    val_entries,
                    split,
                    device=device,
                    batch_size=args.batch_size,
                    eval_sigma=args.eval_sigma,
                    zero_context=zero_context,
                )
                record.update(validation)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                candidate = float(
                    validation["val_aggregate_rmse"]
                    if args.select_metric == "rmse"
                    else validation["val_fixed_flow_loss"]
                )
                early_stopping, improved, stop = early_stopping.update(candidate)
                if improved:
                    best_metric = candidate
                    best_step = step
                record["early_stopping_bad_evaluations"] = early_stopping.bad_evaluations
                print(
                    f"step={step} train={record['train_loss']:.6g} "
                    f"val_flow={validation['val_fixed_flow_loss']:.6g} "
                    f"val_rmse={validation['val_aggregate_rmse']:.6g} "
                    f"selected={args.select_metric}:{candidate:.6g} lr={record['learning_rate']:.3g} "
                    f"seconds={elapsed:.3f} samples/s={record['samples_per_sec']:.2f} "
                    f"gnorm={record['grad_norm']:.1f} clipx={record['grad_clip_factor']:.3f} "
                    f"peak_vram={peak / 2**30:.2f}GiB{anchor_suffix}",
                    flush=True,
                )
            else:
                print(
                    f"step={step} train={record['train_loss']:.6g} lr={record['learning_rate']:.3g} "
                    f"seconds={elapsed:.3f} samples/s={record['samples_per_sec']:.2f} "
                    f"gnorm={record['grad_norm']:.1f} clipx={record['grad_clip_factor']:.3f} "
                    f"peak_vram={peak / 2**30:.2f}GiB{anchor_suffix}",
                    flush=True,
                )
            wandb_logger.log_metrics(record, step)
            if should_validate:
                wandb_logger.update_summary(best_metric=best_metric, best_step=best_step)
            _write_log(metrics, record)
            should_save = should_save_checkpoint(args, step, should_validate, stop)
            if should_save:
                state_path = output_dir / "last.state.pt"
                metadata = checkpoint_metadata(
                    kind="last",
                    step=step,
                    best_step=best_step,
                    best_metric=best_metric,
                    manifest=manifest_path,
                    split=split_path,
                    normalizer=normalizer_path,
                    decoder=decoder,
                    args=args,
                    optimizer_settings=optimizer_settings,
                    scheduler_settings=scheduler_settings,
                    backbone=backbone,
                    resume_state=state_path,
                    normalizer_source=normalizer_source,
                )
                metadata["hyperparameters"]["early_stopping_bad_evaluations"] = early_stopping.bad_evaluations
                if anchor_sampler is not None:
                    metadata["hyperparameters"]["anchor_pool_size"] = anchor_sampler.anchor_count
                    metadata["hyperparameters"]["unique_anchors_seen"] = len(seen_anchors)
                save_training_checkpoint(
                    decoder,
                    optimizer,
                    scheduler,
                    path=last_path,
                    metadata=metadata,
                    rng_state=_capture_rng(train_generator),
                    overwrite=True,
                )
                if should_validate and improved:
                    best_path = output_dir / "best.safetensors"
                    best_state_path = output_dir / "best.state.pt"
                    best_metadata = copy.deepcopy(metadata)
                    best_metadata["checkpoint_kind"] = "best"
                    best_metadata["resume_state"] = str(best_state_path.resolve())
                    save_training_checkpoint(
                        decoder,
                        optimizer,
                        scheduler,
                        path=best_path,
                        metadata=best_metadata,
                        rng_state=_capture_rng(train_generator),
                        overwrite=True,
                    )
            if should_validate and stop:
                print(
                    f"early stopping at step={step} after {early_stopping.bad_evaluations} non-improving validations",
                    flush=True,
                )
                break
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        raise RuntimeError(
            f"CUDA OOM; peak allocated VRAM={peak} bytes; peak reserved VRAM={reserved} bytes; "
            "reduce --batch-size"
        ) from exc
    finally:
        metrics.close()
    if getattr(args, "no_save_checkpoints", False):
        print("checkpoints: disabled by --no-save-checkpoints", flush=True)
    else:
        print(f"last checkpoint: {last_path}", flush=True)
        print(f"best checkpoint: {output_dir / 'best.safetensors'}", flush=True)
    print("rollout readiness: false", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
