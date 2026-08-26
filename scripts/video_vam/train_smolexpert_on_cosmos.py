#!/usr/bin/env python3
"""Train SmolVLA's pretrained action expert on validated cached backbone features."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from lerobot.policies.vam.context_transform import apply_context_transform
from lerobot.policies.vam.cosmos_cache_dataset import (
    CacheDatasetItem,
    CacheManifest,
    CacheManifestEntry,
    CosmosFeatureCacheDataset,
    load_cache_manifest,
)
from lerobot.policies.vam.ltx_feature_cache import (
    LTXFeatureCacheDataset,
    load_ltx_cache_manifest,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLVLA_CHECKPOINT,
    SMOLVLM_CONFIG,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)
from lerobot.policies.vam.vam_split import VAMSplit, load_vam_split
from scripts.video_vam.train_cosmos_world2action import build_training_loader

DEFAULT_TRAIN_MANIFEST = Path("/home/anton/.cache/video-vam/cosmos-train0-31-stride3-sigma80/manifest.json")
DEFAULT_VAL_MANIFEST = Path("/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma80/manifest.json")
DEFAULT_SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/runs/smolexpert-cosmos/train")
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_STEPS = 500_000
DEFAULT_VAL_EVERY = 1_000
DEFAULT_PATIENCE = 10
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-10
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_WARMUP_STEPS = 1_000
DEFAULT_NUM_STEPS = 10
DEFAULT_WANDB_PROJECT = "video-vam-world2action"
DEFAULT_RUN_NAME = "smolexpert-on-cosmos-pool2"
TRAIN_EPISODES = tuple(range(32))
CONTEXT_TRANSFORM = "pool2"
CANONICAL_DATASET = "hubnemo/cube_out_of_box_dataset"
CANONICAL_REVISION = "243370c3c08bcbd860133c4a0d658ea7c1d2e77e"
LTX_PROMPT_ARTIFACT = (
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)


@dataclass(frozen=True, slots=True)
class ContextSpec:
    backbone: str
    artifact: str
    stored_transform: str
    model_transform: str
    stored_tokens: int
    model_tokens: int
    channels: int
    dtype: str


def context_spec_from_manifest(manifest: CacheManifest, *, requested_transform: str = "auto") -> ContextSpec:
    """Validate cache provenance and derive the exact decoder input shape."""
    payload = manifest.payload
    if manifest.dataset_repo_id != CANONICAL_DATASET or manifest.dataset_revision != CANONICAL_REVISION:
        raise ValueError(
            "cache manifest dataset provenance must match the canonical cube-out-of-box revision"
        )
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("cache manifest provenance must be an object")
    artifact = str(payload.get("artifact") or "cosmos_feature_manifest")
    if artifact == "ltx25_frozen_feature_manifest":
        if (
            provenance.get("backbone") != "LTX-2.5-22B-distilled"
            or provenance.get("hidden_layer") != 34
            or provenance.get("high_noise_sigma") != 1.0
        ):
            raise ValueError("LTX cache manifest has incompatible backbone provenance")
        prompt = provenance.get("prompt")
        if (
            not isinstance(prompt, dict)
            or prompt.get("artifact_path") != LTX_PROMPT_ARTIFACT
            or prompt.get("embedding_shape") != [1, 1024, 4096]
            or prompt.get("embedding_dtype") != "bfloat16"
            or prompt.get("fallback_allowed") is not False
        ):
            raise ValueError("LTX cache must use the provenance-checked real Gemma-4 prompt artifact")
        stored_transform = str(provenance.get("context_transform"))
        stored_tokens = provenance.get("context_tokens")
        channels = provenance.get("context_channels")
        dtype = provenance.get("context_dtype")
        model_transform = stored_transform if requested_transform == "auto" else requested_transform
        if model_transform != stored_transform:
            raise ValueError(
                "LTX expert training must consume the representation declared by the cache manifest"
            )
        backbone = "LTX-2.5-22B-distilled"
        model_tokens = stored_tokens
    else:
        sigma = provenance.get("high_noise_sigma")
        if sigma is None or abs(float(sigma) - 80.0) > 1e-9:
            raise ValueError(f"Cosmos cache must contain sigma-80 features, got {sigma!r}")
        storage = provenance.get("context_storage")
        match = re.match(r"detached bfloat16 \[B, ([0-9]+), ([0-9]+)\]", str(storage))
        if match is None:
            raise ValueError("Cosmos cache manifest must declare exact bfloat16 context storage shape")
        stored_tokens, channels = (int(value) for value in match.groups())
        dtype = "bfloat16"
        stored_transform = manifest.context_transform
        model_transform = CONTEXT_TRANSFORM if requested_transform == "auto" else requested_transform
        if stored_transform == "none" and model_transform == "pool2" and stored_tokens == 19_200:
            model_tokens = 4_800
        elif stored_transform == model_transform:
            model_tokens = stored_tokens
        else:
            raise ValueError(
                f"unsupported Cosmos context transform {stored_transform!r} -> {model_transform!r}"
            )
        backbone = "Cosmos-Predict2-2B"
    if (
        type(stored_tokens) is not int
        or stored_tokens <= 0
        or type(model_tokens) is not int
        or model_tokens <= 0
        or type(channels) is not int
        or channels <= 0
        or dtype != "bfloat16"
        or stored_transform not in {"none", "pool2"}
        or model_transform not in {"none", "pool2"}
    ):
        raise ValueError("cache context tokens, channels, dtype, or transform are invalid")
    return ContextSpec(
        backbone,
        artifact,
        stored_transform,
        model_transform,
        stored_tokens,
        model_tokens,
        channels,
        dtype,
    )


def load_context_manifest(path: Path) -> CacheManifest:
    payload = json.loads(path.read_text())
    if payload.get("artifact") == "ltx25_frozen_feature_manifest":
        return load_ltx_cache_manifest(path)
    return load_cache_manifest(path)


def build_context_dataset(manifest: CacheManifest):
    if manifest.payload.get("artifact") == "ltx25_frozen_feature_manifest":
        return LTXFeatureCacheDataset(manifest)
    return CosmosFeatureCacheDataset(manifest, shuffle_seed=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--train-episodes",
        type=int,
        nargs="+",
        help="Optional sorted subset of split train episodes used for scaling experiments.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expert-checkpoint", default=SMOLVLA_CHECKPOINT)
    parser.add_argument("--vlm-config", default=SMOLVLM_CONFIG)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--context-transform", choices=("auto", "none", "pool2"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.max_steps <= 0 or args.val_every <= 0 or args.patience <= 0:
        raise ValueError("batch-size, max-steps, val-every, and patience must be positive")
    if args.max_hours is not None and args.max_hours <= 0:
        raise ValueError("max-hours must be positive")
    if args.seed < 0 or args.lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("seed must be non-negative; lr positive; weight-decay and grad-clip non-negative")
    if args.train_episodes is not None and (
        not args.train_episodes
        or any(episode < 0 for episode in args.train_episodes)
        or args.train_episodes != sorted(set(args.train_episodes))
    ):
        raise ValueError("train-episodes must be a non-empty sorted list of unique non-negative integers")
    if not math.isfinite(args.min_delta) or args.min_delta < 0:
        raise ValueError("min-delta must be finite and non-negative")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup-steps must be non-negative and smaller than max-steps")
    if args.num_steps <= 0 or args.num_workers < 0 or args.prefetch_factor <= 0:
        raise ValueError(
            "num-steps must be positive; worker and prefetch counts must be non-negative/positive"
        )
    if args.device != "cuda":
        raise ValueError("the SmolExpert trainer requires --device cuda")


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is required")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def load_items(dataset: Any, entries: tuple[CacheManifestEntry, ...]) -> list[CacheDatasetItem]:
    if not entries:
        raise ValueError("cannot load an empty cache batch")
    return [dataset.load_entry(entry) for entry in entries]


def prepare_cached_context(context: torch.Tensor, spec: ContextSpec) -> torch.Tensor:
    """Return the exact manifest-derived representation expected by the decoder."""
    if tuple(context.shape[1:]) != (spec.stored_tokens, spec.channels):
        raise ValueError(
            f"cached context shape must end in [{spec.stored_tokens}, {spec.channels}], "
            f"got {tuple(context.shape)}"
        )
    if context.dtype != torch.bfloat16:
        raise ValueError(f"cached context dtype must be bfloat16, got {context.dtype}")
    if spec.stored_transform == spec.model_transform:
        transformed = context
    elif spec.stored_transform == "none" and spec.model_transform == "pool2":
        transformed = apply_context_transform(context, "pool2")
    else:
        raise ValueError(
            f"unsupported cached context transform: {spec.stored_transform!r} -> {spec.model_transform!r}"
        )
    if tuple(transformed.shape[1:]) != (spec.model_tokens, spec.channels):
        raise ValueError("transformed context shape does not match manifest-derived decoder input")
    return transformed


def batch_tensors(
    items: list[CacheDatasetItem],
    device: torch.device,
    *,
    context_spec: ContextSpec,
) -> tuple[torch.Tensor, ...]:
    state = torch.cat([item.state for item in items], dim=0).to(device=device, dtype=torch.float32)
    action = torch.cat([item.target_action for item in items], dim=0).to(device=device, dtype=torch.float32)
    context = torch.cat([item.context for item in items], dim=0).to(device=device, dtype=torch.bfloat16)
    padding = torch.cat([item.action_is_pad for item in items], dim=0).to(device=device)
    return state, action, prepare_cached_context(context, context_spec), padding


def compute_training_normalizer(
    dataset: Any,
    entries: tuple[CacheManifestEntry, ...],
    train_episodes: Sequence[int],
) -> tuple[SmolVLANormalizer, dict[str, Any]]:
    """Compute SmolVLA mean/std stats from the selected training episodes."""
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    padding: list[torch.Tensor] = []
    for entry in entries:
        item = dataset.load_entry(entry)
        states.append(item.state)
        actions.append(item.target_action)
        padding.append(item.action_is_pad)
    normalizer = SmolVLANormalizer.from_training_tensors(
        torch.cat(states, dim=0),
        torch.cat(actions, dim=0),
        action_is_pad=torch.cat(padding, dim=0),
    )
    return normalizer, {
        "derivation": "cache_manifest_train_windows",
        "episodes": list(train_episodes),
        "anchor_count": len(entries),
        "padded_actions_excluded": True,
        "normalization": "(value - mean) / (std + 1e-8), population std",
    }


def save_normalizer(normalizer: SmolVLANormalizer, output: Path, source: dict[str, Any]) -> None:
    save_file(normalizer.state_dict(), str(output), metadata={"artifact": "smolvla_mean_std_normalizer"})
    payload = {
        "artifact": "smolvla_mean_std_normalizer",
        "source_split": normalizer.source_split,
        "eps": normalizer.eps,
        "source": source,
        "state_mean": normalizer.state_mean.tolist(),
        "state_std": normalizer.state_std.tolist(),
        "action_mean": normalizer.action_mean.tolist(),
        "action_std": normalizer.action_std.tolist(),
    }
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")


def build_optimizer(model: torch.nn.Module, args: argparse.Namespace, device: torch.device):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "capturable": device.type == "cuda",
    }
    fused = False
    if "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        try:
            optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
            fused = True
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(parameters, **kwargs)
    else:
        optimizer = torch.optim.AdamW(parameters, **kwargs)
    settings = {
        "name": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "fused": fused,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
    }
    return optimizer, settings


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace):
    def factor(step: int) -> float:
        if step < args.warmup_steps:
            return max(step, 1) / args.warmup_steps
        progress = min(
            1.0,
            (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1),
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), {
        "name": "linear_warmup_cosine_decay",
        "warmup_steps": args.warmup_steps,
        "decay_steps": args.max_steps,
        "minimum_factor": 0.1,
    }


@torch.no_grad()
def evaluate_validation(
    decoder: SmolExpertActionDecoder,
    dataset: Any,
    entries: tuple[CacheManifestEntry, ...],
    split: VAMSplit,
    *,
    device: torch.device,
    batch_size: int,
    context_spec: ContextSpec,
) -> dict[str, float | int]:
    if not entries:
        raise ValueError("validation split contains no entries")
    decoder.eval()
    squared_total = 0.0
    valid_scalars = 0
    prefix_squared = [0.0] * 5
    prefix_valid = [0] * 5
    flow_total = 0.0
    flow_count = 0
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start : start + batch_size]
        state, action, context, padding = batch_tensors(
            load_items(dataset, batch_entries), device, context_spec=context_spec
        )
        probes = [split.probe_for(entry.sample_id) for entry in batch_entries]
        noise = torch.cat([decoder._noise_for_seed(1, device, probe.sample_seed) for probe in probes], dim=0)
        prediction = decoder.sample_actions(state, context, noise=noise)
        valid = (~padding).unsqueeze(-1).expand_as(prediction)
        squared_error = (prediction.float() - action.float()).square()
        squared_total += float((squared_error * valid).sum().item())
        valid_scalars += int(valid.sum().item())
        for horizon_index in range(5):
            step_mask = valid[:, horizon_index]
            prefix_squared[horizon_index] += float((squared_error[:, horizon_index] * step_mask).sum().item())
            prefix_valid[horizon_index] += int(step_mask.sum().item())
        tau = torch.tensor([probe.tau_a for probe in probes], device=device, dtype=torch.float32)
        epsilon = torch.stack([probe.epsilon_tensor() for probe in probes]).to(device=device)
        flow_loss = decoder.flow_matching_loss(
            state,
            action,
            context,
            t=tau,
            epsilon=epsilon,
            action_is_pad=padding,
        )
        flow_total += float(flow_loss.float().item()) * max(int(valid.sum().item()), 1)
        flow_count += int(valid.sum().item())
    if valid_scalars <= 0:
        raise ValueError("validation padding leaves no valid scalar actions")
    if any(count <= 0 for count in prefix_valid):
        raise ValueError("validation padding leaves no valid executed-prefix actions")
    prefix_rmse = [
        math.sqrt(squared / count) for squared, count in zip(prefix_squared, prefix_valid, strict=True)
    ]
    return {
        "val_aggregate_rmse_deg": math.sqrt(squared_total / valid_scalars),
        "val_prefix_h1_rmse_deg": prefix_rmse[0],
        "val_prefix_first5_mean_rmse_deg": sum(prefix_rmse) / len(prefix_rmse),
        "val_fixed_flow_loss": flow_total / max(flow_count, 1),
        "val_samples": len(entries),
    }


def checkpoint_metadata(
    decoder: SmolExpertActionDecoder,
    *,
    step: int,
    best_step: int,
    best_rmse: float,
    best_validation: dict[str, float | int],
    args: argparse.Namespace,
    train_manifest: CacheManifest,
    val_manifest: CacheManifest,
    split: VAMSplit,
    train_episodes: Sequence[int],
    normalizer_source: dict[str, Any],
    context_spec: ContextSpec,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "smolexpert_on_cached_context_training_checkpoint",
        "checkpoint_kind": "best",
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": step,
        "best_step": best_step,
        "best_val_aggregate_rmse_deg": best_rmse,
        "best_validation": best_validation,
        "train_manifest": str(train_manifest.path.resolve()),
        "val_manifest": str(val_manifest.path.resolve()),
        "split": str(split.path.resolve()),
        "split_name": split.split_name,
        "train_episodes": list(train_episodes),
        "val_episodes": list(split.val_episodes),
        "normalizer_source": normalizer_source,
        "expert_checkpoint": args.expert_checkpoint,
        "expert_checkpoint_tensor_digests": dict(decoder.loaded_checkpoint_digests),
        "injection": {
            "point": "expert prefix K/V boundary",
            "backbone": context_spec.backbone,
            "stored_context_transform": context_spec.stored_transform,
            "model_context_transform": context_spec.model_transform,
            "input_shape": ["B", context_spec.model_tokens, context_spec.channels],
            "adapter": (f"LayerNorm({context_spec.channels}) -> Linear({context_spec.channels}, 960)"),
            "prefix_kv_projection": "Linear(960, 320) for key and value",
            "vlm_tower_loaded": False,
            "deviation": (
                "Cosmos K/V are shared across expert layers; even layers concatenate them with action self-K/V, "
                "and odd layers pass them through the pretrained expert cross-K/V projections."
            ),
        },
        "action_semantics": {
            "horizon": ACTION_HORIZON,
            "action_dim": ACTION_DIM,
            "internal_action_dim": decoder.max_action_dim,
            "flow_matching": "SmolVLA x_t=(1-t)*action+t*noise; target=noise-action",
            "normalization": "SmolVLA MEAN_STD from split train episodes",
            "euler_steps": decoder.num_steps,
        },
        "hyperparameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "trainable_parameters": sum(
            parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
        ),
        "frozen_parameters": sum(
            parameter.numel() for parameter in decoder.parameters() if not parameter.requires_grad
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unavailable",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
    }


def save_checkpoint(decoder: SmolExpertActionDecoder, path: Path, metadata: dict[str, Any]) -> None:
    tensors = {
        f"model.{name}": value.detach().cpu().contiguous() for name, value in decoder.state_dict().items()
    }
    save_file(tensors, str(path))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no model or workload is run on CPU")
    device = torch.device(args.device)
    set_reproducible_seeds(args.seed)
    train_manifest = load_context_manifest(args.manifest.expanduser().resolve())
    val_manifest = load_context_manifest(args.val_manifest.expanduser().resolve())
    train_context_spec = context_spec_from_manifest(
        train_manifest, requested_transform=args.context_transform
    )
    val_context_spec = context_spec_from_manifest(val_manifest, requested_transform=args.context_transform)
    if train_context_spec != val_context_spec:
        raise ValueError(
            f"train and validation context contracts differ: {train_context_spec!r} != {val_context_spec!r}"
        )
    context_spec = train_context_spec
    split = load_vam_split(args.split.expanduser().resolve(), val_manifest, allow_partial=True)
    train_episodes = (
        tuple(args.train_episodes) if args.train_episodes is not None else tuple(split.train_episodes)
    )
    if not set(train_episodes).issubset(set(split.train_episodes)):
        raise ValueError(
            f"requested train episodes must be a subset of the split train episodes: {train_episodes}"
        )
    available_train_episodes = {entry.episode_index for entry in train_manifest.entries}
    if not set(train_episodes).issubset(available_train_episodes):
        raise ValueError(
            f"split train episodes are absent from the training manifest: "
            f"{sorted(set(train_episodes) - available_train_episodes)}"
        )
    train_entries = tuple(entry for entry in train_manifest.entries if entry.episode_index in train_episodes)
    # The validation cache is intentionally held-out-only; allow_partial above already validated
    # its fixed probes, so do not ask get_val_entries() to require the absent train episodes.
    val_entries = tuple(entry for entry in val_manifest.entries if entry.episode_index in split.val_episodes)
    if not train_entries or not val_entries:
        raise ValueError("both train and validation splits must contain entries")
    train_dataset = build_context_dataset(train_manifest)
    val_dataset = build_context_dataset(val_manifest)
    for label, dataset, entries in (
        ("train", train_dataset, train_entries),
        ("validation", val_dataset, val_entries),
    ):
        sample = dataset.load_entry(entries[0])
        try:
            prepare_cached_context(sample.context, context_spec)
        except ValueError as exc:
            raise ValueError(f"{label} cache sample violates manifest context contract: {exc}") from exc
    normalizer, normalizer_source = compute_training_normalizer(train_dataset, train_entries, train_episodes)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output directory; pass --overwrite: {output_dir}")
    if not args.no_save_checkpoints:
        save_normalizer(normalizer, output_dir / "normalizer.safetensors", normalizer_source)
    print(
        f"injection: expert prefix K/V boundary; adapter LayerNorm+Linear "
        f"{context_spec.channels}->960; VLM tower not loaded; "
        f"backbone={context_spec.backbone}; context={context_spec.model_transform} "
        f"({context_spec.model_tokens} tokens)",
        flush=True,
    )
    decoder = SmolExpertActionDecoder.from_pretrained(
        args.expert_checkpoint,
        normalizer=normalizer,
        vlm_config_name=args.vlm_config,
        device=device,
        num_steps=args.num_steps,
        input_channels=context_spec.channels,
    )
    optimizer, _optimizer_settings = build_optimizer(decoder, args, device)
    scheduler, _scheduler_settings = build_scheduler(optimizer, args)
    loader = build_training_loader(
        train_dataset,
        train_entries,
        batch_size=args.batch_size,
        start_microbatch=0,
        batches=args.max_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    iterator = iter(loader)
    wandb_run = None
    wandb_url: str | None = None
    if not args.no_wandb and os.environ.get("WANDB_DISABLED", "").lower() not in {"1", "true", "yes"}:
        try:
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "manifest": str(train_manifest.path),
                    "val_manifest": str(val_manifest.path),
                    "split": split.split_name,
                    "cache_artifact": context_spec.artifact,
                    "backbone": context_spec.backbone,
                    "stored_context_transform": context_spec.stored_transform,
                    "model_context_transform": context_spec.model_transform,
                    "stored_context_tokens": context_spec.stored_tokens,
                    "model_context_tokens": context_spec.model_tokens,
                    "context_channels": context_spec.channels,
                    "context_dtype": context_spec.dtype,
                    "dataset_repo_id": train_manifest.dataset_repo_id,
                    "dataset_revision": train_manifest.dataset_revision,
                    "train_stride": train_manifest.stride,
                    "val_stride": val_manifest.stride,
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "train_episodes": list(train_episodes),
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
                    ),
                },
            )
            wandb_url = str(wandb_run.url)
        except Exception as exc:
            print(f"W&B unavailable; continuing locally: {exc}", flush=True)
    started = time.perf_counter()
    best_rmse = math.inf
    best_step = 0
    best_validation: dict[str, float | int] = {}
    bad_validations = 0
    final_step = 0
    stop_reason = "max_steps"
    latest = output_dir / "last.safetensors"
    best = output_dir / "best.safetensors"
    try:
        for step in range(1, args.max_steps + 1):
            if args.max_hours is not None and time.perf_counter() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            final_step = step
            decoder.train()
            optimizer.zero_grad(set_to_none=True)
            batch = next(iterator)
            state = batch["state"].to(device=device, dtype=torch.float32)
            action = batch["action"].to(device=device, dtype=torch.float32)
            context = prepare_cached_context(
                batch["context"].to(device=device, dtype=torch.bfloat16), context_spec
            )
            padding = batch["padding"].to(device=device)
            with autocast_context(device):
                loss = decoder.flow_matching_loss(state, action, context, action_is_pad=padding)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("training loss is non-finite")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in decoder.parameters() if parameter.requires_grad], args.grad_clip
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": float(loss.detach().float().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "wall_clock_seconds": time.perf_counter() - started,
            }
            should_validate = step % args.val_every == 0 or step == args.max_steps
            should_stop = False
            if should_validate:
                validation = evaluate_validation(
                    decoder,
                    val_dataset,
                    val_entries,
                    split,
                    device=device,
                    batch_size=args.batch_size,
                    context_spec=context_spec,
                )
                record.update(validation)
                rmse = float(validation["val_aggregate_rmse_deg"])
                improved = rmse < best_rmse - args.min_delta
                if improved:
                    best_rmse = rmse
                    best_step = step
                    best_validation = dict(validation)
                    bad_validations = 0
                else:
                    bad_validations += 1
                record["early_stopping_bad_validations"] = bad_validations
                print(
                    f"step={step} train={record['train_loss']:.6g} "
                    f"val_rmse={rmse:.6g} "
                    f"val_h1={float(validation['val_prefix_h1_rmse_deg']):.6g} "
                    f"val_first5_mean="
                    f"{float(validation['val_prefix_first5_mean_rmse_deg']):.6g} "
                    f"val_flow={float(validation['val_fixed_flow_loss']):.6g} "
                    f"lr={record['learning_rate']:.3g} gnorm={record['grad_norm']:.3g}",
                    flush=True,
                )
                if improved and not args.no_save_checkpoints:
                    metadata = checkpoint_metadata(
                        decoder,
                        step=step,
                        best_step=best_step,
                        best_rmse=best_rmse,
                        best_validation=best_validation,
                        args=args,
                        train_manifest=train_manifest,
                        val_manifest=val_manifest,
                        train_episodes=train_episodes,
                        split=split,
                        normalizer_source=normalizer_source,
                        context_spec=context_spec,
                    )
                    save_checkpoint(decoder, best, metadata)
                if bad_validations >= args.patience:
                    print(f"early stopping at step={step}", flush=True)
                    stop_reason = "validation_plateau"
                    should_stop = True
            elif step == 1 or step % 100 == 0:
                print(
                    f"step={step} train={record['train_loss']:.6g} "
                    f"lr={record['learning_rate']:.3g} gnorm={record['grad_norm']:.3g}",
                    flush=True,
                )
            if wandb_run is not None:
                wandb_run.log(record, step=step)
            if not args.no_save_checkpoints and should_validate:
                metadata = checkpoint_metadata(
                    decoder,
                    step=step,
                    best_step=best_step,
                    best_rmse=best_rmse,
                    best_validation=best_validation,
                    args=args,
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                    train_episodes=train_episodes,
                    split=split,
                    normalizer_source=normalizer_source,
                    context_spec=context_spec,
                )
                metadata["checkpoint_kind"] = "last"
                save_checkpoint(decoder, latest, metadata)
            if should_stop:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    if args.no_save_checkpoints:
        print("checkpoints: disabled by --no-save-checkpoints", flush=True)
    else:
        print(f"best checkpoint: {best}", flush=True)
        print(f"last checkpoint: {latest}", flush=True)
    result = {
        "schema_version": 1,
        "run_name": args.run_name,
        "wandb_url": wandb_url,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "best_step": best_step,
        "best_validation": best_validation,
        "wall_clock_seconds": time.perf_counter() - started,
        "bad_validations_at_stop": bad_validations,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "context": {
            "backbone": context_spec.backbone,
            "transform": context_spec.model_transform,
            "tokens": context_spec.model_tokens,
            "channels": context_spec.channels,
            "train_stride": train_manifest.stride,
            "val_stride": val_manifest.stride,
        },
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)
    print("rollout readiness: false", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
