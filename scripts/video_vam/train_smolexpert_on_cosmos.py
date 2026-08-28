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
from safetensors.torch import load_file, save_file

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
from lerobot.policies.vam.ltx_layer_mix import (
    LTX_LAYER_PROBE_DEPTHS,
    LTXAttentionDiagnosticsAccumulator,
    LTXLayerAttentionMix,
)
from lerobot.policies.vam.ltx_layer_mix_cache import (
    LTXLayerMixCacheDataset,
    load_layer_mix_manifest,
    sha256_file,
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
DEFAULT_ATTN_WIDTH = 2_048
DEFAULT_ATTN_HEADS = 8
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
    tapped_layers: tuple[int, ...] = ()


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
    tapped_layers: tuple[int, ...] = ()
    if artifact == "ltx25_multidepth_pool2_feature_manifest":
        stored_transform = str(provenance.get("context_transform"))
        stored_tokens = provenance.get("context_tokens_per_layer")
        channels = provenance.get("context_channels")
        dtype = provenance.get("context_dtype")
        model_transform = stored_transform if requested_transform == "auto" else requested_transform
        if model_transform != stored_transform:
            raise ValueError("multidepth LTX training must consume the cache-declared representation")
        if provenance.get("tapped_layers") != list(LTX_LAYER_PROBE_DEPTHS):
            raise ValueError("multidepth LTX cache does not contain the canonical tapped layers")
        tapped_layers = LTX_LAYER_PROBE_DEPTHS
        backbone = "LTX-2.5-22B-distilled"
        model_tokens = stored_tokens
    elif artifact == "ltx25_frozen_feature_manifest":
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
        tapped_layers,
    )


def load_context_manifest(path: Path) -> CacheManifest:
    payload = json.loads(path.read_text())
    if str(payload.get("artifact", "")).startswith("ltx25_multidepth_"):
        return load_layer_mix_manifest(path)
    if payload.get("artifact") == "ltx25_frozen_feature_manifest":
        return load_ltx_cache_manifest(path)
    return load_cache_manifest(path)


def build_context_dataset(manifest: CacheManifest):
    if str(manifest.payload.get("artifact", "")).startswith("ltx25_multidepth_"):
        return LTXLayerMixCacheDataset(manifest)
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
    parser.add_argument("--grad-accum-steps", type=int, default=1)
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
    parser.add_argument("--mixer", choices=("none", "attention"), default="none")
    parser.add_argument("--attn-width", type=int, default=DEFAULT_ATTN_WIDTH)
    parser.add_argument("--attn-heads", type=int, default=DEFAULT_ATTN_HEADS)
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
    if (
        args.batch_size <= 0
        or args.grad_accum_steps <= 0
        or args.max_steps <= 0
        or args.val_every <= 0
        or args.patience <= 0
    ):
        raise ValueError("batch-size, grad-accum-steps, max-steps, val-every, and patience must be positive")
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
    if args.attn_width <= 0 or args.attn_heads <= 0 or args.attn_width % args.attn_heads:
        raise ValueError("attn-width must be positive and divisible by attn-heads")
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


def prepare_model_context(
    context: torch.Tensor,
    spec: ContextSpec,
    mixer: LTXLayerAttentionMix | None,
) -> torch.Tensor:
    """Prepare either a single cached context or a jointly trained multidepth mix."""
    if mixer is None:
        if spec.tapped_layers:
            raise ValueError("multidepth cache requires --mixer attention")
        return prepare_cached_context(context, spec)
    if not spec.tapped_layers:
        raise ValueError("--mixer attention requires a multidepth LTX cache")
    expected = (len(spec.tapped_layers), spec.stored_tokens, spec.channels)
    if tuple(context.shape[1:]) != expected or context.dtype != torch.bfloat16:
        raise ValueError(
            f"multidepth context must be BF16 [B, {expected[0]}, {expected[1]}, {expected[2]}], "
            f"got {context.dtype} {tuple(context.shape)}"
        )
    contexts = {layer: context[:, index] for index, layer in enumerate(spec.tapped_layers)}
    return mixer(contexts)


def layer_context_mapping(context: torch.Tensor, spec: ContextSpec) -> dict[int, torch.Tensor]:
    if not spec.tapped_layers:
        raise ValueError("layer mapping requires a multidepth context")
    return {layer: context[:, index] for index, layer in enumerate(spec.tapped_layers)}


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
    return state, action, context, padding


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


def assert_finite_nonzero_gradients(module: torch.nn.Module, label: str) -> None:
    """Fail a smoke run if any trainable tensor is disconnected or numerically invalid."""
    missing: list[str] = []
    nonfinite: list[str] = []
    zero: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all().item():
            nonfinite.append(name)
        elif not torch.count_nonzero(parameter.grad).item():
            zero.append(name)
    if missing or nonfinite or zero:
        raise FloatingPointError(
            f"{label} gradient health failed: missing={missing[:8]}, "
            f"nonfinite={nonfinite[:8]}, zero={zero[:8]}"
        )


@torch.no_grad()
def evaluate_validation(
    decoder: SmolExpertActionDecoder,
    mixer: LTXLayerAttentionMix | None,
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
    if mixer is not None:
        mixer.eval()
    diagnostics = LTXAttentionDiagnosticsAccumulator(mixer) if mixer is not None else None
    squared_total = 0.0
    valid_scalars = 0
    prefix_squared = [0.0] * 5
    prefix_valid = [0] * 5
    flow_total = 0.0
    flow_count = 0
    for start in range(0, len(entries), batch_size):
        batch_entries = entries[start : start + batch_size]
        state, action, raw_context, padding = batch_tensors(
            load_items(dataset, batch_entries), device, context_spec=context_spec
        )
        if diagnostics is not None:
            diagnostics.update(layer_context_mapping(raw_context, context_spec))
        with autocast_context(device):
            context = prepare_model_context(raw_context, context_spec, mixer)
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
    result: dict[str, float | int] = {
        "val_aggregate_rmse_deg": math.sqrt(squared_total / valid_scalars),
        "val_prefix_h1_rmse_deg": prefix_rmse[0],
        "val_prefix_first5_mean_rmse_deg": sum(prefix_rmse) / len(prefix_rmse),
        "val_fixed_flow_loss": flow_total / max(flow_count, 1),
        "val_samples": len(entries),
    }
    if diagnostics is not None:
        mix_diagnostics = diagnostics.compute()
        weights = mix_diagnostics["weights"]
        contribution_norms = mix_diagnostics["mean_contribution_norms"]
        if not isinstance(weights, dict) or not isinstance(contribution_norms, dict):
            raise TypeError("layer-mix diagnostics must contain dict weights and norms")
        for layer, weight in weights.items():
            if not isinstance(weight, (int, float)):
                raise TypeError(f"layer weight for {layer} must be numeric")
            result[f"layer_weight_{layer}"] = weight
        for layer, norm in contribution_norms.items():
            if not isinstance(norm, (int, float)):
                raise TypeError(f"layer contribution norm for {layer} must be numeric")
            result[f"layer_contribution_norm_{layer}"] = norm
        dispersion = mix_diagnostics["weight_dispersion"]
        gain = mix_diagnostics["gain"]
        if not isinstance(dispersion, (int, float)) or not isinstance(gain, (int, float)):
            raise TypeError("layer-mix dispersion and gain must be numeric")
        result["layer_weight_dispersion"] = dispersion
        result["layer_mix_gain"] = gain
        result["layer_weight_sum"] = sum(float(value) for value in weights.values())
    return result


def checkpoint_metadata(
    decoder: SmolExpertActionDecoder,
    mixer: LTXLayerAttentionMix | None,
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
        "train_manifest_sha256": sha256_file(train_manifest.path),
        "val_manifest": str(val_manifest.path.resolve()),
        "val_manifest_sha256": sha256_file(val_manifest.path),
        "split": str(split.path.resolve()),
        "split_sha256": sha256_file(split.path),
        "split_name": split.split_name,
        "train_episodes": list(train_episodes),
        "val_episodes": list(split.val_episodes),
        "normalizer_source": normalizer_source,
        "normalizer_sha256": sha256_file(args.output_dir.resolve() / "normalizer.safetensors"),
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
        "attention_mixer": (
            {
                "layers": list(mixer.layers),
                "architecture": (
                    "separate non-affine LayerNorm per layer; broadcast learned query; "
                    "layer-axis multi-head SDPA; projected residual around uniform mean; global learned gain"
                ),
                "attn_width": mixer.attn_width,
                "attn_heads": mixer.num_heads,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in mixer.parameters() if parameter.requires_grad
                ),
            }
            if mixer is not None
            else None
        ),
        "expert_trainable_parameters": sum(
            parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
        ),
        "expert_frozen_parameters": sum(
            parameter.numel() for parameter in decoder.parameters() if not parameter.requires_grad
        ),
        "total_trainable_parameters": sum(
            parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
        )
        + (
            sum(parameter.numel() for parameter in mixer.parameters() if parameter.requires_grad)
            if mixer is not None
            else 0
        ),
        "frozen_backbone": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unavailable",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
    }


def save_checkpoint(
    decoder: SmolExpertActionDecoder,
    path: Path,
    metadata: dict[str, Any],
    mixer: LTXLayerAttentionMix | None = None,
) -> None:
    tensors = {
        f"model.{name}": value.detach().cpu().contiguous() for name, value in decoder.state_dict().items()
    }
    if mixer is not None:
        tensors.update(
            {f"mixer.{name}": value.detach().cpu().contiguous() for name, value in mixer.state_dict().items()}
        )
    save_file(tensors, str(path))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_joint_checkpoint(
    decoder: SmolExpertActionDecoder,
    mixer: LTXLayerAttentionMix,
    path: Path,
) -> None:
    tensors = load_file(str(path), device="cpu")
    decoder_state = {
        key.removeprefix("model."): value for key, value in tensors.items() if key.startswith("model.")
    }
    mixer_state = {
        key.removeprefix("mixer."): value for key, value in tensors.items() if key.startswith("mixer.")
    }
    if len(decoder_state) + len(mixer_state) != len(tensors):
        raise ValueError("joint checkpoint contains unrecognized tensor keys")
    decoder.load_state_dict(decoder_state, strict=True)
    mixer.load_state_dict(mixer_state, strict=True)


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
    if (args.mixer == "attention") != bool(context_spec.tapped_layers):
        raise ValueError(
            "canonical multidepth manifests require --mixer attention, and single-context manifests require --mixer none"
        )
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
    mixer = (
        LTXLayerAttentionMix(
            context_spec.tapped_layers,
            hidden_width=context_spec.channels,
            attn_width=args.attn_width,
            num_heads=args.attn_heads,
            seed=args.seed,
        ).to(device=device, dtype=torch.float32)
        if args.mixer == "attention"
        else None
    )
    for label, dataset, entries in (
        ("train", train_dataset, train_entries),
        ("validation", val_dataset, val_entries),
    ):
        sample = dataset.load_entry(entries[0])
        try:
            with torch.no_grad(), autocast_context(device):
                prepare_model_context(sample.context.to(device=device), context_spec, mixer)
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
    expert_trainable_parameters = sum(
        parameter.numel() for parameter in decoder.parameters() if parameter.requires_grad
    )
    mixer_trainable_parameters = (
        sum(parameter.numel() for parameter in mixer.parameters() if parameter.requires_grad)
        if mixer is not None
        else 0
    )
    total_trainable_parameters = expert_trainable_parameters + mixer_trainable_parameters
    trainable_model = torch.nn.ModuleList([decoder] + ([mixer] if mixer is not None else []))
    optimizer, optimizer_settings = build_optimizer(trainable_model, args, device)
    scheduler, scheduler_settings = build_scheduler(optimizer, args)
    print(
        f"EXPERT_TRAINABLE_PARAMETERS={expert_trainable_parameters} "
        f"MIXER_TRAINABLE_PARAMETERS={mixer_trainable_parameters} "
        f"TOTAL_TRAINABLE_PARAMETERS={total_trainable_parameters}",
        flush=True,
    )
    loader = build_training_loader(
        train_dataset,
        train_entries,
        batch_size=args.batch_size,
        start_microbatch=0,
        batches=args.max_steps * args.grad_accum_steps,
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
                    "train_manifest_sha256": sha256_file(train_manifest.path),
                    "val_manifest_sha256": sha256_file(val_manifest.path),
                    "split_sha256": sha256_file(split.path),
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    },
                    "train_episodes": list(train_episodes),
                    "mixer_layers": list(context_spec.tapped_layers),
                    "mixer_trainable_parameters": mixer_trainable_parameters,
                    "expert_trainable_parameters": expert_trainable_parameters,
                    "total_trainable_parameters": total_trainable_parameters,
                    "optimizer": optimizer_settings,
                    "scheduler": scheduler_settings,
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
    metrics = (output_dir / "metrics.jsonl").open("w")
    try:
        for step in range(1, args.max_steps + 1):
            if args.max_hours is not None and time.perf_counter() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            final_step = step
            decoder.train()
            if mixer is not None:
                mixer.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            for _accumulation in range(args.grad_accum_steps):
                batch = next(iterator)
                state = batch["state"].to(device=device, dtype=torch.float32)
                action = batch["action"].to(device=device, dtype=torch.float32)
                raw_context = batch["context"].to(device=device, dtype=torch.bfloat16)
                padding = batch["padding"].to(device=device)
                with autocast_context(device):
                    context = prepare_model_context(raw_context, context_spec, mixer)
                    loss = decoder.flow_matching_loss(state, action, context, action_is_pad=padding)
                    scaled_loss = loss / args.grad_accum_steps
                if not torch.isfinite(scaled_loss).item():
                    raise FloatingPointError("training loss is non-finite")
                scaled_loss.backward()
                losses.append(float(loss.detach().float().item()))
            if step == 1:
                assert_finite_nonzero_gradients(decoder, "expert")
                if mixer is not None:
                    assert_finite_nonzero_gradients(mixer, "mixer")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in trainable_model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("gradient norm is non-finite")
            expert_grad_norm = torch.linalg.vector_norm(
                torch.stack(
                    [
                        parameter.grad.detach().float().norm()
                        for parameter in decoder.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                )
            )
            mixer_grad_norm = (
                torch.linalg.vector_norm(
                    torch.stack(
                        [
                            parameter.grad.detach().float().norm()
                            for parameter in mixer.parameters()
                            if parameter.requires_grad and parameter.grad is not None
                        ]
                    )
                )
                if mixer is not None
                else torch.zeros((), device=device)
            )
            if (
                not torch.isfinite(expert_grad_norm).item()
                or expert_grad_norm.item() <= 0
                or (
                    mixer is not None
                    and (not torch.isfinite(mixer_grad_norm).item() or mixer_grad_norm.item() <= 0)
                )
            ):
                raise FloatingPointError("expert and mixer gradient norms must be finite and nonzero")
            optimizer.step()
            scheduler.step()
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": sum(losses) / len(losses),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "expert_grad_norm": float(expert_grad_norm.item()),
                "mixer_grad_norm": float(mixer_grad_norm.item()),
                "mixer_trainable_parameters": mixer_trainable_parameters,
                "expert_trainable_parameters": expert_trainable_parameters,
                "total_trainable_parameters": total_trainable_parameters,
                "effective_batch_size": args.batch_size * args.grad_accum_steps,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "wall_clock_seconds": time.perf_counter() - started,
            }
            should_validate = step % args.val_every == 0 or step == args.max_steps
            should_stop = False
            if should_validate:
                validation = evaluate_validation(
                    decoder,
                    mixer,
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
                        mixer,
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
                    save_checkpoint(decoder, best, metadata, mixer)
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
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            if not args.no_save_checkpoints and should_validate:
                metadata = checkpoint_metadata(
                    decoder,
                    mixer,
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
                save_checkpoint(decoder, latest, metadata, mixer)
            if should_stop:
                break
    finally:
        metrics.close()
        if wandb_run is not None:
            wandb_run.finish()
    if not args.no_save_checkpoints:
        for metadata_path in (best.with_suffix(".json"), latest.with_suffix(".json")):
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                metadata.update(
                    {
                        "stop_reason": stop_reason,
                        "final_step": final_step,
                        "best_step": best_step,
                        "best_val_aggregate_rmse_deg": best_rmse,
                        "best_validation": best_validation,
                        "wall_clock_seconds": time.perf_counter() - started,
                        "wandb_url": wandb_url,
                    }
                )
                metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
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
        "mixer": args.mixer,
        "mixer_trainable_parameters": mixer_trainable_parameters,
        "expert_trainable_parameters": expert_trainable_parameters,
        "total_trainable_parameters": total_trainable_parameters,
        "train_manifest_sha256": sha256_file(train_manifest.path),
        "val_manifest_sha256": sha256_file(val_manifest.path),
        "split_sha256": sha256_file(split.path),
        "expert_checkpoint_tensor_digests": dict(decoder.loaded_checkpoint_digests),
        "context": {
            "backbone": context_spec.backbone,
            "transform": context_spec.model_transform,
            "tokens": context_spec.model_tokens,
            "channels": context_spec.channels,
            "train_stride": train_manifest.stride,
            "val_stride": val_manifest.stride,
            "tapped_layers": list(context_spec.tapped_layers),
            "attention_width": mixer.attn_width if mixer is not None else None,
            "attention_heads": mixer.num_heads if mixer is not None else None,
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
