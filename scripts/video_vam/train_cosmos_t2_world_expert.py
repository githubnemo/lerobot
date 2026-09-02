#!/usr/bin/env python3
"""Train a T=2 Cosmos LoRA and a latent WorldExpert together."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import nullcontext, suppress
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets import LeRobotDataset
from lerobot.policies.vam.cosmos_cache_dataset import (
    CacheManifest,
    CacheManifestEntry,
    load_cache_manifest,
)
from lerobot.policies.vam.cosmos_lora import (
    LATENT_FRAMES,
    freeze_base_parameters,
    full_clip_delta_timestamps,
    inject_lora,
    latent_valid_mask,
    load_lora_state_dict,
    lora_parameters,
    merge_lora_file_into_base,
    prepare_full_clip_sample,
    sample_upstream_sigma,
    save_lora_state_dict,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.world_expert import LinearWorldReadout, WorldExpert
from scripts.video_vam.smoke_test_cosmos_extractor import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOKENIZER_PATH,
)

DEFAULT_TRAIN_MANIFEST = Path(
    "/home/anton/.cache/video-vam/cosmos-videolora-train0-31-stride3-statet2-unpooled/manifest.json"
)
DEFAULT_VAL_MANIFEST = Path(
    "/home/anton/.cache/video-vam/cosmos-videolora-val32-39-stride20-statet2-unpooled/manifest.json"
)
DEFAULT_SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
DEFAULT_MERGE_LORA_WEIGHTS = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/paused-step6000/best_lora.safetensors"
)
DEFAULT_VERIFY_MERGE_AGAINST = Path(
    "/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828/fused-step6000.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    f"/home/anton/.cache/video-vam/runs/cosmos-t2-we-{datetime.now().strftime('%Y%m%d')}"
)
DEFAULT_EXPERT_CHECKPOINT = "lerobot/smolvla_base"
DEFAULT_MAX_STEPS = 500_000
DEFAULT_MAX_HOURS = 12.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_VAL_EVERY = 1_000
DEFAULT_SAVE_EVERY = 1_000
DEFAULT_PATIENCE = 10
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_LR = 1.0e-4
DEFAULT_EXPERT_LR = 1.0e-4
DEFAULT_EVAL_SIGMA = 80.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_WANDB_PROJECT = "video-vam-world2action"


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.best = math.inf
        self.bad_evaluations = 0
        self.patience = patience
        self.min_delta = min_delta

    def update(self, metric: float) -> tuple[bool, bool]:
        if not math.isfinite(metric):
            raise ValueError("validation world loss must be finite")
        if metric < self.best - self.min_delta:
            self.best = metric
            self.bad_evaluations = 0
            return True, False
        self.bad_evaluations += 1
        return False, self.bad_evaluations >= self.patience


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--merge-lora-weights", type=Path, default=DEFAULT_MERGE_LORA_WEIGHTS)
    parser.add_argument("--verify-merge-against", type=Path)
    parser.add_argument("--expert-checkpoint", default=DEFAULT_EXPERT_CHECKPOINT)
    parser.add_argument("--head", choices=("expert", "linear"), default="expert")
    parser.add_argument("--vlm-config", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-cache-dir", type=Path)
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(32)))
    parser.add_argument("--val-episodes", type=int, nargs="+", default=list(range(32, 40)))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-lr", type=float, default=DEFAULT_LORA_LR)
    parser.add_argument("--expert-lr", type=float, default=DEFAULT_EXPERT_LR)
    parser.add_argument("--latent-patch-size", type=int, default=2)
    parser.add_argument("--eval-sigma", type=float, default=DEFAULT_EVAL_SIGMA)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=f"cosmos-t2-we-{datetime.now().strftime('%Y%m%d')}")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Run at most 30 steps, validate at most 8 anchors."
    )
    parser.add_argument("--smoke-steps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise ValueError("the trainable Cosmos world-expert runtime requires --device cuda")
    if args.max_steps <= 0 or args.max_hours <= 0 or args.batch_size <= 0:
        raise ValueError("max-steps, max-hours, and batch-size must be positive")
    if args.val_every <= 0 or args.save_every <= 0 or args.patience <= 0:
        raise ValueError("val-every, save-every, and patience must be positive")
    if args.seed < 0 or args.lora_rank <= 0 or args.latent_patch_size <= 0:
        raise ValueError("seed, lora rank, and latent patch size must be positive/non-negative")
    for name in ("lora_lr", "expert_lr", "eval_sigma", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.lora_alpha is not None and (not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0):
        raise ValueError("lora-alpha must be finite and positive")
    if args.min_delta < 0 or args.smoke_steps <= 0:
        raise ValueError("min-delta must be non-negative and smoke-steps must be positive")
    for name in ("train_episodes", "val_episodes"):
        episodes = getattr(args, name)
        if not episodes or any(type(value) is not int or value < 0 for value in episodes):
            raise ValueError(f"{name} must contain non-negative integers")
        if len(set(episodes)) != len(episodes):
            raise ValueError(f"{name} must not contain duplicates")


def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is required")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class SafeWandbLogger:
    """Best-effort W&B logging; metrics.jsonl remains authoritative."""

    def __init__(self, *, project: str, run_name: str, config: dict[str, Any], disabled: bool) -> None:
        self.run = None
        if disabled or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            return
        try:
            import wandb

            self.run = wandb.init(project=project, name=run_name, config=config)
            print(f"wandb_url={self.run.url}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"W&B unavailable; continuing locally: {exc}", flush=True)

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class LatentCache:
    """Backbone-independent full-clip VAE latents for repeated anchors."""

    def __init__(self, root: Path, *, overwrite: bool) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.overwrite = overwrite

    def path_for(self, entry: CacheManifestEntry) -> Path:
        return self.root / f"episode-{entry.episode_index:04d}-frame-{entry.frame_index:06d}.safetensors"

    def load(
        self, entry: CacheManifestEntry, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self.path_for(entry)
        if not path.is_file():
            return None
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"latents", "latent_valid"}:
            raise ValueError(f"latent cache keys are malformed: {path}")
        latents, valid = tensors["latents"], tensors["latent_valid"]
        if tuple(latents.shape) != (1, 16, LATENT_FRAMES, 60, 80) or latents.dtype != torch.bfloat16:
            raise ValueError(f"latent cache latents have invalid shape/dtype: {path}")
        if tuple(valid.shape) != (1, LATENT_FRAMES) or valid.dtype != torch.bool:
            raise ValueError(f"latent cache mask has invalid shape/dtype: {path}")
        return latents.to(device=device), valid.to(device=device)

    def save(self, entry: CacheManifestEntry, latents: torch.Tensor, valid: torch.Tensor) -> None:
        path = self.path_for(entry)
        if path.exists() and not self.overwrite:
            return
        if tuple(latents.shape) != (1, 16, LATENT_FRAMES, 60, 80) or latents.dtype != torch.bfloat16:
            raise ValueError("only full Cosmos bfloat16 latents can be cached")
        if tuple(valid.shape) != (1, LATENT_FRAMES) or valid.dtype != torch.bool:
            raise ValueError("only [1, 16] latent validity masks can be cached")
        temporary = path.with_name(f".{path.name}.tmp")
        save_file(
            {
                "latents": latents.detach().cpu().contiguous(),
                "latent_valid": valid.detach().cpu().contiguous(),
            },
            str(temporary),
        )
        os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _build_full_dataset(
    manifest: CacheManifest,
    args: argparse.Namespace,
    episodes: Sequence[int],
) -> LeRobotDataset:
    dataset = manifest.payload["dataset"]
    return LeRobotDataset(
        dataset["repo_id"],
        root=args.dataset_root.expanduser(),
        episodes=sorted({int(value) for value in episodes}),
        delta_timestamps=full_clip_delta_timestamps(),
        revision=dataset["revision"],
        return_uint8=True,
        download_videos=False,
    )


def _relative_index(dataset: LeRobotDataset, absolute_index: int) -> int:
    mapping = dataset.absolute_to_relative_idx
    return absolute_index if mapping is None else int(mapping[absolute_index])


def _load_full_sample(dataset: LeRobotDataset, entry: CacheManifestEntry):
    sample = cast(dict[str, Any], dataset[_relative_index(dataset, entry.frame_index)])
    prepared = prepare_full_clip_sample(sample, frame_index=entry.frame_index)
    if prepared.episode_index != entry.episode_index:
        raise ValueError(f"dataset sample for {entry.sample_id} belongs to episode {prepared.episode_index}")
    return prepared


def _images_to_cosmos(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    rgb = rgb.to(device=device)
    if rgb.dtype == torch.uint8:
        return rgb.to(dtype=torch.bfloat16) / 127.5 - 1.0
    if not torch.is_floating_point(rgb):
        raise TypeError("RGB clips must be uint8 or floating point")
    minimum, maximum = rgb.amin().item(), rgb.amax().item()
    if minimum >= 0.0 and maximum <= 1.0:
        rgb = rgb * 2.0 - 1.0
    elif minimum < -1.0 or maximum > 1.0:
        raise ValueError("floating-point RGB clips must be in [0, 1] or [-1, 1]")
    return rgb.to(dtype=torch.bfloat16)


def _encode_full_clip(
    tokenizer: Any,
    sample: Any,
    entry: CacheManifestEntry,
    cache: LatentCache,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = cache.load(entry, device)
    if cached is not None:
        return cached
    images = _images_to_cosmos(sample.rgb_clip, device)
    with torch.no_grad():
        latents = tokenizer.encode(images)
    expected = (images.shape[0], 16, LATENT_FRAMES, 60, 80)
    if not isinstance(latents, torch.Tensor) or tuple(latents.shape) != expected:
        raise ValueError(
            f"full-clip tokenizer output must have shape {expected}; "
            f"zero-expansion of observed-only output is forbidden, got "
            f"{tuple(latents.shape) if isinstance(latents, torch.Tensor) else type(latents)}"
        )
    latents = latents.to(device=device, dtype=torch.bfloat16)
    valid = latent_valid_mask(sample.camera_is_pad.to(device=device), latent_frames=LATENT_FRAMES)
    cache.save(entry, latents, valid)
    return latents, valid


def _noise_like(shape: Sequence[int], *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (2**63))
    return torch.randn(tuple(shape), device=device, dtype=torch.bfloat16, generator=generator)


def _sigma_for(seed: int, step: int, batch_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed((seed * 2_654_435_761 + step + 1) % (2**63))
    return sample_upstream_sigma(batch_size, device=device, generator=generator)


def _enable_trunk_checkpointing(backbone: torch.nn.Module) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 28:
        raise ValueError("Cosmos backbone must expose exactly 28 blocks")
    for index in range(19):
        block = blocks[index]
        if not getattr(block, "_lerobot_t2_world_checkpoint", False):
            wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
            wrapped._lerobot_t2_world_checkpoint = True
            blocks[index] = wrapped


def _state_mapping(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "net", "net_ema"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                try:
                    return _state_mapping(nested)
                except ValueError:
                    pass
        if all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
            return dict(payload)
    raise ValueError("merge verification checkpoint must contain a tensor state dict")


def _normalise_reference_state(state: dict[str, torch.Tensor], keys: set[str]) -> dict[str, torch.Tensor]:
    if set(state) == keys:
        return state
    if keys.issubset(state):
        return {key: state[key] for key in keys}
    for prefix in ("net_ema.", "net.", "model.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            stripped = {key[len(prefix) :]: value for key, value in state.items()}
            if set(stripped) == keys:
                return stripped
    raise ValueError(
        f"merge verification checkpoint keys differ: missing={sorted(keys - set(state))[:4]}, "
        f"unexpected={sorted(set(state) - keys)[:4]}"
    )


def _verify_merged_backbone(backbone: torch.nn.Module, reference_path: Path) -> float:
    payload = torch.load(reference_path, map_location="cpu", weights_only=True)
    current = {name: parameter.detach().cpu() for name, parameter in backbone.state_dict().items()}
    reference = _normalise_reference_state(_state_mapping(payload), set(current))
    max_abs_diff = 0.0
    for name, value in current.items():
        left = value.to(dtype=torch.bfloat16)
        right = reference[name].to(dtype=torch.bfloat16)
        if left.numel() == 0 and right.numel() == 0:
            continue
        max_abs_diff = max(max_abs_diff, float((left.float() - right.float()).abs().max().item()))
        if not torch.allclose(left, right):
            raise ValueError(f"merged backbone differs from reference at {name}; max_abs_diff={max_abs_diff}")
    print(f"merge_verification={reference_path} max_abs_diff={max_abs_diff:.6g}", flush=True)
    return max_abs_diff


def _world_loss(
    prediction: torch.Tensor,
    clean_future: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
    valid_future: torch.Tensor,
) -> torch.Tensor:
    from lerobot.policies.vam._vendor.cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling

    target = noise - clean_future
    weights = RectifiedFlowScaling(sigma_data=1.0, t_scaling_factor=1.0).sigma_loss_weights(sigma)
    scalar_mask = valid_future[:, None, :, None, None].to(dtype=prediction.dtype)
    squared = (prediction.float() - target.float()).square()
    weighted = squared * scalar_mask * weights.view(-1, 1, 1, 1, 1)
    count = scalar_mask.sum() * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
    if int(count.item()) <= 0:
        return prediction.float().sum() * 0.0
    return weighted.sum() / count


def _linear_loss(
    prediction: torch.Tensor,
    clean_future: torch.Tensor,
    valid_future: torch.Tensor,
) -> torch.Tensor:
    scalar_mask = valid_future[:, None, :, None, None].to(dtype=prediction.dtype)
    squared = (prediction.float() - clean_future.float()).square()
    count = scalar_mask.sum() * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
    if int(count.item()) <= 0:
        return prediction.float().sum() * 0.0
    return (squared * scalar_mask).sum() / count


@torch.no_grad()
def evaluate_world(
    extractor: CosmosPredict2Extractor,
    world_head: WorldExpert | LinearWorldReadout,
    tokenizer: Any,
    dataset: LeRobotDataset,
    entries: Sequence[CacheManifestEntry],
    prompt: torch.Tensor,
    latent_cache: LatentCache,
    *,
    device: torch.device,
    eval_sigma: float,
    head_kind: str = "expert",
) -> dict[str, float | int]:
    if not entries:
        raise ValueError("validation split contains no entries")
    extractor.backbone.eval()
    world_head.eval()
    total = 0.0
    count = 0
    for entry in entries:
        sample = _load_full_sample(dataset, entry)
        full_latents, valid = _encode_full_clip(tokenizer, sample, entry, latent_cache, device)
        clean_future = full_latents[:, :, 2:]
        valid_future = valid[:, 2:]
        observed = sample.rgb_clip[:, :, :5]
        extraction = extractor.forward_features(
            observed,
            prompt.expand(observed.shape[0], -1, -1),
            noise_seed=entry.noise_seed,
            sigma=torch.full((observed.shape[0],), 80.0, device=device),
        )
        if head_kind == "linear":
            with autocast_context(device):
                prediction = world_head(extraction.tokens)
                loss = _linear_loss(prediction, clean_future, valid_future)
        else:
            sigma = torch.full((1,), eval_sigma, device=device, dtype=torch.float32)
            prefix = world_head.prepare_prefix_kv(extraction.tokens)
            noise = _noise_like(clean_future.shape, device=device, seed=entry.noise_seed + 4_000_003)
            x_sigma = clean_future + noise * sigma.to(dtype=clean_future.dtype).view(1, 1, 1, 1, 1)
            with autocast_context(device):
                prediction = world_head.denoise(x_sigma, sigma, prefix)
                loss = _world_loss(prediction, clean_future, noise, sigma, valid_future)
        valid_count = int(
            valid_future.sum().item() * clean_future.shape[1] * clean_future.shape[3] * clean_future.shape[4]
        )
        if valid_count:
            total += float(loss.item()) * valid_count
            count += valid_count
    if count <= 0:
        raise ValueError("validation split contains no non-conditioning latent frames")
    result: dict[str, float | int] = {
        "val_world_loss": total / count,
        "val_world_metric": total / count,
        "val_world_samples": len(entries),
    }
    if head_kind == "expert":
        result["val_world_sigma"] = eval_sigma
    return result


def _args_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _checkpoint_metadata(
    args: argparse.Namespace,
    *,
    kind: str,
    step: int,
    best_step: int,
    best_metric: float | None,
    stop_reason: str,
    train_manifest: CacheManifest,
    val_manifest: CacheManifest,
    merge_info: Mapping[str, Any],
    adapter_names: Sequence[str],
    extractor: CosmosPredict2Extractor,
    world_head: WorldExpert | LinearWorldReadout,
    checkpoint_sha256: str,
    train_episodes: Sequence[int],
    val_episodes: Sequence[int],
) -> dict[str, Any]:
    alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
    head_metadata: dict[str, Any] = {
        "type": args.head,
        "class": type(world_head).__name__,
        "config": world_head.config_dict(),
        "compute_dtype": "bfloat16",
        "trainable_parameters": sum(parameter.numel() for parameter in world_head.parameters()),
    }
    if args.head == "expert":
        head_metadata.update(
            {
                "checkpoint": args.expert_checkpoint,
                "vlm_config": args.vlm_config,
            }
        )
        objective = {
            "name": "rectified_flow_future_latent_loss",
            "target": "noise - clean_future",
            "future_latent_frames": 14,
            "sigma_distribution": "4*exp(N(0,1)) with 5% log-uniform tail [200,100000]",
            "sigma_weighting": "(1 + sigma)^2 / sigma^2",
            "padding": "latent_valid_mask excludes bins containing repeated-last pixel frames",
            "eval_sigma": args.eval_sigma,
        }
    else:
        objective = {
            "name": "clean_future_latent_mse",
            "target": "clean_future",
            "future_latent_frames": 14,
            "sigma_distribution": "none",
            "sigma_weighting": "none",
            "padding": "latent_valid_mask excludes bins containing repeated-last pixel frames",
        }
    return {
        "schema_version": 1,
        "artifact": "cosmos_t2_world_expert_training_checkpoint",
        "head": head_metadata,
        "checkpoint_kind": kind,
        "step": step,
        "best_step": best_step,
        "best_val_world_loss": best_metric,
        "stop_reason": stop_reason,
        "base": {
            "identity": "cosmos-predict2-2B",
            "checkpoint_path": str(args.backbone_checkpoint.expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "dtype": "bfloat16",
            "state_t": 2,
            "hidden_layer": 20,
            "executed_blocks": list(range(20)),
            "gradient_checkpointing": "blocks 0-18; block 19 tap uncheckpointed",
        },
        "merged_old_lora": dict(merge_info),
        "new_lora": {
            "rank": args.lora_rank,
            "alpha": alpha,
            "block_indices": list(range(20)),
            "adapter_targets": list(adapter_names),
            "adapter_dtype": "float32",
        },
        **({"world_expert": head_metadata} if args.head == "expert" else {"linear_readout": head_metadata}),
        "data": {
            "dataset_repo_id": train_manifest.dataset_repo_id,
            "dataset_revision": train_manifest.dataset_revision,
            "train_manifest": str(train_manifest.path.resolve()),
            "train_manifest_sha256": _sha256(train_manifest.path),
            "val_manifest": str(val_manifest.path.resolve()),
            "val_manifest_sha256": _sha256(val_manifest.path),
            "split": str(args.split.expanduser().resolve()),
            "split_sha256": _sha256(args.split.expanduser().resolve()),
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_episodes),
            "clip_contract": "61 pixel frames -> exactly 16 VAE latent frames",
            "observed_contract": "five RGB frames -> exactly two observed latent frames",
        },
        "objective": objective,
        "hyperparameters": _args_payload(args),
        "provenance": {
            "seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
            "python_version": platform.python_version(),
            "extractor_vae_input_mode": extractor.config.vae_input_mode,
        },
    }


def _head_state_dict(world_head: WorldExpert | LinearWorldReadout) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in world_head.state_dict().items()}


def _save_checkpoint(
    output_dir: Path,
    backbone: torch.nn.Module,
    world_head: WorldExpert | LinearWorldReadout,
    metadata: Mapping[str, Any],
    *,
    kind: str,
    head_kind: str,
) -> None:
    lora_path = output_dir / f"{kind}_lora.safetensors"
    head_path = output_dir / (
        f"{kind}_world_expert.safetensors" if head_kind == "expert" else f"{kind}_linear_readout.safetensors"
    )
    lora_temporary = lora_path.with_name(f".{lora_path.name}.tmp")
    head_temporary = head_path.with_name(f".{head_path.name}.tmp")
    save_lora_state_dict(backbone, lora_temporary)
    save_file(_head_state_dict(world_head), str(head_temporary))
    os.replace(lora_temporary, lora_path)
    os.replace(head_temporary, head_path)
    _atomic_json(output_dir / f"{kind}.json", metadata)


def _entries_for_episodes(manifest: CacheManifest, episodes: Sequence[int]) -> tuple[CacheManifestEntry, ...]:
    allowed = set(episodes)
    return tuple(entry for entry in manifest.entries if entry.episode_index in allowed)


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this trainer deliberately has no CPU training path")
    device = torch.device(args.device)
    set_reproducible_seeds(args.seed)
    train_manifest = load_cache_manifest(args.train_manifest.expanduser().resolve())
    val_manifest = load_cache_manifest(args.val_manifest.expanduser().resolve())
    if (train_manifest.dataset_repo_id, train_manifest.dataset_revision) != (
        val_manifest.dataset_repo_id,
        val_manifest.dataset_revision,
    ):
        raise ValueError("train and validation manifests must describe the same dataset revision")
    train_episodes = tuple(args.train_episodes)
    val_episodes = tuple(args.val_episodes)
    train_entries = _entries_for_episodes(train_manifest, train_episodes)
    val_entries = _entries_for_episodes(val_manifest, val_episodes)
    if not train_entries or not val_entries:
        raise ValueError("both manifests must contain selected train and validation anchors")
    if args.smoke:
        val_entries = val_entries[:8]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite and not args.resume:
        raise FileExistsError(f"refusing non-empty output directory; pass --overwrite: {output_dir}")
    latent_cache = LatentCache(args.latent_cache_dir or output_dir / "latent-cache", overwrite=args.overwrite)
    dataset = _build_full_dataset(train_manifest, args, (*train_episodes, *val_episodes))

    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=args.backbone_checkpoint.expanduser().resolve(),
            tokenizer_path=args.tokenizer.expanduser().resolve(),
            device=str(device),
            dtype="bfloat16",
            hidden_layer=20,
            high_noise_sigma=80.0,
            seed=args.seed,
            state_t=2,
            vae_input_mode="observed_prefix",
            input_frames=5,
            sac_mode="none",
        )
    )
    merge_info = merge_lora_file_into_base(extractor.backbone, args.merge_lora_weights)
    verify_path = args.verify_merge_against
    if verify_path is None and args.smoke:
        verify_path = DEFAULT_VERIFY_MERGE_AGAINST
    if verify_path is not None:
        _verify_merged_backbone(extractor.backbone, verify_path.expanduser().resolve())
    alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
    adapter_names = inject_lora(
        extractor.backbone,
        rank=args.lora_rank,
        alpha=alpha,
        block_indices=range(20),
    )
    freeze_base_parameters(extractor.backbone)
    _enable_trunk_checkpointing(extractor.backbone)
    extractor.backbone.train()
    if args.head == "expert":
        world_kwargs: dict[str, Any] = {"device": device, "latent_patch_size": args.latent_patch_size}
        if args.vlm_config is not None:
            world_kwargs["vlm_config_name"] = args.vlm_config
        world_head: WorldExpert | LinearWorldReadout = WorldExpert.from_pretrained(
            args.expert_checkpoint,
            **world_kwargs,
        ).to(device=device, dtype=torch.bfloat16)
    else:
        world_head = LinearWorldReadout(latent_patch_size=args.latent_patch_size).to(
            device=device, dtype=torch.bfloat16
        )
    start_step = 1
    if args.resume:
        last_meta_path = output_dir / "last.json"
        if not last_meta_path.is_file():
            raise FileNotFoundError(f"--resume requires {last_meta_path}")
        last_meta = json.loads(last_meta_path.read_text())
        start_step = int(last_meta["step"]) + 1
        load_lora_state_dict(extractor.backbone, output_dir / "last_lora.safetensors")
        head_path = output_dir / (
            "last_world_expert.safetensors" if args.head == "expert" else "last_linear_readout.safetensors"
        )
        world_head.load_state_dict(load_file(str(head_path), device="cpu"), strict=True)
        world_head.to(device=device, dtype=torch.bfloat16)
        print(f"RESUME from step {start_step}", flush=True)
    prompt = load_prompt_embedding(args.prompt.expanduser()).embedding.to(device=device, dtype=torch.bfloat16)
    if tuple(prompt.shape) != (1, 512, 1024):
        raise ValueError(f"prompt embedding must have shape [1, 512, 1024], got {tuple(prompt.shape)}")
    adapter_params = lora_parameters(extractor.backbone)
    head_params = [parameter for parameter in world_head.parameters() if parameter.requires_grad]
    if not head_params:
        raise ValueError(f"{type(world_head).__name__} has no trainable parameters")
    print(
        f"TRAINABLE_PARAMETERS lora={sum(parameter.numel() for parameter in adapter_params)} "
        f"{'expert' if args.head == 'expert' else 'linear_readout'}="
        f"{sum(parameter.numel() for parameter in head_params)}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_params, "lr": args.lora_lr, "weight_decay": 0.1, "name": "lora"},
            {
                "params": head_params,
                "lr": args.expert_lr,
                "weight_decay": 1.0e-10,
                "name": "world_expert" if args.head == "expert" else "linear_readout",
            },
        ],
        betas=(0.9, 0.99),
        eps=1.0e-8,
        capturable=True,
    )
    max_steps = min(args.max_steps, args.smoke_steps) if args.smoke else args.max_steps
    warmup_steps = min(1_000, max(max_steps - 1, 0))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(float(step) / max(warmup_steps, 1), 1.0e-3)
        return max(0.0, 1.0 - (step - warmup_steps) / max(max_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    if args.resume and start_step > 1:
        for _ in range(start_step - 1):
            scheduler.step()
    checkpoint_sha256 = _sha256(args.backbone_checkpoint.expanduser().resolve())
    logger = SafeWandbLogger(
        project=args.wandb_project,
        run_name=args.run_name,
        disabled=args.no_wandb or args.smoke,
        config={
            "train_manifest": str(train_manifest.path),
            "val_manifest": str(val_manifest.path),
            "objective": (
                "future latent rectified-flow velocity"
                if args.head == "expert"
                else "clean future latent MSE"
            ),
            "head": args.head,
            "lora_blocks": list(range(20)),
            **_args_payload(args),
        },
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics_mode = "a" if args.resume else "w"
    metrics = metrics_path.open(metrics_mode)
    started = time.perf_counter()
    best_metric: float | None = None
    best_step = 0
    if args.resume:
        best_meta_path = output_dir / "best.json"
        if best_meta_path.is_file():
            best_meta = json.loads(best_meta_path.read_text())
            best_step = int(best_meta.get("step", 0))
            val_rows = []
            if metrics_path.is_file():
                for line in metrics_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "val_world_metric" in row:
                        val_rows.append(row)
            if val_rows:
                best_row = min(val_rows, key=lambda r: r["val_world_metric"])
                best_metric = float(best_row["val_world_metric"])
                best_step = int(best_row["step"])
    early = EarlyStopping(args.patience, args.min_delta)
    final_step = 0
    stop_reason = "max_steps"
    try:
        for step in range(start_step, max_steps + 1):
            if time.perf_counter() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            step_started = time.perf_counter()
            extractor.backbone.train()
            world_head.train()
            optimizer.zero_grad(set_to_none=True)
            batch_entries = tuple(
                train_entries[((step - 1) * args.batch_size + position) % len(train_entries)]
                for position in range(args.batch_size)
            )
            samples = [_load_full_sample(dataset, entry) for entry in batch_entries]
            encoded = [
                _encode_full_clip(
                    tokenizer=extractor.tokenizer,
                    sample=sample,
                    entry=entry,
                    cache=latent_cache,
                    device=device,
                )
                for sample, entry in zip(samples, batch_entries, strict=True)
            ]
            full_latents = torch.cat([item[0] for item in encoded], dim=0)
            valid = torch.cat([item[1] for item in encoded], dim=0)
            clean_future = full_latents[:, :, 2:]
            valid_future = valid[:, 2:]
            if args.head == "expert":
                sigma = _sigma_for(args.seed, step - 1, clean_future.shape[0], device)
                noise = _noise_like(
                    clean_future.shape,
                    device=device,
                    seed=args.seed * 1_000_003 + step + 6,
                )
                x_sigma = clean_future + noise * sigma.to(dtype=clean_future.dtype).view(-1, 1, 1, 1, 1)
            observed = torch.cat([sample.rgb_clip[:, :, :5] for sample in samples], dim=0)
            trunk_sigma = torch.full((observed.shape[0],), 80.0, device=device, dtype=torch.float32)
            with autocast_context(device):
                extraction = extractor.forward_features(
                    observed,
                    prompt.expand(observed.shape[0], -1, -1),
                    noise_seed=args.seed,
                    sigma=trunk_sigma,
                )
                if args.head == "expert":
                    prefix = world_head.prepare_prefix_kv(extraction.tokens)
                    prediction = world_head.denoise(x_sigma, sigma, prefix)
                    world_loss = _world_loss(prediction, clean_future, noise, sigma, valid_future)
                else:
                    prediction = world_head(extraction.tokens)
                    world_loss = _linear_loss(prediction, clean_future, valid_future)
            if not torch.isfinite(world_loss).item():
                raise FloatingPointError("world loss is non-finite")
            trainable = tuple(adapter_params) + tuple(head_params)
            # Match the video-LoRA trainer: TE expects exactly one backward scoped to trainables.
            torch.autograd.backward(world_loss, inputs=trainable)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("gradient norm is non-finite")
            if step == 1:
                lora_b_grad_max = max(
                    (parameter.grad.abs().max().item() if parameter.grad is not None else 0.0)
                    for parameter in adapter_params
                    if parameter.ndim == 2 and parameter.shape[0] != 16
                )
                if lora_b_grad_max == 0.0:
                    raise RuntimeError("lora_B received zero gradient after step 1; LoRA adapter is a no-op")
                print(f"lora_B grad sanity check passed: max_abs_grad={lora_b_grad_max:.6g}", flush=True)
            optimizer.step()
            if step == 1:
                from lerobot.policies.vam.cosmos_lora import lora_state_dict

                lora_b_max = max(
                    tensor.abs().max().item()
                    for name, tensor in lora_state_dict(extractor.backbone).items()
                    if name.endswith("lora_B")
                )
                if lora_b_max == 0.0:
                    raise RuntimeError("lora_B stayed at zero after step 1; LoRA adapter is a no-op")
                print(f"lora_B weight sanity check passed: max_abs={lora_b_max:.6g}", flush=True)
            scheduler.step()
            final_step = step
            elapsed = time.perf_counter() - step_started
            peak = int(torch.cuda.max_memory_allocated(device))
            record: dict[str, Any] = {
                "step": step,
                "video_loss": float(world_loss.detach().float().item()),
                "world_loss": float(world_loss.detach().float().item()),
                "learning_rate_lora": float(optimizer.param_groups[0]["lr"]),
                "learning_rate_expert": float(optimizer.param_groups[1]["lr"]),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "seconds_per_step": elapsed,
                "peak_vram_bytes": peak,
            }
            if args.head == "expert":
                record.update(
                    {
                        "sigma_mean": float(sigma.mean().item()),
                        "sigma_min": float(sigma.min().item()),
                        "sigma_max": float(sigma.max().item()),
                    }
                )
            should_validate = step % args.val_every == 0 or step == max_steps
            improved = False
            should_stop = False
            if should_validate:
                validation = evaluate_world(
                    extractor,
                    world_head,
                    extractor.tokenizer,
                    dataset,
                    val_entries,
                    prompt,
                    latent_cache,
                    device=device,
                    eval_sigma=args.eval_sigma,
                    head_kind=args.head,
                )
                record.update(validation)
                candidate = float(validation["val_world_metric"])
                improved, should_stop = early.update(candidate)
                if improved:
                    best_metric = candidate
                    best_step = step
                record["early_stopping_bad_evaluations"] = early.bad_evaluations
                print(
                    f"step={step} world={record['world_loss']:.6g} val_world={candidate:.6g} "
                    f"lr_lora={record['learning_rate_lora']:.3g} seconds={elapsed:.3f} "
                    f"peak_vram={peak / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                if args.head == "expert":
                    print(
                        f"step={step} world={record['world_loss']:.6g} sigma={record['sigma_mean']:.4g} "
                        f"lr_lora={record['learning_rate_lora']:.3g} seconds={elapsed:.3f} "
                        f"peak_vram={peak / 2**30:.2f}GiB",
                        flush=True,
                    )
                else:
                    print(
                        f"step={step} world={record['world_loss']:.6g} "
                        f"lr_lora={record['learning_rate_lora']:.3g} seconds={elapsed:.3f} "
                        f"peak_vram={peak / 2**30:.2f}GiB",
                        flush=True,
                    )
            logger.log(record, step)
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            metadata = _checkpoint_metadata(
                args,
                kind="last",
                step=step,
                best_step=best_step,
                best_metric=best_metric,
                stop_reason=stop_reason,
                train_manifest=train_manifest,
                val_manifest=val_manifest,
                merge_info=merge_info,
                adapter_names=adapter_names,
                extractor=extractor,
                world_head=world_head,
                checkpoint_sha256=checkpoint_sha256,
                train_episodes=train_episodes,
                val_episodes=val_episodes,
            )
            if should_validate or step % args.save_every == 0:
                _save_checkpoint(
                    output_dir,
                    extractor.backbone,
                    world_head,
                    metadata,
                    kind="last",
                    head_kind=args.head,
                )
                if should_validate and improved:
                    best_metadata = dict(metadata)
                    best_metadata["checkpoint_kind"] = "best"
                    _save_checkpoint(
                        output_dir,
                        extractor.backbone,
                        world_head,
                        best_metadata,
                        kind="best",
                        head_kind=args.head,
                    )
            if should_stop:
                stop_reason = "validation_plateau"
                break
            del extraction, prediction, world_loss, full_latents, clean_future
            if args.head == "expert":
                del prefix, noise, sigma, x_sigma
            torch.cuda.empty_cache()
    finally:
        metrics.close()
        logger.finish()
    if final_step > 0:
        metadata = _checkpoint_metadata(
            args,
            kind="last",
            step=final_step,
            best_step=best_step,
            best_metric=best_metric,
            stop_reason=stop_reason,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            merge_info=merge_info,
            adapter_names=adapter_names,
            extractor=extractor,
            world_head=world_head,
            checkpoint_sha256=checkpoint_sha256,
            train_episodes=train_episodes,
            val_episodes=val_episodes,
        )
        _save_checkpoint(
            output_dir,
            extractor.backbone,
            world_head,
            metadata,
            kind="last",
            head_kind=args.head,
        )
        if not (output_dir / "best.json").is_file():
            best_metadata = dict(metadata)
            best_metadata["checkpoint_kind"] = "best"
            best_metadata["best_step"] = final_step
            _save_checkpoint(
                output_dir,
                extractor.backbone,
                world_head,
                best_metadata,
                kind="best",
                head_kind=args.head,
            )
    print(f"last checkpoint: {output_dir / 'last_lora.safetensors'}", flush=True)
    head_checkpoint = (
        "best_world_expert.safetensors" if args.head == "expert" else "best_linear_readout.safetensors"
    )
    print(f"best head checkpoint: {output_dir / head_checkpoint}", flush=True)
    return output_dir


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
