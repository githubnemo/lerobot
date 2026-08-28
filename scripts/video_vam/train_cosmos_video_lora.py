#!/usr/bin/env python3
"""Video-only LoRA fine-tuning for the Cosmos Predict2-2B video backbone."""

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
from collections.abc import Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file

from lerobot.datasets import LeRobotDataset
from lerobot.policies.vam.cosmos_cache_dataset import CacheManifest, CacheManifestEntry, load_cache_manifest
from lerobot.policies.vam.cosmos_lora import (
    LATENT_FRAMES,
    build_cosmos_noised_input,
    freeze_base_parameters,
    full_clip_delta_timestamps,
    inject_lora,
    latent_valid_mask,
    lora_parameters,
    prepare_full_clip_sample,
    rectified_flow_video_loss,
    sample_upstream_sigma,
    save_lora_state_dict,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.vam_split import load_vam_split
from scripts.video_vam.smoke_test_cosmos_extractor import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOKENIZER_PATH,
)

DEFAULT_TRAIN_MANIFEST = Path(
    "/home/anton/.cache/video-vam/cosmos-train0-31-stride3-sigma80-prefix-pool2/manifest.json"
)
DEFAULT_VAL_MANIFEST = Path(
    "/home/anton/.cache/video-vam/cosmos-rehearsal-stride20-sigma80-prefix-pool2/manifest.json"
)
DEFAULT_SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/runs/cosmos2b-video-lora-20260828")
DEFAULT_MAX_STEPS = 500_000
DEFAULT_MAX_HOURS = 12.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_VAL_EVERY = 1_000
DEFAULT_SAVE_EVERY = 1_000
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 0.0
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_EVAL_SIGMA = 80.0
DEFAULT_WANDB_PROJECT = "video-vam-world2action"
DEFAULT_RUN_NAME = "cosmos2b-video-lora-20260828"


@dataclass(frozen=True, slots=True)
class EarlyStopping:
    """State machine for minimization-based validation early stopping."""

    best: float = math.inf
    bad_evaluations: int = 0
    patience: int = DEFAULT_PATIENCE
    min_delta: float = DEFAULT_MIN_DELTA

    def update(self, metric: float) -> tuple[EarlyStopping, bool, bool]:
        if not math.isfinite(metric):
            raise ValueError("validation video metric must be finite")
        if metric < self.best - self.min_delta:
            return EarlyStopping(metric, 0, self.patience, self.min_delta), True, False
        bad = self.bad_evaluations + 1
        return EarlyStopping(self.best, bad, self.patience, self.min_delta), False, bad >= self.patience


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
        except Exception as exc:
            print(f"W&B unavailable; continuing locally: {exc}", flush=True)

    def log(self, values: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class LatentCache:
    """Store backbone-independent full-clip VAE latents per anchor."""

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
        expected_latents = (1, 16, LATENT_FRAMES, 60, 80)
        if tuple(latents.shape) != expected_latents or latents.dtype != torch.bfloat16:
            raise ValueError(f"latent cache latents have invalid shape/dtype: {path}")
        if tuple(valid.shape) != (1, LATENT_FRAMES) or valid.dtype != torch.bool:
            raise ValueError(f"latent cache mask has invalid shape/dtype: {path}")
        return latents.to(device=device), valid.to(device=device)

    def save(self, entry: CacheManifestEntry, latents: torch.Tensor, valid: torch.Tensor) -> None:
        path = self.path_for(entry)
        if path.exists() and not self.overwrite:
            return
        expected = (1, 16, LATENT_FRAMES, 60, 80)
        if tuple(latents.shape) != expected or latents.dtype != torch.bfloat16:
            raise ValueError("only full Cosmos bfloat16 latents can be cached")
        if tuple(valid.shape) != (1, LATENT_FRAMES) or valid.dtype != torch.bool:
            raise ValueError("only [1, 16] latent validity masks can be cached")
        temporary = path.with_name(f".{path.name}.tmp")
        save_file(
            {"latents": latents.cpu().contiguous(), "latent_valid": valid.cpu().contiguous()}, str(temporary)
        )
        os.replace(temporary, path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "sample_id": entry.sample_id,
                    "episode_index": entry.episode_index,
                    "frame_index": entry.frame_index,
                    "window_indices": list(entry.window_indices),
                    "latents_shape": list(latents.shape),
                    "latents_dtype": "bfloat16",
                    "padding_policy": "repeat_last_frame_and_mask_latent_video_loss",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_TRAIN_MANIFEST,
        help="Train cache manifest used for frame selection.",
    )
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--latent-cache-dir", type=Path)
    parser.add_argument("--train-episodes", type=int, nargs="+")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-lr", type=float, default=DEFAULT_LORA_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--eval-sigma", type=float, default=DEFAULT_EVAL_SIGMA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        raise ValueError("the trainable Cosmos video LoRA runtime requires --device cuda")
    if args.max_steps <= 0 or args.max_hours <= 0 or args.batch_size <= 0:
        raise ValueError("max-steps, max-hours, and batch-size must be positive")
    if args.val_every <= 0 or args.save_every <= 0 or args.patience <= 0:
        raise ValueError("val-every, save-every, and patience must be positive")
    if args.seed < 0 or args.lora_rank <= 0:
        raise ValueError("seed must be non-negative and LoRA rank must be positive")
    for name in ("lora_lr", "grad_clip", "eval_sigma"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if args.lora_alpha is not None and (not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0):
        raise ValueError("lora-alpha must be finite and positive")
    if args.weight_decay < 0 or args.min_delta < 0:
        raise ValueError("weight-decay and min-delta must be non-negative")
    if args.train_episodes is not None and (
        not args.train_episodes or any(type(value) is not int or value < 0 for value in args.train_episodes)
    ):
        raise ValueError("train-episodes must contain non-negative integers")
    if args.train_episodes is not None and len(set(args.train_episodes)) != len(args.train_episodes):
        raise ValueError("train-episodes must not contain duplicates")


def _enable_full_block_checkpointing(backbone: torch.nn.Module) -> None:
    """Checkpoint all DiT blocks, preserving the TE-safe single-backward graph."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 28:
        raise ValueError("Cosmos backbone must expose exactly 28 blocks")
    for index, block in enumerate(blocks):
        if not getattr(block, "_lerobot_full_block_checkpoint", False):
            wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
            wrapped._lerobot_full_block_checkpoint = True
            blocks[index] = wrapped


def _build_full_dataset(
    manifest: CacheManifest, args: argparse.Namespace, episodes: Sequence[int]
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


def _encode_latents(
    tokenizer: Any, sample: Any, entry: CacheManifestEntry, cache: LatentCache, device: torch.device
):
    cached = cache.load(entry, device)
    if cached is not None:
        return cached
    images = _images_to_cosmos(sample.rgb_clip, device)
    with torch.no_grad():
        latents = tokenizer.encode(images)
    if not isinstance(latents, torch.Tensor):
        raise TypeError("Cosmos tokenizer encode() must return a tensor")
    expected = (images.shape[0], 16, LATENT_FRAMES, 60, 80)
    prefix = (images.shape[0], 16, 2, 60, 80)
    if tuple(latents.shape) == prefix:
        full = torch.zeros(expected, device=device, dtype=torch.bfloat16)
        full[:, :, :2] = latents.to(device=device, dtype=torch.bfloat16)
        latents = full
    elif tuple(latents.shape) != expected:
        raise ValueError(f"Cosmos tokenizer output must have shape {expected}, got {tuple(latents.shape)}")
    else:
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


def _backbone_kwargs(
    network_input: torch.Tensor,
    condition_mask: torch.Tensor,
    c_noise: torch.Tensor,
    prompt: torch.Tensor,
    data_type: Any,
) -> dict[str, Any]:
    batch_size = network_input.shape[0]
    padding_mask = torch.zeros(
        (batch_size, 1, network_input.shape[-2], network_input.shape[-1]),
        device=network_input.device,
        dtype=network_input.dtype,
    )
    return {
        "x_B_C_T_H_W": network_input,
        "timesteps_B_T": c_noise,
        "crossattn_emb": prompt.expand(batch_size, -1, -1),
        "condition_video_input_mask_B_C_T_H_W": condition_mask,
        "fps": torch.full((batch_size, 1), 10.0, device=network_input.device, dtype=torch.float32),
        "padding_mask": padding_mask,
        "data_type": data_type,
    }


def _prediction(backbone: torch.nn.Module, **kwargs: Any) -> torch.Tensor:
    output = backbone(**kwargs, use_cuda_graphs=False)
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Cosmos backbone prediction must be a tensor")
    return output


@torch.no_grad()
def evaluate_video(
    backbone: torch.nn.Module,
    tokenizer: Any,
    dataset: LeRobotDataset,
    entries: Sequence[CacheManifestEntry],
    prompt: torch.Tensor,
    data_type: Any,
    cache: LatentCache,
    *,
    device: torch.device,
    sigma_value: float,
) -> dict[str, float | int]:
    if not entries:
        raise ValueError("validation split contains no entries")
    backbone.eval()
    total = 0.0
    count = 0
    sigma = torch.full((1,), sigma_value, device=device, dtype=torch.float32)
    for entry in entries:
        sample = _load_full_sample(dataset, entry)
        clean, valid = _encode_latents(tokenizer, sample, entry, cache, device)
        noise = _noise_like(clean.shape, device=device, seed=entry.noise_seed + 4_000_003)
        network_input, condition_mask, c_noise = build_cosmos_noised_input(clean, noise, sigma)
        with autocast_context(device):
            prediction = _prediction(
                backbone,
                **_backbone_kwargs(network_input, condition_mask, c_noise, prompt, data_type),
            )
            loss = rectified_flow_video_loss(prediction, clean, noise, sigma, valid)
        valid_count = int((valid[:, 2:]).sum().item() * clean.shape[1] * clean.shape[3] * clean.shape[4])
        if valid_count:
            total += float(loss.float().item()) * valid_count
            count += valid_count
        del prediction, clean, noise
    if count <= 0:
        raise ValueError("validation split contains no non-conditioning latent frames")
    return {
        "val_video_loss": total / count,
        "val_video_sigma": sigma_value,
        "val_video_samples": len(entries),
        "val_video_metric": total / count,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _args_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _checkpoint_metadata(
    args: argparse.Namespace,
    *,
    kind: str,
    step: int,
    best_step: int,
    best_metric: float | None,
    adapter_names: Sequence[str],
    train_manifest: CacheManifest,
    val_manifest: CacheManifest,
    split_path: Path,
    checkpoint_sha256: str,
    trainable_parameters: int,
    stop_reason: str,
    final_step: int,
    val_episodes: Sequence[int],
    train_episodes: Sequence[int],
) -> dict[str, Any]:
    alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
    return {
        "schema_version": 1,
        "artifact": "cosmos_video_lora_training_checkpoint",
        "checkpoint_kind": kind,
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": step,
        "best_step": best_step,
        "best_val_video_loss": best_metric,
        "stop_reason": stop_reason,
        "final_step": final_step,
        "lora": {
            "rank": args.lora_rank,
            "alpha": alpha,
            "trainable_parameters": trainable_parameters,
            "adapter_targets": list(adapter_names),
            "adapter_dtype": "float32",
        },
        "backbone": {
            "identity": "cosmos-predict2-2B",
            "checkpoint": str(args.backbone_checkpoint.expanduser().resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "base_dtype": "bfloat16",
            "num_blocks": 28,
            "hidden_layer": 20,
            "gradient_checkpointing": "LeRobot full-block checkpointing on all 28 DiT blocks",
        },
        "dataset": {
            "repo_id": train_manifest.dataset_repo_id,
            "revision": train_manifest.dataset_revision,
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_episodes),
            "train_manifest": str(train_manifest.path.resolve()),
            "train_manifest_sha256": _sha256(train_manifest.path),
            "val_manifest": str(val_manifest.path.resolve()),
            "val_manifest_sha256": _sha256(val_manifest.path),
            "split": str(split_path.resolve()),
            "split_sha256": _sha256(split_path),
            "latent_contract": "61 pixel frames -> 16 latent frames; conditioning is first 2 latent frames",
        },
        "video_objective": {
            "name": "rectified_flow_video_loss",
            "target": "noise - clean",
            "conditioning": "observed first 5 pixel frames encoded to first 2 clean latent slots",
            "future_input": "future latent slots are noised; future clean pixels are used only as loss targets",
            "sigma_distribution": "4*exp(N(0,1)) with 5% log-uniform tail [200,100000]",
            "sigma_weighting": "RectifiedFlowScaling.sigma_loss_weights",
            "padding": "repeat-last pixel frames are excluded through latent_valid_mask",
            "eval_metric": "mean masked rectified-flow MSE at fixed sigma",
        },
        "hyperparameters": _args_payload(args),
        "provenance": {
            "seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
            "python_version": platform.python_version(),
        },
    }


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    *,
    kind: str,
) -> Path:
    path = output_dir / f"{kind}_lora.safetensors"
    temporary = path.with_name(f".{path.name}.tmp")
    save_lora_state_dict(model, temporary)
    os.replace(temporary, path)
    _atomic_json(path.with_suffix(".json"), metadata)
    return path


def _entries_for_episodes(manifest: CacheManifest, episodes: Sequence[int]) -> tuple[CacheManifestEntry, ...]:
    allowed = set(episodes)
    return tuple(entry for entry in manifest.entries if entry.episode_index in allowed)


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; CPU validation is provided by tests/policies/test_cosmos_lora.py"
        )
    device = torch.device(args.device)
    set_reproducible_seeds(args.seed)
    train_manifest = load_cache_manifest(args.manifest.expanduser().resolve())
    val_manifest = load_cache_manifest(args.val_manifest.expanduser().resolve())
    if (train_manifest.dataset_repo_id, train_manifest.dataset_revision) != (
        val_manifest.dataset_repo_id,
        val_manifest.dataset_revision,
    ):
        raise ValueError("train and validation manifests must describe the same dataset revision")
    split_path = args.split.expanduser().resolve()
    val_split = load_vam_split(split_path, val_manifest, allow_partial=True)
    train_episodes = tuple(args.train_episodes or val_split.train_episodes)
    if set(train_episodes) - set(val_split.train_episodes):
        raise ValueError("requested train episodes must be a subset of split train episodes")
    train_entries = _entries_for_episodes(train_manifest, train_episodes)
    val_entries = _entries_for_episodes(val_manifest, val_split.val_episodes)
    if not train_entries or not val_entries:
        raise ValueError("both train and validation manifests must contain selected entries")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing non-empty output directory; pass --overwrite: {output_dir}")
    latent_cache = LatentCache(args.latent_cache_dir or output_dir / "latent-cache", overwrite=args.overwrite)
    dataset = _build_full_dataset(train_manifest, args, (*train_episodes, *val_split.val_episodes))
    checkpoint_path = args.backbone_checkpoint.expanduser().resolve()
    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=checkpoint_path,
            tokenizer_path=args.tokenizer.expanduser().resolve(),
            device=str(device),
            dtype="bfloat16",
            hidden_layer=20,
            sac_mode="none",
            high_noise_sigma=4.0,
            seed=args.seed,
        )
    )
    backbone = extractor.backbone
    tokenizer = extractor.tokenizer
    alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
    adapter_names = inject_lora(backbone, rank=args.lora_rank, alpha=alpha)
    freeze_base_parameters(backbone)
    _enable_full_block_checkpointing(backbone)
    adapter_params = lora_parameters(backbone)
    trainable_parameters = sum(parameter.numel() for parameter in adapter_params)
    print(f"TRAINABLE_PARAMETERS={trainable_parameters}", flush=True)
    print(f"LORA_RANK={args.lora_rank} LORA_ALPHA={alpha} ADAPTERS={len(adapter_names)}", flush=True)
    prompt = load_prompt_embedding(args.prompt.expanduser()).embedding.to(device=device, dtype=torch.bfloat16)
    if tuple(prompt.shape) != (1, 512, 1024):
        raise ValueError(f"prompt embedding must have shape [1, 512, 1024], got {tuple(prompt.shape)}")
    data_type = extractor._data_type()
    optimizer = torch.optim.AdamW(
        adapter_params,
        lr=args.lora_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1.0e-8,
        capturable=True,
    )
    warmup_steps = min(1_000, max(args.max_steps - 1, 0))

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return max(float(step) / max(warmup_steps, 1), 1.0e-3)
        return max(0.0, 1.0 - (step - warmup_steps) / max(args.max_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    checkpoint_sha256 = _sha256(checkpoint_path)
    logger = SafeWandbLogger(
        project=args.wandb_project,
        run_name=args.run_name,
        disabled=args.no_wandb,
        config={
            "train_manifest": str(train_manifest.path),
            "val_manifest": str(val_manifest.path),
            "split": str(split_path),
            "train_episodes": list(train_episodes),
            "val_episodes": list(val_split.val_episodes),
            "lora_rank": args.lora_rank,
            "lora_alpha": alpha,
            "trainable_parameters": trainable_parameters,
            "objective": "video-only rectified-flow; no action decoder constructed",
            **_args_payload(args),
        },
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics = metrics_path.open("w")
    started = time.perf_counter()
    best_metric: float | None = None
    best_step = 0
    best_validation: dict[str, float | int] = {}
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    final_step = 0
    stop_reason = "max_steps"
    try:
        for step in range(1, args.max_steps + 1):
            if time.perf_counter() - started >= args.max_hours * 3600:
                stop_reason = "max_hours"
                break
            final_step = step
            backbone.train()
            optimizer.zero_grad(set_to_none=True)
            batch_entries = tuple(
                train_entries[((step - 1) * args.batch_size + position) % len(train_entries)]
                for position in range(args.batch_size)
            )
            samples = [_load_full_sample(dataset, entry) for entry in batch_entries]
            encoded = [
                _encode_latents(tokenizer, sample, entry, latent_cache, device)
                for sample, entry in zip(samples, batch_entries, strict=True)
            ]
            clean = torch.cat([item[0] for item in encoded], dim=0)
            valid = torch.cat([item[1] for item in encoded], dim=0)
            sigma = _sigma_for(args.seed, step - 1, clean.shape[0], device)
            noise = _noise_like(clean.shape, device=device, seed=args.seed * 1_000_003 + step + 6)
            network_input, condition_mask, c_noise = build_cosmos_noised_input(clean, noise, sigma)
            with autocast_context(device):
                prediction = _prediction(
                    backbone,
                    **_backbone_kwargs(network_input, condition_mask, c_noise, prompt, data_type),
                )
                video_loss = rectified_flow_video_loss(prediction, clean, noise, sigma, valid)
            if not torch.isfinite(video_loss).item():
                raise FloatingPointError("video loss is non-finite")
            # Keep Transformer Engine's graph traversal to exactly one backward per optimizer step.
            torch.autograd.backward(video_loss, inputs=tuple(adapter_params))
            grad_norm = torch.nn.utils.clip_grad_norm_(adapter_params, args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("LoRA gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            elapsed = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(device))
            record: dict[str, Any] = {
                "step": step,
                "video_loss": float(video_loss.detach().float().item()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "sigma_mean": float(sigma.mean().item()),
                "sigma_min": float(sigma.min().item()),
                "sigma_max": float(sigma.max().item()),
                "seconds_per_step": elapsed if step == 1 else elapsed / step,
                "peak_vram_bytes": peak,
                "trainable_parameters": trainable_parameters,
            }
            should_validate = step % args.val_every == 0 or step == args.max_steps
            improved = False
            should_stop = False
            if should_validate:
                validation = evaluate_video(
                    backbone,
                    tokenizer,
                    dataset,
                    val_entries,
                    prompt,
                    data_type,
                    latent_cache,
                    device=device,
                    sigma_value=args.eval_sigma,
                )
                record.update(validation)
                candidate = float(validation["val_video_metric"])
                early, improved, should_stop = early.update(candidate)
                if improved:
                    best_metric = candidate
                    best_step = step
                    best_validation = dict(validation)
                record["early_stopping_bad_evaluations"] = early.bad_evaluations
                print(
                    f"step={step} video={record['video_loss']:.6g} val_video={candidate:.6g} "
                    f"lr={record['learning_rate']:.3g} seconds={record['seconds_per_step']:.3f} "
                    f"peak_vram={peak / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                print(
                    f"step={step} video={record['video_loss']:.6g} sigma={record['sigma_mean']:.4g} "
                    f"lr={record['learning_rate']:.3g} seconds={record['seconds_per_step']:.3f} "
                    f"peak_vram={peak / 2**30:.2f}GiB",
                    flush=True,
                )
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            logger.log(record, step)
            if should_validate or step % args.save_every == 0:
                last_metadata = _checkpoint_metadata(
                    args,
                    kind="last",
                    step=step,
                    best_step=best_step,
                    best_metric=best_metric,
                    adapter_names=adapter_names,
                    train_manifest=train_manifest,
                    val_manifest=val_manifest,
                    split_path=split_path,
                    checkpoint_sha256=checkpoint_sha256,
                    trainable_parameters=trainable_parameters,
                    stop_reason=stop_reason,
                    final_step=final_step,
                    train_episodes=train_episodes,
                    val_episodes=val_split.val_episodes,
                )
                _save_checkpoint(output_dir, backbone, last_metadata, kind="last")
                if should_validate and improved:
                    best_metadata = dict(last_metadata)
                    best_metadata["checkpoint_kind"] = "best"
                    _save_checkpoint(output_dir, backbone, best_metadata, kind="best")
            if should_stop:
                stop_reason = "validation_plateau"
                print(f"early stopping at step={step}", flush=True)
                break
            del prediction, video_loss, clean, noise
            torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        raise RuntimeError(
            f"CUDA OOM; peak allocated VRAM={peak} bytes; peak reserved VRAM={reserved} bytes; "
            "try --batch-size 1 or a smaller LoRA rank"
        ) from exc
    finally:
        metrics.close()
        logger.finish()
    final_step = max(final_step, step if "step" in locals() else 0)
    if final_step > 0:
        last_metadata = _checkpoint_metadata(
            args,
            kind="last",
            step=final_step,
            best_step=best_step,
            best_metric=best_metric,
            adapter_names=adapter_names,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            split_path=split_path,
            checkpoint_sha256=checkpoint_sha256,
            trainable_parameters=trainable_parameters,
            stop_reason=stop_reason,
            final_step=final_step,
            train_episodes=train_episodes,
            val_episodes=val_split.val_episodes,
        )
        _save_checkpoint(output_dir, backbone, last_metadata, kind="last")
        if best_metric is None:
            best_metadata = dict(last_metadata)
            best_metadata["checkpoint_kind"] = "best"
            best_metadata["best_step"] = final_step
            _save_checkpoint(output_dir, backbone, best_metadata, kind="best")
            best_step = final_step
    result = {
        "artifact": "cosmos_video_lora_training_result",
        "status": "validation_trained_non_rollout",
        "stop_reason": stop_reason,
        "final_step": final_step,
        "best_step": best_step,
        "best_val_video_loss": best_metric,
        "best_validation": best_validation,
        "trainable_parameters": trainable_parameters,
        "lora_rank": args.lora_rank,
        "lora_alpha": alpha,
        "train_episodes": list(train_episodes),
        "val_episodes": list(val_split.val_episodes),
        "checkpoint_sha256": checkpoint_sha256,
        "wall_clock_seconds": time.perf_counter() - started,
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.run_name,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"last checkpoint: {output_dir / 'last_lora.safetensors'}", flush=True)
    print(f"best checkpoint: {output_dir / 'best_lora.safetensors'}", flush=True)
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
