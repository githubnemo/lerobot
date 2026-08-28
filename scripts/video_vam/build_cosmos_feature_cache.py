#!/usr/bin/env python3
# Build one strict frozen Cosmos cache artifact per selected causal frame.
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata
from lerobot.policies.vam.context_transform import (
    CONTEXT_TRANSFORMS,
    apply_context_transform,
    context_transform_metadata,
)
from lerobot.policies.vam.cosmos_cache_dataset import (
    MANIFEST_SCHEMA_VERSION,
    load_cache_manifest,
    sha256_file,
    write_cache_manifest,
)
from lerobot.policies.vam.cosmos_feature_cache import (
    ACTION_IS_PAD_KEY,
    CACHE_SCHEMA_VERSION,
    CosmosFeatureCacheArtifact,
    build_feature_cache_provenance,
    derive_window_seed,
    save_feature_cache,
    verify_feature_cache,
)
from lerobot.policies.vam.cosmos_lora import inject_lora, load_lora_state_dict
from lerobot.policies.vam.cosmos_predict2_extractor import (
    UPSTREAM_COMMIT,
    VAE_INPUT_MODE_LEGACY_PADDED,
    VAE_INPUT_MODE_OBSERVED_PREFIX,
    VAE_INPUT_MODES,
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from scripts.video_vam.smoke_test_cosmos_extractor import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOKENIZER_PATH,
    MIMIC_VIDEO_PREPROCESS,
    _cuda_peak,
    _episode_row,
    _git_commit,
    _relative_index,
    _runtime,
    _synchronize,
    conditioning_description,
    prepare_sample,
    run_timed_extraction,
    validate_extraction_output,
    verify_pinned_checkpoints,
)

DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/cosmos-features")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episodes", type=int, nargs="+", default=[0])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--frame-start", type=int, help="Absolute current-frame start, inclusive.")
    parser.add_argument("--frame-end", type=int, help="Absolute current-frame end, exclusive.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Positive episode-local stride. Windows are enumerated from each episode's first "
            "valid frame (episode start + 4), then filtered by frame-start/end; max-samples "
            "truncates the resulting episode-major ordered list."
        ),
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--lora-weights", type=Path, help="Adapter-only Cosmos LoRA safetensors checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, help="Manifest path; defaults to OUTPUT_DIR/manifest.json.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vae-input-mode",
        choices=VAE_INPUT_MODES,
        default=VAE_INPUT_MODE_OBSERVED_PREFIX,
        help=(
            "VAE input contract for new artifacts: observed_prefix encodes only the five "
            "observed pixels; legacy_padded_vae restores 5->61 padding."
        ),
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help=(
            "Video-diffusion noise level at which the frozen Cosmos context is extracted. "
            "The decoder must be told the same value through context_timestep."
        ),
    )
    parser.add_argument(
        "--random-init-seed",
        type=int,
        help="Use a seeded random Cosmos initialization instead of checkpoint weights.",
    )
    parser.add_argument(
        "--context-transform",
        choices=CONTEXT_TRANSFORMS,
        default="none",
        help=(
            "Transform the extracted [B, 19200, 2048] context before writing it. "
            "pool4 stores all 16 latent frames at an adaptive 8x10 spatial grid "
            "([B, 1280, 2048])."
        ),
    )
    parser.add_argument("--resume", action="store_true", help="Strictly verify and skip existing artifacts.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.episodes or any(type(episode) is not int or episode < 0 for episode in args.episodes):
        raise ValueError("--episodes must contain unique non-negative integers")
    if len(set(args.episodes)) != len(args.episodes):
        raise ValueError("--episodes must not contain duplicates")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.frame_start is not None and args.frame_start < 0:
        raise ValueError("--frame-start must be non-negative")
    if args.frame_end is not None and args.frame_end < 0:
        raise ValueError("--frame-end must be non-negative")
    if args.frame_start is not None and args.frame_end is not None and args.frame_end <= args.frame_start:
        raise ValueError("--frame-end must be greater than --frame-start")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.vae_input_mode not in VAE_INPUT_MODES:
        raise ValueError(f"--vae-input-mode must be one of {VAE_INPUT_MODES!r}")
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("--sigma must be finite and positive")
    if args.random_init_seed is not None and args.random_init_seed < 0:
        raise ValueError("--random-init-seed must be non-negative")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.lora_weights is not None and args.random_init_seed is not None:
        raise ValueError("--lora-weights cannot be combined with --random-init-seed")


def _lora_checkpoint_provenance(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA weights not found: {path}")
    metadata_path = path.with_suffix(".json")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"LoRA provenance sidecar not found: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read LoRA provenance sidecar {metadata_path}: {exc}") from exc
    lora = payload.get("lora")
    if not isinstance(lora, dict):
        raise ValueError(f"LoRA provenance sidecar has no lora object: {metadata_path}")
    rank = lora.get("rank")
    alpha = lora.get("alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"LoRA provenance rank must be a positive integer: {metadata_path}")
    if (
        not isinstance(alpha, (int, float))
        or isinstance(alpha, bool)
        or not math.isfinite(float(alpha))
        or alpha <= 0
    ):
        raise ValueError(f"LoRA provenance alpha must be finite and positive: {metadata_path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rank": rank,
        "alpha": float(alpha),
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "adapters_applied": True,
    }


def _selected_frames(dataset: Any, args: argparse.Namespace):
    for episode in args.episodes:
        row = _episode_row(dataset, episode)
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        first = start + len(CUBE_OUT_OF_BOX_CONTRACT.history_offsets) - 1
        last = end
        if first >= last:
            raise ValueError(f"episode {episode} has no selected causal current frames")
        for frame in range(first, last, args.stride):
            if args.frame_start is not None and frame < args.frame_start:
                continue
            if args.frame_end is not None and frame >= args.frame_end:
                continue
            yield episode, frame


def _artifact_path(output_dir: Path, episode: int, frame: int) -> Path:
    return output_dir / f"episode-{episode:04d}-frame-{frame:06d}.safetensors"


def _existing_entry(
    output_dir: Path,
    episode: int,
    frame: int,
    *,
    expected_seed: int,
    expected_transform: str,
    expected_sigma: float,
    expected_random_init_seed: int | None,
    expected_vae_input_mode: str,
    expected_lora_sha256: str | None,
):
    path = _artifact_path(output_dir, episode, frame)
    artifact = verify_feature_cache(path)
    dataset = artifact.provenance.payload["dataset"]
    extractor = artifact.provenance.payload["extractor"]
    if (
        dataset["episode_index"] != episode
        or dataset["frame_index"] != frame
        or dataset["window_indices"] != list(range(frame - 4, frame + 1))
        or extractor["noise_seed"] != expected_seed
    ):
        raise ValueError(f"existing cache provenance does not match episode/frame/seed: {path}")
    stored_transform = artifact.provenance.payload["output"].get("context_transform", "none")
    if stored_transform != expected_transform:
        raise ValueError(f"existing cache context transform does not match request: {path}")
    if abs(float(extractor["high_noise_sigma"]) - expected_sigma) > 1e-9:
        raise ValueError(f"existing cache sigma does not match request: {path}")
    if extractor.get("random_init_seed") != expected_random_init_seed:
        raise ValueError(f"existing cache random-init seed does not match request: {path}")
    if extractor.get("vae_input_mode", VAE_INPUT_MODE_LEGACY_PADDED) != expected_vae_input_mode:
        raise ValueError(f"existing cache VAE input mode does not match request: {path}")
    stored_lora = extractor.get("lora_weights")
    stored_lora_sha256 = stored_lora.get("sha256") if isinstance(stored_lora, dict) else None
    if stored_lora_sha256 != expected_lora_sha256:
        raise ValueError(f"existing cache LoRA provenance does not match request: {path}")
    return artifact


def _manifest_payload(
    *,
    dataset: Any,
    args: argparse.Namespace,
    entries: list[dict[str, Any]],
    runtime: dict[str, Any],
    weights: Mapping[str, Any],
    prompt_embedding: Mapping[str, Any],
) -> dict[str, Any]:
    context = context_transform_metadata(args.context_transform)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "dataset": {"repo_id": dataset.repo_id, "revision": dataset.revision},
        "subset": {
            "episodes": list(args.episodes),
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "max_samples": args.max_samples,
            "stride": args.stride,
        },
        "provenance": {
            "builder": "scripts/video_vam/build_cosmos_feature_cache.py",
            "dataset_revision_seed_input": "dataset.revision + episode + frame + global_seed via SHA256",
            "global_seed": args.seed,
            "high_noise_sigma": args.sigma,
            "vae_input_mode": args.vae_input_mode,
            "random_init_seed": args.random_init_seed,
            "context_transform": args.context_transform,
            "context_tokens": context["output_tokens"],
            "context_grid": context["output_grid"],
            "context_input_grid": context["input_grid"],
            "context_storage": (
                f"detached bfloat16 [B, {context['output_tokens']}, 2048] "
                f"with grid {context['output_grid']['temporal']}x"
                f"{context['output_grid']['height']}x{context['output_grid']['width']}"
            ),
            "weights": dict(weights),
            "lora_weights": weights.get("lora"),
            "adapters_applied": bool(weights.get("adapters_applied", False)),
            "prompt_embedding": dict(prompt_embedding),
            "extractor_input_keys": ["rgb_history", "prompt_embedding"],
            "excluded_from_extractor": ["state", "target_action", ACTION_IS_PAD_KEY],
            "action_padding_semantics": "action_is_pad=true means padded and excluded from decoder loss/statistics",
            "raw_context_storage": "detached bfloat16 [B, 19200, 2048]",
            "status": "diagnostic_only_non_rollout",
            "selection": {
                "stride": args.stride,
                "ordered_pairs": [
                    [int(entry["episode_index"]), int(entry["frame_index"])] for entry in entries
                ],
            },
        },
        "global_seed": args.seed,
        "entries": entries,
        "total_bytes": sum(int(entry["bytes"]) for entry in entries),
        "runtime": runtime,
    }


def build(args: argparse.Namespace) -> Path:
    _validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("The real cache builder requires CUDA; use offline unit tests for CPU validation")
    device = torch.device("cuda")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest.expanduser() if args.manifest else output_dir / "manifest.json").resolve()
    if manifest_path.parent != output_dir:
        raise ValueError("--manifest must be inside --output-dir so artifact paths stay relative and atomic")
    if manifest_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing manifest; pass --resume or --overwrite: {manifest_path}"
        )
    old_manifest = load_cache_manifest(manifest_path) if args.resume and manifest_path.exists() else None

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from lerobot.datasets import LeRobotDataset

    checkpoint_path = args.checkpoint.expanduser().resolve()
    tokenizer_path = args.tokenizer.expanduser().resolve()
    lora_info = _lora_checkpoint_provenance(args.lora_weights) if args.lora_weights is not None else None
    if lora_info is not None and "bridge" in checkpoint_path.name:
        raise ValueError(
            "--lora-weights requires the generic Cosmos checkpoint, not a fused bridge checkpoint"
        )
    weights = verify_pinned_checkpoints(
        DEFAULT_CHECKPOINT_PATH if "bridge" in checkpoint_path.name else checkpoint_path,
        tokenizer_path,
    )
    if "bridge" in checkpoint_path.name:
        weights = dict(weights)
        weights.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint_kind": "bridge_fused",
                "bridge_lora": {
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "rank": 256,
                    "fused": True,
                },
            }
        )
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=args.root.expanduser(),
        episodes=list(args.episodes),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    metadata_report = validate_metadata(dataset.meta, CUBE_OUT_OF_BOX_CONTRACT)
    metadata_report.raise_if_invalid()
    if dataset.revision != CUBE_OUT_OF_BOX_CONTRACT.revision:
        raise ValueError(f"dataset revision is {dataset.revision!r}, expected pinned revision")

    prompt_start = time.perf_counter()
    prompt_artifact = load_prompt_embedding(args.prompt.expanduser())
    prompt_load_seconds = time.perf_counter() - prompt_start
    if tuple(prompt_artifact.embedding.shape) != (1, 512, 1024):
        raise ValueError("The pinned prompt artifact must have shape [1, 512, 1024]")
    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=checkpoint_path,
        tokenizer_path=args.tokenizer.expanduser(),
        device="cuda",
        dtype="bfloat16",
        high_noise_sigma=args.sigma,
        seed=args.seed,
        hidden_layer=20,
        stop_after_step=0,
        random_init_seed=args.random_init_seed,
        vae_input_mode=args.vae_input_mode,
    )
    _synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    extractor = CosmosPredict2Extractor(config)
    if lora_info is not None:
        adapter_names = inject_lora(
            extractor.backbone,
            rank=int(lora_info["rank"]),
            alpha=float(lora_info["alpha"]),
        )
        load_lora_state_dict(extractor.backbone, args.lora_weights.expanduser().resolve())
        extractor.backbone.eval()
        lora_info["adapter_modules"] = list(adapter_names)
        lora_info["adapters_applied"] = True
    weights = dict(weights)
    weights["lora"] = lora_info
    weights["adapters_applied"] = lora_info is not None
    _synchronize(device)
    load_seconds = time.perf_counter() - load_start
    load_peak = _cuda_peak(device)

    entries: list[dict[str, Any]] = []
    total_extract_seconds = 0.0
    total_extract_allocated = 0
    total_extract_reserved = 0
    selected_count = 0
    for selected_count, (episode, frame) in enumerate(_selected_frames(dataset, args)):
        if args.max_samples is not None and selected_count >= args.max_samples:
            break
        noise_seed = derive_window_seed(dataset.revision, episode, frame, args.seed)
        path = _artifact_path(output_dir, episode, frame)
        if args.resume and path.exists():
            artifact = _existing_entry(
                output_dir,
                episode,
                frame,
                expected_seed=noise_seed,
                expected_transform=args.context_transform,
                expected_sigma=args.sigma,
                expected_random_init_seed=args.random_init_seed,
                expected_vae_input_mode=args.vae_input_mode,
                expected_lora_sha256=None if lora_info is None else str(lora_info["sha256"]),
            )
            print(f"resume verified {episode}/{frame}: {path}")
        else:
            if (path.exists() or path.with_suffix(".json").exists()) and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing cache; pass --overwrite: {path}")
            sample = dataset[_relative_index(dataset, frame)]
            prepared = prepare_sample(sample, frame_index=frame, config=CUBE_OUT_OF_BOX_CONTRACT)
            timing = run_timed_extraction(
                extractor,
                prepared,
                prompt_artifact.embedding,
                device=device,
                sigma=args.sigma,
                noise_seed=noise_seed,
            )
            validate_extraction_output(timing.extraction, batch_size=1, sigma=args.sigma)
            extraction = timing.extraction
            context = (
                apply_context_transform(extraction.tokens, args.context_transform)
                .detach()
                .to(device="cpu", dtype=torch.bfloat16)
                .contiguous()
            )
            state = prepared.state.detach().cpu().contiguous()
            target_action = prepared.target_action.detach().cpu().contiguous()
            action_is_pad = prepared.action_is_pad.detach().cpu().contiguous()
            provenance = build_feature_cache_provenance(
                dataset={
                    "repo_id": dataset.repo_id,
                    "revision": dataset.revision,
                    "episode_index": episode,
                    "frame_index": frame,
                    "window_indices": list(prepared.window_indices),
                    "window_offsets": list(CUBE_OUT_OF_BOX_CONTRACT.history_offsets),
                    "fps": CUBE_OUT_OF_BOX_CONTRACT.fps,
                },
                source_shapes={
                    "sample_camera": list(prepared.sample[CUBE_OUT_OF_BOX_CONTRACT.camera_key].shape),
                    "sample_state": list(prepared.sample[CUBE_OUT_OF_BOX_CONTRACT.state_key].shape),
                    "sample_action": list(prepared.sample[CUBE_OUT_OF_BOX_CONTRACT.action_key].shape),
                    "rgb_history": list(prepared.rgb_history.shape),
                    "prompt_embedding": list(prompt_artifact.embedding.shape),
                    "raw_hidden": list(extraction.hidden_grid.shape),
                    "context": list(context.shape),
                    "state": list(state.shape),
                    "target_action": list(target_action.shape),
                    ACTION_IS_PAD_KEY: list(action_is_pad.shape),
                },
                weights=weights,
                prompt_embedding={
                    "artifact_path": str(args.prompt.expanduser().resolve()),
                    "output_sha256": prompt_artifact.provenance.output_sha256,
                    "token_ids_sha256": prompt_artifact.provenance.token_ids_sha256,
                    "shape": list(prompt_artifact.embedding.shape),
                    "dtype": str(prompt_artifact.embedding.dtype).removeprefix("torch."),
                },
                extractor={
                    "device": str(device),
                    "dtype": "bfloat16",
                    "backend": config.backend,
                    "high_noise_sigma": config.high_noise_sigma,
                    "seed": config.seed,
                    "noise_seed": extraction.provenance.noise_seed,
                    "hidden_layer": config.hidden_layer,
                    "stop_after_step": config.stop_after_step,
                    "vae_input_mode": config.vae_input_mode,
                    "input_shape": list(prepared.rgb_history.shape),
                    "preprocess": MIMIC_VIDEO_PREPROCESS,
                    "conditioning": conditioning_description(args.vae_input_mode),
                    "official_resolution": "480",
                    "official_positional_latent_max_h": 240,
                    "official_positional_latent_max_w": 240,
                    "bridge_lora": weights.get("bridge_lora"),
                    "lora_weights": weights.get("lora"),
                    "adapters_applied": bool(weights.get("adapters_applied", False)),
                    "extractor_input_keys": ["rgb_history", "prompt_embedding"],
                    "excluded_from_extractor": ["state", "target_action"],
                    "checkpoint_ignored_metadata_keys": list(
                        extraction.provenance.checkpoint_ignored_metadata_keys
                    ),
                    "checkpoint_ignored_metadata_count": extraction.provenance.checkpoint_ignored_metadata_count,
                    "random_init_seed": extraction.provenance.random_init_seed,
                },
                upstream_commits={
                    "lerobot": _git_commit(),
                    "mimic_video": UPSTREAM_COMMIT,
                    "vendor_manifest": UPSTREAM_COMMIT,
                },
                runtime=_runtime(
                    load_seconds=load_seconds,
                    prompt_load_seconds=prompt_load_seconds,
                    extraction_timing=timing,
                    load_peak=load_peak,
                ),
                context=context,
                state=state,
                target_action=target_action,
                action_is_pad=action_is_pad,
                raw_hidden_shape=tuple(extraction.hidden_grid.shape),
                raw_hidden_dtype=extraction.hidden_grid.dtype,
                context_transform=args.context_transform,
            )
            artifact = CosmosFeatureCacheArtifact(context, state, target_action, action_is_pad, provenance)
            save_feature_cache(artifact, path, overwrite=args.overwrite)
            total_extract_seconds += timing.seconds
            total_extract_allocated = max(total_extract_allocated, timing.peak_allocated_bytes)
            total_extract_reserved = max(total_extract_reserved, timing.peak_reserved_bytes)
            artifact = verify_feature_cache(path)
            print(f"cached {episode}/{frame}: {path}")
        tensor_path = path
        sidecar_path = path.with_suffix(".json")
        entries.append(
            {
                "sample_id": f"episode-{episode:04d}-frame-{frame:06d}",
                "episode_index": episode,
                "frame_index": frame,
                "window_indices": list(range(frame - 4, frame + 1)),
                "noise_seed": noise_seed,
                "safetensors": tensor_path.relative_to(output_dir).as_posix(),
                "sidecar": sidecar_path.relative_to(output_dir).as_posix(),
                "safetensors_sha256": sha256_file(tensor_path),
                "sidecar_sha256": sha256_file(sidecar_path),
                "bytes": tensor_path.stat().st_size + sidecar_path.stat().st_size,
            }
        )

    if not entries:
        raise ValueError("No cache samples were selected")
    if old_manifest is not None:
        expected_subset = {
            "episodes": list(args.episodes),
            "frame_start": args.frame_start,
            "frame_end": args.frame_end,
            "max_samples": args.max_samples,
            "stride": args.stride,
        }
        old_subset = dict(old_manifest.payload["subset"])
        old_subset.setdefault("stride", 1)
        if old_manifest.global_seed != args.seed or old_manifest.dataset_revision != dataset.revision:
            raise ValueError("resume manifest dataset revision or global seed does not match the request")
        if old_manifest.context_transform != args.context_transform:
            raise ValueError("resume manifest context transform does not match the request")
        old_provenance = old_manifest.payload["provenance"]
        if old_provenance.get("lora_weights") != weights.get("lora"):
            raise ValueError("resume manifest LoRA provenance does not match the request")
        if bool(old_provenance.get("adapters_applied", False)) != bool(
            weights.get("adapters_applied", False)
        ):
            raise ValueError("resume manifest adapter state does not match the request")
        if old_manifest.vae_input_mode != args.vae_input_mode:
            raise ValueError("resume manifest VAE input mode does not match the request")
        if (
            "high_noise_sigma" in old_provenance
            and abs(float(old_provenance["high_noise_sigma"]) - args.sigma) > 1e-9
        ):
            raise ValueError("resume manifest sigma does not match the request")
        if (
            "random_init_seed" in old_provenance
            and old_provenance["random_init_seed"] != args.random_init_seed
        ):
            raise ValueError("resume manifest random-init seed does not match the request")
        if old_subset != expected_subset:
            raise ValueError("resume manifest subset does not match the requested selection")
        old_ids = [entry.sample_id for entry in old_manifest.entries]
        new_ids = [entry["sample_id"] for entry in entries]
        if old_ids != new_ids:
            raise ValueError("resume manifest subset/order does not match the requested selection")
        for old_entry, new_entry in zip(old_manifest.entries, entries, strict=True):
            if (
                old_entry.safetensors_sha256 != new_entry["safetensors_sha256"]
                or old_entry.sidecar_sha256 != new_entry["sidecar_sha256"]
                or old_entry.bytes != new_entry["bytes"]
            ):
                raise ValueError(f"resume manifest hash/size mismatch for {old_entry.sample_id}")
    runtime = {
        "load_seconds": load_seconds,
        "prompt_load_seconds": prompt_load_seconds,
        "extraction_seconds": total_extract_seconds,
        "total_runtime_seconds": time.perf_counter() - load_start,
        "load_peak_allocated_bytes": load_peak[0],
        "load_peak_reserved_bytes": load_peak[1],
        "extraction_peak_allocated_bytes": total_extract_allocated,
        "extraction_peak_reserved_bytes": total_extract_reserved,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unavailable",
        "gpu_name": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
    }
    prompt_payload = {
        "artifact_path": str(args.prompt.expanduser().resolve()),
        "output_sha256": prompt_artifact.provenance.output_sha256,
        "token_ids_sha256": prompt_artifact.provenance.token_ids_sha256,
        "shape": list(prompt_artifact.embedding.shape),
        "dtype": str(prompt_artifact.embedding.dtype).removeprefix("torch."),
    }
    payload = _manifest_payload(
        dataset=dataset,
        args=args,
        entries=entries,
        runtime=runtime,
        weights=weights,
        prompt_embedding=prompt_payload,
    )
    write_cache_manifest(payload, manifest_path, overwrite=bool(args.resume or args.overwrite))
    print(f"manifest: {manifest_path}")
    print(f"samples: {len(entries)} total_bytes: {payload['total_bytes']}")
    print("status: diagnostic_only_non_rollout")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
