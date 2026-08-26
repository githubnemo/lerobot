#!/usr/bin/env python3
"""Build resumable transformed caches for the six-depth LTX layer-selection probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata
from lerobot.policies.vam.ltx_extractor import LTXExtractor, LTXExtractorConfig, derive_window_seed
from lerobot.policies.vam.ltx_layer_mix import (
    LTX_CONTEXT_TRANSFORMS,
    LTX_LAYER_PROBE_DEPTHS,
    apply_ltx_context_transform,
    ltx_context_grid,
    ltx_layer_probe_tokens,
)
from lerobot.policies.vam.ltx_layer_mix_cache import (
    CACHE_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    LTXLayerMixCacheError,
    LTXLayerMixCacheItem,
    layer_mix_artifact_name,
    load_layer_mix_manifest,
    save_layer_mix_feature_cache,
    sha256_file,
    verify_layer_mix_feature_cache,
    write_layer_mix_manifest,
)
from lerobot.policies.vam.ltx_prompt_embedding import load_ltx_prompt_artifact
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample

DEFAULT_DATASET = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_TRANSFORMER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_VAE = Path("/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors")
DEFAULT_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)
DEFAULT_CACHE_ROOT = Path("/home/anton/.cache/video-vam")
TRAIN_EPISODES = tuple(range(32))
VAL_EPISODES = tuple(range(32, 40))
TRAIN_STRIDE = 3
VAL_STRIDE = 20
CHECKPOINT_RESERVE_BYTES = 2 * 2**30


def expected_artifact_bytes(context_transform: str) -> int:
    context_bytes = len(LTX_LAYER_PROBE_DEPTHS) * ltx_layer_probe_tokens(context_transform) * 4096 * 2
    return context_bytes + (1 * 6 + 1 * 30 * 6) * 4 + 30 + 32_768


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--train-output-dir", type=Path)
    parser.add_argument("--val-output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument("--train-stride", type=int, default=TRAIN_STRIDE)
    parser.add_argument("--val-stride", type=int, default=VAL_STRIDE)
    parser.add_argument("--context-transform", choices=LTX_CONTEXT_TRANSFORMS, default="pool2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.train_output_dir is None:
        args.train_output_dir = DEFAULT_CACHE_ROOT / (
            f"ltx25-layer-probe-train0-31-stride3-{args.context_transform}"
        )
    if args.val_output_dir is None:
        args.val_output_dir = DEFAULT_CACHE_ROOT / (
            f"ltx25-layer-probe-val32-39-stride20-{args.context_transform}"
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.train_stride <= 0 or args.val_stride <= 0:
        raise ValueError("--train-stride and --val-stride must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib < 10:
        raise ValueError("--min-free-gib must be finite and at least 10 GiB")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.train_output_dir.expanduser().resolve() == args.val_output_dir.expanduser().resolve():
        raise ValueError("train and validation output directories must differ")


def episode_rows(dataset: LeRobotDataset) -> dict[int, tuple[int, int]]:
    metadata = dataset.meta.episodes
    return {
        int(episode): (int(start), int(stop))
        for episode, start, stop in zip(
            metadata["episode_index"],
            metadata["dataset_from_index"],
            metadata["dataset_to_index"],
            strict=True,
        )
    }


def selected_windows(
    dataset: LeRobotDataset, *, train_stride: int, val_stride: int
) -> list[tuple[str, int, int]]:
    rows = episode_rows(dataset)
    selected: list[tuple[str, int, int]] = []
    for split, episodes, stride in (
        ("train", TRAIN_EPISODES, train_stride),
        ("validation", VAL_EPISODES, val_stride),
    ):
        for episode in episodes:
            start, stop = rows[episode]
            selected.extend((split, episode, frame) for frame in range(start + 4, stop, stride))
    return selected


def artifact_path(output_dir: Path, episode: int, frame: int) -> Path:
    return output_dir / f"episode-{episode:04d}-frame-{frame:06d}.safetensors"


def disk_guard(args: argparse.Namespace, selected: list[tuple[str, int, int]]) -> dict[str, Any]:
    free = shutil.disk_usage(args.train_output_dir.expanduser().resolve().parent).free
    min_free = int(args.min_free_gib * 2**30)
    artifact_bytes = expected_artifact_bytes(args.context_transform)
    missing = sum(
        not artifact_path(
            args.train_output_dir.expanduser().resolve()
            if split == "train"
            else args.val_output_dir.expanduser().resolve(),
            episode,
            frame,
        ).is_file()
        for split, episode, frame in selected
    )
    projected_selected = missing * artifact_bytes
    safe_requirement = projected_selected + CHECKPOINT_RESERVE_BYTES + min_free
    if safe_requirement > free:
        raise RuntimeError(
            f"insufficient disk for provenance-tagged {args.context_transform} multi-depth cache: "
            f"free={free / 2**30:.2f} GiB, projected={projected_selected / 2**30:.2f} GiB, "
            f"checkpoint_reserve={CHECKPOINT_RESERVE_BYTES / 2**30:.2f} GiB, "
            f"guard={min_free / 2**30:.2f} GiB"
        )
    decision = {
        "free_bytes_before": free,
        "minimum_free_guard_bytes": min_free,
        "checkpoint_reserve_bytes": CHECKPOINT_RESERVE_BYTES,
        "selected_windows": len(selected),
        "missing_windows": missing,
        "context_transform": args.context_transform,
        "projected_cache_bytes": projected_selected,
        "safe_requirement_bytes": safe_requirement,
        "storage_strategy": (
            "six aligned BF16 contexts per entry; adaptive average pool "
            f"15x20 -> {ltx_context_grid(args.context_transform)[1]}x"
            f"{ltx_context_grid(args.context_transform)[2]} per frame"
        ),
        "representation_change": (
            f"{args.context_transform} spatial reduction; temporal grid and 4096 channels "
            "preserved independently per layer"
        ),
    }
    print("DISK_AUDIT=" + json.dumps(decision, sort_keys=True), flush=True)
    return decision


def prompt_payload(path: Path, artifact: Any) -> dict[str, Any]:
    sidecar = json.loads(path.with_suffix(".json").read_text())
    return {
        "artifact_path": str(path.resolve()),
        "artifact_sha256": sha256_file(path),
        "sidecar_sha256": sha256_file(path.with_suffix(".json")),
        "embedding_shape": list(artifact.embedding.shape),
        "embedding_dtype": str(artifact.embedding.dtype).removeprefix("torch."),
        "embedding_sha256": artifact.provenance.embedding_sha256,
        "prompt": artifact.provenance.prompt,
        "valid_tokens": artifact.provenance.valid_tokens,
        "transformer_sha256": artifact.provenance.transformer_sha256,
        "source_commit": artifact.provenance.source_commit,
        "model_revision": artifact.provenance.model_revision,
        "validated_sidecar": sidecar,
        "fallback_allowed": False,
    }


def sidecar_payload(
    *,
    split: str,
    dataset: LeRobotDataset,
    episode: int,
    frame: int,
    prepared: Any,
    prompt: dict[str, Any],
    extraction: Any,
    contexts: dict[int, torch.Tensor],
    context_transform: str,
    seconds: float,
    stage_seconds: dict[str, float],
    runner_profile: dict[str, Any],
) -> dict[str, Any]:
    provenance = extraction.provenance.to_dict()
    context_grid = ltx_context_grid(context_transform)
    context_tokens = ltx_layer_probe_tokens(context_transform)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "artifact": layer_mix_artifact_name(context_transform),
        "dataset": {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "episode_index": episode,
            "frame_index": frame,
            "window_indices": list(prepared.window_indices),
            "history_offsets": list(CUBE_OUT_OF_BOX_CONTRACT.history_offsets),
            "fps": CUBE_OUT_OF_BOX_CONTRACT.fps,
        },
        "split": {
            "name": split,
            "train_episodes": list(TRAIN_EPISODES),
            "validation_episodes": list(VAL_EPISODES),
            "leakage_guard": "episode-disjoint canonical split",
        },
        "prompt": prompt,
        "backbone": provenance,
        "extraction": {
            "noise_seed": extraction.provenance.noise_seed,
            "sigma": float(extraction.sigma[0].item()),
            "tapped_layers": list(extraction.tapped_layers),
            "deepest_layer": extraction.deepest_layer,
            "raw_shapes": {
                str(layer): list(extraction.tokens_by_layer[layer].shape)
                for layer in extraction.tapped_layers
            },
            "raw_grid_shapes": {
                str(layer): list(extraction.hidden_grids[layer].shape) for layer in extraction.tapped_layers
            },
            "seconds": seconds,
            "stage_seconds": stage_seconds,
            "runner_profile": runner_profile,
            "extractor_inputs": ["five_causal_rgb_frames", "real_gemma4_prompt_embedding"],
            "excluded_inputs": ["state", "target_action", "action_is_pad"],
        },
        "temporal_contract": {
            "observed_rgb_frames": 5,
            "vae_input_frames": 9,
            "vae_padding": "repeat latest observed frame four times",
            "clean_latent_frames": 2,
            "sigma1_noise_latent_frames": 6,
            "future_pixels_used": False,
            "target_frame_count": 57,
        },
        "output": {
            "backbone": "LTX-2.5-22B-distilled",
            "tapped_layers": list(LTX_LAYER_PROBE_DEPTHS),
            "deepest_layer": LTX_LAYER_PROBE_DEPTHS[-1],
            "num_blocks": 48,
            "high_noise_sigma": 1.0,
            "one_forward_pass": True,
            "raw_context_shape_per_layer": [1, 2400, 4096],
            "context_transform": context_transform,
            "transform_definition": (
                f"per-latent-frame adaptive_avg_pool2d 15x20 -> {context_grid[1]}x{context_grid[2]}"
            ),
            "context_shape_per_layer": [1, context_tokens, 4096],
            "context_tokens_per_layer": context_tokens,
            "context_channels": 4096,
            "context_grid": list(context_grid),
            "context_dtype": "bfloat16",
            "flatten_order": "T,H,W,C",
        },
        "tensors": {
            "contexts": {str(layer): list(contexts[layer].shape) for layer in LTX_LAYER_PROBE_DEPTHS},
            "state": list(prepared.state.shape),
            "target_action": list(prepared.target_action.shape),
            "action_is_pad": list(prepared.action_is_pad.shape),
            "action_padding_semantics": "true means padded; excluded from loss, normalization, and RMSE",
        },
    }


def manifest_payload(
    *,
    split: str,
    dataset: LeRobotDataset,
    output_dir: Path,
    stride: int,
    entries: list[dict[str, Any]],
    args: argparse.Namespace,
    producer: dict[str, Any],
    prompt: dict[str, Any],
    disk: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    episodes = TRAIN_EPISODES if split == "train" else VAL_EPISODES
    context_grid = ltx_context_grid(args.context_transform)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "artifact": layer_mix_artifact_name(args.context_transform, manifest=True),
        "dataset": {"repo_id": dataset.repo_id, "revision": dataset.revision},
        "subset": {"split": split, "episodes": list(episodes), "stride": stride},
        "provenance": {
            "builder": "scripts/video_vam/build_ltx_layer_mix_cache.py",
            "backbone": "LTX-2.5-22B-distilled",
            "checkpoint_path": producer["checkpoint_path"],
            "checkpoint_sha256": producer["checkpoint_sha256"],
            "video_vae_path": producer["video_vae_path"],
            "video_vae_sha256": producer["video_vae_sha256"],
            "source_commit": producer["source_commit"],
            "model_revision": producer["model_revision"],
            "tapped_layers": list(LTX_LAYER_PROBE_DEPTHS),
            "deepest_layer": LTX_LAYER_PROBE_DEPTHS[-1],
            "num_blocks": 48,
            "high_noise_sigma": 1.0,
            "noise_parameterization": "normalized_rectified_flow_sigma",
            "global_seed": args.seed,
            "one_forward_pass": True,
            "context_transform": args.context_transform,
            "context_tokens_per_layer": ltx_layer_probe_tokens(args.context_transform),
            "context_channels": 4096,
            "context_grid": list(context_grid),
            "context_dtype": "bfloat16",
            "prompt": prompt,
            "disk_audit": disk,
            "selection": {
                "stride": stride,
                "ordered_pairs": [[entry["episode_index"], entry["frame_index"]] for entry in entries],
            },
            "split_contract": "train episodes 0-31; validation episodes 32-39; no overlap",
            "action_contract": "causal target [30,6], masks preserved; state [1,6]",
        },
        "global_seed": args.seed,
        "entries": entries,
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "runtime": runtime,
    }


def existing_entry(
    path: Path,
    *,
    split: str,
    episode: int,
    frame: int,
    noise_seed: int,
    context_transform: str,
) -> dict[str, Any]:
    item = verify_layer_mix_feature_cache(path)
    dataset = item.provenance["dataset"]
    if (
        item.provenance["split"]["name"] != split
        or dataset["episode_index"] != episode
        or dataset["frame_index"] != frame
        or dataset["window_indices"] != list(range(frame - 4, frame + 1))
        or item.provenance["extraction"]["noise_seed"] != noise_seed
        or item.provenance["output"]["context_transform"] != context_transform
    ):
        raise ValueError(f"resume provenance mismatch: {path}")
    sidecar = path.with_suffix(".json")
    return {
        "sample_id": f"episode-{episode:04d}-frame-{frame:06d}",
        "episode_index": episode,
        "frame_index": frame,
        "window_indices": list(range(frame - 4, frame + 1)),
        "noise_seed": noise_seed,
        "safetensors": path.name,
        "sidecar": sidecar.name,
        "safetensors_sha256": sha256_file(path),
        "sidecar_sha256": sha256_file(sidecar),
        "bytes": path.stat().st_size + sidecar.stat().st_size,
    }


def verify_all_manifest_entries(manifest_path: Path) -> None:
    """Hash every tensor after the complete cache is built, deleting corrupt pairs."""
    manifest = load_layer_mix_manifest(manifest_path)
    corrupt: list[str] = []
    for index, entry in enumerate(manifest.entries, 1):
        path = manifest.root / entry.safetensors
        if sha256_file(path) != entry.safetensors_sha256:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            corrupt.append(entry.sample_id)
        else:
            verify_layer_mix_feature_cache(path, expected_entry=entry)
        if index % 100 == 0 or index == len(manifest.entries):
            print(
                f"HASH_AUDIT {manifest_path.parent.name} {index}/{len(manifest.entries)} "
                f"corrupt={len(corrupt)}",
                flush=True,
            )
    if corrupt:
        raise RuntimeError(
            f"deleted {len(corrupt)} corrupt cache tensor/sidecar pairs; rerun with --resume: "
            + ",".join(corrupt)
        )


def build(args: argparse.Namespace) -> tuple[Path, Path]:
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real LTX feature extraction")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    train_output = args.train_output_dir.expanduser().resolve()
    val_output = args.val_output_dir.expanduser().resolve()
    for output in (train_output, val_output):
        if output.exists() and any(output.iterdir()) and not (args.resume or args.overwrite):
            raise FileExistsError(
                f"refusing non-empty cache directory without --resume/--overwrite: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=args.dataset_root.expanduser(),
        episodes=list(range(40)),
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    validate_metadata(dataset.meta, CUBE_OUT_OF_BOX_CONTRACT).raise_if_invalid()
    if dataset.revision != CUBE_OUT_OF_BOX_CONTRACT.revision:
        raise ValueError("dataset revision does not match the canonical cube-out-of-box contract")
    selected = selected_windows(dataset, train_stride=args.train_stride, val_stride=args.val_stride)
    expected_train = sum(1 for split, _, _ in selected if split == "train")
    expected_val = sum(1 for split, _, _ in selected if split == "validation")
    if not expected_train or not expected_val:
        raise ValueError("train and validation selections must both be non-empty")
    if {episode for split, episode, _ in selected if split == "train"} & {
        episode for split, episode, _ in selected if split == "validation"
    }:
        raise ValueError("train/validation episode leakage detected")
    disk = disk_guard(args, selected)
    prompt_path = args.prompt.expanduser().resolve()
    prompt_artifact = load_ltx_prompt_artifact(prompt_path)
    prompt = prompt_payload(prompt_path, prompt_artifact)
    config = LTXExtractorConfig(
        checkpoint_path=args.transformer.expanduser().resolve(),
        video_vae_path=args.video_vae.expanduser().resolve(),
        device="cuda",
        dtype="bfloat16",
        hidden_layer=LTX_LAYER_PROBE_DEPTHS[-1],
        high_noise_sigma=1.0,
        seed=args.seed,
        input_frames=5,
        padded_input_frames=9,
        offload_mode="cpu",
        quantization="fp8-cast",
        persistent_transformer=True,
        explicit_prefix_execution=True,
        vae_compile_mode=None,
    )
    torch.cuda.reset_peak_memory_stats()
    constructed = time.perf_counter()
    extractor = LTXExtractor(config)
    construct_seconds = time.perf_counter() - constructed
    if prompt_artifact.provenance.transformer_sha256 != extractor.checkpoint_sha256:
        raise ValueError("prompt embedding transformer hash does not match LTX extractor")
    opened = extractor.open_transformer()
    producer: dict[str, Any] | None = None
    entries_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    times_by_split: dict[str, list[float]] = {"train": [], "validation": []}
    started = time.perf_counter()
    try:
        for index, (split, episode, frame) in enumerate(selected, 1):
            output_dir = train_output if split == "train" else val_output
            path = artifact_path(output_dir, episode, frame)
            noise_seed = derive_window_seed(dataset.revision, episode, frame, args.seed)
            entry: dict[str, Any] | None = None
            if args.resume and path.exists():
                try:
                    entry = existing_entry(
                        path,
                        split=split,
                        episode=episode,
                        frame=frame,
                        noise_seed=noise_seed,
                        context_transform=args.context_transform,
                    )
                except LTXLayerMixCacheError as exc:
                    if "hash mismatch" not in str(exc):
                        raise
                    path.unlink(missing_ok=True)
                    path.with_suffix(".json").unlink(missing_ok=True)
                    print(f"deleted corrupt resume pair {path}: {exc}", flush=True)
                else:
                    print(
                        f"resume verified {index}/{len(selected)} {split} {episode}/{frame}",
                        flush=True,
                    )
            if entry is None:
                if (path.exists() or path.with_suffix(".json").exists()) and not args.overwrite:
                    raise FileExistsError(f"refusing existing partial artifact without --overwrite: {path}")
                required_floor = int(args.min_free_gib * 2**30) + CHECKPOINT_RESERVE_BYTES
                artifact_bytes = expected_artifact_bytes(args.context_transform)
                free_now = shutil.disk_usage(output_dir).free
                if free_now - artifact_bytes < required_floor:
                    raise RuntimeError(
                        "disk guard tripped before extraction; refusing to cross the free-space floor"
                    )
                sample = dataset[_relative_index(dataset, frame)]
                prepared = prepare_sample(sample, frame_index=frame, config=CUBE_OUT_OF_BOX_CONTRACT)
                if tuple(prepared.window_indices) != tuple(range(frame - 4, frame + 1)):
                    raise RuntimeError("prepared sample violates five-frame causal history")
                stage_seconds: dict[str, float] = {}
                extraction_started = time.perf_counter()
                extraction = extractor.extract_layers(
                    prepared.rgb_history,
                    prompt_artifact.embedding,
                    layer_indices=LTX_LAYER_PROBE_DEPTHS,
                    noise_seed=noise_seed,
                    timings=stage_seconds,
                )
                torch.cuda.synchronize()
                seconds = time.perf_counter() - extraction_started
                if extraction.provenance.prompt_source != "embedding":
                    raise RuntimeError("LTX extraction did not use the prompt embedding artifact")
                if any(
                    tuple(extraction.tokens_by_layer[layer].shape) != (1, 2400, 4096)
                    for layer in LTX_LAYER_PROBE_DEPTHS
                ):
                    raise RuntimeError("unexpected raw LTX multi-depth shape")
                contexts = {
                    layer: apply_ltx_context_transform(extraction.hidden_grids[layer], args.context_transform)
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous()
                    for layer in LTX_LAYER_PROBE_DEPTHS
                }
                provenance = sidecar_payload(
                    split=split,
                    dataset=dataset,
                    episode=episode,
                    frame=frame,
                    prepared=prepared,
                    prompt=prompt,
                    extraction=extraction,
                    contexts=contexts,
                    context_transform=args.context_transform,
                    seconds=seconds,
                    stage_seconds=stage_seconds,
                    runner_profile=dict(getattr(extractor.backbone, "last_profile", {})),
                )
                save_layer_mix_feature_cache(
                    LTXLayerMixCacheItem(
                        contexts,
                        prepared.state.to(dtype=torch.float32),
                        prepared.target_action.to(dtype=torch.float32),
                        prepared.action_is_pad.to(dtype=torch.bool),
                        provenance,
                    ),
                    path,
                    overwrite=args.overwrite,
                )
                verify_layer_mix_feature_cache(path)
                entry = existing_entry(
                    path,
                    split=split,
                    episode=episode,
                    frame=frame,
                    noise_seed=noise_seed,
                    context_transform=args.context_transform,
                )
                times_by_split[split].append(seconds)
                producer = extraction.provenance.to_dict()
                print(
                    f"cached {index}/{len(selected)} {split} {episode}/{frame} "
                    f"seconds={seconds:.3f} size_mib={entry['bytes'] / 2**20:.3f}",
                    flush=True,
                )
            entries_by_split[split].append(entry)
            if index % 10 == 0 or index == len(selected):
                progress = {
                    "completed": index,
                    "total": len(selected),
                    "train_completed": len(entries_by_split["train"]),
                    "validation_completed": len(entries_by_split["validation"]),
                    "elapsed_seconds": time.perf_counter() - started,
                    "free_bytes": shutil.disk_usage(output_dir).free,
                }
                (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
    finally:
        extractor.close()
    if producer is None:
        first = verify_layer_mix_feature_cache(train_output / entries_by_split["train"][0]["safetensors"])
        producer = dict(first.provenance["backbone"])
    elapsed_total = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    manifests = []
    for split, output_dir, stride in (
        ("train", train_output, args.train_stride),
        ("validation", val_output, args.val_stride),
    ):
        times = times_by_split[split]
        runtime = {
            "construct_seconds": construct_seconds,
            "transformer_open_seconds": opened,
            "new_extraction_seconds": sum(times),
            "new_extraction_count": len(times),
            "all_entries_count": len(entries_by_split[split]),
            "new_extraction_p50_seconds": statistics.median(times) if times else None,
            "overall_wall_seconds": elapsed_total,
            "peak_vram_allocated_bytes": peak_allocated,
            "peak_vram_reserved_bytes": peak_reserved,
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "python_version": platform.python_version(),
            "exact_eager_vae": True,
        }
        payload = manifest_payload(
            split=split,
            dataset=dataset,
            output_dir=output_dir,
            stride=stride,
            entries=entries_by_split[split],
            args=args,
            producer=producer,
            prompt=prompt,
            disk=disk,
            runtime=runtime,
        )
        manifest = output_dir / "manifest.json"
        write_layer_mix_manifest(payload, manifest, overwrite=bool(args.resume or args.overwrite))
        loaded = load_layer_mix_manifest(manifest)
        expected_count = expected_train if split == "train" else expected_val
        if len(loaded.entries) != expected_count:
            raise RuntimeError(f"{split} manifest count mismatch")
        manifests.append(manifest)
        print(
            f"MANIFEST_{split.upper()}={manifest} count={len(loaded.entries)} bytes={payload['total_bytes']}",
            flush=True,
        )
    for manifest in manifests:
        verify_all_manifest_entries(manifest)
    print("CACHE_HASH_AUDIT=all_entries_verified", flush=True)
    print(f"CACHE_PEAK_VRAM_GIB={peak_allocated / 2**30:.3f}", flush=True)
    print(f"CACHE_WALL_SECONDS={elapsed_total:.3f}", flush=True)
    print(f"CACHE_FREE_GIB_AFTER={shutil.disk_usage(train_output).free / 2**30:.3f}", flush=True)
    return manifests[0], manifests[1]


def main(argv: list[str] | None = None) -> int:
    try:
        build(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
