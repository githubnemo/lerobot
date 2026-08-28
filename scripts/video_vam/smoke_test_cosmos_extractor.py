#!/usr/bin/env python3
"""Run one real pinned-dataset Cosmos extraction and cache its representation."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, cast

import torch

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata, validate_sample
from lerobot.policies.vam.cosmos_feature_cache import (
    CosmosFeatureCacheArtifact,
    build_feature_cache_provenance,
    save_feature_cache,
    verify_feature_cache,
)
from lerobot.policies.vam.cosmos_predict2_extractor import (
    UPSTREAM_COMMIT,
    VAE_INPUT_MODE_LEGACY_PADDED,
    VAE_INPUT_MODE_OBSERVED_PREFIX,
    VAE_INPUT_MODES,
    CosmosPredict2Extraction,
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.download_cosmos_checkpoints import ARTIFACTS, build_plan, verify_download

DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_PROMPT_PATH = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER_PATH = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_CACHE_DIR = Path("/home/anton/.cache/video-vam/cosmos-features")

# The generic checkpoint and preprocessing follow the pinned mimic-video 480p contract;
# the pixel VAE input mode below is an intentional local optimization choice.
MIMIC_VIDEO_PREPROCESS = "mimic-video 480p RGB uint8 -> [-1, 1], resize_online=False"
MIMIC_VIDEO_CONDITIONING = (
    "frame_replace, five observed pixel frames -> two latent frames via prefix-only VAE, "
    "zero-expanded to 16; latent_conditional_frames=2 for T=5"
)


def conditioning_description(vae_input_mode: str, state_t: int = 16) -> str:
    """Describe the selected pixel-to-latent conditioning contract for provenance."""
    if vae_input_mode == VAE_INPUT_MODE_OBSERVED_PREFIX:
        if state_t == 2:
            return (
                "frame_replace, five observed pixel frames -> two latent frames via prefix-only VAE, "
                "kept as the complete observed-only state; latent_conditional_frames=2"
            )
        return MIMIC_VIDEO_CONDITIONING
    if vae_input_mode == VAE_INPUT_MODE_LEGACY_PADDED:
        return (
            "frame_replace, zero-padded 61 pixel frames with first T observed, "
            "encoded whole; latent_conditional_frames=2 for T=5"
        )
    raise ValueError(f"unsupported VAE input mode: {vae_input_mode!r}")


@dataclass(frozen=True, slots=True)
class PreparedSample:
    """Validated sample tensors with labels kept outside the extractor call."""

    rgb_history: torch.Tensor
    state: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    frame_index: int
    window_indices: tuple[int, ...]
    sample: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionTiming:
    """Measured extraction result and CUDA memory peaks."""

    extraction: CosmosPredict2Extraction
    seconds: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse read-only smoke inputs and cache output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--frame-index",
        type=int,
        help="Absolute dataset frame to use as the current frame; default is episode start + 4.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        help="Absolute first frame of the five-frame causal window; overrides --frame-index.",
    )
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, help="Explicit .safetensors output path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vae-input-mode",
        choices=VAE_INPUT_MODES,
        default=VAE_INPUT_MODE_OBSERVED_PREFIX,
        help=(
            "VAE input contract: observed_prefix encodes only the five observed pixels "
            "(default); legacy_padded_vae restores 5->61 padding for compatibility/debugging."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _scalar_int(value: Any, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar, got shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _episode_row(dataset: Any, episode_index: int) -> Any:
    """Return an episode metadata row from the LeRobot metadata table."""
    try:
        return dataset.meta.episodes[episode_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"Episode {episode_index} is unavailable in dataset metadata") from exc


def resolve_frame_index(
    dataset: Any,
    *,
    episode_index: int,
    frame_index: int | None = None,
    window_start: int | None = None,
) -> int:
    """Resolve a causal five-frame window, defaulting to its first non-padded end."""
    if episode_index < 0:
        raise ValueError("episode must be non-negative")
    row = _episode_row(dataset, episode_index)
    episode_start = _scalar_int(row["dataset_from_index"], "dataset_from_index")
    episode_end = _scalar_int(row["dataset_to_index"], "dataset_to_index")
    if episode_end <= episode_start:
        raise ValueError(f"Episode {episode_index} has no frames")
    if window_start is not None:
        current = window_start + 4
    elif frame_index is not None:
        current = frame_index
    else:
        current = episode_start + 4
    if current - 4 < episode_start or current >= episode_end:
        raise ValueError(
            f"Causal window [{current - 4}, {current}] is outside episode {episode_index} "
            f"range [{episode_start}, {episode_end})"
        )
    if frame_index is not None and window_start is not None and frame_index != current:
        raise ValueError("--window-start and --frame-index describe different current frames")
    return current


def _relative_index(dataset: Any, absolute_index: int) -> int:
    mapping = dataset.absolute_to_relative_idx
    return absolute_index if mapping is None else int(mapping[absolute_index])


def prepare_sample(
    sample: dict[str, Any],
    *,
    frame_index: int,
    config=CUBE_OUT_OF_BOX_CONTRACT,
) -> PreparedSample:
    """Validate a dataset sample, then build RGB/state/action tensors."""
    report = validate_sample(sample, config)
    report.raise_if_invalid()
    camera = sample[config.camera_key]
    state = sample[config.state_key]
    target_action = sample[config.action_key]
    action_is_pad = sample[f"{config.action_key}_is_pad"]
    if not all(isinstance(value, torch.Tensor) for value in (camera, state, target_action, action_is_pad)):
        raise TypeError("LeRobot sample camera, state, action, and action_is_pad must be torch tensors")
    # LeRobot's expanded camera contract is TCHW; Cosmos's extractor is BCTHW.
    rgb_history = camera.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    state_input = state.unsqueeze(0).contiguous()
    action_label = target_action.unsqueeze(0).contiguous()
    action_padding = action_is_pad.unsqueeze(0).contiguous()
    window_indices = tuple(frame_index + offset for offset in config.history_offsets)
    return PreparedSample(
        rgb_history, state_input, action_label, action_padding, frame_index, window_indices, dict(sample)
    )


def load_real_sample(
    *,
    root: Path,
    episode_index: int = 0,
    frame_index: int | None = None,
    window_start: int | None = None,
    config=CUBE_OUT_OF_BOX_CONTRACT,
) -> tuple[Any, PreparedSample]:
    """Load one local pinned sample without downloading or mutating the dataset."""
    from lerobot.datasets import LeRobotDataset

    dataset = LeRobotDataset(
        config.repo_id,
        root=root.expanduser(),
        episodes=[episode_index],
        delta_timestamps=config.delta_timestamps(),
        revision=config.revision,
        return_uint8=True,
        download_videos=False,
    )
    metadata_report = validate_metadata(dataset.meta, config)
    metadata_report.raise_if_invalid()
    current = resolve_frame_index(
        dataset,
        episode_index=episode_index,
        frame_index=frame_index,
        window_start=window_start,
    )
    sample = cast(dict[str, Any], dataset[_relative_index(dataset, current)])
    prepared = prepare_sample(sample, frame_index=current, config=config)
    sample_episode = prepared.sample.get("episode_index")
    if sample_episode is not None and _scalar_int(sample_episode, "sample episode_index") != episode_index:
        raise ValueError(f"Loaded sample belongs to episode {sample_episode!r}, expected {episode_index}")
    return dataset, prepared


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cuda_peak(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 0, 0
    return torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)


def validate_extraction_output(
    extraction: CosmosPredict2Extraction, *, batch_size: int, sigma: float
) -> None:
    """Assert the frozen output contract before writing any cache."""
    hidden = extraction.hidden_grid
    context = extraction.tokens
    if (
        not isinstance(hidden, torch.Tensor)
        or not isinstance(context, torch.Tensor)
        or not isinstance(extraction.sigma, torch.Tensor)
    ):
        raise TypeError("Cosmos extraction must expose tensor hidden_grid, tokens, and sigma")
    if hidden.requires_grad or context.requires_grad:
        raise RuntimeError("Cosmos hidden output must be detached and gradient-free")
    if not torch.is_floating_point(hidden) or not torch.is_floating_point(context):
        raise TypeError("Cosmos hidden output must be floating point")
    if not torch.isfinite(hidden).all().item() or not torch.isfinite(context).all().item():
        raise RuntimeError("Cosmos hidden output must contain only finite values")
    if hidden.ndim != 5 or tuple(hidden.shape[2:]) != (30, 40, 2048):
        raise ValueError(f"Cosmos hidden shape must be [B, T, 30, 40, 2048], got {tuple(hidden.shape)}")
    state_t = hidden.shape[1]
    if state_t not in (2, 16):
        raise ValueError(f"Cosmos hidden temporal state must be 2 or 16, got {state_t}")
    expected_hidden_shape = (batch_size, state_t, 30, 40, 2048)
    expected_context_shape = (batch_size, state_t * 30 * 40, 2048)
    if tuple(hidden.shape) != expected_hidden_shape or tuple(context.shape) != expected_context_shape:
        raise ValueError(
            f"Cosmos output shapes must be raw [B, {state_t}, 30, 40, 2048] and "
            f"flattened [B, {state_t * 30 * 40}, 2048]; raw={tuple(hidden.shape)}, "
            f"flattened={tuple(context.shape)}"
        )
    if context.dtype != torch.bfloat16:
        raise TypeError(f"Cosmos context must be bfloat16, got {context.dtype}")
    provenance = extraction.provenance
    if (
        extraction.layer != 20
        or extraction.sigma.shape != (batch_size,)
        or not torch.allclose(extraction.sigma.float(), torch.full_like(extraction.sigma.float(), sigma))
        or provenance.high_noise_sigma != sigma
        or provenance.stop_after_step != 0
    ):
        raise ValueError(
            f"Cosmos extraction metadata does not match sigma={sigma}, layer 20, and stop_after_step=0"
        )


def run_timed_extraction(
    extractor: Any,
    prepared: PreparedSample,
    prompt_embedding: torch.Tensor,
    *,
    device: torch.device,
    sigma: float = 10.0,
    noise_seed: int | None = None,
) -> ExtractionTiming:
    """Call the extractor with RGB and prompt only, never state or action labels."""
    if tuple(prompt_embedding.shape) != (1, 512, 1024):
        raise ValueError(
            f"prompt_embedding must have shape [1, 512, 1024], got {tuple(prompt_embedding.shape)}"
        )
    if prompt_embedding.dtype != torch.bfloat16 or not torch.isfinite(prompt_embedding).all().item():
        raise ValueError("prompt_embedding must be finite bfloat16")
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    # Deliberately only RGB, prompt, and explicit extractor noise seed cross the boundary.
    if noise_seed is None:
        extraction = extractor.extract(prepared.rgb_history, prompt_embedding)
    else:
        extraction = extractor.extract(prepared.rgb_history, prompt_embedding, noise_seed=noise_seed)
    _synchronize(device)
    seconds = time.perf_counter() - start
    validate_extraction_output(extraction, batch_size=prepared.rgb_history.shape[0], sigma=sigma)
    allocated, reserved = _cuda_peak(device)
    return ExtractionTiming(extraction, seconds, allocated, reserved)


def _git_commit() -> str:
    # Fixed argv; resolving git from PATH is intentional, like the rest of the repo tooling.
    result = subprocess.run(  # nosec B607
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    )
    return result.stdout.strip()


def _runtime(
    *,
    load_seconds: float,
    prompt_load_seconds: float,
    extraction_timing: ExtractionTiming,
    load_peak: tuple[int, int],
) -> dict[str, Any]:
    cuda_version = torch.version.cuda or "unavailable"
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "load_seconds": load_seconds,
        "prompt_load_seconds": prompt_load_seconds,
        "extraction_seconds": extraction_timing.seconds,
        "load_peak_allocated_bytes": load_peak[0],
        "load_peak_reserved_bytes": load_peak[1],
        "extraction_peak_allocated_bytes": extraction_timing.peak_allocated_bytes,
        "extraction_peak_reserved_bytes": extraction_timing.peak_reserved_bytes,
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "python_version": platform.python_version(),
    }


def verify_pinned_checkpoints(checkpoint_path: Path, tokenizer_path: Path) -> dict[str, Any]:
    """Stream-check the generic checkpoint and tokenizer against downloader pins."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    tokenizer_path = tokenizer_path.expanduser().resolve()
    output_dir = checkpoint_path.parent.parent
    expected_checkpoint = output_dir / ARTIFACTS["generic"].relative_path
    if checkpoint_path != expected_checkpoint:
        raise ValueError(f"Checkpoint must be the pinned generic checkpoint at {expected_checkpoint}")
    verify_download(output_dir, build_plan())
    expected_tokenizer = output_dir / ARTIFACTS["tokenizer"].relative_path
    if tokenizer_path != expected_tokenizer:
        raise ValueError(f"Tokenizer must be the pinned generic tokenizer at {expected_tokenizer}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": ARTIFACTS["generic"].size,
        "checkpoint_sha256": ARTIFACTS["generic"].sha256,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_size_bytes": ARTIFACTS["tokenizer"].size,
        "tokenizer_sha256": ARTIFACTS["tokenizer"].sha256,
        "checkpoint_kind": "generic_cosmos",
        "bridge_lora": None,
    }


def _default_output(output_dir: Path, episode_index: int, frame_index: int) -> Path:
    return output_dir / f"cube-out-of-box-episode-{episode_index:04d}-frame-{frame_index:06d}.safetensors"


def smoke(args: argparse.Namespace) -> CosmosFeatureCacheArtifact:
    """Run the pinned first-forward smoke and write one strict cache artifact."""
    if args.seed != 0:
        raise ValueError(
            "This first-real-forward cache is fixed to seed 0; later builders assign deterministic per-window seeds"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("The real smoke requires CUDA; use the offline unit tests for CPU validation")
    device = torch.device("cuda")
    checkpoint = args.checkpoint.expanduser()
    tokenizer = args.tokenizer.expanduser()
    weights = verify_pinned_checkpoints(checkpoint, tokenizer)
    prompt_path = args.prompt.expanduser()

    dataset, prepared = load_real_sample(
        root=args.root,
        episode_index=args.episode,
        frame_index=args.frame_index,
        window_start=args.window_start,
    )
    output_path = (
        args.output.expanduser()
        if args.output
        else _default_output(args.output_dir.expanduser(), args.episode, prepared.frame_index)
    )
    if not args.overwrite and (output_path.exists() or output_path.with_suffix(".json").exists()):
        raise FileExistsError(
            f"Refusing to overwrite existing feature cache; pass --overwrite: {output_path}"
        )

    prompt_start = time.perf_counter()
    prompt_artifact = load_prompt_embedding(prompt_path)
    prompt_load_seconds = time.perf_counter() - prompt_start
    if prompt_artifact.embedding.shape[0] != 1:
        raise ValueError("The pinned prompt artifact must have batch size 1")

    config = CosmosPredict2ExtractorConfig(
        checkpoint_path=checkpoint,
        tokenizer_path=tokenizer,
        device="cuda",
        dtype="bfloat16",
        high_noise_sigma=10.0,
        seed=0,
        hidden_layer=20,
        stop_after_step=0,
        vae_input_mode=args.vae_input_mode,
    )
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_start = time.perf_counter()
    extractor = CosmosPredict2Extractor(config)
    _synchronize(device)
    load_seconds = time.perf_counter() - load_start
    load_peak = _cuda_peak(device)

    timing = run_timed_extraction(extractor, prepared, prompt_artifact.embedding, device=device)
    extraction = timing.extraction
    raw_hidden = extraction.hidden_grid
    context = extraction.tokens.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    state = prepared.state.detach().cpu().contiguous()
    target_action = prepared.target_action.detach().cpu().contiguous()
    action_is_pad = prepared.action_is_pad.detach().cpu().contiguous()
    provenance = build_feature_cache_provenance(
        dataset={
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "episode_index": args.episode,
            "frame_index": prepared.frame_index,
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
            "raw_hidden": list(raw_hidden.shape),
            "context": list(context.shape),
            "state": list(state.shape),
            "target_action": list(target_action.shape),
            "action_is_pad": list(action_is_pad.shape),
        },
        weights=weights,
        prompt_embedding={
            "artifact_path": str(prompt_path.resolve()),
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
            "conditioning": conditioning_description(config.vae_input_mode, config.state_t),
            "official_resolution": "480",
            "official_positional_latent_max_h": 240,
            "official_positional_latent_max_w": 240,
            "bridge_lora": None,
            "extractor_input_keys": ["rgb_history", "prompt_embedding"],
            "excluded_from_extractor": ["state", "target_action"],
            "checkpoint_ignored_metadata_keys": list(extraction.provenance.checkpoint_ignored_metadata_keys),
            "checkpoint_ignored_metadata_count": extraction.provenance.checkpoint_ignored_metadata_count,
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
        raw_hidden_shape=tuple(raw_hidden.shape),
        raw_hidden_dtype=raw_hidden.dtype,
    )
    artifact = CosmosFeatureCacheArtifact(context, state, target_action, action_is_pad, provenance)
    save_feature_cache(artifact, output_path, overwrite=args.overwrite)
    reloaded = verify_feature_cache(output_path)

    print("Cosmos extractor first-real-forward smoke succeeded")
    print(f"  dataset:            {dataset.repo_id}@{dataset.revision}")
    print(f"  episode/frame:      {args.episode}/{prepared.frame_index}")
    print(f"  window_indices:     {list(prepared.window_indices)}")
    print(f"  raw_hidden_shape:   {tuple(raw_hidden.shape)}")
    print(f"  context_shape:      {tuple(reloaded.context.shape)}")
    print(f"  context_dtype:      {reloaded.context.dtype}")
    print(f"  load_seconds:       {load_seconds:.3f}")
    print(f"  prompt_load_seconds:{prompt_load_seconds:.3f}")
    print(f"  extraction_seconds: {timing.seconds:.3f}")
    print(f"  load_peak_vram:     allocated={load_peak[0]} reserved={load_peak[1]}")
    print(
        "  extraction_peak_vram:"
        f" allocated={timing.peak_allocated_bytes} reserved={timing.peak_reserved_bytes}"
    )
    print(f"  cache:              {output_path}")
    print(f"  cache_provenance:   {output_path.with_suffix('.json')}")
    print("  labels:             state=decoder conditioning; target_action=label only")
    return reloaded


def main(argv: list[str] | None = None) -> int:
    """Run the real forward smoke, returning a concise CLI error."""
    try:
        smoke(parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - smoke CLI should report a concise failure.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
