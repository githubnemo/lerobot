#!/usr/bin/env python3
"""Preview and score full Cosmos Video2World predictions on cube episodes."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_predict2_extractor import VAE_INPUT_MODE_OBSERVED_PREFIX, VAE_INPUT_MODES
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.cosmos_video_prediction import (
    CosmosVideo2WorldBackend,
    build_prediction_report,
    compute_baseline_metrics,
    compute_frame_metrics,
    compute_optional_lpips,
    cosmos_2b_backbone_spec,
    file_sha256,
    set_deterministic_seed,
    temporal_alignment,
    write_strict_json,
)

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
DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/cosmos-prediction-previews")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument(
        "--vae-input-mode",
        choices=VAE_INPUT_MODES,
        default=VAE_INPUT_MODE_OBSERVED_PREFIX,
        help=(
            "VAE input contract: observed_prefix encodes only real observed pixels (default); "
            "legacy_padded_vae restores 5->61 padding for compatibility/debugging."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _episode_row(dataset: Any, episode_index: int) -> Any:
    try:
        return dataset.meta.episodes[episode_index]
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(f"episode {episode_index} is unavailable") from exc


def _relative_index(dataset: Any, absolute_index: int) -> int:
    mapping = dataset.absolute_to_relative_idx
    return absolute_index if mapping is None else int(mapping[absolute_index])


def _load_future_frames(dataset: Any, frame_indices: tuple[int, ...]) -> torch.Tensor:
    """Read current-frame RGB values while retaining the pinned dataset loader."""
    frames: list[torch.Tensor] = []
    for absolute_index in frame_indices:
        sample = dataset[_relative_index(dataset, absolute_index)]
        camera = sample[CUBE_OUT_OF_BOX_CONTRACT.camera_key]
        if not isinstance(camera, torch.Tensor):
            raise TypeError("dataset camera sample must be a torch.Tensor")
        if camera.ndim == 4:
            frame = camera[-1]
        elif camera.ndim == 3:
            frame = camera
        else:
            raise ValueError(f"dataset camera sample must be TCHW or CHW, got {tuple(camera.shape)}")
        if tuple(frame.shape) != (3, 480, 640) or frame.dtype != torch.uint8:
            raise ValueError(
                f"dataset RGB frame must be uint8 [3, 480, 640], got {tuple(frame.shape)} {frame.dtype}"
            )
        frames.append(frame.contiguous())
    return torch.stack(frames)


def _decoded_to_uint8(decoded: torch.Tensor) -> np.ndarray:
    if decoded.ndim != 5 or decoded.shape[0] != 1:
        raise ValueError(f"decoded video must have shape [1, C, T, H, W], got {tuple(decoded.shape)}")
    return (
        ((decoded[0].float().clamp(-1, 1) + 1.0) * 127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


def _rgb_tensor_to_numpy(frames: torch.Tensor) -> np.ndarray:
    return frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _vae_round_trip(
    backend: CosmosVideo2WorldBackend, ground_truth: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    """Encode/decode the future with only the Cosmos VAE."""
    original_count = ground_truth.shape[0]
    padding = (-(original_count - 1)) % backend.spec.temporal_compression_factor
    if padding:
        padded = torch.cat([ground_truth, ground_truth[-1:].expand(padding, -1, -1, -1)], dim=0)
    else:
        padded = ground_truth
    video_bcthw = padded.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()
    video_bcthw = video_bcthw.to(backend.device, backend.dtype) / 127.5 - 1.0
    tokenizer = backend.extractor.tokenizer
    if tokenizer is None:
        raise RuntimeError("Cosmos extractor tokenizer is not loaded")
    with torch.no_grad():
        latent = tokenizer.encode(video_bcthw)
        decoded = _decoded_to_uint8(backend.decode_latent(latent))[:original_count]
    reconstructed = torch.from_numpy(decoded).permute(0, 3, 1, 2).contiguous()
    result = compute_frame_metrics(reconstructed, ground_truth)
    result.update(
        {
            "input_frame_count": original_count,
            "padded_input_frame_count": int(padded.shape[0]),
            "latent_frame_count": int(latent.shape[2]),
            "decoded_frame_count": int(decoded.shape[0]),
            "scored_frame_count": original_count,
            "padding_frames_repeated": padding,
        }
    )
    return result, reconstructed


def _write_pixel_path_probe(path: Path, source_rgb: np.ndarray, fps: int) -> dict[str, Any]:
    """Write/read one RGB frame through the MP4 path and score the result."""
    import av

    from lerobot.utils.io_utils import write_video

    write_video(path, [source_rgb], fps)
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0)).to_ndarray(format="rgb24")
    source = torch.from_numpy(source_rgb).permute(2, 0, 1).unsqueeze(0).contiguous()
    round_trip = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).contiguous()
    result = compute_frame_metrics(round_trip, source, frame_indices=(0,))
    result.update(
        {
            "source_layout": "HWC RGB uint8 [0,255]",
            "writer_format": "PyAV rgb24 -> libx264 yuv420p",
            "reader_format": "PyAV rgb24 -> HWC RGB uint8 [0,255]",
            "source_shape": list(source_rgb.shape),
            "read_shape": list(frame.shape),
            "max_absolute_channel_error": int(
                np.abs(frame.astype(np.int16) - source_rgb.astype(np.int16)).max()
            ),
            "vertical_flip": False,
            "resize": False,
        }
    )
    return result


def _write_contact_sheet(
    path: Path, ground_truth: np.ndarray, predicted: np.ndarray, frame_indices: tuple[int, ...]
) -> None:
    """Write GT-over-predicted evenly sampled contact sheet with burned-in indices."""
    from PIL import Image, ImageDraw

    positions = np.linspace(0, len(frame_indices) - 1, num=min(8, len(frame_indices)), dtype=int)
    tile_width, tile_height, label_height = 200, 150, 24
    sheet = Image.new("RGB", (len(positions) * tile_width, 2 * (tile_height + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for column, position in enumerate(positions):
        x = column * tile_width
        for row, (frames, label) in enumerate(((ground_truth, "GT"), (predicted, "Pred"))):
            image = Image.fromarray(frames[position], mode="RGB").resize(
                (tile_width, tile_height), Image.Resampling.BILINEAR
            )
            y = row * (tile_height + label_height)
            sheet.paste(image, (x, y))
            draw.rectangle((x, y + tile_height, x + tile_width, y + tile_height + label_height), fill="black")
            draw.text((x + 4, y + tile_height + 4), f"{label} frame={frame_indices[position]}", fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, format="PNG")


def _alignment_probe(
    backend: CosmosVideo2WorldBackend, condition: Any, rgb_history: torch.Tensor
) -> tuple[dict[str, Any], np.ndarray]:
    """Empirically check the VAE frame-zero convention before rollout."""
    with torch.no_grad():
        reconstruction = _decoded_to_uint8(backend.decode_latent(condition.conditional_latent))
    input_rgb = rgb_history[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
    candidate_offsets: dict[str, float] = {}
    for offset in range(max(1, reconstruction.shape[0] - input_rgb.shape[0] + 1)):
        if offset > 8:
            break
        candidate_offsets[str(offset)] = float(
            np.mean(
                (
                    reconstruction[offset : offset + input_rgb.shape[0]].astype(np.float32)
                    - input_rgb.astype(np.float32)
                )
                ** 2
            )
        )
    expected_mse = candidate_offsets["0"]
    best_offset = min(candidate_offsets, key=lambda offset: candidate_offsets[offset])
    return {
        "method": "encode five RGB frames to two clean latents, zero-pad to state_t, decode",
        "decoded_frame_count": int(reconstruction.shape[0]),
        "expected_decoded_frame_count": int(backend.spec.pixel_frames),
        "expected_offset": 0,
        "expected_offset_mse_uint8": expected_mse,
        "best_offset_among_0_to_8": int(best_offset),
        "best_offset_mse_uint8": candidate_offsets[best_offset],
        "candidate_offset_mse_uint8": candidate_offsets,
        "verified_expected_frame_zero": reconstruction.shape[0] == backend.spec.pixel_frames,
    }, reconstruction


def _runtime_details(
    *,
    clip_seconds: float,
    rollout_seconds: float,
    load_seconds: float,
    peak_allocated: int,
    peak_reserved: int,
) -> dict[str, Any]:
    return {
        "clip_seconds": clip_seconds,
        "rollout_seconds": rollout_seconds,
        "model_load_seconds": load_seconds,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "peak_vram_allocated_gib": peak_allocated / 1024**3,
        "peak_vram_reserved_gib": peak_reserved / 1024**3,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unavailable",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python_version": platform.python_version(),
    }


def _git_commit() -> str:
    try:
        # Fixed argv; resolving git from PATH is intentional, like the rest of the repo tooling.
        result = subprocess.run(  # nosec B607
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def run_preview(args: argparse.Namespace) -> Path:
    set_deterministic_seed(args.seed)
    clip_start = time.perf_counter()
    spec = cosmos_2b_backbone_spec(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        guidance=args.guidance,
        sampling_steps=args.steps,
        vae_input_mode=args.vae_input_mode,
    )
    dataset, prepared = _load_prepared_sample(args)
    alignment = temporal_alignment(
        current_frame_index=prepared.frame_index,
        input_frames=spec.input_frames,
        state_t=spec.state_t,
        latent_conditional_frames=spec.latent_conditional_frames,
        temporal_compression_factor=spec.temporal_compression_factor,
    )
    episode = _episode_row(dataset, args.episode)
    episode_start = int(episode["dataset_from_index"])
    episode_end = int(episode["dataset_to_index"])
    if alignment.predicted_future_indices[-1] >= episode_end:
        raise ValueError(
            f"episode {args.episode} ends at {episode_end - 1}, but this 61-pixel-frame rollout "
            f"needs future frame {alignment.predicted_future_indices[-1]}"
        )
    if alignment.conditioning_indices[0] < episode_start:
        raise ValueError("conditioning window crossed the episode boundary")

    ground_truth = _load_future_frames(dataset, alignment.predicted_future_indices)
    prompt_artifact = load_prompt_embedding(args.prompt.expanduser())
    checkpoint_info = file_sha256(spec.checkpoint_path)

    load_start = time.perf_counter()
    backend = CosmosVideo2WorldBackend(spec, seed=args.seed)
    load_seconds = time.perf_counter() - load_start
    condition = backend.prepare_conditioning(prepared.rgb_history, prompt_artifact.embedding)
    alignment_probe, conditioning_reconstruction_rgb = _alignment_probe(
        backend, condition, prepared.rgb_history
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(backend.device)
    rollout_start = time.perf_counter()
    predicted_full = _decoded_to_uint8(backend.rollout(condition, seed=args.seed))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated(backend.device)
        peak_reserved = torch.cuda.max_memory_reserved(backend.device)
    else:
        peak_allocated = peak_reserved = 0
    rollout_seconds = time.perf_counter() - rollout_start

    predicted_future = (
        torch.from_numpy(predicted_full[spec.conditional_pixel_frames :]).permute(0, 3, 1, 2).contiguous()
    )
    ground_truth_float = ground_truth
    metric_result = compute_frame_metrics(
        predicted_future,
        ground_truth_float,
        frame_indices=alignment.predicted_future_indices,
    )
    conditioning_reconstruction = (
        torch.from_numpy(conditioning_reconstruction_rgb[: spec.conditional_pixel_frames])
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    conditioning_ground_truth = prepared.rgb_history[0].permute(1, 0, 2, 3).contiguous()
    conditioning_metrics = compute_frame_metrics(
        conditioning_reconstruction,
        conditioning_ground_truth,
        frame_indices=alignment.conditioning_indices,
    )
    baseline_metrics = compute_baseline_metrics(
        conditioning_ground_truth,
        ground_truth_float,
        frame_indices=alignment.predicted_future_indices,
    )
    vae_roundtrip_metrics, _ = _vae_round_trip(backend, ground_truth_float)
    lpips_result = compute_optional_lpips(
        predicted_future.float() / 255.0,
        ground_truth_float.float() / 255.0,
    )
    metrics = {
        **metric_result,
        "lpips": lpips_result,
        "conditioning_reconstruction": conditioning_metrics,
        "baselines": baseline_metrics,
        "vae_roundtrip": vae_roundtrip_metrics,
    }

    output_dir = args.output_dir.expanduser()
    stem = f"episode-{args.episode:04d}-frame-{prepared.frame_index:06d}"
    output_paths = {
        "predicted_full": output_dir / f"{stem}-predicted-full.mp4",
        "predicted_future": output_dir / f"{stem}-predicted-future.mp4",
        "ground_truth_future": output_dir / f"{stem}-ground-truth-future.mp4",
        "side_by_side": output_dir / f"{stem}-side-by-side.mp4",
        "contact_sheet": output_dir / f"{stem}-contact-sheet.png",
        "pixel_path_roundtrip": output_dir / f"{stem}-pixel-path-roundtrip.mp4",
        "report": output_dir / f"{stem}.json",
    }
    if not args.overwrite:
        existing = [str(path) for path in output_paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"outputs already exist; pass --overwrite: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    from lerobot.utils.io_utils import write_video

    predicted_future_rgb = predicted_full[spec.conditional_pixel_frames :]
    ground_truth_rgb = _rgb_tensor_to_numpy(ground_truth)
    side_by_side = np.concatenate([predicted_future_rgb, ground_truth_rgb], axis=2)
    write_video(output_paths["predicted_full"], list(predicted_full), spec.fps)
    write_video(output_paths["predicted_future"], list(predicted_future_rgb), spec.fps)
    write_video(output_paths["ground_truth_future"], list(ground_truth_rgb), spec.fps)
    write_video(output_paths["side_by_side"], list(side_by_side), spec.fps)
    _write_contact_sheet(
        output_paths["contact_sheet"],
        ground_truth_rgb,
        predicted_future_rgb,
        alignment.predicted_future_indices,
    )
    pixel_path_metrics = _write_pixel_path_probe(
        output_paths["pixel_path_roundtrip"], ground_truth_rgb[0], spec.fps
    )
    metrics["pixel_path"] = pixel_path_metrics
    metrics["pixel_path_contract"] = {
        "dataset_load": "LeRobot return_uint8 camera TCHW RGB [0,255]",
        "conditioning_reorder": "TCHW -> BCTHW, no channel swap",
        "vae_input": "BCTHW BF16 [-1,1], native 480x640, no resize or flip",
        "vae_output": "BCTHW [-1,1] -> uint8 [0,255] -> THWC RGB",
        "mp4_writer": "THWC RGB uint8 -> PyAV rgb24 -> libx264 yuv420p",
        "mp4_reader": "PyAV rgb24 -> THWC RGB uint8",
        "vertical_flip": False,
        "resize": False,
    }

    runtime = _runtime_details(
        clip_seconds=time.perf_counter() - clip_start,
        rollout_seconds=rollout_seconds,
        load_seconds=load_seconds,
        peak_allocated=peak_allocated,
        peak_reserved=peak_reserved,
    )
    sampler = {
        "name": "RectifiedFlowAB2Scheduler",
        "steps": spec.sampling_steps,
        "sigma_min": spec.sigma_min,
        "sigma_max": spec.sigma_max,
        "schedule_order": spec.schedule_order,
        "guidance": spec.guidance,
        "negative_prompt": "",
        "cfg_note": "official empty-negative-prompt path makes conditional and unconditional text embeddings identical",
        "final_clean_pass": True,
    }
    report = build_prediction_report(
        spec=spec,
        seed=args.seed,
        checkpoint=checkpoint_info,
        episode_index=args.episode,
        current_frame_index=prepared.frame_index,
        alignment=alignment,
        sampler=sampler,
        metrics=metrics,
        runtime=runtime,
        artifacts={key: str(path) for key, path in output_paths.items()},
        alignment_probe=alignment_probe,
    )
    report["runtime"]["repo_commit"] = _git_commit()
    write_strict_json(report, output_paths["report"])
    print(
        json.dumps(
            {
                "report": str(output_paths["report"]),
                "predicted_full": str(output_paths["predicted_full"]),
                "predicted_future": str(output_paths["predicted_future"]),
                "ground_truth_future": str(output_paths["ground_truth_future"]),
                "side_by_side": str(output_paths["side_by_side"]),
                "contact_sheet": str(output_paths["contact_sheet"]),
                "pixel_path_roundtrip": str(output_paths["pixel_path_roundtrip"]),
                "mean_psnr_db": metrics["mean_psnr_db"],
                "mean_ssim": metrics["mean_ssim"],
                "lpips": lpips_result,
                "rollout_seconds": rollout_seconds,
                "clip_seconds": runtime["clip_seconds"],
                "peak_vram_allocated_gib": runtime["peak_vram_allocated_gib"],
                "peak_vram_reserved_gib": runtime["peak_vram_reserved_gib"],
                "alignment_probe": alignment_probe,
            },
            indent=2,
        )
    )
    return output_paths["report"]


def _load_prepared_sample(args: argparse.Namespace) -> tuple[Any, Any]:
    from smoke_test_cosmos_extractor import load_real_sample

    return load_real_sample(
        root=args.root.expanduser(),
        episode_index=args.episode,
        frame_index=args.frame_index,
        window_start=args.window_start,
        config=CUBE_OUT_OF_BOX_CONTRACT,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_preview(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
