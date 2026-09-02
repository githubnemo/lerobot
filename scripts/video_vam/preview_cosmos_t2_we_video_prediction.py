#!/usr/bin/env python3
"""Generate a 61-frame video from a trained T=2 Cosmos WorldExpert.

The world expert is conditioned only on the Cosmos layer-20 features.  There
is deliberately no classifier-free guidance or prompt input in this sampler:
the text prompt is consumed once by the trunk before the future-latent loop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.cosmos_lora import inject_lora, load_lora_state_dict, merge_lora_file_into_base
from lerobot.policies.vam.cosmos_predict2_extractor import (
    CosmosPredict2Extractor,
    CosmosPredict2ExtractorConfig,
)
from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding
from lerobot.policies.vam.cosmos_video_prediction import (
    OfficialRectifiedFlowAB2Scheduler,
    cosmos_2b_backbone_spec,
    temporal_alignment,
)
from lerobot.policies.vam.world_expert import LinearWorldReadout, WorldExpert
from scripts.video_vam.preview_cosmos_video_prediction import (
    _decoded_to_uint8,
    _episode_row,
    _load_future_frames,
    _rgb_tensor_to_numpy,
    _write_contact_sheet,
    _write_pixel_path_probe,
)
from scripts.video_vam.smoke_test_cosmos_extractor import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PROMPT_PATH,
    DEFAULT_TOKENIZER_PATH,
    load_real_sample,
)

DEFAULT_OUTPUT_DIR = Path("/home/anton/.cache/video-vam/cosmos-t2-we-prediction-previews")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-index", type=int)
    parser.add_argument("--window-start", type=int)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--merge-lora-weights", type=Path)
    parser.add_argument("--expert-checkpoint")
    parser.add_argument("--vlm-config")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _strict_load_world_expert(world_expert: WorldExpert, path: Path) -> None:
    _strict_load_module(world_expert, path, "world expert")


def _strict_load_module(module: torch.nn.Module, path: Path, label: str) -> None:
    tensors = load_file(str(path), device="cpu")
    expected = module.state_dict()
    if set(tensors) != set(expected):
        raise ValueError(
            f"{label} checkpoint keys mismatch: missing={sorted(set(expected) - set(tensors))[:8]}, "
            f"unexpected={sorted(set(tensors) - set(expected))[:8]}"
        )
    for name, expected_tensor in expected.items():
        if tuple(tensors[name].shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"{label} tensor {name} has shape {tuple(tensors[name].shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )
    module.load_state_dict(tensors, strict=True)


def _load_metadata(run_dir: Path) -> dict[str, Any]:
    path = run_dir.expanduser().resolve() / "best.json"
    if not path.is_file():
        raise FileNotFoundError(f"world-expert sidecar not found: {path}")
    payload = json.loads(path.read_text())
    if payload.get("artifact") != "cosmos_t2_world_expert_training_checkpoint":
        raise ValueError(f"unsupported world-expert sidecar artifact: {path}")
    return payload


def _head_kind(metadata: dict[str, Any]) -> str:
    value = metadata.get("head", "expert")
    if isinstance(value, dict):
        value = value.get("type", "expert")
    if value not in {"expert", "linear"}:
        raise ValueError(f"unsupported head type in best.json: {value!r}")
    return str(value)


def _load_world_expert(
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> WorldExpert:
    config = metadata["world_expert"]["config"]
    checkpoint = args.expert_checkpoint or metadata["world_expert"]["checkpoint"]
    kwargs: dict[str, Any] = {
        "device": args.device,
        "latent_patch_size": int(config["latent_patch_size"]),
    }
    vlm_config = args.vlm_config or metadata["world_expert"].get("vlm_config")
    if vlm_config is not None:
        kwargs["vlm_config_name"] = vlm_config
    world_expert = WorldExpert.from_pretrained(checkpoint, **kwargs)
    _strict_load_world_expert(
        world_expert,
        args.run_dir.expanduser().resolve() / "best_world_expert.safetensors",
    )
    return world_expert.to(device=args.device, dtype=torch.bfloat16).eval()


def _load_linear_readout(
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> LinearWorldReadout:
    head_metadata = metadata.get("linear_readout", metadata.get("head"))
    if not isinstance(head_metadata, dict) or not isinstance(head_metadata.get("config"), dict):
        raise ValueError("linear readout sidecar is missing its config")
    config = head_metadata["config"]
    readout = LinearWorldReadout(
        feature_channels=int(config["feature_channels"]),
        feature_frames=int(config["feature_frames"]),
        token_height=int(config["token_height"]),
        token_width=int(config["token_width"]),
        latent_channels=int(config["latent_channels"]),
        latent_frames=int(config["latent_frames"]),
        latent_height=int(config["latent_height"]),
        latent_width=int(config["latent_width"]),
        latent_patch_size=int(config["latent_patch_size"]),
    ).to(device=args.device, dtype=torch.bfloat16)
    _strict_load_module(
        readout,
        args.run_dir.expanduser().resolve() / "best_linear_readout.safetensors",
        "linear readout",
    )
    return readout.eval()


def run_preview(args: argparse.Namespace) -> Path:
    if args.steps <= 0 or args.seed < 0:
        raise ValueError("--steps must be positive and --seed must be non-negative")
    run_dir = args.run_dir.expanduser().resolve()
    metadata = _load_metadata(run_dir)
    head_kind = _head_kind(metadata)
    merge_path = args.merge_lora_weights
    if merge_path is None:
        merge_path = Path(metadata["merged_old_lora"]["path"])
    new_lora = metadata["new_lora"]
    block_indices = tuple(int(index) for index in new_lora["block_indices"])
    if block_indices != tuple(range(20)):
        raise ValueError(f"preview requires new LoRA blocks 0-19, got {block_indices}")
    spec = cosmos_2b_backbone_spec(
        args.checkpoint,
        args.tokenizer,
        device=args.device,
        sampling_steps=args.steps,
        vae_input_mode="observed_prefix",
    )
    dataset, prepared = load_real_sample(
        root=args.root.expanduser(),
        episode_index=args.episode,
        frame_index=args.frame_index,
        window_start=args.window_start,
        config=CUBE_OUT_OF_BOX_CONTRACT,
    )
    alignment = temporal_alignment(
        current_frame_index=prepared.frame_index,
        input_frames=5,
        state_t=16,
        latent_conditional_frames=2,
        temporal_compression_factor=4,
    )
    episode = _episode_row(dataset, args.episode)
    episode_start = int(episode["dataset_from_index"])
    episode_end = int(episode["dataset_to_index"])
    if alignment.predicted_future_indices[-1] >= episode_end:
        raise ValueError(
            f"episode {args.episode} ends at {episode_end - 1}, but rollout needs "
            f"future frame {alignment.predicted_future_indices[-1]}"
        )
    if alignment.conditioning_indices[0] < episode_start:
        raise ValueError("conditioning window crossed the episode boundary")
    ground_truth = _load_future_frames(dataset, alignment.predicted_future_indices)
    prompt = load_prompt_embedding(args.prompt.expanduser()).embedding
    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=args.checkpoint.expanduser().resolve(),
            tokenizer_path=args.tokenizer.expanduser().resolve(),
            device=args.device,
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
    merge_info = merge_lora_file_into_base(extractor.backbone, merge_path)
    if merge_info["sha256"] != metadata["merged_old_lora"]["sha256"]:
        raise ValueError("loaded merged old-LoRA artifact does not match best.json")
    inject_lora(
        extractor.backbone,
        rank=int(new_lora["rank"]),
        alpha=float(new_lora["alpha"]),
        block_indices=block_indices,
    )
    load_lora_state_dict(extractor.backbone, run_dir / "best_lora.safetensors")
    extractor.backbone.eval()
    if head_kind == "expert":
        world_expert = _load_world_expert(metadata, args)
    else:
        linear_readout = _load_linear_readout(metadata, args)
    prompt = prompt.to(device=extractor.device, dtype=extractor.dtype)
    observed = prepared.rgb_history
    rollout_start = time.perf_counter()
    with torch.no_grad():
        extraction = extractor.extract(
            observed,
            prompt,
            noise_seed=args.seed,
            sigma=torch.full((1,), 80.0, device=extractor.device, dtype=torch.float32),
        )
        observed_normalized = extractor._preprocess_images(observed)
        observed_latents = extractor.encode_observed_pixels(observed_normalized)
        if head_kind == "linear":
            generated_future = linear_readout(extraction.tokens)
        else:
            prefix = world_expert.prepare_prefix_kv(extraction.tokens)
            scheduler = OfficialRectifiedFlowAB2Scheduler(
                sigma_min=0.002,
                sigma_max=80.0,
                order=7.0,
                steps=args.steps,
            )
            scheduler.set_timesteps(extractor.device)
            from lerobot.policies.vam.cosmos_predict2_extractor import arch_invariant_rand

            future_shape = (1, 16, 14, 60, 80)
            sample = (
                arch_invariant_rand(future_shape, args.seed).to(device=extractor.device, dtype=torch.float32)
                * scheduler.sigmas[0]
            )
            x0_prev: torch.Tensor | None = None
            assert scheduler.timesteps is not None and scheduler.sigmas is not None
            for index in range(args.steps):
                sigma = scheduler.sigmas[index].to(extractor.device, torch.float32).repeat(1)
                x0_pred = world_expert.predict_x0(sample, sigma, prefix)
                sample, x0_prev = scheduler.step(x0_pred, index, sample, x0_prev)
            sigma_min = scheduler.sigmas[-1].to(extractor.device, torch.float32).repeat(1)
            generated_future = world_expert.predict_x0(sample, sigma_min, prefix)
        full_latents = torch.cat([observed_latents, generated_future], dim=2)
        decoded = extractor.tokenizer.decode(full_latents).clamp(-1.0, 1.0)
        predicted_full = _decoded_to_uint8(decoded)
    rollout_seconds = time.perf_counter() - rollout_start
    predicted_future = predicted_full[spec.conditional_pixel_frames :]
    ground_truth_rgb = _rgb_tensor_to_numpy(ground_truth)
    output_dir = args.output_dir.expanduser().resolve()
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
    from lerobot.policies.vam.cosmos_video_prediction import compute_frame_metrics
    from lerobot.utils.io_utils import write_video

    side_by_side = np.concatenate([predicted_future, ground_truth_rgb], axis=2)
    write_video(output_paths["predicted_full"], list(predicted_full), spec.fps)
    write_video(output_paths["predicted_future"], list(predicted_future), spec.fps)
    write_video(output_paths["ground_truth_future"], list(ground_truth_rgb), spec.fps)
    write_video(output_paths["side_by_side"], list(side_by_side), spec.fps)
    _write_contact_sheet(
        output_paths["contact_sheet"], ground_truth_rgb, predicted_future, alignment.predicted_future_indices
    )
    pixel_path_metrics = _write_pixel_path_probe(
        output_paths["pixel_path_roundtrip"], ground_truth_rgb[0], spec.fps
    )
    predicted_tensor = torch.from_numpy(predicted_future).permute(0, 3, 1, 2).contiguous()
    target_tensor = torch.from_numpy(ground_truth_rgb).permute(0, 3, 1, 2).contiguous()
    metrics = compute_frame_metrics(
        predicted_tensor,
        target_tensor,
        frame_indices=alignment.predicted_future_indices,
    )
    report = {
        "artifact": "cosmos_t2_world_expert_video_prediction",
        "run_dir": str(run_dir),
        "episode": args.episode,
        "frame_index": prepared.frame_index,
        "seed": args.seed,
        "head": head_kind,
        "steps": args.steps,
        "guidance": 0.0,
        "cfg": "disabled; prompt is baked into T=2 trunk features",
        "alignment": {
            "conditioning_indices": list(alignment.conditioning_indices),
            "predicted_future_indices": list(alignment.predicted_future_indices),
        },
        "metrics": {**metrics, "pixel_path": pixel_path_metrics},
        "rollout_seconds": rollout_seconds,
        "artifacts": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"report": str(output_paths["report"]), "side_by_side": str(output_paths["side_by_side"])},
            indent=2,
        )
    )
    return output_paths["report"]


def main(argv: list[str] | None = None) -> int:
    run_preview(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
