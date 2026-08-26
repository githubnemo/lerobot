#!/usr/bin/env python3
"""Bounded online LTX-to-World2Action tiny-overfit research gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT, validate_metadata
from lerobot.policies.vam.action_rmse import aggregate_action_rmse
from lerobot.policies.vam.ltx_action import LTXContextAdapter, frame_mean_ltx_context
from lerobot.policies.vam.ltx_extractor import LTXExtractor, LTXExtractorConfig, derive_window_seed
from lerobot.policies.vam.ltx_prompt_embedding import load_ltx_prompt_artifact
from lerobot.policies.vam.world2action import ActionStateNormalizer, World2ActionConfig, World2ActionDecoder
from scripts.video_vam.smoke_test_cosmos_extractor import _relative_index, prepare_sample
from scripts.video_vam.train_cosmos_world2action import (
    autocast_context,
    build_optimizer,
    build_scheduler,
    convert_decoder_parameters_to_fp32,
)

DEFAULT_DATASET = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_TRANSFORMER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_VAE = Path("/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors")
DEFAULT_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)
DEFAULT_NORMALIZER = Path("/home/anton/.cache/video-vam/runs/vam-hour-k8/normalizer.safetensors")
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/runs/ltx25-tiny-overfit/result.json")


@dataclass(frozen=True, slots=True)
class Window:
    episode_index: int
    frame_index: int

    @property
    def sample_id(self) -> str:
        return f"episode-{self.episode_index:04d}-frame-{self.frame_index:06d}"


def parse_windows(value: str) -> tuple[Window, ...]:
    windows = []
    for item in value.split(","):
        try:
            episode, frame = (int(part) for part in item.strip().split(":", maxsplit=1))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("windows must use episode:frame comma syntax") from exc
        if episode < 0 or frame < 4:
            raise argparse.ArgumentTypeError("window episode must be non-negative and frame at least four")
        windows.append(Window(episode, frame))
    if not 2 <= len(windows) <= 16 or len(set(windows)) != len(windows):
        raise argparse.ArgumentTypeError("provide 2-16 distinct windows")
    return tuple(windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--transformer", type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument("--video-vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows", type=parse_windows, default=parse_windows("0:4,0:24,0:44,0:64"))
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-minutes", type=float, default=25.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--fit-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0 or args.max_minutes <= 0 or args.eval_every <= 0:
        raise ValueError("step/time/evaluation limits must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must be non-negative and below max steps")
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not 0 < args.fit_ratio < 1:
        raise ValueError("fit ratio must be in (0, 1)")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing existing result; pass --overwrite: {args.output}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_normalizer_source(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require the established all-anchor train-episode normalization provenance."""
    expected = {
        "anchor_count": 4688,
        "derivation": "all_valid_training_episode_anchors",
        "episodes": list(range(32)),
        "padded_actions_excluded": True,
    }
    if dict(payload) != expected:
        raise ValueError(f"normalizer provenance does not match the established Cosmos path: {payload}")
    return expected


def load_established_normalizer(path: Path) -> tuple[ActionStateNormalizer, dict[str, Any]]:
    path = path.expanduser().resolve()
    source_path = path.with_name("normalizer_source.json")
    if not source_path.is_file():
        raise FileNotFoundError(f"normalizer provenance is required: {source_path}")
    payload = json.loads(source_path.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("normalizer provenance must be a JSON object")
    return ActionStateNormalizer.load(path), validate_normalizer_source(payload)


def synchronize() -> None:
    torch.cuda.synchronize()


def load_windows(args: argparse.Namespace) -> tuple[LeRobotDataset, list[Any]]:
    episodes = sorted({window.episode_index for window in args.windows})
    dataset = LeRobotDataset(
        CUBE_OUT_OF_BOX_CONTRACT.repo_id,
        root=args.dataset_root.expanduser(),
        episodes=episodes,
        delta_timestamps=CUBE_OUT_OF_BOX_CONTRACT.delta_timestamps(),
        revision=CUBE_OUT_OF_BOX_CONTRACT.revision,
        return_uint8=True,
        download_videos=False,
    )
    validate_metadata(dataset.meta, CUBE_OUT_OF_BOX_CONTRACT).raise_if_invalid()
    rows = {
        int(row["episode_index"]): (int(row["dataset_from_index"]), int(row["dataset_to_index"]))
        for row in dataset.meta.episodes
    }
    prepared = []
    for window in args.windows:
        start, stop = rows[window.episode_index]
        if window.frame_index - 4 < start or window.frame_index >= stop:
            raise ValueError(f"causal window for {window.sample_id} is outside episode [{start}, {stop})")
        sample = dataset[_relative_index(dataset, window.frame_index)]
        item = prepare_sample(sample, frame_index=window.frame_index)
        if tuple(item.window_indices) != tuple(range(window.frame_index - 4, window.frame_index + 1)):
            raise RuntimeError("causal window indices are not the five real observed frames")
        prepared.append(item)
    return dataset, prepared


def extract_contexts(
    args: argparse.Namespace, prepared: list[Any]
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    prompt = load_ltx_prompt_artifact(args.prompt)
    config = LTXExtractorConfig(
        checkpoint_path=args.transformer,
        video_vae_path=args.video_vae,
        device="cuda",
        dtype="bfloat16",
        hidden_layer=34,
        high_noise_sigma=1.0,
        seed=args.seed,
        input_frames=5,
        padded_input_frames=9,
        offload_mode="cpu",
        quantization="fp8-cast",
        persistent_transformer=True,
    )
    torch.cuda.reset_peak_memory_stats()
    constructed = time.perf_counter()
    extractor = LTXExtractor(config)
    construct_seconds = time.perf_counter() - constructed
    if prompt.provenance.transformer_sha256 != extractor.checkpoint_sha256:
        raise ValueError("prompt artifact and extractor transformer hashes differ")
    records = []
    contexts = []
    open_seconds = extractor.open_transformer()
    try:
        for window, item in zip(args.windows, prepared, strict=True):
            timings: dict[str, float] = {}
            noise_seed = derive_window_seed(
                CUBE_OUT_OF_BOX_CONTRACT.revision,
                window.episode_index,
                window.frame_index,
                args.seed,
            )
            started = time.perf_counter()
            extraction = extractor.extract(
                item.rgb_history,
                prompt.embedding,
                noise_seed=noise_seed,
                timings=timings,
            )
            synchronize()
            elapsed = time.perf_counter() - started
            if extraction.provenance.prompt_source != "embedding":
                raise RuntimeError("LTX extraction did not consume the real prompt embedding artifact")
            if tuple(extraction.tokens.shape) != (1, 2400, 4096):
                raise RuntimeError(f"unexpected raw LTX context shape: {tuple(extraction.tokens.shape)}")
            frame_context = frame_mean_ltx_context(extraction.hidden_grid)
            contexts.append(frame_context.to(device="cpu", dtype=torch.bfloat16).contiguous())
            records.append(
                {
                    "sample_id": window.sample_id,
                    "episode_index": window.episode_index,
                    "frame_index": window.frame_index,
                    "window_indices": list(item.window_indices),
                    "noise_seed": extraction.provenance.noise_seed,
                    "raw_shape": list(extraction.tokens.shape),
                    "raw_grid_shape": list(extraction.hidden_grid.shape),
                    "frame_mean_shape": list(frame_context.shape),
                    "seconds": elapsed,
                    "stage_seconds": timings,
                }
            )
    finally:
        extractor.close()
    peak = torch.cuda.max_memory_allocated()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    return (
        torch.cat(contexts, dim=0),
        records,
        {
            "construct_seconds": construct_seconds,
            "transformer_open_seconds": open_seconds,
            "peak_vram_bytes": peak,
            "prompt_path": str(args.prompt.expanduser().resolve()),
            "prompt_shape": list(prompt.embedding.shape),
            "prompt_dtype": str(prompt.embedding.dtype).removeprefix("torch."),
            "prompt_embedding_sha256": prompt.provenance.embedding_sha256,
            "prompt_valid_tokens": prompt.provenance.valid_tokens,
            "source_commit": prompt.provenance.source_commit,
            "model_revision": prompt.provenance.model_revision,
            "ephemeral_reuse": "frame-mean LTX features retained only in process memory",
        },
    )


def tensors(
    prepared: list[Any], indices: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state = torch.cat([prepared[index].state for index in indices]).to(device=device, dtype=torch.float32)
    action = torch.cat([prepared[index].target_action for index in indices]).to(
        device=device, dtype=torch.float32
    )
    padding = torch.cat([prepared[index].action_is_pad for index in indices]).to(device=device)
    return state, action, padding


def masked_normalized_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding: torch.Tensor,
    normalizer: ActionStateNormalizer,
) -> float:
    pred = normalizer.normalize_action(prediction)
    truth = normalizer.normalize_action(target)
    valid = (~padding).unsqueeze(-1).expand_as(pred)
    return float(torch.sqrt(((pred.float() - truth.float()).square() * valid).sum() / valid.sum()).item())


def fixed_flow_loss(
    decoder: World2ActionDecoder,
    adapter: LTXContextAdapter,
    state: torch.Tensor,
    action: torch.Tensor,
    padding: torch.Tensor,
    raw_context: torch.Tensor,
    tau: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    backward: bool = False,
) -> torch.Tensor:
    with autocast_context(state.device):
        context = adapter(raw_context)
        loss = decoder.flow_matching_loss(
            state,
            action,
            context,
            t=tau,
            epsilon=epsilon,
            action_is_pad=padding,
            context_timestep=torch.ones((state.shape[0], 1), device=state.device),
            obs_dropout=0.0,
            detach_context=not backward,
        )
    return loss


def evaluate_actions(
    decoder: World2ActionDecoder,
    adapter: LTXContextAdapter,
    state: torch.Tensor,
    action: torch.Tensor,
    padding: torch.Tensor,
    raw_context: torch.Tensor,
    normalizer: ActionStateNormalizer,
    *,
    seed: int,
) -> dict[str, Any]:
    decoder.eval()
    adapter.eval()
    with torch.no_grad(), autocast_context(state.device):
        context = adapter(raw_context)
        prediction = decoder.sample_actions(
            state,
            context,
            seed=seed,
            context_timestep=torch.ones((state.shape[0], 1), device=state.device),
        )
    physical = aggregate_action_rmse(prediction, action, padding)
    return {
        "aggregate_rmse_deg": physical["aggregate_rmse_deg"],
        "per_joint_rmse_deg": physical["per_joint_rmse_deg"],
        "normalized_rmse": masked_normalized_rmse(prediction, action, padding, normalizer),
    }


def parameter_probe(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = [parameter.detach().flatten()[:1024].float().cpu() for parameter in parameters[:8]]
    return torch.cat(pieces)


def train_gate(
    args: argparse.Namespace,
    prepared: list[Any],
    contexts_cpu: torch.Tensor,
) -> dict[str, Any]:
    device = torch.device("cuda")
    train_count = len(prepared) - 1 if len(prepared) >= 4 else len(prepared)
    train_indices = list(range(train_count))
    heldout_indices = list(range(train_count, len(prepared)))
    normalizer, normalizer_source = load_established_normalizer(args.normalizer)
    adapter = LTXContextAdapter().to(device)
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cuda", dtype=torch.bfloat16), normalizer=normalizer
    )
    convert_decoder_parameters_to_fp32(decoder)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in decoder.denoiser.parameters() if parameter.requires_grad)
    optimizer, optimizer_settings = build_optimizer(
        parameters, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    scheduler, scheduler_settings = build_scheduler(optimizer, args.max_steps, args.warmup_steps)
    state, action, padding = tensors(prepared, train_indices, device)
    raw_context = contexts_cpu[train_indices].to(device=device, dtype=torch.bfloat16)
    heldout = tensors(prepared, heldout_indices, device) if heldout_indices else None
    heldout_context = (
        contexts_cpu[heldout_indices].to(device=device, dtype=torch.bfloat16) if heldout_indices else None
    )
    fixed_generator = torch.Generator(device=device).manual_seed(args.seed + 7001)
    tau = torch.rand(len(train_indices), device=device, generator=fixed_generator) * 0.999 + 0.001
    epsilon = torch.randn(action.shape, device=device, dtype=action.dtype, generator=fixed_generator)
    train_generator = torch.Generator(device=device).manual_seed(args.seed)
    initial_probe = parameter_probe(parameters)
    torch.cuda.reset_peak_memory_stats()

    decoder.eval()
    adapter.eval()
    with torch.no_grad():
        initial_fixed = float(
            fixed_flow_loss(decoder, adapter, state, action, padding, raw_context, tau, epsilon)
            .float()
            .item()
        )
    initial_action = evaluate_actions(
        decoder, adapter, state, action, padding, raw_context, normalizer, seed=args.seed + 99
    )

    optimizer.zero_grad(set_to_none=True)
    smoke_loss = fixed_flow_loss(
        decoder,
        adapter,
        state[:1],
        action[:1],
        padding[:1],
        raw_context[:1],
        tau[:1],
        epsilon[:1],
        backward=True,
    )
    smoke_loss.backward()

    def gradient_norm(module: torch.nn.Module) -> tuple[float, int]:
        gradients = [
            parameter.grad.float() for parameter in module.parameters() if parameter.grad is not None
        ]
        if not gradients:
            return 0.0, 0
        squared = sum(float(gradient.square().sum().item()) for gradient in gradients)
        nonzero = sum(int(torch.count_nonzero(gradient).item()) for gradient in gradients)
        return math.sqrt(squared), nonzero

    adapter_grad, adapter_grad_nonzero = gradient_norm(adapter)
    decoder_grad, decoder_grad_nonzero = gradient_norm(decoder.denoiser)
    print(
        f"consumer smoke gradients: adapter={adapter_grad:.6g}/{adapter_grad_nonzero} nonzero, "
        f"decoder={decoder_grad:.6g}/{decoder_grad_nonzero} nonzero",
        flush=True,
    )
    if not all(math.isfinite(value) for value in (adapter_grad, decoder_grad)) or decoder_grad <= 0:
        raise RuntimeError("consumer smoke produced no finite decoder gradients")
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()
    history = []
    fit_demonstrated = False
    stop_reason = "max_steps"
    completed_steps = 0
    last_grad_norm = 0.0
    for step in range(1, args.max_steps + 1):
        if time.perf_counter() - started >= args.max_minutes * 60:
            stop_reason = "max_minutes"
            break
        decoder.train()
        adapter.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            context = adapter(raw_context)
            loss = decoder.flow_matching_loss(
                state,
                action,
                context,
                action_is_pad=padding,
                context_timestep=torch.ones((state.shape[0], 1), device=device),
                generator=train_generator,
                detach_context=False,
            )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("training loss is non-finite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        last_grad_norm = float(grad_norm.float().item())
        optimizer.step()
        scheduler.step()
        completed_steps = step
        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            decoder.eval()
            adapter.eval()
            with torch.no_grad():
                fixed = float(
                    fixed_flow_loss(decoder, adapter, state, action, padding, raw_context, tau, epsilon)
                    .float()
                    .item()
                )
            record = {
                "step": step,
                "stochastic_loss": float(loss.detach().float().item()),
                "fixed_flow_loss": fixed,
                "fixed_loss_ratio": fixed / initial_fixed,
                "grad_norm": last_grad_norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if step >= args.eval_every and fixed <= initial_fixed * args.fit_ratio:
                fit_demonstrated = True
                stop_reason = "fit_ratio"
                break

    wall_seconds = time.perf_counter() - started
    decoder.eval()
    adapter.eval()
    with torch.no_grad():
        final_fixed = float(
            fixed_flow_loss(decoder, adapter, state, action, padding, raw_context, tau, epsilon)
            .float()
            .item()
        )
    final_action = evaluate_actions(
        decoder, adapter, state, action, padding, raw_context, normalizer, seed=args.seed + 99
    )
    zero_action = evaluate_actions(
        decoder,
        adapter,
        state,
        action,
        padding,
        torch.zeros_like(raw_context),
        normalizer,
        seed=args.seed + 99,
    )
    shuffled_action = evaluate_actions(
        decoder,
        adapter,
        state,
        action,
        padding,
        raw_context.roll(1, dims=0),
        normalizer,
        seed=args.seed + 99,
    )
    heldout_metrics = None
    if heldout is not None and heldout_context is not None:
        heldout_metrics = evaluate_actions(
            decoder,
            adapter,
            *heldout,
            heldout_context,
            normalizer,
            seed=args.seed + 199,
        )
    final_probe = parameter_probe(parameters)
    movement = final_probe - initial_probe
    result = {
        "fit_demonstrated": fit_demonstrated,
        "stop_reason": stop_reason,
        "steps": completed_steps,
        "wall_seconds": wall_seconds,
        "steps_per_second": completed_steps / wall_seconds if wall_seconds else 0.0,
        "initial_fixed_flow_loss": initial_fixed,
        "final_fixed_flow_loss": final_fixed,
        "fixed_loss_ratio": final_fixed / initial_fixed,
        "initial_train_action": initial_action,
        "final_train_action": final_action,
        "final_zero_context_action": zero_action,
        "final_shuffled_context_action": shuffled_action,
        "heldout_action": heldout_metrics,
        "heldout_is_generalization_evidence": False,
        "consumer_smoke": {
            "loss": float(smoke_loss.detach().float().item()),
            "adapter_grad_norm": adapter_grad,
            "adapter_grad_nonzero": adapter_grad_nonzero,
            "decoder_grad_norm": decoder_grad,
            "decoder_grad_nonzero": decoder_grad_nonzero,
        },
        "parameter_probe_l2_movement": float(movement.norm().item()),
        "parameter_probe_max_abs_movement": float(movement.abs().max().item()),
        "last_grad_norm": last_grad_norm,
        "peak_training_vram_bytes": torch.cuda.max_memory_allocated(),
        "train_indices": train_indices,
        "heldout_indices": heldout_indices,
        "history": history,
        "optimizer": optimizer_settings,
        "scheduler": scheduler_settings,
        "normalizer_path": str(args.normalizer.expanduser().resolve()),
        "normalizer_source": normalizer_source,
        "action_flow": "World2Action Beta(1,1), masked 30x6 action target, state conditioned",
        "context_timestep": 1.0,
        "adapter": {
            "input_shape": [len(prepared), 8, 4096],
            "output_shape": [len(prepared), 8, 2048],
            "reduction": "spatial mean over each 15x20 LTX latent frame",
            "projection": "non-affine LayerNorm plus trainable bias-free Linear(4096, 2048)",
        },
    }
    del optimizer, scheduler, decoder, adapter, parameters
    del state, action, padding, raw_context, tau, epsilon, smoke_loss, loss, context
    if heldout is not None:
        del heldout
    if heldout_context is not None:
        del heldout_context
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("the real LTX tiny-overfit gate requires CUDA")
    set_seed(args.seed)
    overall_started = time.perf_counter()
    dataset, prepared = load_windows(args)
    contexts, extraction_records, extraction_summary = extract_contexts(args, prepared)
    extraction_seconds = sum(record["seconds"] for record in extraction_records)
    steady_samples = [record["seconds"] for record in extraction_records[1:]] or [
        extraction_records[0]["seconds"]
    ]
    steady_p50 = statistics.median(steady_samples)
    print(
        f"LTX extraction complete: {len(extraction_records)} windows, "
        f"steady p50 {steady_p50:.3f}s ({1.0 / steady_p50:.3f} Hz)",
        flush=True,
    )
    training = train_gate(args, prepared, contexts)
    result = {
        "experiment": "ltx25_online_tiny_overfit",
        "research_only": True,
        "seed": args.seed,
        "dataset": {
            "repo_id": dataset.repo_id,
            "revision": dataset.revision,
            "root": str(args.dataset_root.expanduser().resolve()),
            "task": CUBE_OUT_OF_BOX_CONTRACT.task,
            "fps": CUBE_OUT_OF_BOX_CONTRACT.fps,
        },
        "windows": extraction_records,
        "extraction": {
            **extraction_summary,
            "window_count": len(extraction_records),
            "all_window_average_seconds": extraction_seconds / len(extraction_records),
            "warmup_window_seconds": extraction_records[0]["seconds"],
            "steady_window_seconds_p50": steady_p50,
            "steady_hz": 1.0 / steady_p50,
            "hard_runtime_target_hz": 10.0,
            "meets_runtime_target": False,
        },
        "training": training,
        "overall_wall_seconds": time.perf_counter() - overall_started,
        "interpretation": (
            "Training-fit and context-sensitivity evidence only; the one fixed held-out window is too small "
            "to support a generalization claim."
        ),
        "artifacts": {
            "feature_cache_written": False,
            "checkpoint_written": False,
            "result_json_only": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT_JSON={args.output}", flush=True)
    print(f"FIT_DEMONSTRATED={training['fit_demonstrated']}", flush=True)
    print("PROCESS_CLEANUP_COMPLETE=True", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
