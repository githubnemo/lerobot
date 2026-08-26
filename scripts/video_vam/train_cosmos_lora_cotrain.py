#!/usr/bin/env python3
"""Jointly train Cosmos Predict2 LoRA adapters and the World2Action decoder."""

from __future__ import annotations

import argparse
import copy
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
from collections.abc import Iterable, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save_file

from lerobot.datasets import LeRobotDataset
from lerobot.policies.vam.context_transform import CONTEXT_TRANSFORMS, apply_context_transform
from lerobot.policies.vam.cosmos_cache_dataset import (
    CacheManifest,
    CacheManifestEntry,
    load_cache_manifest,
)
from lerobot.policies.vam.cosmos_lora import (
    FULL_CLIP_LENGTH,
    LATENT_FRAMES,
    action_context_for_arm,
    build_cosmos_noised_input,
    capture_prediction_and_layer20,
    full_clip_delta_timestamps,
    inject_lora,
    latent_valid_mask,
    load_lora_state_dict,
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
from lerobot.policies.vam.vam_split import get_train_entries, get_val_entries, load_vam_split
from lerobot.policies.vam.world2action import ActionStateNormalizer, World2ActionConfig, World2ActionDecoder
from scripts.video_vam.train_cosmos_world2action import (
    EarlyStopping,
    SafeWandbLogger,
    autocast_context,
    build_scheduler,
    compute_training_normalizer_from_entries,
    set_reproducible_seeds,
)

DEFAULT_MAX_STEPS = 500_000
DEFAULT_MAX_HOURS = 8.0
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_FLOW_DRAWS = 8
DEFAULT_VAL_EVERY = 100
DEFAULT_SAVE_EVERY = 100
DEFAULT_PATIENCE = 20
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_LR = 1.0e-4
DEFAULT_DECODER_LR = 1.0e-4
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_LAMBDA_VIDEO = 0.5
DEFAULT_EVAL_SIGMA = 80.0
DEFAULT_WANDB_PROJECT = "video-vam-world2action"
DEFAULT_DATASET_ROOT = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")
DEFAULT_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_PROMPT = Path("/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors")


class LatentCache:
    """Store backbone-independent full-clip VAE latents per anchor."""

    def __init__(self, root: Path | None, *, overwrite: bool = False) -> None:
        self.root = None if root is None else root.expanduser().resolve()
        self.overwrite = overwrite
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, entry: CacheManifestEntry) -> Path:
        if self.root is None:
            raise RuntimeError("latent cache is disabled")
        return self.root / f"episode-{entry.episode_index:04d}-frame-{entry.frame_index:06d}.safetensors"

    def load(
        self, entry: CacheManifestEntry, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.root is None:
            return None
        path = self.path_for(entry)
        if not path.is_file():
            return None
        tensors = load_file(str(path), device="cpu")
        if set(tensors) != {"latents", "latent_valid"}:
            raise ValueError(f"latent cache keys are malformed: {path}")
        latents = tensors["latents"]
        valid = tensors["latent_valid"]
        if tuple(latents.shape) != (1, 16, LATENT_FRAMES, 60, 80) or latents.dtype != torch.bfloat16:
            raise ValueError(f"latent cache latents have an invalid shape/dtype: {path}")
        if tuple(valid.shape) != (1, LATENT_FRAMES) or valid.dtype != torch.bool:
            raise ValueError(f"latent cache latent_valid has an invalid shape/dtype: {path}")
        return latents.to(device=device), valid.to(device=device)

    def save(
        self,
        entry: CacheManifestEntry,
        latents: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        if self.root is None:
            return
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
        sidecar = path.with_suffix(".json")
        sidecar.write_text(
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
    """Parse the explicit training and provenance contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latent-cache-dir", type=Path)
    parser.add_argument("--precompute-latents-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum-steps", type=int, default=DEFAULT_GRAD_ACCUM_STEPS)
    parser.add_argument("--flow-draws", type=int, default=DEFAULT_FLOW_DRAWS)
    parser.add_argument("--val-every", type=int, default=DEFAULT_VAL_EVERY)
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=float)
    parser.add_argument("--lora-lr", type=float, default=DEFAULT_LORA_LR)
    parser.add_argument("--decoder-lr", type=float, default=DEFAULT_DECODER_LR)
    parser.add_argument("--decoder-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--lambda-video", type=float, default=DEFAULT_LAMBDA_VIDEO)
    parser.add_argument("--action-grad-to-backbone", choices=("on", "off"), default="on")
    parser.add_argument(
        "--context-transform",
        choices=CONTEXT_TRANSFORMS,
        default="pool2",
        help="Reduce layer-20 tokens before the action decoder. Video loss still uses the full DiT.",
    )
    parser.add_argument("--eval-sigma", type=float, default=DEFAULT_EVAL_SIGMA)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Reject unsafe or ambiguous training configurations before model load."""
    if args.device != "cuda":
        raise ValueError("the trainable Cosmos co-training runtime requires --device cuda")
    if args.max_steps <= 0 or args.max_hours <= 0:
        raise ValueError("--max-steps and --max-hours must be positive")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0 or args.flow_draws <= 0:
        raise ValueError("batch-size, grad-accum-steps, and flow-draws must be positive")
    if args.val_every <= 0 or args.save_every <= 0 or args.patience <= 0:
        raise ValueError("val-every, save-every, and patience must be positive")
    if args.seed < 0 or args.lora_rank <= 0:
        raise ValueError("seed must be non-negative and LoRA rank must be positive")
    if args.lora_lr <= 0 or args.decoder_lr <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("learning rates and grad clip must be positive; weight decay must be non-negative")
    if not math.isfinite(args.lambda_video) or args.lambda_video < 0:
        raise ValueError("--lambda-video must be finite and non-negative")
    if not math.isfinite(args.eval_sigma) or args.eval_sigma <= 0:
        raise ValueError("--eval-sigma must be finite and positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.precompute_latents_only and args.latent_cache_dir is None:
        raise ValueError("--precompute-latents-only requires --latent-cache-dir")


def _enable_full_block_checkpointing(backbone: torch.nn.Module, *, leave_tap_uncheckpointed: bool) -> None:
    """Checkpoint DiT blocks while keeping layer 20 differentiable for arm A."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) != 28:
        raise ValueError("Cosmos backbone must expose exactly 28 DiT blocks")
    for index, block in enumerate(blocks):
        if leave_tap_uncheckpointed and index == 19:
            continue
        if not getattr(block, "_lerobot_full_block_checkpoint", False):
            wrapped = checkpoint_wrapper(block, preserve_rng_state=False)
            wrapped._lerobot_full_block_checkpoint = True
            blocks[index] = wrapped


def _decoder_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported decoder dtype: {name}")


def _json_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _strict_decoder_load(decoder: World2ActionDecoder, path: Path) -> None:
    tensors = load_file(str(path), device="cpu")
    expected = decoder.denoiser.state_dict()
    if set(tensors) != set(expected):
        raise ValueError(
            f"decoder checkpoint keys mismatch: missing={sorted(set(expected) - set(tensors))}, "
            f"unexpected={sorted(set(tensors) - set(expected))}"
        )
    decoder.denoiser.load_state_dict(tensors, strict=True)


def _decoder_checkpoint_state(decoder: World2ActionDecoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous() for name, value in decoder.denoiser.state_dict().items()}


def _save_decoder(path: Path, decoder: World2ActionDecoder) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(_decoder_checkpoint_state(decoder), str(temporary))
    os.replace(temporary, path)


def _capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "action_generator": generator.get_state(),
    }


def _restore_rng(payload: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["action_generator"])


def _args_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _build_full_dataset(
    manifest: CacheManifest,
    args: argparse.Namespace,
    episodes: Sequence[int],
) -> LeRobotDataset:
    dataset = manifest.payload["dataset"]
    return LeRobotDataset(
        dataset["repo_id"],
        root=args.dataset_root.expanduser(),
        episodes=sorted({int(episode) for episode in episodes}),
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


def _images_to_cosmos(rgb: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = rgb.to(device=device)
    if rgb.dtype == torch.uint8:
        return rgb.to(dtype=dtype) / 127.5 - 1.0
    if not torch.is_floating_point(rgb):
        raise TypeError("RGB clips must be uint8 or floating point")
    minimum, maximum = rgb.amin().item(), rgb.amax().item()
    if minimum >= 0.0 and maximum <= 1.0:
        rgb = rgb * 2.0 - 1.0
    elif minimum < -1.0 or maximum > 1.0:
        raise ValueError("floating-point RGB clips must be in [0, 1] or [-1, 1]")
    return rgb.to(dtype=dtype)


def _encode_latents(tokenizer: Any, rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = _images_to_cosmos(rgb, device, torch.bfloat16)
    with torch.no_grad():
        latents = tokenizer.encode(images)
    if not isinstance(latents, torch.Tensor):
        raise TypeError("Cosmos tokenizer encode() must return a tensor")
    expected = (images.shape[0], 16, LATENT_FRAMES, 60, 80)
    prefix = (images.shape[0], 16, 2, 60, 80)
    if tuple(latents.shape) == prefix:
        full = torch.zeros(expected, device=device, dtype=torch.bfloat16)
        full[:, :, :2] = latents.to(device=device, dtype=torch.bfloat16)
        return full
    if tuple(latents.shape) != expected:
        raise ValueError(f"Cosmos tokenizer output must have shape {expected}, got {tuple(latents.shape)}")
    return latents.to(device=device, dtype=torch.bfloat16)


def _encode_training_latents(
    tokenizer: Any,
    sample: Any,
    entry: CacheManifestEntry,
    latent_cache: LatentCache,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cached = latent_cache.load(entry, device)
    if cached is not None:
        return cached
    latents = _encode_latents(tokenizer, sample.rgb_clip, device)
    valid = latent_valid_mask(sample.camera_is_pad.to(device=device), latent_frames=LATENT_FRAMES)
    latent_cache.save(entry, latents, valid)
    return latents, valid


def _encode_causal_conditional_latents(
    tokenizer: Any, history: torch.Tensor, device: torch.device
) -> torch.Tensor:
    if history.ndim != 5 or history.shape[2] != 5:
        raise ValueError("causal RGB history must have shape [B, 3, 5, H, W]")
    padded = torch.zeros(
        (history.shape[0], history.shape[1], FULL_CLIP_LENGTH, history.shape[3], history.shape[4]),
        device=history.device,
        dtype=history.dtype,
    )
    padded[:, :, :5] = history
    return _encode_latents(tokenizer, padded, device)


def _noise_like(shape: Sequence[int], *, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (2**63))
    return torch.randn(tuple(shape), device=device, dtype=torch.bfloat16, generator=generator)


def _sigma_for(seed: int, microbatch_index: int, batch_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed((seed * 2_654_435_761 + microbatch_index + 1) % (2**63))
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


def _flatten_hidden(hidden: torch.Tensor, transform: str = "none") -> torch.Tensor:
    if hidden.ndim != 5 or hidden.shape[1:4] != (LATENT_FRAMES, 30, 40) or hidden.shape[-1] != 2048:
        raise ValueError(f"layer-20 hidden state has unexpected shape {tuple(hidden.shape)}")
    flat = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1]).contiguous()
    return apply_context_transform(flat, transform)


def _add_scaled_gradients(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
    scale: float,
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = gradient.detach().mul(scale)
        else:
            parameter.grad.add_(gradient.detach(), alpha=scale)


def _stream_action_gradients(
    decoder: World2ActionDecoder,
    state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    sigma: torch.Tensor,
    action_is_pad: torch.Tensor,
    adapter_params: Sequence[torch.nn.Parameter],
    decoder_params: Sequence[torch.nn.Parameter],
    *,
    draws: int,
    grad_accum_steps: int,
    generator: torch.Generator,
    detach_context: bool,
) -> tuple[float, float, torch.Tensor | None]:
    """Compute K action draws without retaining K decoder graphs.

    The context endpoint is differentiated directly, so each decoder graph can
    be released after its gradients are accumulated.  The summed context
    gradient is then propagated through the shared layer-20 graph, preserving
    the single-backbone-forward contract for arm A.
    """
    context_gradient_sum: torch.Tensor | None = None
    decoder_squared_norm = 0.0
    loss_total = 0.0
    draw_scale = 1.0 / (draws * grad_accum_steps)
    for _ in range(draws):
        loss = decoder.flow_matching_loss(
            state,
            action,
            context,
            action_is_pad=action_is_pad,
            context_timestep=sigma[:, None],
            generator=generator,
            detach_context=detach_context,
        )
        loss_total += float(loss.detach().float().item())
        if context.requires_grad:
            gradients = torch.autograd.grad(
                loss,
                (context, *decoder_params),
                allow_unused=True,
            )
            context_gradient = gradients[0]
            decoder_gradients = gradients[1:]
            if context_gradient is not None:
                if context_gradient_sum is None:
                    context_gradient_sum = context_gradient.detach().clone()
                else:
                    context_gradient_sum.add_(context_gradient.detach())
        else:
            decoder_gradients = torch.autograd.grad(
                loss,
                decoder_params,
                allow_unused=True,
            )
        decoder_squared_norm += _gradient_norm(decoder_gradients) ** 2
        _add_scaled_gradients(decoder_params, decoder_gradients, draw_scale)

    if grad_accum_steps == 1:
        action_norm = _gradient_norm(parameter.grad for parameter in decoder_params)
    else:
        action_norm = math.sqrt(decoder_squared_norm / draws / (grad_accum_steps**2))
    return loss_total / draws, action_norm, context_gradient_sum


def _gradient_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    total = 0.0
    for gradient in gradients:
        if gradient is not None:
            total += float(gradient.detach().float().square().sum().item())
    return math.sqrt(total)


def _accumulate_component_grads(
    parameters: Sequence[torch.nn.Parameter],
    action_gradients: Sequence[torch.Tensor | None],
    video_gradients: Sequence[torch.Tensor | None],
    *,
    lambda_video: float,
    divisor: int,
) -> None:
    for parameter, action_gradient, video_gradient in zip(
        parameters, action_gradients, video_gradients, strict=True
    ):
        if action_gradient is None and video_gradient is None:
            continue
        result = torch.zeros_like(parameter)
        if action_gradient is not None:
            result.add_(action_gradient)
        if video_gradient is not None:
            result.add_(video_gradient, alpha=lambda_video)
        result.div_(divisor)
        if parameter.grad is None:
            parameter.grad = result.detach()
        else:
            parameter.grad.add_(result.detach())


def _fixed_probe_loss(
    decoder: World2ActionDecoder,
    state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    padding: torch.Tensor,
    probe: Any,
    eval_sigma: float,
) -> torch.Tensor:
    tau = torch.tensor([probe.tau_a], device=state.device, dtype=torch.float32)
    epsilon = probe.epsilon_tensor().unsqueeze(0).to(device=state.device, dtype=torch.float32)
    return decoder.flow_matching_loss(
        state,
        action,
        context,
        t=tau,
        epsilon=epsilon,
        action_is_pad=padding,
        context_timestep=torch.full((1, 1), eval_sigma, device=state.device),
        obs_dropout=0.0,
        detach_context=True,
    )


@torch.no_grad()
def evaluate_validation(
    backbone: torch.nn.Module,
    tokenizer: Any,
    decoder: World2ActionDecoder,
    dataset: LeRobotDataset,
    entries: Sequence[CacheManifestEntry],
    split: Any,
    prompt: torch.Tensor,
    data_type: Any,
    *,
    device: torch.device,
    eval_sigma: float,
    context_transform: str = "pool2",
) -> dict[str, Any]:
    """Run the established fixed probes through LoRA at pure-noise sigma 80."""
    if not entries:
        raise ValueError("validation split contains no entries")
    decoder.eval()
    backbone.eval()
    squared_sum = torch.zeros(6, dtype=torch.float64, device=device)
    scalar_count = torch.zeros(6, dtype=torch.float64, device=device)
    flow_total = 0.0
    flow_count = 0.0
    for entry in entries:
        sample = _load_full_sample(dataset, entry)
        history = sample.rgb_clip[:, :, :5]
        conditional = _encode_causal_conditional_latents(tokenizer, history, device)
        clean = conditional.clone()
        clean[:, :, 2:] = 0
        noise = _noise_like(clean.shape, device=device, seed=entry.noise_seed + 4_000_003)
        sigma = torch.full((1,), eval_sigma, device=device, dtype=torch.float32)
        network_input, condition_mask, c_noise = build_cosmos_noised_input(clean, noise, sigma)
        with autocast_context(device):
            prediction, hidden = capture_prediction_and_layer20(
                backbone,
                **_backbone_kwargs(network_input, condition_mask, c_noise, prompt, data_type),
            )
        del prediction
        context = _flatten_hidden(hidden, context_transform)
        state = sample.state.to(device=device, dtype=torch.float32)
        action = sample.target_action.to(device=device, dtype=torch.float32)
        padding = sample.action_is_pad.to(device=device)
        probe = split.probe_for(entry.sample_id)
        with autocast_context(device):
            flow = _fixed_probe_loss(decoder, state, action, context, padding, probe, eval_sigma)
            sampled = decoder.sample_actions(
                state,
                context,
                seed=probe.sample_seed,
                context_timestep=torch.full((1, 1), eval_sigma, device=device),
            )
        valid = float((~padding).sum().item() * action.shape[-1])
        flow_total += float(flow.item()) * valid
        flow_count += valid
        valid_mask = (~padding).unsqueeze(-1).expand_as(sampled)
        squared_sum += (
            ((sampled.float() - action.float()).square() * valid_mask).sum(dim=(0, 1)).to(torch.float64)
        )
        scalar_count += valid_mask.sum(dim=(0, 1)).to(torch.float64)
    if flow_count <= 0 or bool((scalar_count <= 0).any().item()):
        raise ValueError("validation probes leave no valid action positions")
    per_joint = torch.sqrt(squared_sum / scalar_count).cpu().tolist()
    aggregate = math.sqrt(float(squared_sum.sum().item() / scalar_count.sum().item()))
    return {
        "val_fixed_flow_loss": flow_total / flow_count,
        "val_per_joint_rmse": [float(value) for value in per_joint],
        "val_aggregate_rmse": aggregate,
        "val_samples": len(entries),
        "val_sigma": float(eval_sigma),
        "val_protocol": "5 real frames + pure-noise future, Cosmos sigma 80, fixed action probes",
    }


def _checkpoint_metadata(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    split_path: Path,
    normalizer_path: Path,
    backbone_checkpoint: Path,
    adapter_names: Sequence[str],
    decoder: World2ActionDecoder,
    backbone: torch.nn.Module,
    step: int,
    best_step: int,
    best_rmse: float | None,
    optimizer_settings: dict[str, Any],
    scheduler_settings: dict[str, Any],
    kind: str,
    resume_state: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    return {
        "schema_version": 1,
        "artifact": "cosmos_lora_cotraining_checkpoint",
        "checkpoint_kind": kind,
        "status": "validation_trained_non_rollout",
        "rollout_readiness": False,
        "robot_ready": False,
        "step": step,
        "best_step": best_step,
        "best_val_rmse": best_rmse,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _json_hash(manifest_path),
        "split": str(split_path.resolve()),
        "split_sha256": _json_hash(split_path),
        "normalizer": str(normalizer_path.resolve()),
        "backbone": {
            "identity": "cosmos-predict2-2B-lora-cotrain",
            "checkpoint": str(backbone_checkpoint.resolve()),
            "checkpoint_sha256": _json_hash(backbone_checkpoint),
            "base_dtype": "bfloat16",
            "adapter_dtype": "float32",
            "hidden_layer": 20,
            "num_blocks": 28,
            "gradient_checkpointing": (
                "LeRobot non-reentrant full-block checkpoint on blocks 1-19 and 21-28; "
                "layer-20 tap left uncheckpointed for arm A"
                if args.action_grad_to_backbone == "on"
                else "LeRobot non-reentrant full-block checkpoint on every DiT block"
            ),
            "adapter_targets": list(adapter_names),
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank),
            "trainable_parameters": sum(parameter.numel() for parameter in lora_parameters(backbone)),
        },
        "decoder_config": {
            "context_layer": 20,
            "context_channels": 2048,
            "action_horizon": 30,
            "state_dim": 6,
            "flow_draws_per_context": args.flow_draws,
        },
        "trainable_parameters": {
            "lora": sum(parameter.numel() for parameter in lora_parameters(backbone)),
            "decoder": sum(parameter.numel() for parameter in decoder.denoiser.parameters()),
        },
        "hyperparameters": _args_payload(args),
        "sigma_distribution": {
            "main": "sigma = 4 * exp(N(0,1))",
            "tail_probability": 0.05,
            "tail": "exp(U(log(200), log(100000)))",
            "video_loss_target": "noise - clean (vendored RectifiedFlowScaling c_out=-t)",
            "video_loss_sigma_weight": "(1 + sigma)^2 / sigma^2",
        },
        "loss": {
            "formula": "L_action + lambda_video * L_video",
            "lambda_video": args.lambda_video,
            "action_grad_to_backbone": args.action_grad_to_backbone,
        },
        "padding": {
            "pixel_policy": "LeRobot repeat-last-frame for future timestamps beyond episode end",
            "latent_loss_policy": "mask any latent bin containing a padded pixel frame",
            "conditioning_frames": 2,
        },
        "optimizer": optimizer_settings,
        "scheduler": scheduler_settings,
        "provenance": {
            "seed": args.seed,
            "sigma_seed": args.seed,
            "action_grad_seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "python_version": platform.python_version(),
            "upstream_mimic_video_commit": "e3355dbc93132b576c02f920a59b4fc18a4f5906",
        },
        "resume_state": str(resume_state.resolve()),
    }


def _save_checkpoint(
    *,
    output_dir: Path,
    backbone: torch.nn.Module,
    decoder: World2ActionDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metadata: dict[str, Any],
    rng_state: dict[str, Any],
    kind: str,
) -> None:
    prefix = "last" if kind == "last" else "best"
    lora_path = output_dir / f"{prefix}_lora.safetensors"
    decoder_path = output_dir / f"{prefix}_decoder.safetensors"
    state_path = output_dir / f"{prefix}.state.pt"
    save_lora_state_dict(backbone, lora_path)
    _save_decoder(decoder_path, decoder)
    metadata = copy.deepcopy(metadata)
    metadata["checkpoint_kind"] = kind
    metadata["resume_state"] = str(state_path.resolve())
    _atomic_json(output_dir / f"{prefix}.json", metadata)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng_state": rng_state},
        state_path,
    )


def _load_resume(
    *,
    output_dir: Path,
    backbone: torch.nn.Module,
    decoder: World2ActionDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    manifest_path: Path,
    split_path: Path,
    generator: torch.Generator,
) -> tuple[int, int, float | None, EarlyStopping]:
    metadata_path = output_dir / "last.json"
    state_path = output_dir / "last.state.pt"
    if not metadata_path.is_file() or not state_path.is_file():
        raise FileNotFoundError("--resume requires last.json and last.state.pt")
    metadata = json.loads(metadata_path.read_text())
    if metadata["manifest"] != str(manifest_path.resolve()) or metadata["manifest_sha256"] != _json_hash(
        manifest_path
    ):
        raise ValueError("resume manifest provenance does not match")
    if metadata["split"] != str(split_path.resolve()) or metadata["split_sha256"] != _json_hash(split_path):
        raise ValueError("resume split provenance does not match")
    load_lora_state_dict(backbone, output_dir / "last_lora.safetensors")
    _strict_decoder_load(decoder, output_dir / "last_decoder.safetensors")
    state = torch.load(state_path, map_location="cpu", weights_only=False)  # nosec B614
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    _restore_rng(state["rng_state"], generator)
    best_rmse = metadata.get("best_val_rmse")
    best_rmse = None if best_rmse is None else float(best_rmse)
    bad = int(metadata.get("hyperparameters", {}).get("early_stopping_bad_evaluations", 0))
    early = EarlyStopping(
        math.inf if best_rmse is None else best_rmse,
        bad,
        20,
        float(metadata.get("hyperparameters", {}).get("min_delta", 0.0)),
    )
    return int(metadata["step"]), int(metadata["best_step"]), best_rmse, early


def train(args: argparse.Namespace) -> Path:
    """Run co-training, or only the optional latent-cache precomputation."""
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required; CPU validation is provided by tests/policies/test_cosmos_lora.py"
        )
    device = torch.device(args.device)
    manifest_path = args.manifest.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = load_cache_manifest(manifest_path)
    split = load_vam_split(split_path, manifest)
    train_entries = get_train_entries(manifest, split)
    val_entries = get_val_entries(manifest, split)
    if not train_entries or not val_entries:
        raise ValueError("both train and validation splits must contain entries")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.resume and not args.overwrite:
        raise FileExistsError(f"Refusing non-empty output directory; pass --overwrite: {output_dir}")
    set_reproducible_seeds(args.seed)
    dataset = _build_full_dataset(manifest, args, tuple(split.train_episodes) + tuple(split.val_episodes))
    latent_cache = LatentCache(args.latent_cache_dir, overwrite=args.overwrite)

    extractor = CosmosPredict2Extractor(
        CosmosPredict2ExtractorConfig(
            checkpoint_path=args.backbone_checkpoint.expanduser(),
            tokenizer_path=args.tokenizer.expanduser(),
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
    _enable_full_block_checkpointing(backbone, leave_tap_uncheckpointed=args.action_grad_to_backbone == "on")
    adapter_names = inject_lora(backbone, rank=args.lora_rank, alpha=args.lora_alpha)
    backbone.train()
    data_type = extractor._data_type()
    prompt = load_prompt_embedding(args.prompt.expanduser()).embedding.to(device=device, dtype=torch.bfloat16)
    if tuple(prompt.shape) != (1, 512, 1024):
        raise ValueError(f"prompt embedding must have shape [1, 512, 1024], got {tuple(prompt.shape)}")

    if args.precompute_latents_only:
        unique_entries = {entry.sample_id: entry for entry in (*train_entries, *val_entries)}
        for position, entry in enumerate(unique_entries.values(), 1):
            sample = _load_full_sample(dataset, entry)
            latents, valid = _encode_training_latents(tokenizer, sample, entry, latent_cache, device)
            print(
                f"latent-cache {position}/{len(unique_entries)} {entry.sample_id} shape={tuple(latents.shape)} valid={int(valid.sum())}",
                flush=True,
            )
        print(f"latent cache: {latent_cache.root}", flush=True)
        return latent_cache.root if latent_cache.root is not None else output_dir

    normalizer_path = output_dir / "normalizer.safetensors"
    normalizer_source_path = output_dir / "normalizer_source.json"
    if args.resume:
        normalizer = ActionStateNormalizer.load(normalizer_path)
        normalizer_source = (
            json.loads(normalizer_source_path.read_text()) if normalizer_source_path.is_file() else None
        )
    else:
        normalizer = compute_training_normalizer_from_entries(manifest, train_entries)
        normalizer.save(normalizer_path, overwrite=args.overwrite)
        normalizer_source = {
            "derivation": "causal_cache_manifest_train_windows",
            "episodes": list(split.train_episodes),
            "anchor_count": len(train_entries),
            "padded_actions_excluded": True,
        }
        _atomic_json(normalizer_source_path, normalizer_source)

    decoder_dtype = _decoder_dtype(args.decoder_dtype)
    decoder = World2ActionDecoder(
        World2ActionConfig(device=str(device), dtype=decoder_dtype), normalizer=normalizer
    )
    decoder.train()
    decoder.denoiser.to(dtype=decoder_dtype)
    for parameter in decoder.denoiser.parameters():
        parameter.requires_grad_(True)
    adapter_params = lora_parameters(backbone)
    decoder_params = [parameter for parameter in decoder.denoiser.parameters() if parameter.requires_grad]
    optimizer_kwargs: dict[str, Any] = {
        "weight_decay": args.weight_decay,
        "betas": (0.9, 0.99),
        "eps": 1.0e-8,
        "capturable": True,
    }
    parameter_groups = [
        {"params": adapter_params, "lr": args.lora_lr, "name": "lora"},
        {"params": decoder_params, "lr": args.decoder_lr, "name": "decoder"},
    ]
    optimizer_settings = {
        "name": "AdamW",
        "lora_lr": args.lora_lr,
        "decoder_lr": args.decoder_lr,
        "weight_decay": args.weight_decay,
        "betas": [0.9, 0.99],
        "eps": 1.0e-8,
        "base_dtype": "bfloat16",
        "lora_dtype": "float32",
        "decoder_dtype": args.decoder_dtype,
    }
    if "fused" in __import__("inspect").signature(torch.optim.AdamW).parameters:
        try:
            optimizer = torch.optim.AdamW(parameter_groups, fused=True, **optimizer_kwargs)
            optimizer_settings["fused"] = True
        except (RuntimeError, TypeError):
            optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
            optimizer_settings["fused"] = False
    else:
        optimizer = torch.optim.AdamW(parameter_groups, **optimizer_kwargs)
        optimizer_settings["fused"] = False
    scheduler, scheduler_settings = build_scheduler(
        optimizer, args.max_steps, warm_up_steps=min(1000, args.max_steps - 1)
    )
    action_generator = torch.Generator(device=device)
    action_generator.manual_seed(args.seed)
    step = 0
    best_step = 0
    best_rmse: float | None = None
    early = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    if args.resume:
        step, best_step, best_rmse, old_early = _load_resume(
            output_dir=output_dir,
            backbone=backbone,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            manifest_path=manifest_path,
            split_path=split_path,
            generator=action_generator,
        )
        early = EarlyStopping(old_early.best, old_early.bad_evaluations, args.patience, args.min_delta)

    wandb_logger = SafeWandbLogger(
        output_dir=output_dir,
        project=args.wandb_project,
        run_name=f"cosmos-lora-cotrain-{args.action_grad_to_backbone}-s{args.seed}",
        tags=["cosmos-lora-cotrain", f"arm:{args.action_grad_to_backbone}"],
        config={
            "manifest": str(manifest_path),
            "split": str(split_path),
            "training": _args_payload(args),
            "sigma_distribution": "4*exp(N(0,1)) + 5% loguniform tail [200,100000]",
            "video_target": "noise-clean",
            "padding_policy": "repeat_last_frame; mask padded latent bins",
        },
        resume=args.resume,
        disabled=args.no_wandb,
    )
    metrics_path = output_dir / "metrics.jsonl"
    metrics = metrics_path.open("a" if args.resume else "w")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        while step < args.max_steps:
            if time.perf_counter() - started >= args.max_hours * 3600:
                print(f"max-hours reached at step={step}", flush=True)
                break
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            action_losses: list[float] = []
            video_losses: list[float] = []
            action_grad_norms: list[float] = []
            video_grad_norms: list[float] = []
            sigma_values: list[float] = []
            samples_this_step = 0
            for accumulation in range(args.grad_accum_steps):
                microbatch_index = step * args.grad_accum_steps + accumulation
                batch_entries = tuple(
                    train_entries[(microbatch_index * args.batch_size + position) % len(train_entries)]
                    for position in range(args.batch_size)
                )
                samples = [_load_full_sample(dataset, entry) for entry in batch_entries]
                encoded = [
                    _encode_training_latents(tokenizer, sample, entry, latent_cache, device)
                    for sample, entry in zip(samples, batch_entries, strict=True)
                ]
                clean = torch.cat([item[0] for item in encoded], dim=0)
                valid = torch.cat([item[1] for item in encoded], dim=0)
                sigma = _sigma_for(args.seed, microbatch_index, clean.shape[0], device)
                noise = _noise_like(
                    clean.shape, device=device, seed=args.seed * 1_000_003 + microbatch_index + 7
                )
                network_input, condition_mask, c_noise = build_cosmos_noised_input(clean, noise, sigma)
                with autocast_context(device):
                    prediction, hidden = capture_prediction_and_layer20(
                        backbone,
                        **_backbone_kwargs(network_input, condition_mask, c_noise, prompt, data_type),
                    )
                    video_loss = rectified_flow_video_loss(prediction, clean, noise, sigma, valid)
                    context = _flatten_hidden(hidden, args.context_transform)
                    state = torch.cat([sample.state for sample in samples], dim=0).to(
                        device=device, dtype=torch.float32
                    )
                    action = torch.cat([sample.target_action for sample in samples], dim=0).to(
                        device=device, dtype=torch.float32
                    )
                    padding = torch.cat([sample.action_is_pad for sample in samples], dim=0).to(device=device)
                    action_context = action_context_for_arm(
                        context, action_grad_to_backbone=args.action_grad_to_backbone == "on"
                    )
                    parameters = tuple(adapter_params) + tuple(decoder_params)
                    action_loss, action_grad_norm, action_context_gradient = _stream_action_gradients(
                        decoder,
                        state,
                        action,
                        action_context,
                        sigma,
                        padding,
                        adapter_params,
                        decoder_params,
                        draws=args.flow_draws,
                        grad_accum_steps=args.grad_accum_steps,
                        generator=action_generator,
                        detach_context=args.action_grad_to_backbone != "on",
                    )
                    if action_context_gradient is not None:
                        action_context_gradient.mul_(1.0 / (args.flow_draws * args.grad_accum_steps))
                if not math.isfinite(action_loss) or not torch.isfinite(video_loss).item():
                    raise FloatingPointError("non-finite action or video loss")
                # One TE-safe traversal of the shared Cosmos graph. A second
                # backward (retain_graph + video autograd.grad) clears
                # Transformer Engine ctx.tensor_objects and crashes Arm A.
                backbone_roots: list[torch.Tensor] = [video_loss]
                backbone_grad_outputs: list[torch.Tensor] = [
                    video_loss.new_tensor(args.lambda_video / args.grad_accum_steps)
                ]
                if action_context_gradient is not None:
                    backbone_roots.append(action_context)
                    backbone_grad_outputs.append(action_context_gradient)
                torch.autograd.backward(
                    tuple(backbone_roots),
                    grad_tensors=tuple(backbone_grad_outputs),
                    inputs=tuple(adapter_params),
                )
                video_grad_norms.append(_gradient_norm(parameter.grad for parameter in adapter_params))
                del parameters
                action_grad_norms.append(action_grad_norm)
                action_losses.append(action_loss)
                video_losses.append(float(video_loss.detach().float().item()))
                sigma_values.extend(float(value) for value in sigma.detach().cpu().tolist())
                samples_this_step += len(samples)
                del prediction, hidden, context, action_context, action_loss, video_loss, clean, noise
            combined_grad_norm = torch.nn.utils.clip_grad_norm_(
                tuple(adapter_params) + tuple(decoder_params), args.grad_clip
            )
            if not torch.isfinite(torch.as_tensor(combined_grad_norm)).item():
                raise FloatingPointError("combined gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
            step += 1
            elapsed = time.perf_counter() - step_started
            peak = torch.cuda.max_memory_allocated(device)
            record: dict[str, Any] = {
                "step": step,
                "loss": float(sum(action_losses) / len(action_losses))
                + args.lambda_video * float(sum(video_losses) / len(video_losses)),
                "action_loss": float(sum(action_losses) / len(action_losses)),
                "video_loss": float(sum(video_losses) / len(video_losses)),
                "lambda_video": args.lambda_video,
                "grad_norm_action": float(sum(action_grad_norms) / len(action_grad_norms)),
                "grad_norm_video": float(sum(video_grad_norms) / len(video_grad_norms)),
                "grad_norm_combined_preclip": float(torch.as_tensor(combined_grad_norm).item()),
                "learning_rate_lora": float(optimizer.param_groups[0]["lr"]),
                "learning_rate_decoder": float(optimizer.param_groups[1]["lr"]),
                "sigma_mean": float(sum(sigma_values) / len(sigma_values)),
                "sigma_min": min(sigma_values),
                "sigma_max": max(sigma_values),
                "seconds_per_step": elapsed,
                "samples_per_sec": samples_this_step / max(elapsed, 1e-9),
                "peak_vram_bytes": int(peak),
                "action_grad_to_backbone": args.action_grad_to_backbone == "on",
                "context_transform": args.context_transform,
                "flow_draws": args.flow_draws,
            }
            should_validate = step % args.val_every == 0 or step == args.max_steps
            improved = False
            stop = False
            if should_validate:
                validation = evaluate_validation(
                    backbone,
                    tokenizer,
                    decoder,
                    dataset,
                    val_entries,
                    split,
                    prompt,
                    data_type,
                    device=device,
                    eval_sigma=args.eval_sigma,
                    context_transform=args.context_transform,
                )
                backbone.train()
                decoder.train()
                record.update(validation)
                candidate = float(validation["val_aggregate_rmse"])
                early, improved, stop = early.update(candidate)
                if improved:
                    best_rmse = candidate
                    best_step = step
                record["early_stopping_bad_evaluations"] = early.bad_evaluations
                print(
                    f"step={step} action={record['action_loss']:.6g} video={record['video_loss']:.6g} "
                    f"total={record['loss']:.6g} val_rmse={candidate:.6g} "
                    f"action_gnorm={record['grad_norm_action']:.3g} video_gnorm={record['grad_norm_video']:.3g} "
                    f"seconds={elapsed:.3f} peak_vram={peak / 2**30:.2f}GiB",
                    flush=True,
                )
            else:
                print(
                    f"step={step} action={record['action_loss']:.6g} video={record['video_loss']:.6g} "
                    f"total={record['loss']:.6g} sigma={record['sigma_mean']:.4g} "
                    f"seconds={elapsed:.3f} peak_vram={peak / 2**30:.2f}GiB",
                    flush=True,
                )
            wandb_logger.log_metrics(record, step)
            if should_validate:
                wandb_logger.update_summary(best_metric=best_rmse, best_step=best_step)
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            if step % args.save_every == 0 or should_validate or step == args.max_steps:
                metadata = _checkpoint_metadata(
                    args=args,
                    manifest_path=manifest_path,
                    split_path=split_path,
                    normalizer_path=normalizer_path,
                    backbone_checkpoint=args.backbone_checkpoint,
                    adapter_names=adapter_names,
                    decoder=decoder,
                    backbone=backbone,
                    step=step,
                    best_step=best_step,
                    best_rmse=best_rmse,
                    optimizer_settings=optimizer_settings,
                    scheduler_settings=scheduler_settings,
                    kind="last",
                    resume_state=output_dir / "last.state.pt",
                )
                metadata["normalizer_source"] = normalizer_source
                metadata["hyperparameters"]["early_stopping_bad_evaluations"] = early.bad_evaluations
                _save_checkpoint(
                    output_dir=output_dir,
                    backbone=backbone,
                    decoder=decoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metadata=metadata,
                    rng_state=_capture_rng(action_generator),
                    kind="last",
                )
                if should_validate and improved:
                    _save_checkpoint(
                        output_dir=output_dir,
                        backbone=backbone,
                        decoder=decoder,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        metadata=metadata,
                        rng_state=_capture_rng(action_generator),
                        kind="best",
                    )
            if should_validate and stop:
                print(f"early stopping at step={step}", flush=True)
                break
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.max_memory_reserved(device)
        raise RuntimeError(
            f"CUDA OOM; peak allocated VRAM={peak} bytes; peak reserved VRAM={reserved} bytes; "
            "try --grad-accum-steps or a smaller LoRA rank, without changing the 61-frame contract"
        ) from exc
    finally:
        metrics.close()
        wandb_logger.finish()
    print(f"last checkpoint: {output_dir / 'last_lora.safetensors'}", flush=True)
    print(f"best checkpoint: {output_dir / 'best_lora.safetensors'}", flush=True)
    print("rollout readiness: false", flush=True)
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
