#!/usr/bin/env python3
# Run the explicit tiny-overfit diagnostic against a frozen Cosmos cache.
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import random
import sys
import tempfile
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from lerobot.policies.vam.cosmos_cache_dataset import (
    CosmosFeatureCacheDataset,
    manifest_sha256,
)
from lerobot.policies.vam.world2action import (
    ActionStateNormalizer,
    World2ActionConfig,
    World2ActionDecoder,
)

DEFAULT_STEPS = 300
DEFAULT_BATCH_SIZE = 1
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_BETAS = (0.9, 0.99)
DEFAULT_EPS = 1e-8
DEFAULT_GRAD_CLIP = 1.0
EXPECTED_NATIVE_DECODER_PARAMETERS = 499_171_958
_CHECKPOINT_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "status",
        "robot_ready",
        "weights_only",
        "optimizer_resume_supported",
        "step",
        "manifest",
        "manifest_sha256",
        "normalizer",
        "decoder_config",
        "parameter_count",
        "optimizer",
        "autocast",
        "parameter_dtype",
        "frozen_context_dtype",
        "action_padding_semantics",
        "provenance",
    }
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--i-understand-diagnostic", action="store_true", help="Acknowledge diagnostic-only status."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--eval-every", type=int, default=0, help="Flow-loss eval interval; zero means start/end only."
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--grad-clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_train_args(args: argparse.Namespace) -> None:
    if not args.i_understand_diagnostic:
        raise ValueError("Pass --i-understand-diagnostic; checkpoints are diagnostic-only and non-rollout")
    if args.steps <= 0 or args.batch_size != 1:
        raise ValueError("The approved diagnostic requires positive steps and batch_size=1")
    if args.seed < 0 or args.save_every <= 0 or args.eval_every < 0:
        raise ValueError("seed/save-every must be positive and eval-every non-negative")
    if (
        args.lr != DEFAULT_LR
        or args.weight_decay != DEFAULT_WEIGHT_DECAY
        or args.grad_clip != DEFAULT_GRAD_CLIP
    ):
        raise ValueError(
            "The approved diagnostic optimizer settings must remain lr=3e-4, weight_decay=0, grad_clip=1"
        )
    if args.device != "cuda":
        raise ValueError("The approved diagnostic training path requires CUDA")


def set_reproducible_seeds(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is required for the approved diagnostic")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def optimizer_settings() -> dict[str, Any]:
    return {
        "name": "AdamW",
        "fused_requested": True,
        "lr": DEFAULT_LR,
        "weight_decay": DEFAULT_WEIGHT_DECAY,
        "betas": list(DEFAULT_BETAS),
        "eps": DEFAULT_EPS,
        "fused_fallback": False,
    }


def build_optimizer(parameters, *, lr: float = DEFAULT_LR, weight_decay: float = DEFAULT_WEIGHT_DECAY):
    settings = optimizer_settings()
    settings["lr"] = lr
    settings["weight_decay"] = weight_decay
    kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "betas": DEFAULT_BETAS,
        "eps": DEFAULT_EPS,
    }
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if supports_fused:
        try:
            optimizer = torch.optim.AdamW(parameters, fused=True, **kwargs)
            settings["fused"] = True
            return optimizer, settings
        except (TypeError, RuntimeError):
            pass
    optimizer = torch.optim.AdamW(parameters, **kwargs)
    settings["fused"] = False
    settings["fused_fallback"] = True
    return optimizer, settings


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"{name} contains non-finite values")


def assert_finite_parameters(module: torch.nn.Module) -> None:
    for name, parameter in module.named_parameters():
        _require_finite_tensor(parameter.detach(), f"parameter {name}")


def convert_decoder_parameters_to_fp32(decoder: World2ActionDecoder) -> None:
    decoder.denoiser.float()
    for parameter in decoder.denoiser.parameters():
        parameter.requires_grad_(True)
        parameter.data = parameter.data.float()
        if parameter.dtype != torch.float32:
            raise TypeError("all trainable decoder parameters must be FP32")


def compute_training_normalizer(dataset: CosmosFeatureCacheDataset) -> ActionStateNormalizer:
    if len(dataset) == 0:
        raise ValueError("Cannot compute a normalizer from an empty cache manifest")
    states = []
    actions = []
    masks = []
    for item in dataset:
        states.append(item.state)
        actions.append(item.target_action)
        masks.append(item.action_is_pad)
    return ActionStateNormalizer.from_training_tensors(
        torch.cat(states, dim=0),
        torch.cat(actions, dim=0),
        action_is_pad=torch.cat(masks, dim=0),
        source_split="train",
    )


def _masked_physical_mse(
    prediction: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or tuple(prediction.shape[1:]) != (30, 6):
        raise ValueError("physical action tensors must both have shape [B, 30, 6]")
    if action_is_pad.shape != prediction.shape[:2] or action_is_pad.dtype != torch.bool:
        raise ValueError("action_is_pad must have boolean shape [B, 30]")
    valid = (~action_is_pad).unsqueeze(-1).to(torch.float32)
    count = valid.sum() * prediction.shape[-1]
    if count.item() <= 0:
        raise ValueError("action_is_pad leaves no valid positions")
    return ((prediction.float() - target.float()).square() * valid).sum() / count


def _fixed_flow_inputs(item: Any, device: torch.device, seed: int):
    action = item.target_action.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    epsilon = torch.randn(action.shape, device=device, dtype=action.dtype, generator=generator)
    t = torch.full((action.shape[0],), 0.5, device=device, dtype=torch.float32)
    return (
        item.state.to(device=device, dtype=torch.float32),
        action,
        item.context.to(device=device),
        item.action_is_pad.to(device=device),
        t,
        epsilon,
    )


@torch.no_grad()
def evaluate_fixed_sample(
    decoder: World2ActionDecoder, item: Any, device: torch.device, seed: int, *, sample_actions: bool
):
    decoder.eval()
    state, action, context, padding, t, epsilon = _fixed_flow_inputs(item, device, seed)
    with select_autocast_context(device):
        loss = decoder.flow_matching_loss(
            state,
            action,
            context,
            t=t,
            epsilon=epsilon,
            action_is_pad=padding,
        )
    result = {"fixed_flow_loss": float(loss.float().item())}
    if sample_actions:
        with select_autocast_context(device):
            sampled = decoder.sample_actions(state, context, seed=seed)
        result["sampled_action_physical_mse"] = float(_masked_physical_mse(sampled, action, padding).item())
    return result


def _atomic_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite metadata; pass --overwrite: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise


def checkpoint_metadata(
    *,
    step: int,
    manifest_path: Path,
    normalizer_path: Path,
    decoder: World2ActionDecoder,
    optimizer_settings_payload: dict[str, Any],
    autocast: str,
) -> dict[str, Any]:
    report = decoder.parameter_count_report()
    return {
        "schema_version": 1,
        "artifact": "world2action_decoder_weights",
        "status": "diagnostic_only_non_rollout",
        "robot_ready": False,
        "weights_only": True,
        "optimizer_resume_supported": False,
        "step": step,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256(manifest_path),
        "normalizer": str(normalizer_path.resolve()),
        "decoder_config": {
            "state_shape": list(decoder.config.state_shape),
            "action_shape": list(decoder.config.action_shape),
            "context_channels": decoder.config.context_channels,
            "context_layer": decoder.config.context_layer,
            "model_channels": decoder.config.model_channels,
            "num_blocks": decoder.config.num_blocks,
            "num_heads": decoder.config.num_heads,
            "full_native_capacity": True,
        },
        "parameter_count": {
            "decoder": report.decoder_parameters,
            "trainable_decoder": report.trainable_decoder_parameters,
        },
        "optimizer": optimizer_settings_payload,
        "autocast": autocast,
        "parameter_dtype": "float32",
        "frozen_context_dtype": "bfloat16",
        "action_padding_semantics": "action_is_pad=true excluded from loss and train statistics",
        "provenance": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "gpu_name": torch.cuda.get_device_name(0),
            "python_version": platform.python_version(),
        },
    }


def validate_checkpoint_metadata(payload: dict[str, Any]) -> None:
    if set(payload) != _CHECKPOINT_METADATA_KEYS:
        raise ValueError("checkpoint metadata keys are not strict")
    if payload["schema_version"] != 1 or payload["artifact"] != "world2action_decoder_weights":
        raise ValueError("checkpoint metadata schema/artifact is invalid")
    if payload["status"] != "diagnostic_only_non_rollout" or payload["robot_ready"] is not False:
        raise ValueError("checkpoint must remain diagnostic-only and non-rollout")
    if payload["weights_only"] is not True or payload["optimizer_resume_supported"] is not False:
        raise ValueError("checkpoint metadata must declare weights-only non-resumable storage")
    if payload["parameter_dtype"] != "float32" or payload["frozen_context_dtype"] != "bfloat16":
        raise ValueError("checkpoint dtype provenance is invalid")
    if payload["parameter_count"]["decoder"] != EXPECTED_NATIVE_DECODER_PARAMETERS:
        raise ValueError("checkpoint is not the full native decoder capacity")


def save_decoder_checkpoint(
    decoder: World2ActionDecoder,
    path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if (path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint; pass --overwrite: {path}")
    validate_checkpoint_metadata(metadata)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        tensors = {
            name: value.detach().cpu().contiguous() for name, value in decoder.denoiser.state_dict().items()
        }
        save_file(tensors, temporary_name)
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    _atomic_json(metadata_path, metadata, overwrite=overwrite)


def _write_metric(stream, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True) + "\n")
    stream.flush()


def train(args: argparse.Namespace) -> Path:
    validate_train_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("The real diagnostic trainer requires CUDA; no model or workload is run on CPU")
    device = torch.device(args.device)
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to write into non-empty output directory; pass --overwrite: {output_dir}"
        )
    set_reproducible_seeds(args.seed)
    dataset = CosmosFeatureCacheDataset(manifest_path, shuffle_seed=args.seed)
    normalizer = compute_training_normalizer(dataset)
    normalizer_path = output_dir / "normalizer.safetensors"
    normalizer.save(normalizer_path, overwrite=args.overwrite)

    # Scratch construction intentionally follows the seed; no reduced test architecture is permitted.
    decoder = World2ActionDecoder(
        World2ActionConfig(device=str(device), dtype=torch.bfloat16),
        normalizer=normalizer,
    )
    convert_decoder_parameters_to_fp32(decoder)
    report = decoder.parameter_count_report()
    if report.decoder_parameters != EXPECTED_NATIVE_DECODER_PARAMETERS:
        raise RuntimeError(
            f"native decoder parameter count is {report.decoder_parameters}, expected {EXPECTED_NATIVE_DECODER_PARAMETERS}; "
            "reduced diagnostic architectures are not permitted"
        )
    decoder.train()
    parameters = [parameter for parameter in decoder.denoiser.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("native decoder has no trainable parameters")
    optimizer, opt_settings = build_optimizer(parameters, lr=args.lr, weight_decay=args.weight_decay)
    train_generator = torch.Generator(device=device)
    train_generator.manual_seed(args.seed)
    fixed_item = dataset[0]
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite metrics; pass --overwrite: {metrics_path}")
    metrics_mode = "w" if args.overwrite else "x"
    autocast_name = "cuda_bfloat16"
    try:
        with metrics_path.open(metrics_mode) as metrics:
            initial = evaluate_fixed_sample(
                decoder, fixed_item, device, args.seed + 1000, sample_actions=False
            )
            _write_metric(metrics, {"type": "eval", "step": 0, "sample_id": fixed_item.sample_id, **initial})
            for step in range(1, args.steps + 1):
                started = time.perf_counter()
                item = dataset.sample_for_step(step - 1)
                state = item.state.to(device=device, dtype=torch.float32)
                action = item.target_action.to(device=device, dtype=torch.float32)
                context = item.context.to(device=device, dtype=torch.bfloat16).detach()
                padding = item.action_is_pad.to(device=device)
                optimizer.zero_grad(set_to_none=True)
                with select_autocast_context(device):
                    loss = decoder.flow_matching_loss(
                        state,
                        action,
                        context,
                        action_is_pad=padding,
                        generator=train_generator,
                    )
                _require_finite_tensor(loss.float(), "loss")
                loss.backward()
                for name, parameter in decoder.denoiser.named_parameters():
                    if parameter.grad is not None:
                        _require_finite_tensor(parameter.grad, f"gradient {name}")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
                _require_finite_tensor(torch.as_tensor(grad_norm), "gradient norm")
                optimizer.step()
                assert_finite_parameters(decoder.denoiser)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                _write_metric(
                    metrics,
                    {
                        "type": "train",
                        "step": step,
                        "sample_id": item.sample_id,
                        "loss": float(loss.float().item()),
                        "grad_norm": float(torch.as_tensor(grad_norm).item()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "step_seconds": time.perf_counter() - started,
                        "allocated_vram_bytes": allocated,
                        "reserved_vram_bytes": reserved,
                    },
                )
                if args.eval_every and step % args.eval_every == 0 and step < args.steps:
                    evaluation = evaluate_fixed_sample(
                        decoder, fixed_item, device, args.seed + 1000, sample_actions=False
                    )
                    _write_metric(
                        metrics,
                        {"type": "eval", "step": step, "sample_id": fixed_item.sample_id, **evaluation},
                    )
                if step % args.save_every == 0 and step < args.steps:
                    checkpoint = output_dir / f"checkpoint-step-{step:06d}.safetensors"
                    metadata = checkpoint_metadata(
                        step=step,
                        manifest_path=manifest_path,
                        normalizer_path=normalizer_path,
                        decoder=decoder,
                        optimizer_settings_payload=opt_settings,
                        autocast=autocast_name,
                    )
                    save_decoder_checkpoint(
                        decoder,
                        checkpoint,
                        checkpoint.with_suffix(".json"),
                        metadata,
                        overwrite=args.overwrite,
                    )
            final_eval = evaluate_fixed_sample(
                decoder, fixed_item, device, args.seed + 1000, sample_actions=True
            )
            _write_metric(
                metrics, {"type": "eval", "step": args.steps, "sample_id": fixed_item.sample_id, **final_eval}
            )
            checkpoint = output_dir / f"checkpoint-step-{args.steps:06d}.safetensors"
            metadata = checkpoint_metadata(
                step=args.steps,
                manifest_path=manifest_path,
                normalizer_path=normalizer_path,
                decoder=decoder,
                optimizer_settings_payload=opt_settings,
                autocast=autocast_name,
            )
            save_decoder_checkpoint(
                decoder,
                checkpoint,
                checkpoint.with_suffix(".json"),
                metadata,
                overwrite=args.overwrite,
            )
    except torch.cuda.OutOfMemoryError as exc:
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        raise RuntimeError(
            f"CUDA OOM in diagnostic trainer; peak_allocated={peak_allocated} peak_reserved={peak_reserved}. "
            "No checkpoint was marked usable for rollout."
        ) from exc
    checkpoint_path = output_dir / f"checkpoint-step-{args.steps:06d}.safetensors"
    print(f"checkpoint: {checkpoint_path}")
    print("status: diagnostic_only_non_rollout")
    return output_dir


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
