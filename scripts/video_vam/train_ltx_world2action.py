#!/usr/bin/env python3
"""Train the native World2Action decoder on frozen cached LTX-2.5 features."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from lerobot.policies.vam.ltx_action import LTXContextAdapter
from lerobot.policies.vam.ltx_feature_cache import (
    LTXFeatureCacheDataset,
    load_ltx_cache_manifest,
    sha256_file,
)
from lerobot.policies.vam.vam_split import load_vam_split
from lerobot.policies.vam.world2action import ActionStateNormalizer, World2ActionConfig, World2ActionDecoder
from scripts.video_vam.train_cosmos_world2action import (
    EarlyStopping,
    autocast_context,
    build_optimizer,
    build_scheduler,
    convert_decoder_parameters_to_fp32,
)
from scripts.video_vam.train_ltx_world2action_tiny import load_established_normalizer

DEFAULT_TRAIN_MANIFEST = Path("/home/anton/.cache/video-vam/ltx25-train0-31-stride3-pool2/manifest.json")
DEFAULT_VAL_MANIFEST = Path("/home/anton/.cache/video-vam/ltx25-val32-39-stride20-pool2/manifest.json")
DEFAULT_SPLIT = Path("/home/anton/.cache/video-vam/splits/rehearsal-stride20.json")
DEFAULT_NORMALIZER = Path("/home/anton/.cache/video-vam/runs/vam-hour-k8/normalizer.safetensors")
DEFAULT_OUTPUT = Path("/home/anton/.cache/video-vam/runs/ltx25-frozen-pool2-w2a-plateau-20260825")
EXPECTED_DECODER_PARAMETERS = 499_171_958
CHECKPOINT_RESERVE_BYTES = 9 * 2**30


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--normalizer", type=Path, default=DEFAULT_NORMALIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--max-hours", type=float, default=72.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--flow-draws", type=int, default=8)
    parser.add_argument("--flow-draw-chunk", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="video-vam-world2action")
    parser.add_argument("--wandb-run-name", default="ltx25-frozen-pool2-w2a-plateau-s0-20260825")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "max_hours",
        "batch_size",
        "grad_accum_steps",
        "flow_draws",
        "flow_draw_chunk",
        "val_every",
        "patience",
        "lr",
        "grad_clip",
        "loss_scale",
    )
    for name in positive:
        value = getattr(args, name)
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.weight_decay < 0 or args.min_delta < 0 or args.seed < 0:
        raise ValueError("weight decay, min delta, and seed must be non-negative")
    if args.warmup_steps < 0 or args.warmup_steps >= args.max_steps:
        raise ValueError("warmup steps must be non-negative and below max steps")
    if args.flow_draw_chunk > args.flow_draws:
        raise ValueError("flow draw chunk cannot exceed flow draws")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
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


class SafeWandb:
    """Best-effort W&B transport; metrics.jsonl remains authoritative."""

    def __init__(self, args: argparse.Namespace, output: Path, config: dict[str, Any]) -> None:
        self.run = None
        self.url: str | None = None
        if args.no_wandb or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
            print("W&B disabled; local metrics remain enabled", flush=True)
            return
        try:
            import wandb

            run_id_path = output / "wandb_run_id.txt"
            run_id = run_id_path.read_text().strip() if args.resume and run_id_path.is_file() else None
            kwargs: dict[str, Any] = {
                "project": args.wandb_project,
                "name": args.wandb_run_name,
                "tags": ["ltx-2.5", "frozen-features", "world2action", "plateau"],
                "config": config,
                "dir": str(output),
                "job_type": "train",
                "settings": wandb.Settings(init_timeout=10),
            }
            if run_id:
                kwargs.update(id=run_id, resume="must")
            self.run = wandb.init(**kwargs)
            if self.run is None:
                raise RuntimeError("wandb.init returned no run")
            run_id_path.write_text(str(self.run.id) + "\n")
            self.url = getattr(self.run, "url", None)
            print(f"WANDB_URL={self.url}", flush=True)
        except Exception as exc:
            print(f"W&B unavailable; continuing locally: {exc}", file=sys.stderr, flush=True)
            self.run = None

    def log(self, record: Mapping[str, Any], step: int) -> None:
        if self.run is None:
            return
        try:
            payload = {
                key: value
                for key, value in record.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value is not None
            }
            self.run.log(payload, step=step)
        except Exception as exc:
            print(f"W&B logging failed; continuing locally: {exc}", file=sys.stderr, flush=True)
            self.run = None

    def summary(self, *, best_step: int, best_degrees: float, stop_reason: str | None = None) -> None:
        if self.run is None:
            return
        try:
            self.run.summary["best_val_action_rmse_degrees"] = best_degrees
            self.run.summary["best_val_step"] = best_step
            if stop_reason is not None:
                self.run.summary["stop_reason"] = stop_reason
        except Exception as exc:
            print(f"W&B summary failed: {exc}", file=sys.stderr, flush=True)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as exc:
                print(f"W&B finish failed: {exc}", file=sys.stderr, flush=True)


def batch_entries(entries: Sequence[Any], batch_size: int, microbatch: int, seed: int) -> tuple[Any, ...]:
    batches = max(math.ceil(len(entries) / batch_size), 1)
    epoch = microbatch // batches
    offset = (microbatch % batches) * batch_size
    order = list(range(len(entries)))
    random.Random(seed + epoch).shuffle(order)
    return tuple(entries[order[(offset + index) % len(order)]] for index in range(batch_size))


def batch_tensors(
    dataset: LTXFeatureCacheDataset, entries: Sequence[Any], device: torch.device
) -> tuple[torch.Tensor, ...]:
    items = [dataset.load_entry(entry) for entry in entries]
    return (
        torch.cat([item.state for item in items]).to(device=device, dtype=torch.float32),
        torch.cat([item.target_action for item in items]).to(device=device, dtype=torch.float32),
        torch.cat([item.context for item in items]).to(device=device, dtype=torch.bfloat16),
        torch.cat([item.action_is_pad for item in items]).to(device=device, dtype=torch.bool),
    )


def normalized_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    padding: torch.Tensor,
    normalizer: ActionStateNormalizer,
) -> tuple[float, float, int]:
    pred = normalizer.normalize_action(prediction)
    truth = normalizer.normalize_action(target)
    valid = (~padding).unsqueeze(-1).expand_as(pred)
    squared_sum = float(((pred.float() - truth.float()).square() * valid).sum().item())
    count = int(valid.sum().item())
    return squared_sum, math.sqrt(squared_sum / count), count


@torch.no_grad()
def evaluate(
    decoder: World2ActionDecoder,
    adapter: LTXContextAdapter,
    dataset: LTXFeatureCacheDataset,
    entries: Sequence[Any],
    split: Any,
    normalizer: ActionStateNormalizer,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    decoder.eval()
    adapter.eval()
    flow_weighted = 0.0
    flow_count = 0
    physical_squared = torch.zeros(6, dtype=torch.float64, device=device)
    physical_count = torch.zeros(6, dtype=torch.float64, device=device)
    normalized_squared = 0.0
    normalized_count = 0
    for start in range(0, len(entries), batch_size):
        selected = tuple(entries[start : start + batch_size])
        state, action, raw_context, padding = batch_tensors(dataset, selected, device)
        probes = [split.probe_for(entry.sample_id) for entry in selected]
        tau = torch.tensor([probe.tau_a for probe in probes], device=device, dtype=torch.float32)
        epsilon = torch.stack([probe.epsilon_tensor() for probe in probes]).to(
            device=device, dtype=torch.float32
        )
        with autocast_context(device):
            context = adapter(raw_context)
            flow = decoder.flow_matching_loss(
                state,
                action,
                context,
                t=tau,
                epsilon=epsilon,
                action_is_pad=padding,
                context_timestep=torch.ones((state.shape[0], 1), device=device),
                obs_dropout=0.0,
                detach_context=False,
            )
        valid_scalars = int((~padding).sum().item() * action.shape[-1])
        flow_weighted += float(flow.float().item()) * valid_scalars
        flow_count += valid_scalars
        for index, probe in enumerate(probes):
            with autocast_context(device):
                prediction = decoder.sample_actions(
                    state[index : index + 1],
                    context[index : index + 1],
                    seed=probe.sample_seed,
                    context_timestep=torch.ones((1, 1), device=device),
                )
            valid = (~padding[index : index + 1]).unsqueeze(-1).expand_as(prediction)
            squared = (prediction.float() - action[index : index + 1].float()).square()
            physical_squared += (squared * valid).sum(dim=(0, 1)).to(torch.float64)
            physical_count += valid.sum(dim=(0, 1)).to(torch.float64)
            norm_sum, _, norm_count = normalized_rmse(
                prediction, action[index : index + 1], padding[index : index + 1], normalizer
            )
            normalized_squared += norm_sum
            normalized_count += norm_count
    if flow_count == 0 or normalized_count == 0 or bool((physical_count == 0).any().item()):
        raise ValueError("validation masks contain no valid action scalars")
    per_joint = torch.sqrt(physical_squared / physical_count).cpu().tolist()
    return {
        "val_fixed_flow_loss_normalized": flow_weighted / flow_count,
        "val_action_rmse_degrees": math.sqrt(
            float(physical_squared.sum().item() / physical_count.sum().item())
        ),
        "val_action_rmse_normalized": math.sqrt(normalized_squared / normalized_count),
        "val_action_rmse_per_joint_degrees": [float(value) for value in per_joint],
        "val_samples": len(entries),
        "val_valid_action_scalars": int(physical_count.sum().item()),
        "val_metric_contract": "global masked RMSE over physical degree-valued SO-101 actions",
        "val_flow_equals_rmse": False,
    }


def model_tensors(decoder: World2ActionDecoder, adapter: LTXContextAdapter) -> dict[str, torch.Tensor]:
    tensors = {
        f"decoder.{name}": value.detach().cpu().contiguous()
        for name, value in decoder.denoiser.state_dict().items()
    }
    tensors.update(
        {f"adapter.{name}": value.detach().cpu().contiguous() for name, value in adapter.state_dict().items()}
    )
    return tensors


def load_model(path: Path, decoder: World2ActionDecoder, adapter: LTXContextAdapter) -> None:
    tensors = load_file(str(path), device="cpu")
    decoder_state = {
        key.removeprefix("decoder."): value for key, value in tensors.items() if key.startswith("decoder.")
    }
    adapter_state = {
        key.removeprefix("adapter."): value for key, value in tensors.items() if key.startswith("adapter.")
    }
    if len(decoder_state) + len(adapter_state) != len(tensors):
        raise ValueError("checkpoint has unrecognized tensor keys")
    decoder.denoiser.load_state_dict(decoder_state, strict=True)
    adapter.load_state_dict(adapter_state, strict=True)


def save_weights(path: Path, decoder: World2ActionDecoder, adapter: LTXContextAdapter) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(model_tensors(decoder, adapter), str(temporary))
    os.replace(temporary, path)


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "train_generator": generator.get_state(),
    }


def restore_rng(payload: Mapping[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["train_generator"])


def save_resume_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    rng: dict[str, Any],
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(
        {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng}, temporary
    )
    os.replace(temporary, path)


def config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value.expanduser().resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def train(args: argparse.Namespace) -> Path:
    validate_args(args)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 support is required")
    set_seed(args.seed)
    device = torch.device("cuda")
    train_manifest_path = args.train_manifest.expanduser().resolve()
    val_manifest_path = args.val_manifest.expanduser().resolve()
    split_path = args.split.expanduser().resolve()
    normalizer_path = args.normalizer.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    train_manifest = load_ltx_cache_manifest(train_manifest_path)
    val_manifest = load_ltx_cache_manifest(val_manifest_path)
    if train_manifest.dataset_revision != val_manifest.dataset_revision:
        raise ValueError("train and validation manifests have different dataset revisions")
    if train_manifest.payload["subset"] != {"split": "train", "episodes": list(range(32)), "stride": 3}:
        raise ValueError("training manifest does not match canonical episodes 0-31 stride 3")
    if val_manifest.payload["subset"] != {
        "split": "validation",
        "episodes": list(range(32, 40)),
        "stride": 20,
    }:
        raise ValueError("validation manifest does not match canonical episodes 32-39 stride 20")
    if len(train_manifest.entries) != 1572 or len(val_manifest.entries) != 88:
        raise ValueError("canonical cache counts must be 1572 train and 88 validation")
    train_ids = {entry.sample_id for entry in train_manifest.entries}
    val_ids = {entry.sample_id for entry in val_manifest.entries}
    if train_ids & val_ids:
        raise ValueError("train/validation sample leakage detected")
    split = load_vam_split(split_path, val_manifest, allow_partial=True)
    if tuple(range(32)) != split.train_episodes or tuple(range(32, 40)) != split.val_episodes:
        raise ValueError("split episode IDs are not canonical")
    if [probe.sample_id for probe in split.validation_probe] != [
        entry.sample_id for entry in val_manifest.entries
    ]:
        raise ValueError("fixed validation probes do not exactly match validation cache IDs/order")
    free = shutil.disk_usage(output.parent).free
    if free < CHECKPOINT_RESERVE_BYTES + 20 * 2**30:
        raise RuntimeError(
            f"training disk guard failed: free={free / 2**30:.2f} GiB; "
            "need 9 GiB checkpoint reserve plus 20 GiB free floor"
        )
    if output.exists() and any(output.iterdir()) and not (args.resume or args.overwrite):
        raise FileExistsError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_dataset = LTXFeatureCacheDataset(train_manifest)
    val_dataset = LTXFeatureCacheDataset(val_manifest)
    # Verify one artifact from each split before allocating the trainable model.
    train_dataset.load_entry(train_manifest.entries[0])
    val_dataset.load_entry(val_manifest.entries[0])
    normalizer, normalizer_source = load_established_normalizer(normalizer_path)
    adapter = LTXContextAdapter().to(device=device, dtype=torch.float32)
    decoder = World2ActionDecoder(
        World2ActionConfig(device="cuda", dtype=torch.bfloat16), normalizer=normalizer
    )
    convert_decoder_parameters_to_fp32(decoder)
    report = decoder.parameter_count_report()
    if report.decoder_parameters != EXPECTED_DECODER_PARAMETERS:
        raise RuntimeError("native World2Action decoder parameter count changed")
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    parameters.extend(parameter for parameter in decoder.denoiser.parameters() if parameter.requires_grad)
    optimizer, optimizer_settings = build_optimizer(
        parameters, lr=args.lr, weight_decay=args.weight_decay, device=device
    )
    scheduler, scheduler_settings = build_scheduler(optimizer, args.max_steps, args.warmup_steps)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    step = 0
    best_step = 0
    best_degrees = math.inf
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    last_weights = output / "last.safetensors"
    last_state = output / "last.state.pt"
    last_metadata = output / "last.json"
    if args.resume:
        if not all(path.is_file() for path in (last_weights, last_state, last_metadata)):
            raise FileNotFoundError("resume requires last.safetensors, last.state.pt, and last.json")
        metadata = json.loads(last_metadata.read_text())
        expected = {
            "train_manifest_sha256": sha256_file(train_manifest_path),
            "val_manifest_sha256": sha256_file(val_manifest_path),
            "split_sha256": sha256_file(split_path),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("resume provenance does not match manifests/split")
        load_model(last_weights, decoder, adapter)
        state = torch.load(last_state, map_location="cpu", weights_only=False)  # nosec B614
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], generator)
        step = int(metadata["step"])
        best_step = int(metadata["best_step"])
        best_degrees = float(metadata["best_val_action_rmse_degrees"])
        early_stopping = EarlyStopping(
            best_degrees,
            int(metadata["early_stopping_bad_evaluations"]),
            args.patience,
            args.min_delta,
        )
    provenance = {
        "experiment": "frozen_ltx25_pool2_world2action_plateau",
        "dataset": {
            "repo_id": train_manifest.dataset_repo_id,
            "revision": train_manifest.dataset_revision,
            "train_episodes": list(range(32)),
            "validation_episodes": list(range(32, 40)),
        },
        "train_manifest": str(train_manifest_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "val_manifest": str(val_manifest_path),
        "val_manifest_sha256": sha256_file(val_manifest_path),
        "split": str(split_path),
        "split_sha256": sha256_file(split_path),
        "normalizer": str(normalizer_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "normalizer_source": normalizer_source,
        "backbone": train_manifest.payload["provenance"],
        "adapter": {
            "input_channels": 4096,
            "output_channels": 2048,
            "architecture": "non-affine LayerNorm + bias-free Linear",
            "trainable_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        },
        "decoder_parameters": report.decoder_parameters,
        "frozen_backbone": True,
        "backbone_retrained": False,
        "optimizer": optimizer_settings,
        "scheduler": scheduler_settings,
        "training": {
            **config_payload(args),
            "K_flow_draws": args.flow_draws,
            "batched_flow_draws": True,
            "autocast": "cuda_bfloat16",
            "context_timestep": 1.0,
            "best_selection_metric": "val_action_rmse_degrees",
            "action_horizon": 30,
            "action_dim": 6,
            "observation_dropout_train": 0.2,
            "validation_probe_count": 88,
            "validation_aggregation": "global masked RMSE in degrees",
        },
        "runtime": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "unavailable",
            "python_version": platform.python_version(),
        },
    }
    atomic_json(output / "config.json", provenance)
    wandb = SafeWandb(args, output, provenance)
    metrics_path = output / "metrics.jsonl"
    metrics = metrics_path.open("a" if args.resume else "w")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    stop_reason = "max_steps"
    try:
        while step < args.max_steps:
            if time.perf_counter() - started >= args.max_hours * 3600:
                stop_reason = "safety_cap_max_hours"
                break
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            for accumulation in range(args.grad_accum_steps):
                microbatch = step * args.grad_accum_steps + accumulation
                entries = batch_entries(train_manifest.entries, args.batch_size, microbatch, args.seed)
                state, action, raw_context, padding = batch_tensors(train_dataset, entries, device)
                remaining = args.flow_draws
                while remaining > 0:
                    k = min(remaining, args.flow_draw_chunk)
                    remaining -= k

                    def repeat(tensor: torch.Tensor, copies: int = k) -> torch.Tensor:
                        return tensor.repeat(copies, *([1] * (tensor.ndim - 1)))

                    with autocast_context(device):
                        context = adapter(repeat(raw_context))
                        loss = decoder.flow_matching_loss(
                            repeat(state),
                            repeat(action),
                            context,
                            action_is_pad=repeat(padding),
                            context_timestep=torch.ones((k * state.shape[0], 1), device=device),
                            generator=generator,
                            detach_context=False,
                        )
                        scaled = loss * args.loss_scale * k / (args.grad_accum_steps * args.flow_draws)
                    if not torch.isfinite(scaled).item():
                        raise FloatingPointError("training loss is non-finite")
                    scaled.backward()
                    losses.append(float(loss.detach().float().item()))
                    del context, loss, scaled
                del state, action, raw_context, padding
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                raise FloatingPointError("gradient norm is non-finite")
            optimizer.step()
            scheduler.step()
            step += 1
            elapsed_step = time.perf_counter() - step_started
            record: dict[str, Any] = {
                "step": step,
                "train_flow_loss_normalized": sum(losses) / len(losses),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grad_norm": float(torch.as_tensor(grad_norm).item()),
                "grad_clip_factor": min(1.0, args.grad_clip / max(float(grad_norm), 1e-12)),
                "step_seconds": elapsed_step,
                "wall_clock_seconds": time.perf_counter() - started,
                "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                "val_fixed_flow_loss_normalized": None,
                "val_action_rmse_degrees": None,
                "val_action_rmse_normalized": None,
            }
            should_validate = step % args.val_every == 0 or step == args.max_steps
            improved = False
            should_stop = False
            if should_validate:
                validation = evaluate(
                    decoder,
                    adapter,
                    val_dataset,
                    val_manifest.entries,
                    split,
                    normalizer,
                    device=device,
                    batch_size=args.batch_size,
                )
                record.update(validation)
                candidate = float(validation["val_action_rmse_degrees"])
                early_stopping, improved, should_stop = early_stopping.update(candidate)
                if improved:
                    best_step = step
                    best_degrees = candidate
                record["early_stopping_bad_evaluations"] = early_stopping.bad_evaluations
                print(
                    f"step={step} train_flow={record['train_flow_loss_normalized']:.6g} "
                    f"val_fixed_flow={validation['val_fixed_flow_loss_normalized']:.6g} "
                    f"val_action_rmse_deg={candidate:.6g} "
                    f"val_action_rmse_norm={validation['val_action_rmse_normalized']:.6g} "
                    f"bad={early_stopping.bad_evaluations}/{args.patience} "
                    f"lr={record['learning_rate']:.3g} step_s={elapsed_step:.3f} "
                    f"peak_vram={record['peak_vram_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
            elif step == 1 or step % 25 == 0:
                print(
                    f"step={step} train_flow={record['train_flow_loss_normalized']:.6g} "
                    f"lr={record['learning_rate']:.3g} step_s={elapsed_step:.3f} "
                    f"peak_vram={record['peak_vram_bytes'] / 2**30:.2f}GiB",
                    flush=True,
                )
            metrics.write(json.dumps(record, sort_keys=True) + "\n")
            metrics.flush()
            wandb.log(record, step)
            if should_validate:
                metadata = {
                    **provenance,
                    "artifact": "ltx25_world2action_checkpoint",
                    "checkpoint_kind": "last",
                    "step": step,
                    "best_step": best_step,
                    "best_val_action_rmse_degrees": best_degrees,
                    "early_stopping_bad_evaluations": early_stopping.bad_evaluations,
                    "wall_clock_seconds": time.perf_counter() - started,
                    "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
                    "wandb_url": wandb.url,
                    "metric_names_are_distinct": True,
                    "val_flow_equals_val_rmse": False,
                }
                save_weights(last_weights, decoder, adapter)
                save_resume_state(last_state, optimizer, scheduler, capture_rng(generator))
                atomic_json(last_metadata, metadata)
                if improved:
                    best_metadata = copy.deepcopy(metadata)
                    best_metadata["checkpoint_kind"] = "best"
                    save_weights(output / "best.safetensors", decoder, adapter)
                    atomic_json(output / "best.json", best_metadata)
                wandb.summary(best_step=best_step, best_degrees=best_degrees)
                if should_stop:
                    stop_reason = "validation_plateau"
                    print(
                        f"EARLY_STOP step={step} best_step={best_step} best_rmse_deg={best_degrees:.6g} "
                        f"post_best_non_improving_evals={early_stopping.bad_evaluations}",
                        flush=True,
                    )
                    break
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"CUDA OOM peak_allocated={torch.cuda.max_memory_allocated()} "
            f"peak_reserved={torch.cuda.max_memory_reserved()}"
        ) from exc
    finally:
        metrics.close()
    result = {
        "stop_reason": stop_reason,
        "step": step,
        "best_step": best_step,
        "best_val_action_rmse_degrees": best_degrees,
        "early_stopping_bad_evaluations": early_stopping.bad_evaluations,
        "wall_clock_seconds": time.perf_counter() - started,
        "peak_vram_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "wandb_url": wandb.url,
        "best_checkpoint": str((output / "best.safetensors").resolve()),
        "last_checkpoint": str(last_weights.resolve()),
        "metric_contract": {
            "val_fixed_flow_loss_normalized": "fixed-probe flow-matching loss in normalized action space",
            "val_action_rmse_degrees": "global masked RMSE in denormalized physical degrees",
            "val_action_rmse_normalized": "global masked RMSE after train-split min/max normalization",
            "metrics_are_not_duplicated": True,
        },
    }
    atomic_json(output / "result.json", result)
    wandb.summary(best_step=best_step, best_degrees=best_degrees, stop_reason=stop_reason)
    wandb.finish()
    print("TRAIN_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    try:
        train(parse_args(argv))
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
