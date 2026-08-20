#!/usr/bin/env python3
"""Run the frozen physical-action comparison for VAM, SmolVLA, and baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.vam.action_rmse import (
    ACTION_DIM,
    ACTION_HORIZON,
    ActionPredictionBackend,
    EvaluationBatch,
    MeanActionBackend,
    SmolVLABackend,
    StateRepeatBackend,
    VAMBackend,
    evaluate_backend,
)
from lerobot.policies.vam.cosmos_cache_dataset import CosmosFeatureCacheDataset, load_cache_manifest
from lerobot.policies.vam.vam_split import get_val_entries, load_vam_split
from lerobot.policies.vam.world2action import (
    ActionStateNormalizer,
    World2ActionConfig,
    World2ActionDecoder,
    load_diagnostic_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--vam-checkpoint", type=Path)
    parser.add_argument("--vam-normalizer", type=Path)
    parser.add_argument("--vam-step", type=int)
    parser.add_argument(
        "--vam-sigma",
        type=float,
        help=(
            "Video sigma the VAM decoder is conditioned on. Must equal the sigma the manifest "
            "contexts were extracted at. Defaults to the training run's recorded evaluation sigma."
        ),
    )
    parser.add_argument(
        "--vam-zero-context",
        action="store_true",
        help=(
            "Replace every cached context with zeros. Required to score a checkpoint trained with "
            "--context-mode state-only, whose decoder never saw an informative context."
        ),
    )
    parser.add_argument("--smolvla-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seed < 0:
        parser.error("--batch-size must be positive and --seed must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if (args.vam_checkpoint is None) != (args.vam_normalizer is None):
        parser.error("--vam-checkpoint and --vam-normalizer must be supplied together")
    return args


def _absolute_frame_lookup(dataset: LeRobotDataset) -> dict[tuple[int, int], int]:
    raw_dataset = dataset.reader.hf_dataset
    if raw_dataset is None:
        raise RuntimeError("dataset reader did not activate")
    lookup = {}
    for relative_index in range(len(raw_dataset)):
        row = raw_dataset[relative_index]
        key = (int(row["episode_index"]), int(row["index"]))
        if key in lookup:
            raise RuntimeError(f"duplicate dataset frame identifier: {key}")
        lookup[key] = relative_index
    return lookup


def _vam_normalization_description(args: argparse.Namespace) -> str:
    """Report how the scored checkpoint actually derived its min/max statistics."""
    if args.vam_checkpoint is None:
        return "no VAM checkpoint scored"
    metadata_path = args.vam_checkpoint.expanduser().resolve().with_suffix(".json")
    derivation = "cache_manifest_train_windows"
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text())
        derivation = payload.get("normalizer_source", {}).get("derivation", derivation)
    return f"{derivation} min/max to [-1, 1]"


def _make_batches(args: argparse.Namespace) -> tuple[list[EvaluationBatch], dict[str, Any]]:
    manifest = load_cache_manifest(args.manifest)
    split = load_vam_split(args.split, manifest)
    cache = CosmosFeatureCacheDataset(manifest, shuffle_seed=0)
    entries = list(get_val_entries(manifest, split))
    if args.max_samples is not None:
        entries = entries[: args.max_samples]
    if not entries:
        raise ValueError("manifest contains no evaluation entries")
    episodes = sorted({entry.episode_index for entry in entries})
    dataset = LeRobotDataset(
        manifest.dataset_repo_id,
        root=args.dataset_root,
        episodes=episodes,
        revision=manifest.dataset_revision,
        delta_timestamps={"action": [index / 10 for index in range(ACTION_HORIZON)]},
        download_videos=True,
    )
    lookup = _absolute_frame_lookup(dataset)
    train_dataset = LeRobotDataset(
        manifest.dataset_repo_id,
        root=args.dataset_root,
        episodes=list(split.train_episodes),
        revision=manifest.dataset_revision,
        download_videos=False,
    )
    train_raw = train_dataset.reader.hf_dataset
    if train_raw is None:
        raise RuntimeError("training dataset reader did not activate")
    mean_action = torch.stack([train_raw[index]["action"].float() for index in range(len(train_raw))]).mean(
        dim=0
    )
    samples = []
    for entry in entries:
        relative_index = lookup.get((entry.episode_index, entry.frame_index))
        if relative_index is None:
            raise RuntimeError(
                f"manifest frame is absent from dataset: {entry.episode_index}/{entry.frame_index}"
            )
        row = dataset[relative_index]
        cache_item = cache.load_entry(entry)
        target = row["action"].float()
        padding = row["action_is_pad"].bool()
        if target.shape != (ACTION_HORIZON, ACTION_DIM):
            raise RuntimeError(f"unexpected target shape for {entry.sample_id}: {tuple(target.shape)}")
        if not torch.allclose(target, cache_item.target_action.float(), atol=1e-4, rtol=1e-4):
            raise RuntimeError(f"cache and dataset action targets differ for {entry.sample_id}")
        if not torch.equal(padding, cache_item.action_is_pad.squeeze(0).bool()):
            raise RuntimeError(f"cache and dataset padding masks differ for {entry.sample_id}")
        if not torch.allclose(
            row["observation.state"].float(), cache_item.state.squeeze(0).float(), atol=1e-4, rtol=1e-4
        ):
            raise RuntimeError(f"cache and dataset states differ for {entry.sample_id}")
        observation = {key: value for key, value in row.items() if key.startswith("observation.")}
        observation["task"] = row["task"]
        samples.append(
            {
                "sample_id": entry.sample_id,
                "target": target,
                "padding": padding,
                "state": row["observation.state"].float(),
                "context": cache_item.context,
                "observation": observation,
            }
        )
    batches = []
    for start in range(0, len(samples), args.batch_size):
        group = samples[start : start + args.batch_size]
        contexts = torch.cat([sample["context"] for sample in group]).to(dtype=torch.bfloat16)
        if args.vam_zero_context:
            contexts = torch.zeros_like(contexts)
        batches.append(
            EvaluationBatch(
                sample_ids=tuple(sample["sample_id"] for sample in group),
                target_actions=torch.stack([sample["target"] for sample in group]),
                action_is_pad=torch.stack([sample["padding"] for sample in group]),
                current_state=torch.stack([sample["state"] for sample in group]),
                contexts=contexts,
                observations=tuple(sample["observation"] for sample in group),
            )
        )
    protocol = {
        "protocol_version": "1.0",
        "dataset_repo_id": manifest.dataset_repo_id,
        "dataset_revision": manifest.dataset_revision,
        "train_episodes": list(split.train_episodes),
        "validation_episodes": list(split.val_episodes),
        "mean_training_action": [float(value) for value in mean_action],
        "ordered_sample_ids": [sample["sample_id"] for sample in samples],
        "ordered_frame_ids": [
            {"episode_index": entry.episode_index, "frame_index": entry.frame_index} for entry in entries
        ],
        "manifest": str(args.manifest.resolve()),
        "action_horizon": ACTION_HORIZON,
        "action_dim": ACTION_DIM,
        "normalization": {
            "vam": _vam_normalization_description(args),
            "smolvla": "checkpoint-native dataset-wide mean/std",
        },
        "vam_context": "zeroed" if args.vam_zero_context else "cosmos_layer20_hidden",
        "padding": "action_is_pad=true tokens excluded",
        "aggregation": "sqrt(sum(masked_squared_error) / n_valid_scalars)",
        "seed": args.seed,
    }
    return batches, protocol


def _enable_cpu_vam_runtime() -> None:
    """Replace the CUDA-only TE RMSNorm import with an equivalent torch module."""
    from types import SimpleNamespace

    from lerobot.policies.vam._vendor.cosmos_predict2.models import world2action_dit

    class TorchRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def reset_parameters(self) -> None:
            torch.nn.init.ones_(self.weight)

        def get_extra_state(self) -> None:
            return None

        def set_extra_state(self, state: object) -> None:
            del state

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            variance = value.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = value * torch.rsqrt(variance + self.eps).to(dtype=value.dtype)
            return normalized * self.weight.to(dtype=value.dtype)

    te_module = SimpleNamespace(pytorch=SimpleNamespace(RMSNorm=TorchRMSNorm))
    world2action_dit._load_transformer_engine = lambda: (te_module, None, None)


def _recorded_sigma(checkpoint: Path) -> dict[str, Any]:
    """Read the sigma configuration a training run recorded next to its weights."""
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.is_file():
        metadata_path = checkpoint.parent / "last.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text())
    hyperparameters = payload.get("hyperparameters", {})
    backbone = payload.get("backbone", {})
    return {
        "metadata_path": str(metadata_path),
        "training_video_sigma": hyperparameters.get("video_sigma"),
        "training_video_sigma_distribution": hyperparameters.get("video_sigma_distribution"),
        "recorded_evaluation_sigma": hyperparameters.get("evaluation_sigma"),
        "backbone_high_noise_sigma": backbone.get("high_noise_sigma"),
    }


def _resolve_vam_sigma(args: argparse.Namespace, recorded: dict[str, Any]) -> float:
    """Pick one fixed, explicit evaluation sigma and never fall through to a default."""
    for candidate in (
        args.vam_sigma,
        recorded.get("recorded_evaluation_sigma"),
        recorded.get("training_video_sigma"),
        recorded.get("backbone_high_noise_sigma"),
    ):
        if candidate is not None:
            return float(candidate)
    raise ValueError("could not determine the VAM evaluation sigma; pass --vam-sigma explicitly")


def _make_vam_backend(args: argparse.Namespace) -> tuple[VAMBackend | None, dict[str, Any]]:
    if args.vam_checkpoint is None:
        return None, {}
    if args.device == "cpu":
        _enable_cpu_vam_runtime()
    normalizer = ActionStateNormalizer.load(args.vam_normalizer)
    decoder = World2ActionDecoder(
        World2ActionConfig(device=args.device, dtype=torch.bfloat16),
        normalizer=normalizer,
    )
    if args.vam_step is None:
        from safetensors.torch import load_file

        state = load_file(str(args.vam_checkpoint), device=args.device)
        expected = decoder.denoiser.state_dict()
        state = {
            key: value.to(dtype=expected[key].dtype)
            if expected[key] is not None and value.dtype != expected[key].dtype
            else value
            for key, value in state.items()
        }
        decoder.denoiser.load_state_dict(state, strict=True, assign=True)
        decoder.denoiser.to(device=args.device)
    else:
        load_diagnostic_checkpoint(
            decoder,
            args.vam_checkpoint,
            manifest_path=args.manifest,
            normalizer_path=args.vam_normalizer,
            expected_step=args.vam_step,
        )
    recorded = _recorded_sigma(args.vam_checkpoint.expanduser().resolve())
    sigma = _resolve_vam_sigma(args, recorded)
    provenance = {
        **recorded,
        "evaluation_sigma": sigma,
        "evaluation_sigma_source": "cli" if args.vam_sigma is not None else "checkpoint_metadata",
    }
    context_timestep = torch.full((args.batch_size, 1), sigma, dtype=torch.float32)
    return VAMBackend(decoder, context_timestep=context_timestep), provenance


def main() -> int:
    args = parse_args()
    batches, protocol = _make_batches(args)
    backends: list[ActionPredictionBackend] = [
        StateRepeatBackend(),
        MeanActionBackend(torch.tensor(protocol["mean_training_action"])),
    ]
    vam, vam_sigma_provenance = _make_vam_backend(args)
    if vam is not None:
        backends.append(vam)
        protocol["vam_sigma"] = vam_sigma_provenance
    if args.smolvla_checkpoint is not None:
        backends.append(
            SmolVLABackend.from_pretrained(str(args.smolvla_checkpoint), device=torch.device(args.device))
        )
    results = {backend.name: evaluate_backend(backend, batches, seed=args.seed) for backend in backends}
    payload = {"protocol": protocol, "backends": results}
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({name: result["aggregate_rmse_deg"] for name, result in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
