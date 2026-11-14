#!/usr/bin/env python3
"""Unified SmolExpert Action Policy Trainer for Video Action Models.

Trains SmolVLA's action expert head on intermediate spatiotemporal features from
any video diffusion/flow-matching backbone (Cosmos 2B, Cosmos 7B, Cosmos 14B,
LTX-2.5, FLUX.2, etc.).

Guarantees:
1. Protocol 1.0 / Scale-100 Split Enforcement: Prevents frame-level data leakage;
   enforces strictly disjoint episode splits (Protocol 1: 0-31 train, 32-39 val;
   Scale-100 Eval-Set 2: strictly 90-99).
2. Strict Cache Invariants: Required state, target_action, action_is_pad, features,
   cryptographic SHA-256 checks, finite tensor validation, and no zero fallbacks.
   Handles legacy state tensor shapes [6], [1, 6], [1, 1, 6] safely.
   Single-pass file integrity verification in constructor avoids repeated per-epoch hash overhead.
3. Train-only Normalization: Derives SmolVLA normalizer exclusively from train episodes;
   eval-only requires pre-existing checkpoint normalizer without recomputation.
4. Independent & Invariant Evaluation: Per-sample seeded noise via SHA256(dataset_id:sample_id:seed),
   deterministic flow t/epsilon, batch-size / order invariance, separate arm_rmse_deg
   and gripper_rmse metrics, and legacy mean first-5 alongside pooled first-5 RMSE.
5. Exact Gradient Accumulation: Optimizer update steps encompass full microbatch loops.
6. Feature Signature Consistency: Strict verification against mixed base vs LoRA caches.
7. Fail-Closed Model Loading: Real model loading fails closed without toy fallback unless --dry-run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torchvision.transforms import functional as tf_f

from lerobot.common.run_artifacts import (
    atomic_write_json,
    code_identity,
    default_run_dir,
    save_weights_artifact,
)
from lerobot.datasets.vam import CUBE_OUT_OF_BOX_CONTRACT
from lerobot.policies.vam.base.split_guard import (
    validate_manifests_for_protocol,
)
from lerobot.policies.vam.smol_expert import (
    ACTION_DIM,
    ACTION_HORIZON,
    SMOLVLA_CHECKPOINT,
    SmolExpertActionDecoder,
    SmolVLANormalizer,
)
from lerobot.utils.rotation import Rotation

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_sample_seed(dataset_id: str, sample_id: str, base_seed: int, offset: int = 0) -> int:
    """Derive a stable 31-bit integer seed from dataset identity, sample_id, and base seed."""
    seed_str = f"{dataset_id}:{sample_id}:{base_seed}:{offset}"
    digest = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


@dataclass(frozen=True, slots=True)
class UnifiedCacheItem:
    """Standardized cached observation sample."""

    sample_id: str
    episode_index: int
    frame_index: int
    context: Tensor  # [S, C]
    state: Tensor  # [ACTION_DIM]
    action: Tensor  # [ACTION_HORIZON, ACTION_DIM]
    action_is_pad: Tensor  # [ACTION_HORIZON] boolean


class UnifiedFeatureCacheDataset:
    """Unified dataset loading cached features with strict schema and finiteness validation.

    Integrity checks (SHA-256 and file existence/finiteness) are executed once per unique
    file path during initialization, preventing catastrophic per-step hash overhead.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        context_transform: str = "auto",
        verify_hashes: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path, encoding="utf-8") as f:
            self.payload = json.load(f)

        if "entries" not in self.payload or not isinstance(self.payload["entries"], list):
            raise ValueError(f"Manifest {self.manifest_path} must contain an 'entries' list")

        self.entries = self.payload["entries"]
        self.context_transform = context_transform
        self.root_dir = self.manifest_path.parent
        self.dataset_id = str(
            self.payload.get("dataset", {}).get("repo_id")
            or self.payload.get("provenance", {}).get("backbone")
            or self.manifest_path.parent.name
        )

        # Pre-verify file existence, duplicate paths, and SHA-256 hashes once in constructor
        seen_sample_ids: set[str] = set()
        seen_files: dict[str, str | None] = {}

        for idx, entry in enumerate(self.entries):
            file_key = entry.get("safetensors") or entry.get("artifact_file")
            if not file_key:
                raise ValueError(f"Entry {idx} missing 'safetensors' or 'artifact_file' key: {entry}")

            sample_id = str(entry.get("sample_id", f"sample_{idx}"))
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id detected in manifest: {sample_id}")
            seen_sample_ids.add(sample_id)

            expected_hash = entry.get("sha256") or entry.get("safetensors_sha256") or entry.get("hash")
            if file_key not in seen_files:
                tensor_path = self.root_dir / file_key
                if not tensor_path.is_file():
                    raise FileNotFoundError(f"Artifact file not found: {tensor_path}")
                if verify_hashes and expected_hash:
                    actual_hash = sha256_file(tensor_path)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"SHA-256 mismatch for {tensor_path}: expected {expected_hash}, got {actual_hash}"
                        )
                seen_files[file_key] = expected_hash

        # Cache of parsed tensors per unique file_key to avoid re-reading and re-verifying
        self._verified_files: set[str] = set()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> UnifiedCacheItem:
        entry = self.entries[idx]
        file_key = entry.get("safetensors") or entry.get("artifact_file")
        tensor_path = self.root_dir / file_key

        data = load_file(str(tensor_path))

        # 1. Resolve context / features (Strict requirement: no fallback)
        raw_context = data.get("context") if "context" in data else data.get("features")
        if raw_context is None:
            raise KeyError(f"Artifact {tensor_path} does not contain required 'context' or 'features' tensor")

        if raw_context.ndim == 3 and raw_context.shape[0] == 1:
            raw_context = raw_context.squeeze(0)

        # Apply spatial pooling if requested and applicable
        if self.context_transform == "pool2" and raw_context.shape[0] == 19200:
            from lerobot.policies.vam.context_transform import apply_context_transform

            raw_context = apply_context_transform(raw_context.unsqueeze(0), "pool2").squeeze(0)

        # 2. Resolve state (Strict requirement: no zero fallback, supports [6], [1, 6], [1, 1, 6])
        state = data.get("state")
        if state is None:
            raise KeyError(f"Artifact {tensor_path} does not contain required 'state' tensor")
        if state.ndim > 1:
            state = state.squeeze()
        state = state.float()

        # 3. Resolve target action (Strict requirement: no zero fallback)
        action = data.get("target_action") if "target_action" in data else data.get("action")
        if action is None:
            raise KeyError(
                f"Artifact {tensor_path} does not contain required 'target_action' or 'action' tensor"
            )
        if action.ndim == 3 and action.shape[0] == 1:
            action = action.squeeze(0)
        action = action.float()

        # 4. Resolve action_is_pad (Strict requirement: no zero fallback)
        action_is_pad = data.get("action_is_pad")
        if action_is_pad is None:
            raise KeyError(f"Artifact {tensor_path} does not contain required 'action_is_pad' tensor")
        if action_is_pad.ndim > 1:
            action_is_pad = action_is_pad.squeeze()
        action_is_pad = action_is_pad.to(dtype=torch.bool)

        # Validate tensor shapes and finiteness (checked once per file path)
        if file_key not in self._verified_files:
            if raw_context.ndim != 2 or raw_context.shape[0] == 0 or raw_context.shape[1] == 0:
                raise ValueError(
                    f"Context tensor in {tensor_path} must have shape [S, C] with S>0, C>0; got {raw_context.shape}"
                )
            if not torch.isfinite(raw_context).all().item():
                raise ValueError(f"Context tensor in {tensor_path} contains non-finite values (NaN / Inf)")

            if state.shape != (ACTION_DIM,):
                raise ValueError(
                    f"State tensor in {tensor_path} must have shape [{ACTION_DIM}], got {state.shape}"
                )
            if not torch.isfinite(state).all().item():
                raise ValueError(f"State tensor in {tensor_path} contains non-finite values (NaN / Inf)")

            if action.shape != (ACTION_HORIZON, ACTION_DIM):
                raise ValueError(
                    f"Action tensor in {tensor_path} must have shape [{ACTION_HORIZON}, {ACTION_DIM}], got {action.shape}"
                )
            if not torch.isfinite(action).all().item():
                raise ValueError(f"Action tensor in {tensor_path} contains non-finite values (NaN / Inf)")

            if action_is_pad.shape != (ACTION_HORIZON,):
                raise ValueError(
                    f"action_is_pad in {tensor_path} must have shape [{ACTION_HORIZON}], got {action_is_pad.shape}"
                )
            self._verified_files.add(file_key)

        sample_id = str(entry.get("sample_id", f"sample_{idx}"))
        episode_index = int(entry.get("episode_index", 0))
        frame_index = int(entry.get("frame_index", 0))

        return UnifiedCacheItem(
            sample_id=sample_id,
            episode_index=episode_index,
            frame_index=frame_index,
            context=raw_context,
            state=state,
            action=action,
            action_is_pad=action_is_pad,
        )


COSMOS3_CHECKPOINT_DIR = Path("/home/anton/.cache/video-vam/cosmos3-edge")
COSMOS3_CACHE_BUILDER = "scripts/video_vam/extract_cosmos3_edge_pure_vision.py"


# Snapshot used by every Scale-100 run so far. NOTE: this snapshot's metadata declares 100 episodes /
# 12,163 frames while 140 episodes / 15,998 rows are present (see docs/video_vam_status.md).
SCALE100_V2_REVISION = "5d0325cc1412f4774223a0beb528958108814962"


def is_cosmos3(backbone: str | None) -> bool:
    return backbone is not None and backbone.replace("-", "_").lower() in {
        "cosmos3",
        "cosmos3_edge",
        "cosmos3_nano",
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_sha256(value: Any, source: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"Missing or invalid SHA-256: {source}")
    return value


def cosmos3_eval_identity(
    *,
    checkpoint_path: Path,
    lora_checkpoint: Path | None,
    lora_rank: int,
    lora_alpha: float,
    hidden_layer: int,
    prompt: str,
    backbone_name: str = "cosmos3-edge",
) -> dict[str, Any]:
    """Resolve current file identity without constructing a backbone or using a GPU."""
    from lerobot.policies.vam.cosmos3_features import checkpoint_digest

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    transformer_dir = checkpoint_path / "transformer"
    if not transformer_dir.is_dir():
        transformer_dir = checkpoint_path
    # The directory digest also includes JSON: metadata alone is not a backbone.
    for component, directory in (("transformer", transformer_dir), ("VAE", checkpoint_path / "vae")):
        if not any(
            path.is_file() and path.stat().st_size > 0
            for pattern in ("diffusion_pytorch_model*.safetensors", "diffusion_pytorch_model*.bin")
            for path in directory.glob(pattern)
        ):
            raise FileNotFoundError(f"Cosmos3 checkpoint lacks nonempty {component} weights: {directory}")

    is_nano = backbone_name.replace("-", "_").lower() == "cosmos3_nano"
    max_layers = 36 if is_nano else 28
    if type(hidden_layer) is not int or not 1 <= hidden_layer <= max_layers or not isinstance(prompt, str):
        raise ValueError("Invalid Cosmos3 layer or prompt")
    lora_sha = None
    rank = None
    alpha = None
    if lora_checkpoint is not None:
        from lerobot.policies.vam.cosmos3_lora import (
            cosmos3_adapter_weights_file,
            read_cosmos3_adapter_hyperparameters,
        )

        lora_checkpoint = Path(lora_checkpoint).expanduser().resolve()
        lora_sha = sha256_file(cosmos3_adapter_weights_file(lora_checkpoint))
        # The loader gives adapter metadata precedence over CLI values. Do not
        # reproduce its silent fallback for malformed metadata in an identity check.
        try:
            recorded = read_cosmos3_adapter_hyperparameters(lora_checkpoint)
            rank, alpha = recorded if recorded is not None else (int(lora_rank), float(lora_alpha))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid Cosmos3 LoRA rank/alpha metadata: {lora_checkpoint}") from exc
        if rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid Cosmos3 LoRA rank/alpha: {lora_checkpoint}")
    return {
        "backbone": "cosmos3-nano" if is_nano else "cosmos3-edge",
        # This producer's legacy field is a directory digest, NOT the single
        # transformer-file hash returned by Cosmos3FeatureExtractor.get_provenance().
        "transformer_sha256": checkpoint_digest(checkpoint_path),
        "lora_sha256": lora_sha,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "hidden_layer": hidden_layer,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "fps": 10.0,
        "base_fps": 24.0,
        "context_tokens": 600,
        "context_dim": 4096 if is_nano else 2048,
    }


def read_adapter_hyperparameters(path: Path) -> tuple[int, float] | None:
    """Return the (rank, alpha) an adapter was trained with, from safetensors metadata or sidecar."""

    from lerobot.policies.vam.cosmos3_lora import is_peft_adapter, read_cosmos3_adapter_hyperparameters
    from lerobot.policies.vam.cosmos_lora import read_lora_sidecar_hyperparameters

    if is_peft_adapter(path):
        return read_cosmos3_adapter_hyperparameters(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
    for rank_key, alpha_key in (("rank", "alpha"), ("lora_rank", "lora_alpha")):
        if rank_key in meta and alpha_key in meta:
            return int(meta[rank_key]), float(meta[alpha_key])
    return read_lora_sidecar_hyperparameters(path)


def resolve_backbone_lora_hyperparameters(args: argparse.Namespace) -> None:
    """Fill ``--backbone-lora-rank/alpha`` from the adapter and reject contradicting CLI values.

    Passing an alpha different from the one the adapter was trained with silently rescales the
    LoRA delta (this happened for the Cosmos 2B step-6000 adapter: trained alpha=16, loaded as 32).
    """
    if args.backbone_lora_weights is None:
        args.backbone_lora_rank = args.backbone_lora_rank or 16
        args.backbone_lora_alpha = args.backbone_lora_alpha or 32.0
        return
    from lerobot.policies.vam.cosmos3_lora import is_peft_adapter

    path = Path(args.backbone_lora_weights).expanduser()
    if not (path.is_file() or is_peft_adapter(path)):
        raise FileNotFoundError(f"--backbone-lora-weights not found (file or PEFT adapter dir): {path}")
    try:
        recorded = read_adapter_hyperparameters(path)
    except Exception as exc:  # noqa: BLE001 - unreadable header: fall back to explicit values
        if args.backbone_lora_rank is None or args.backbone_lora_alpha is None:
            raise ValueError(f"Cannot read rank/alpha from adapter {path}: {exc}") from exc
        return
    if recorded is None:
        if args.backbone_lora_rank is None or args.backbone_lora_alpha is None:
            raise ValueError(
                f"Adapter {path} has no rank/alpha metadata or sidecar; pass "
                "--backbone-lora-rank and --backbone-lora-alpha explicitly."
            )
        return
    rank, alpha = recorded
    if args.backbone_lora_rank is not None and args.backbone_lora_rank != rank:
        raise ValueError(f"--backbone-lora-rank {args.backbone_lora_rank} != adapter rank {rank} ({path})")
    if args.backbone_lora_alpha is not None and not math.isclose(args.backbone_lora_alpha, alpha):
        raise ValueError(
            f"--backbone-lora-alpha {args.backbone_lora_alpha} != adapter alpha {alpha} ({path})"
        )
    args.backbone_lora_rank, args.backbone_lora_alpha = rank, alpha


def _find_lora_record(provenance: dict[str, Any]) -> dict[str, Any] | None:
    """Locate the LoRA provenance record across the manifest layouts written by our cache builders."""
    for key in ("lora_weights", "lora"):
        value = provenance.get(key)
        if isinstance(value, dict):
            return value
    weights = provenance.get("weights")
    if isinstance(weights, dict) and isinstance(weights.get("lora"), dict):
        return weights["lora"]
    return None


def validate_online_eval_cache_identity(path: Path, args: argparse.Namespace, *, source: str) -> None:
    """Fail if an offline eval cache was not produced by the same feature extractor as online training.

    Applies to non-Cosmos3 online backbones (Cosmos3 has its own stricter preflight). Checks the
    fields that previously differed silently: LoRA file hash, LoRA alpha/rank, Cosmos 2B sigma and
    tapped layer, and augmentation.
    """
    payload = _read_json_object(path)
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{source} cache has no provenance; cannot verify online feature identity: {path}")
    problems: list[str] = []

    lora = _find_lora_record(provenance)
    lora_sha = lora.get("sha256") if lora else provenance.get("lora_sha256")
    if args.backbone_lora_weights is None:
        if lora_sha:
            problems.append("cache was built WITH a LoRA but online training uses none")
    else:
        from lerobot.policies.vam.cosmos3_lora import cosmos3_adapter_weights_file

        expected_sha = sha256_file(
            cosmos3_adapter_weights_file(Path(args.backbone_lora_weights).expanduser())
        )
        if lora_sha != expected_sha:
            problems.append(f"LoRA sha256 {lora_sha!r} != online adapter {expected_sha!r}")
        if lora is not None:
            if "alpha" in lora and not math.isclose(float(lora["alpha"]), float(args.backbone_lora_alpha)):
                problems.append(f"LoRA alpha {lora['alpha']} != online {args.backbone_lora_alpha}")
            if "rank" in lora and int(lora["rank"]) != int(args.backbone_lora_rank):
                problems.append(f"LoRA rank {lora['rank']} != online {args.backbone_lora_rank}")

    b_key = (args.online_backbone or "").replace("-", "_").lower()
    if "cosmos" in b_key:
        sigma = provenance.get("high_noise_sigma")
        if sigma is None or not math.isclose(float(sigma), float(args.high_noise_sigma)):
            problems.append(f"high_noise_sigma {sigma!r} != online {args.high_noise_sigma}")
        state_t = provenance.get("state_t")
        if state_t is not None and int(state_t) != 2:
            problems.append(f"state_t {state_t} != online 2")
        layer = provenance.get("hidden_layer")
        expected_layer = args.backbone_layer if args.backbone_layer is not None else 20
        if layer is not None and int(layer) != int(expected_layer):
            problems.append(f"hidden_layer {layer} != online {expected_layer}")
    for key in ("augment", "augmented"):
        if provenance.get(key):
            problems.append(f"evaluation cache is augmented ({key})")
    if problems:
        raise ValueError(
            f"{source} cache {path} does not match the online feature extractor:\n  - "
            + "\n  - ".join(problems)
            + "\nRebuild the cache with matching settings or omit it to validate with the online extractor."
        )


def _cosmos3_identity_from_args(args: argparse.Namespace) -> dict[str, Any]:
    is_nano = (args.online_backbone or "").replace("-", "_").lower() == "cosmos3_nano"
    default_ckpt = Path("/home/anton/.cache/video-vam/cosmos3-nano") if is_nano else COSMOS3_CHECKPOINT_DIR
    return cosmos3_eval_identity(
        checkpoint_path=args.backbone_checkpoint or default_ckpt,
        lora_checkpoint=args.backbone_lora_weights,
        lora_rank=args.backbone_lora_rank,
        lora_alpha=args.backbone_lora_alpha,
        hidden_layer=args.backbone_layer,
        prompt=args.backbone_prompt,
        backbone_name="cosmos3-nano" if is_nano else "cosmos3-edge",
    )


def validate_cosmos3_cache_identity(
    payload: dict[str, Any], expected: dict[str, Any], *, source: str
) -> None:
    """Only the known, unaugmented pure-vision producer has supported cache semantics."""
    provenance = payload.get("provenance")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError(f"Unsupported Cosmos3 cache schema: {source}")
    if not isinstance(provenance, dict) or provenance.get("builder") != COSMOS3_CACHE_BUILDER:
        raise ValueError(f"Unknown Cosmos3 pure-vision cache producer: {source}")
    for key, value in expected.items():
        if key not in provenance:
            raise ValueError(f"Missing Cosmos3 cache identity field {key}: {source}")
        actual = provenance[key]
        valid_type = type(actual) is type(value) or (
            isinstance(value, float) and type(actual) in (int, float)
        )
        if not valid_type or actual != value:
            raise ValueError(
                f"Cosmos3 cache identity mismatch for {key}: {source}; expected {value!r}, got {actual!r}"
            )
    if (
        "lora_checkpoint" not in provenance
        or (expected["lora_sha256"] is None and provenance["lora_checkpoint"] is not None)
        or (
            expected["lora_sha256"] is not None
            and (not isinstance(provenance["lora_checkpoint"], str) or not provenance["lora_checkpoint"])
        )
    ):
        raise ValueError(f"Missing or inconsistent Cosmos3 lora_checkpoint identity: {source}")
    # Legacy schema 1 has no augmentation flag; its supported producer always
    # extracts the unaugmented five-frame window. Never infer this for other builders.
    for key in ("augment", "augmented"):
        if key in provenance and provenance[key] is not False:
            raise ValueError(f"Cosmos3 evaluation cache must be unaugmented ({key}): {source}")
    if provenance.get("preprocessing_version", "cosmos3_native_v2") != "cosmos3_native_v2":
        raise ValueError(f"Unknown Cosmos3 cache preprocessing: {source}")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("dtype") != "bfloat16":
        raise ValueError(f"Missing or mismatched Cosmos3 cache dtype: {source}")


def load_cosmos3_eval_cache(
    path: Path, expected: dict[str, Any], *, context_transform: str, split: str
) -> UnifiedFeatureCacheDataset:
    payload = _read_json_object(path)
    validate_cosmos3_cache_identity(payload, expected, source=str(path))
    dataset_metadata = payload.get("dataset", {})
    if not isinstance(dataset_metadata, dict) or not all(
        isinstance(dataset_metadata.get(k), str) and dataset_metadata[k] for k in ("repo_id", "revision")
    ):
        raise ValueError(f"Missing Cosmos3 cache dataset identity: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Cosmos3 evaluation cache requires nonempty entries: {path}")
    episodes = range(32, 40) if split == "val" else range(90, 100)
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or type(entry.get("episode_index")) is not int
            or entry["episode_index"] not in episodes
        ):
            raise ValueError(f"Invalid {split} episode in Cosmos3 cache: {path}")
        if not isinstance(entry.get("sample_id"), str) or not entry["sample_id"]:
            raise ValueError(f"Missing Cosmos3 evaluation sample_id: {path}")
        _require_sha256(entry.get("safetensors_sha256"), str(path))
    dataset = UnifiedFeatureCacheDataset(path, context_transform=context_transform)
    if dataset.payload != payload:
        raise ValueError(f"Cosmos3 manifest changed during validation: {path}")
    for index in range(len(dataset)):
        context = dataset[index].context
        if tuple(context.shape) != (600, 2048) or context.dtype != torch.bfloat16:
            raise ValueError(f"Cosmos3 cache context must be bfloat16 [600, 2048]: {path}")
    return dataset


def preflight_cosmos3_eval_caches(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, UnifiedFeatureCacheDataset]]:
    paths = {"val": args.val_manifest, "eval2": args.eval2_manifest}
    if not is_cosmos3(args.online_backbone) or not any(path is not None for path in paths.values()):
        return None, {}
    for path in paths.values():
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"Supplied Cosmos3 evaluation manifest not found: {path}")
    identity = _cosmos3_identity_from_args(args)
    datasets = {
        split: load_cosmos3_eval_cache(
            Path(path), identity, context_transform=args.context_transform, split=split
        )
        for split, path in paths.items()
        if path is not None
    }
    return identity, datasets


def compute_training_normalizer(
    train_dataset: UnifiedFeatureCacheDataset,
    train_episodes: Sequence[int],
    output_dir: Path,
) -> SmolVLANormalizer:
    """Compute SmolVLA normalization statistics strictly from training episodes."""
    states: list[Tensor] = []
    actions: list[Tensor] = []
    paddings: list[Tensor] = []

    for idx in range(len(train_dataset)):
        item = train_dataset[idx]
        states.append(item.state)
        actions.append(item.action)
        paddings.append(item.action_is_pad)

    all_states = torch.stack(states, dim=0)
    all_actions = torch.stack(actions, dim=0)
    all_paddings = torch.stack(paddings, dim=0)

    # action_is_pad ensures padded actions are excluded from population mean/std
    normalizer = SmolVLANormalizer.from_training_tensors(
        state=all_states,
        action=all_actions,
        action_is_pad=all_paddings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / "normalizer.safetensors"
    save_file(
        normalizer.state_dict(),
        str(norm_path),
        metadata={"artifact": "smolvla_mean_std_normalizer", "source_split": "train"},
    )

    meta_payload = {
        "artifact": "smolvla_mean_std_normalizer",
        "source_split": "train",
        "train_episodes": list(train_episodes),
        "anchor_count": len(train_dataset),
        "padded_actions_excluded": True,
        "state_mean": normalizer.state_mean.tolist(),
        "state_std": normalizer.state_std.tolist(),
        "action_mean": normalizer.action_mean.tolist(),
        "action_std": normalizer.action_std.tolist(),
    }
    (output_dir / "normalizer.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
    return normalizer


class OnlineVideoItem:
    """Individual sample item streamed from an online video dataset."""

    __slots__ = (
        "rgb",
        "state",
        "action",
        "action_is_pad",
        "sample_id",
        "episode_index",
        "frame_index",
        "context",
    )

    def __init__(
        self,
        rgb: Tensor,
        state: Tensor,
        action: Tensor,
        action_is_pad: Tensor,
        sample_id: str,
        episode_index: int,
        frame_index: int,
    ) -> None:
        self.rgb = rgb
        self.state = state
        self.action = action
        self.action_is_pad = action_is_pad
        self.sample_id = sample_id
        self.episode_index = episode_index
        self.frame_index = frame_index
        self.context: Tensor | None = None


def joints_to_cartesian(joints: torch.Tensor, kin: Any) -> torch.Tensor:
    """Convert joint positions [B, T, 6] or [B, 6] to Cartesian end-effector targets [x, y, z, wx, wy, wz, grip]."""
    orig_ndim = joints.ndim
    if orig_ndim == 2:
        joints = joints.unsqueeze(1)
    b_size, t_size, _ = joints.shape
    q_np = joints.detach().cpu().numpy()
    out = np.zeros((b_size, t_size, 7), dtype=np.float32)
    for b in range(b_size):
        for t in range(t_size):
            pose = kin.forward_kinematics(q_np[b, t])
            xyz = pose[:3, 3]
            tw = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
            grip = q_np[b, t, 5]
            out[b, t] = [xyz[0], xyz[1], xyz[2], tw[0], tw[1], tw[2], grip]
    res = torch.from_numpy(out).to(device=joints.device, dtype=torch.float32)
    return res.squeeze(1) if orig_ndim == 2 else res


def cartesian_to_joints_ik(
    q_current: torch.Tensor,
    cart_chunk: torch.Tensor,
    kin: Any,
    num_iters: int = 5,
    orientation_weight: float = 0.01,
) -> torch.Tensor:
    """Convert Cartesian predictions [B, 30, 7] back to joint angles [B, 30, 6] via iterative differential IK."""
    b_size, t_size, _ = cart_chunk.shape
    q_curr_np = q_current.detach().cpu().numpy()
    cart_np = cart_chunk.detach().cpu().numpy()
    out_joints = np.zeros((b_size, t_size, 6), dtype=np.float32)

    # Enforce physical servo limits (+/- 110 deg) so the QP never jumps to unphysical backward branches.
    # Resolve q-vector indices by joint name instead of assuming placo's floating-base offset.
    model = kin.robot.model
    for joint_name in (name for name in kin.joint_names if name != "gripper"):
        q_idx = model.joints[model.getJointId(joint_name)].idx_q
        model.lowerPositionLimit[q_idx] = -np.deg2rad(110.0)
        model.upperPositionLimit[q_idx] = np.deg2rad(110.0)
    kin.solver.enable_joint_limits(True)

    for b in range(b_size):
        q_step = q_curr_np[b].copy()
        for t in range(t_size):
            xyz = cart_np[b, t, :3]
            tw = cart_np[b, t, 3:6]
            grip = cart_np[b, t, 6]

            target_pose = np.eye(4)
            target_pose[:3, :3] = Rotation.from_rotvec(tw).as_matrix()
            target_pose[:3, 3] = xyz

            for _ in range(num_iters):
                q_step = kin.inverse_kinematics(
                    q_step, target_pose, position_weight=1.0, orientation_weight=orientation_weight
                )
            q_step[5] = grip
            out_joints[b, t] = q_step
    return torch.from_numpy(out_joints).to(device=cart_chunk.device, dtype=torch.float32)


class OnlineVideoDataset(torch.utils.data.Dataset[OnlineVideoItem]):
    """Online video dataset streaming consecutive temporal windows directly from LeRobotDataset."""

    def __init__(
        self,
        repo_id: str,
        *,
        root: Path | str | None = None,
        episodes: Sequence[int],
        stride: int = 1,
        contract: Any = CUBE_OUT_OF_BOX_CONTRACT,
        ram_headroom_gb: int = 26,
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.episodes = tuple(sorted(episodes))
        self.stride = stride
        self.contract = contract
        self.ram_headroom_gb = ram_headroom_gb
        self.manifest_path = Path(f"online://{repo_id}?stride={stride}&episodes={len(episodes)}")
        revision = getattr(contract, "revision", None) if contract.repo_id == repo_id else None
        self.payload: dict[str, Any] = {
            "source": "online_video_dataset",
            "repo_id": repo_id,
            "revision": revision,
            "stride": stride,
            "episodes": list(self.episodes),
        }

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.lerobot_ds = LeRobotDataset(
            self.repo_id,
            root=str(self.root) if self.root is not None else None,
            revision=revision,
            delta_timestamps=self.contract.delta_timestamps(),
            episodes=list(self.episodes),
            return_uint8=True,
            download_videos=False,
        )

        self.ep_starts: dict[int, int] = {}
        cum = 0
        for ep in self.episodes:
            self.ep_starts[ep] = cum
            cum += int(self.lerobot_ds.meta.episodes[ep]["length"])

        self.index_map: list[tuple[int, int, int]] = []
        for ep in self.episodes:
            ep_len = int(self.lerobot_ds.meta.episodes[ep]["length"])
            # Causal 5-frame window: earliest end frame is 4
            for f in range(4, ep_len, self.stride):
                local_idx = self.ep_starts[ep] + f
                self.index_map.append((ep, f, local_idx))

        self._ram_preloaded = False
        self._tabular_preloaded = False
        self._ram_ep_frames: dict[int, Tensor] = {}
        self._ram_ep_frame_pos: dict[int, dict[int, int]] = {}
        self._ram_states: list[Tensor] = []
        self._ram_actions: list[Tensor] = []
        self._ram_paddings: list[Tensor] = []
        self._ep_video_info: dict[int, tuple[Path, float, list[float]]] = {}
        self._cam_delta_indices: list[int] = [-4, -3, -2, -1, 0]
        self._tolerance_s: float = 1e-4
        self._video_backend: str | None = None
        self._maybe_preload_to_ram()

    @staticmethod
    def _get_available_ram_bytes() -> int:
        avail_bytes = 0
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    avail_bytes = int(line.split()[1]) * 1024
                    break
        elif (
            hasattr(os, "sysconf")
            and "SC_PAGE_SIZE" in os.sysconf_names
            and "SC_AVPHYS_PAGES" in os.sysconf_names
        ):
            avail_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))

        for cg_lim_p, cg_stat_p, cg_cur_p in (
            (
                Path("/sys/fs/cgroup/memory.max"),
                Path("/sys/fs/cgroup/memory.stat"),
                Path("/sys/fs/cgroup/memory.current"),
            ),
            (
                Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
                Path("/sys/fs/cgroup/memory/memory.stat"),
                Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
            ),
        ):
            if cg_lim_p.is_file():
                raw = cg_lim_p.read_text().strip()
                if raw.isdigit():
                    cg_lim = int(raw)
                    if 0 < cg_lim < (1 << 50):
                        used_rss = 0
                        if cg_stat_p.is_file():
                            for sline in cg_stat_p.read_text().splitlines():
                                if sline.startswith("total_rss ") or sline.startswith("anon "):
                                    used_rss = max(used_rss, int(sline.split()[1]))
                        if used_rss == 0 and cg_cur_p.is_file():
                            cur_raw = cg_cur_p.read_text().strip()
                            if cur_raw.isdigit():
                                used_rss = int(cur_raw)
                        cg_avail = max(0, cg_lim - used_rss)
                        avail_bytes = min(avail_bytes, cg_avail) if avail_bytes > 0 else cg_avail
        return avail_bytes

    def _maybe_preload_to_ram(self) -> None:
        from collections import defaultdict

        from lerobot.datasets.utils import shift_timestamps
        from lerobot.datasets.video_utils import decode_video_frames

        t0 = time.perf_counter()
        reader = self.lerobot_ds._ensure_reader()
        self._tolerance_s = reader._tolerance_s
        self._video_backend = reader._video_backend
        hf_ds = reader.hf_dataset if reader.hf_dataset is not None else reader.load_and_activate()
        vid_key = self.contract.camera_key
        if reader.delta_indices is not None and vid_key in reader.delta_indices:
            self._cam_delta_indices = list(reader.delta_indices[vid_key])

        rel_indices = [local_idx for _, _, local_idx in self.index_map]
        base = hf_ds[rel_indices]
        ep_idxs = [int(base["episode_index"][i]) for i in range(len(rel_indices))]
        abs_idxs = [int(base["index"][i]) for i in range(len(rel_indices))]
        query_indices_per_item = []
        paddings_per_item = []
        for abs_i, ep_i in zip(abs_idxs, ep_idxs, strict=True):
            q_idx, pad = reader._get_query_indices(abs_i, ep_i)
            query_indices_per_item.append(q_idx)
            paddings_per_item.append(pad)
        tabular_list = reader._query_hf_dataset(query_indices_per_item)
        for tab, pad in zip(tabular_list, paddings_per_item, strict=True):
            self._ram_states.append(tab[self.contract.state_key].squeeze(0).contiguous())
            self._ram_actions.append(tab[self.contract.action_key].contiguous())
            self._ram_paddings.append(pad[f"{self.contract.action_key}_is_pad"].contiguous())

        ts_col_all = reader._column_view("timestamp")[list(range(len(hf_ds)))]["timestamp"]
        for ep in self.episodes:
            ep_meta = reader._meta.episodes[ep]
            ep_len = int(ep_meta["length"])
            start_rel = self.ep_starts[ep]
            query_ts = [float(t) for t in ts_col_all[start_rel : start_rel + ep_len]]
            from_ts = float(ep_meta[f"videos/{vid_key}/from_timestamp"])
            video_path = reader.root / reader._meta.get_video_file_path(ep, vid_key)
            self._ep_video_info[ep] = (video_path, from_ts, query_ts)
        self._tabular_preloaded = True

        sparse_windows = self.stride > len(self._cam_delta_indices)
        ep_needed_frames: dict[int, list[int]] = {}
        if sparse_windows:
            for ep in self.episodes:
                ep_len = int(reader._meta.episodes[ep]["length"])
                ep_needed_frames[ep] = sorted(
                    {
                        max(0, min(ep_len - 1, f + d))
                        for f in range(4, ep_len, self.stride)
                        for d in self._cam_delta_indices
                    }
                )
            total_frames = sum(len(v) for v in ep_needed_frames.values())
        else:
            total_frames = sum(int(self.lerobot_ds.meta.episodes[ep]["length"]) for ep in self.episodes)
        cam_h, cam_w, cam_c = getattr(self.contract, "camera_shape", (480, 640, 3))
        frame_bytes = cam_h * cam_w * cam_c
        required_bytes = total_frames * frame_bytes
        avail_ram_bytes = self._get_available_ram_bytes()
        ram_headroom_bytes = self.ram_headroom_gb * (1024**3)
        usable_ram_bytes = max(0, avail_ram_bytes - ram_headroom_bytes)

        usable_vram_bytes = 0
        req_gb = required_bytes / (1024**3)
        avail_gb = avail_ram_bytes / (1024**3)
        usable_total_gb = (usable_ram_bytes + usable_vram_bytes) / (1024**3)
        print(
            f"[OnlineVideoDataset] Memory check for {len(self.episodes)} episodes ({total_frames} frames): "
            f"requires {req_gb:.2f} GB, free RAM {avail_gb:.2f} GB, usable RAM+VRAM budget {usable_total_gb:.2f} GB.",
            flush=True,
        )
        if usable_ram_bytes + usable_vram_bytes < frame_bytes * min(500, max(1, total_frames)):
            print("[OnlineVideoDataset] Insufficient RAM headroom; streaming videos from disk.", flush=True)
            return

        selected_eps: dict[int, str] = {}
        rem_ram = usable_ram_bytes
        rem_vram = usable_vram_bytes
        preload_frames_count = 0
        for ep in self.episodes:
            ep_frame_cnt = len(ep_needed_frames[ep]) if sparse_windows else int(reader._meta.episodes[ep]["length"])
            ep_b = ep_frame_cnt * frame_bytes
            if ep_b <= rem_ram:
                selected_eps[ep] = "cpu"
                rem_ram -= ep_b
                preload_frames_count += ep_frame_cnt
            elif ep_b <= rem_vram:
                selected_eps[ep] = "cuda"
                rem_vram -= ep_b
                preload_frames_count += ep_frame_cnt

        if not selected_eps:
            return

        file_groups: dict[Path, list[tuple[int, int, list[float]]]] = defaultdict(list)
        for ep in selected_eps:
            video_path, from_ts, query_ts = self._ep_video_info[ep]
            if sparse_windows:
                needed = ep_needed_frames[ep]
                self._ram_ep_frame_pos[ep] = {orig_f: pos for pos, orig_f in enumerate(needed)}
                ep_ts_sel = [query_ts[w] for w in needed]
            else:
                ep_ts_sel = query_ts
            shifted_ts = shift_timestamps(ep_ts_sel, from_ts)
            file_groups[video_path].append((ep, len(ep_ts_sel), shifted_ts))

        chunk_ep_count = 6
        for v_path, ep_entries in file_groups.items():
            ep_entries_sorted = sorted(ep_entries, key=lambda x: x[2][0] if x[2] else 0.0)
            for start_i in range(0, len(ep_entries_sorted), chunk_ep_count):
                sub_entries = ep_entries_sorted[start_i : start_i + chunk_ep_count]
                combined_ts: list[float] = []
                for _, _, s_ts in sub_entries:
                    combined_ts.extend(s_ts)
                all_frames = decode_video_frames(
                    v_path,
                    combined_ts,
                    reader._tolerance_s,
                    reader._video_backend,
                    return_uint8=True,
                ).squeeze(0)
                offset = 0
                for ep_id, ep_len, _ in sub_entries:
                    target_dev = selected_eps[ep_id]
                    ep_slice = all_frames[offset : offset + ep_len]
                    self._ram_ep_frames[ep_id] = (
                        ep_slice.to(device="cuda", non_blocking=True)
                        if target_dev == "cuda"
                        else ep_slice.clone()
                    )
                    offset += ep_len
                del all_frames

        self._ram_preloaded = len(self._ram_ep_frames) == len(self.episodes)
        elapsed = time.perf_counter() - t0
        cpu_eps = sum(1 for d in selected_eps.values() if d == "cpu")
        gpu_eps = sum(1 for d in selected_eps.values() if d == "cuda")
        print(
            f"[OnlineVideoDataset] Preloaded {len(selected_eps)}/{len(self.episodes)} episodes "
            f"(CPU: {cpu_eps}, VRAM: {gpu_eps}, {preload_frames_count} frames) in {elapsed:.1f}s.",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> OnlineVideoItem:
        ep, frame, local_idx = self.index_map[idx]
        sample_id = f"episode-{ep:04d}-frame-{frame:06d}"
        if ep in self._ram_ep_frames and self._tabular_preloaded:
            ep_frames = self._ram_ep_frames[ep]
            frame_pos = self._ram_ep_frame_pos.get(ep)
            if frame_pos is not None:
                ep_len = len(self._ep_video_info[ep][2])
                win = [frame_pos[max(0, min(ep_len - 1, frame + d))] for d in self._cam_delta_indices]
            else:
                ep_len = ep_frames.shape[0]
                win = [max(0, min(ep_len - 1, frame + d)) for d in self._cam_delta_indices]
            camera = ep_frames[win]
            return OnlineVideoItem(
                rgb=camera,
                state=self._ram_states[idx],
                action=self._ram_actions[idx],
                action_is_pad=self._ram_paddings[idx],
                sample_id=sample_id,
                episode_index=ep,
                frame_index=frame,
            )

        if self._tabular_preloaded and ep in self._ep_video_info:
            from lerobot.datasets.utils import shift_timestamps
            from lerobot.datasets.video_utils import decode_video_frames

            video_path, from_ts, ep_ts = self._ep_video_info[ep]
            ep_len = len(ep_ts)
            win = [max(0, min(ep_len - 1, frame + d)) for d in self._cam_delta_indices]
            win_ts = shift_timestamps([ep_ts[w] for w in win], from_ts)
            camera = decode_video_frames(
                video_path,
                win_ts,
                self._tolerance_s,
                self._video_backend,
                return_uint8=True,
            ).squeeze(0)
            return OnlineVideoItem(
                rgb=camera,
                state=self._ram_states[idx],
                action=self._ram_actions[idx],
                action_is_pad=self._ram_paddings[idx],
                sample_id=sample_id,
                episode_index=ep,
                frame_index=frame,
            )

        sample = self.lerobot_ds[local_idx]
        camera = sample[self.contract.camera_key]
        state = sample[self.contract.state_key].squeeze(0).contiguous()
        action = sample[self.contract.action_key].contiguous()
        action_is_pad = sample[f"{self.contract.action_key}_is_pad"].contiguous()

        return OnlineVideoItem(
            rgb=camera,
            state=state,
            action=action,
            action_is_pad=action_is_pad,
            sample_id=sample_id,
            episode_index=ep,
            frame_index=frame,
        )


def augment_temporal_window_gpu(
    batch_frames: Tensor,
    *,
    spatial_crop: bool = False,
    p_blur: float = 0.3,
) -> Tensor:
    """Apply visually coherent data augmentation identically across all T=5 frames.

    Args:
        batch_frames: Tensor [B, T, C, H, W] in uint8 [0, 255] or float [0, 1] on CUDA.

    Returns:
        Augmented tensor [B, 3, T, H, W] in float32 in [0, 1] on CUDA.
    """
    out: list[Tensor] = []
    b_size, t_size, c_size, h_size, w_size = batch_frames.shape
    for i in range(b_size):
        x = batch_frames[i]
        x_float = x.float() / 255.0 if x.dtype == torch.uint8 else x.clone()

        # 1. Photometric Jitter (applied to all T frames identically)
        b = 1.0 + random.uniform(-0.15, 0.15)
        c = 1.0 + random.uniform(-0.15, 0.15)
        s = 1.0 + random.uniform(-0.15, 0.15)
        h = random.uniform(-0.05, 0.05)

        x_jitter = tf_f.adjust_brightness(x_float, b)
        x_jitter = tf_f.adjust_contrast(x_jitter, c)
        x_jitter = tf_f.adjust_saturation(x_jitter, s)
        x_jitter = tf_f.adjust_hue(x_jitter, h)

        # 2. Spatial Translation / Crop of 4-6% (applied only if requested)
        if spatial_crop:
            crop_frac = random.uniform(0.94, 0.96)
            crop_h = int(h_size * crop_frac)
            crop_w = int(w_size * crop_frac)
            top = random.randint(0, h_size - crop_h)
            left = random.randint(0, w_size - crop_w)
            x_crop = tf_f.resized_crop(
                x_jitter, top, left, crop_h, crop_w, size=[h_size, w_size], antialias=True
            )
        else:
            x_crop = x_jitter

        # 3. Sensor Noise / Blur: low-probability mild Gaussian blur sigma in [0.1, 0.8]
        if random.random() < p_blur:
            sigma = random.uniform(0.1, 0.8)
            x_crop = tf_f.gaussian_blur(x_crop, kernel_size=[3, 3], sigma=[sigma, sigma])

        x_clamped = torch.clamp(x_crop, 0.0, 1.0)
        # Permute from [T, C, H, W] to [C, T, H, W] for VAM extractors
        out.append(x_clamped.permute(1, 0, 2, 3))

    return torch.stack(out, dim=0)


def physics_augment_temporal_window_gpu(
    batch_frames: Tensor,
    *,
    temp_range: tuple[float, float] = (-0.10, 0.10),
    p_shadow: float = 0.5,
    gamma_range: tuple[float, float] = (0.90, 1.10),
) -> Tensor:
    out: list[Tensor] = []
    b_size, t_size, c_size, h_size, w_size = batch_frames.shape
    device = batch_frames.device

    for i in range(b_size):
        x = (
            batch_frames[i].float() / 255.0
            if batch_frames[i].dtype == torch.uint8
            else batch_frames[i].clone()
        )

        # 1. Planckian Illuminant Jitter (color temperature shift along Planckian locus)
        temp = random.uniform(temp_range[0], temp_range[1])
        scale = torch.tensor(
            [1.0 + temp, 1.0 - 0.2 * temp, 1.0 - temp],
            device=device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        x = x * scale

        # 2. Smooth Cast Shadow (p=0.5)
        if random.random() < p_shadow:
            opacity = random.uniform(0.20, 0.35)
            x_start = random.randint(0, w_size // 3)
            x_end = random.randint(w_size // 2, w_size)

            mask = torch.ones((1, 1, h_size, w_size), device=device, dtype=x.dtype)
            mask[:, :, :, x_start:x_end] = 1.0 - opacity
            down = F.avg_pool2d(mask, kernel_size=16, stride=16)
            mask = F.interpolate(down, size=(h_size, w_size), mode="bilinear", align_corners=False)
            x = x * mask

        # 3. Gamma / Auto-Exposure Variation
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        x = torch.pow(torch.clamp(x, 1e-4, 1.0), gamma)

        x_clamped = torch.clamp(x, 0.0, 1.0)
        out.append(x_clamped.permute(1, 0, 2, 3))

    return torch.stack(out, dim=0)


def unaugmented_temporal_window_gpu(batch_frames: Tensor) -> Tensor:
    """Format unaugmented frames [B, T, C, H, W] -> [B, 3, T, H, W] in [0, 1] on CUDA."""
    x = batch_frames.float() / 255.0 if batch_frames.dtype == torch.uint8 else batch_frames
    return torch.clamp(x.permute(0, 2, 1, 3, 4), 0.0, 1.0)


def compute_online_training_normalizer(
    train_dataset: OnlineVideoDataset,
    train_episodes: Sequence[int],
    output_dir: Path,
    kinematics: Any | None = None,
) -> SmolVLANormalizer:
    """Compute SmolVLA normalization statistics from online training dataset."""
    states: list[Tensor] = []
    actions: list[Tensor] = []
    paddings: list[Tensor] = []

    # Iterate with stride to sample states and actions representative of the split
    sample_stride = max(1, len(train_dataset) // 2000)
    for idx in range(0, len(train_dataset), sample_stride):
        item = train_dataset[idx]
        states.append(item.state)
        actions.append(item.action)
        paddings.append(item.action_is_pad)

    all_states = torch.stack(states, dim=0)
    all_actions = torch.stack(actions, dim=0)
    all_paddings = torch.stack(paddings, dim=0)

    if kinematics is not None:
        all_states = joints_to_cartesian(all_states, kinematics)
        all_actions = joints_to_cartesian(all_actions, kinematics)

    normalizer = SmolVLANormalizer.from_training_tensors(
        state=all_states,
        action=all_actions,
        action_is_pad=all_paddings,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    norm_path = output_dir / "normalizer.safetensors"
    save_file(
        normalizer.state_dict(),
        str(norm_path),
        metadata={"artifact": "smolvla_mean_std_normalizer", "source_split": "train"},
    )

    meta_payload = {
        "artifact": "smolvla_mean_std_normalizer",
        "source_split": "train",
        "train_episodes": list(train_episodes),
        "anchor_count": len(states),
        "padded_actions_excluded": True,
        "state_mean": normalizer.state_mean.tolist(),
        "state_std": normalizer.state_std.tolist(),
        "action_mean": normalizer.action_mean.tolist(),
        "action_std": normalizer.action_std.tolist(),
    }
    (output_dir / "normalizer.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
    return normalizer


def build_synthetic_tiny_decoder(
    input_channels: int,
    normalizer: SmolVLANormalizer | None = None,
    device: str | torch.device = "cpu",
) -> SmolExpertActionDecoder:
    """Build a fast, lightweight SmolExpertActionDecoder for testing and explicit CPU dry-runs."""

    class _TinyAttention(nn.Module):
        def __init__(self, hidden: int = 16, kv: int = 8) -> None:
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv, bias=False)
            self.v_proj = nn.Linear(hidden, kv, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)

    class _TinyLayer(nn.Module):
        def __init__(self, cross: bool) -> None:
            super().__init__()
            self.input_layernorm = nn.LayerNorm(16)
            self.self_attn = _TinyAttention()
            if cross:
                self.self_attn.k_proj = nn.Linear(8, 8, bias=False)
                self.self_attn.v_proj = nn.Linear(8, 8, bias=False)
            self.post_attention_layernorm = nn.LayerNorm(16)
            self.mlp = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16))

    class _TinyExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_TinyLayer(cross=i % 2 == 1) for i in range(2)])
            self.norm = nn.LayerNorm(16)

    if normalizer is None:
        normalizer = SmolVLANormalizer(
            state_mean=torch.zeros(ACTION_DIM),
            state_std=torch.ones(ACTION_DIM),
            action_mean=torch.zeros(ACTION_DIM),
            action_std=torch.ones(ACTION_DIM),
        )

    decoder = SmolExpertActionDecoder(
        normalizer,
        expert=_TinyExpert(),
        prefix_hidden_size=16,
        expert_hidden_size=16,
        kv_dim=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_steps=2,
        input_channels=input_channels,
        max_action_dim=16,
        max_state_dim=16,
    )
    return decoder.to(device=device)


@torch.no_grad()
def evaluate_validation(
    decoder: SmolExpertActionDecoder,
    val_dataset: Any,
    *,
    device: torch.device,
    batch_size: int = 8,
    num_steps: int = 10,
    seed: int = 42,
    extractor: Any | None = None,
    kinematics: Any | None = None,
) -> dict[str, Any]:
    """Evaluate policy on validation dataset with strict per-sample noise and global masked metrics.

    Guarantees:
    - Stable per-sample seeded noise: SHA256(dataset_id:sample_id:seed) guarantees batch-size and order invariance.
    - Explicit flow t and epsilon: Deterministic flow loss independent of global train RNG.
    - Masked per-joint & aggregate RMSE: Strictly unpadded scalar accounting.
    - Separate metrics: arm_rmse_deg (joints 0..4), gripper_rmse (joint 5), legacy mean first-5, and pooled first-5.
    """
    if len(val_dataset) == 0:
        raise ValueError("Validation dataset contains no samples.")

    cosmos3_cached = False
    config = getattr(extractor, "config", None)
    if (
        config is not None
        and is_cosmos3(getattr(config, "backbone_name", None))
        and val_dataset[0].context is not None
    ):
        if (
            len(config.hidden_layers) != 1
            or config.fps != 10.0
            or config.base_fps != 24.0
            or config.dtype != "bfloat16"
            or config.pool_spatial not in (None, 1)
            or config.vae_path is not None
        ):
            raise ValueError("Unsupported Cosmos3 extractor configuration for cached evaluation")
        identity = getattr(extractor, "_smolexpert_eval_identity", None)
        if identity is None:
            identity = cosmos3_eval_identity(
                checkpoint_path=config.checkpoint_path,
                lora_checkpoint=config.lora_checkpoint,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                hidden_layer=config.hidden_layers[0],
                prompt=config.prompt,
                backbone_name=getattr(config, "backbone_name", "cosmos3-edge"),
            )
            cast(Any, extractor)._smolexpert_eval_identity = identity
        validate_cosmos3_cache_identity(
            getattr(val_dataset, "payload", {}),
            identity,
            source=str(getattr(val_dataset, "manifest_path", "dataset")),
        )
        cosmos3_cached = True

    was_training = decoder.training
    decoder.eval()

    try:
        squared_by_joint = torch.zeros(ACTION_DIM, dtype=torch.float64, device=device)
        prefix_squared_by_step = [
            torch.zeros(ACTION_DIM, dtype=torch.float64, device=device) for _ in range(5)
        ]
        prefix_tokens_by_step = [0 for _ in range(5)]

        valid_tokens_count = 0
        flow_total = 0.0
        flow_tokens = 0
        total_cart_pos_err_mm = 0.0

        max_action_dim = decoder.max_action_dim
        dtype = (
            decoder.action_in_proj.weight.dtype
            if hasattr(decoder, "action_in_proj") and decoder.action_in_proj is not None
            else torch.float32
        )

        dataset_id = getattr(val_dataset, "dataset_id", "default_dataset")

        for start in range(0, len(val_dataset), batch_size):
            end = min(start + batch_size, len(val_dataset))
            batch_items = [val_dataset[i] for i in range(start, end)]

            states = torch.stack([item.state for item in batch_items]).to(device=device, dtype=torch.float32)
            actions = torch.stack([item.action for item in batch_items]).to(
                device=device, dtype=torch.float32
            )
            if batch_items[0].context is not None:
                contexts = torch.stack([item.context for item in batch_items]).to(device=device)
                if cosmos3_cached and (
                    tuple(contexts.shape[1:]) != (600, 2048) or contexts.dtype != torch.bfloat16
                ):
                    raise ValueError("Cosmos3 cache context must be bfloat16 [600, 2048]")
            elif extractor is not None:
                has_trainable_bb = hasattr(extractor, "parameters") and any(
                    p.requires_grad for p in extractor.parameters()
                )
                if not hasattr(extractor, "_val_context_cache"):
                    cast(Any, extractor)._val_context_cache = {}
                if not hasattr(extractor, "_val_latent_cache"):
                    cast(Any, extractor)._val_latent_cache = {}
                val_cache: dict[str, Tensor] = cast(Any, extractor)._val_context_cache
                val_lat_cache: dict[str, Tensor] = cast(Any, extractor)._val_latent_cache
                if not has_trainable_bb and all(item.sample_id in val_cache for item in batch_items):
                    contexts = torch.stack([val_cache[item.sample_id] for item in batch_items]).to(device=device)
                elif (
                    has_trainable_bb
                    and hasattr(extractor, "forward_transformer_blocks")
                    and all(item.sample_id in val_lat_cache for item in batch_items)
                ):
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        lat_b = torch.stack([val_lat_cache[item.sample_id] for item in batch_items]).to(
                            device=device, non_blocking=True
                        )
                        tap_b = extractor.forward_transformer_blocks(lat_b)
                        lyr_k = next(iter(tap_b.keys()))
                        contexts = tap_b[lyr_k].to(device=device)
                else:
                    raw_rgb = torch.stack(
                        [item.rgb.to(device=device, non_blocking=True) for item in batch_items]
                    )
                    rgb_in = unaugmented_temporal_window_gpu(raw_rgb)
                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        if hasattr(extractor, "encode_latents") and hasattr(
                            extractor, "forward_transformer_blocks"
                        ):
                            lat_b = extractor.encode_latents(rgb_in)
                            for b_item, lat_t in zip(batch_items, lat_b, strict=False):
                                val_lat_cache[b_item.sample_id] = lat_t.detach().cpu()
                            tap_b = extractor.forward_transformer_blocks(lat_b)
                            lyr_k = next(iter(tap_b.keys()))
                            contexts = tap_b[lyr_k].to(device=device)
                        else:
                            extracted_eval = []
                            for b_i in range(rgb_in.shape[0]):
                                out_f = extractor.extract(rgb_frames=rgb_in[b_i : b_i + 1])
                                extracted_eval.append(out_f.features)
                            contexts = torch.cat(extracted_eval, dim=0).to(device=device)
                        if not has_trainable_bb:
                            for b_item, ctx_t in zip(batch_items, contexts, strict=False):
                                val_cache[b_item.sample_id] = ctx_t.detach().cpu()
            else:
                raise ValueError(
                    "Evaluation requires cached context or an extractor; random features are forbidden"
                )
            paddings = torch.stack([item.action_is_pad for item in batch_items]).to(
                device=device, dtype=torch.bool
            )

            # 1. Deterministic per-sample seeded noise generation (Stable across reorder and batching)
            sample_noises = []
            sample_epsilons = []
            sample_ts = []
            for item in batch_items:
                # Noise seed isolated per dataset_id + sample_id
                s_seed = stable_sample_seed(dataset_id, item.sample_id, seed, offset=0)
                gen_noise = torch.Generator(device=device).manual_seed(s_seed)
                n = torch.randn(
                    (1, ACTION_HORIZON, max_action_dim), device=device, dtype=dtype, generator=gen_noise
                )
                sample_noises.append(n)

                # Deterministic epsilon & t for flow-matching loss (Beta(1.5, 1.0) * 0.999 + 0.001)
                eps_seed = stable_sample_seed(dataset_id, item.sample_id, seed, offset=100_000)
                gen_eps = torch.Generator(device=device).manual_seed(eps_seed)
                eps = torch.randn(
                    (1, ACTION_HORIZON, max_action_dim), device=device, dtype=dtype, generator=gen_eps
                )
                sample_epsilons.append(eps)

                t_seed = stable_sample_seed(dataset_id, item.sample_id, seed, offset=200_000)
                gen_t = torch.Generator(device="cpu").manual_seed(t_seed)
                u = torch.rand(1, generator=gen_t).item()
                t_val = (u ** (2.0 / 3.0)) * 0.999 + 0.001
                sample_ts.append(t_val)

            noise = torch.cat(sample_noises, dim=0)
            epsilon_tensor = torch.cat(sample_epsilons, dim=0)
            t_tensor = torch.tensor(sample_ts, dtype=torch.float32, device=device)

            if kinematics is not None:
                true_joint_actions = actions
                true_joint_states = states
                actions = joints_to_cartesian(actions, kinematics)
                states = joints_to_cartesian(states, kinematics)

            # 2. Action sampling with explicit noise
            pred_actions = decoder.sample_actions(states, contexts, noise=noise, num_steps=num_steps)

            # 3. Masked error accumulation
            valid = (~paddings).unsqueeze(-1)  # [B, 30, 1]
            valid_b = valid.squeeze(-1)  # [B, 30]

            if kinematics is not None:
                cart_pos_err = torch.norm(pred_actions[..., :3] - actions[..., :3], dim=-1) * 1000.0
                total_cart_pos_err_mm += float((cart_pos_err * valid_b).sum().item())
                pred_joint_actions = cartesian_to_joints_ik(true_joint_states, pred_actions, kinematics)
                squared_err = (pred_joint_actions.double() - true_joint_actions.double()).square() * valid
            else:
                squared_err = (pred_actions.double() - actions.double()).square() * valid  # [B, 30, 6]

            squared_by_joint += squared_err.sum(dim=(0, 1))
            batch_valid_tokens = int(valid_b.sum().item())
            valid_tokens_count += batch_valid_tokens

            # Step-by-step horizons for first 5 steps
            for h in range(5):
                vh = valid_b[:, h : h + 1].unsqueeze(-1)
                prefix_squared_by_step[h] += (squared_err[:, h : h + 1] * vh).sum(dim=(0, 1))
                prefix_tokens_by_step[h] += int(vh.sum().item())

            # 4. Explicit Flow-matching loss computation
            flow_loss = decoder.flow_matching_loss(
                state=states,
                action=actions,
                context=contexts,
                t=t_tensor,
                epsilon=epsilon_tensor,
                action_is_pad=paddings,
            )
            flow_total += float(flow_loss.float().item()) * max(batch_valid_tokens, 1)
            flow_tokens += max(batch_valid_tokens, 1)

    finally:
        decoder.train(was_training)

    valid_scalars = valid_tokens_count * ACTION_DIM
    if valid_scalars == 0:
        raise ValueError("Evaluation contains zero valid action tokens.")

    per_joint_rmse = torch.sqrt(squared_by_joint / max(valid_tokens_count, 1)).cpu().tolist()
    agg_rmse = math.sqrt(float(squared_by_joint.sum().item()) / max(valid_scalars, 1))

    # Arm joints 0..4 (degrees) vs Gripper joint 5 (range 0..100)
    arm_squared = float(squared_by_joint[:5].sum().item())
    arm_valid_scalars = valid_tokens_count * 5
    arm_rmse_deg = math.sqrt(arm_squared / max(arm_valid_scalars, 1))

    gripper_squared = float(squared_by_joint[5].item())
    gripper_valid_scalars = valid_tokens_count * 1
    gripper_rmse = math.sqrt(gripper_squared / max(gripper_valid_scalars, 1))

    # Horizon 1 (First step)
    h1_rmse = (
        math.sqrt(
            float(prefix_squared_by_step[0].sum().item()) / max(prefix_tokens_by_step[0] * ACTION_DIM, 1)
        )
        if prefix_tokens_by_step[0] > 0
        else agg_rmse
    )

    # Legacy Mean per-horizon RMSE across horizons 0..4
    per_step_rmses = []
    prefix_5_total_sq = 0.0
    prefix_5_total_tokens = 0
    for h in range(5):
        if prefix_tokens_by_step[h] > 0:
            step_sq = float(prefix_squared_by_step[h].sum().item())
            step_scalars = prefix_tokens_by_step[h] * ACTION_DIM
            per_step_rmses.append(math.sqrt(step_sq / step_scalars))
            prefix_5_total_sq += step_sq
            prefix_5_total_tokens += prefix_tokens_by_step[h]

    legacy_first5_mean_rmse = (
        sum(per_step_rmses) / max(len(per_step_rmses), 1) if per_step_rmses else agg_rmse
    )
    pooled_first5_rmse = (
        math.sqrt(prefix_5_total_sq / max(prefix_5_total_tokens * ACTION_DIM, 1))
        if prefix_5_total_tokens > 0
        else agg_rmse
    )

    res = {
        "val_rmse": agg_rmse,
        "val_h1": h1_rmse,
        "val_first5": legacy_first5_mean_rmse,
        "val_first5_pooled_rmse": pooled_first5_rmse,
        "arm_rmse_deg": arm_rmse_deg,
        "gripper_rmse": gripper_rmse,
        "val_flow_loss": flow_total / max(flow_tokens, 1),
        "per_joint_rmse_deg": [float(v) for v in per_joint_rmse],
        "per_joint_units": ["degrees", "degrees", "degrees", "degrees", "degrees", "range_0_100"],
        "aggregate_rmse_deg": agg_rmse,
        "n_valid_action_tokens": valid_tokens_count,
        "n_valid_scalars": valid_scalars,
        "val_samples": len(val_dataset),
    }
    if kinematics is not None:
        res["cart_pos_mm"] = total_cart_pos_err_mm / max(valid_tokens_count, 1)
    return res


class EarlyStoppingTracker:
    """Tracks validation performance and signals early stopping when patience runs out."""

    def __init__(self, patience: int = 15, min_steps: int = 0, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_steps = min_steps
        self.min_delta = min_delta
        self.best_metric = float("inf")
        self.best_step = -1
        self.bad_evaluations = 0
        self.should_stop = False

    def update(self, metric: float, step: int) -> bool:
        """Update tracker with new validation metric. Returns True if metric improved."""
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.best_step = step
            self.bad_evaluations = 0
            return True
        self.bad_evaluations += 1
        if self.bad_evaluations >= self.patience and step >= self.min_steps:
            self.should_stop = True
        return False


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Build unified LR scheduler: cosine decay with linear warmup or constant."""
    if scheduler_type == "cosine":

        def factor(step: int) -> float:
            if step < warmup_steps:
                return max(step, 1) / max(warmup_steps, 1)
            progress = min(1.0, (step - warmup_steps) / max(max_steps - warmup_steps, 1))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    elif scheduler_type == "constant":

        def factor(step: int) -> float:
            if step < warmup_steps:
                return max(step, 1) / max(warmup_steps, 1)
            return 1.0

    else:
        raise ValueError(f"Unsupported lr_scheduler: {scheduler_type!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_own_artifacts(output_dir: Path) -> None:
    """Clean only known trainer artifacts when --overwrite is set."""
    own_filenames = {
        "best.safetensors",
        "last.safetensors",
        "best_backbone_lora.safetensors",
        "last_backbone_lora.safetensors",
        "run_manifest.json",
        "normalizer.safetensors",
        "normalizer.json",
        "best_metrics.json",
        "metrics.jsonl",
        "training_summary.json",
        "final_dual_eval_metrics.json",
        "eval_metrics.json",
    }
    for file_name in own_filenames:
        target = output_dir / file_name
        if target.is_file():
            target.unlink()


def parse_episodes(ep_str: str) -> tuple[int, ...]:
    """Parse episode range string like '0:68' or '0,1,2' into a tuple of ints."""
    if ":" in ep_str:
        start, end = map(int, ep_str.split(":"))
        return tuple(range(start, end))
    return tuple(map(int, ep_str.split(",")))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train unified SmolExpert action decoder across Video Action Model backbones."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="cosmos",
        help="Backbone identifier (e.g. cosmos, cosmos7b, cosmos14b, flux2_klein, ltx).",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Path to training manifest.json (Protocol 1.0 / scale100 train episodes).",
    )
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Path to validation manifest.json (Protocol 1.0 val episodes / Eval-Set 1).",
    )
    parser.add_argument(
        "--online-backbone",
        type=str,
        default=None,
        help="Online backbone identifier (e.g. cosmos3_edge). When set, streams raw frames from dataset.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=None,
        help="Path to online backbone checkpoint directory (e.g. /home/anton/.cache/video-vam/cosmos3-edge).",
    )
    parser.add_argument(
        "--backbone-lora-weights",
        type=Path,
        default=None,
        help="Optional path to LoRA weights safetensors for online backbone.",
    )
    parser.add_argument(
        "--backbone-lora-rank",
        type=int,
        default=None,
        help="LoRA rank for online backbone. Default: read from the adapter's metadata/sidecar; "
        "an explicit value must match it.",
    )
    parser.add_argument(
        "--backbone-lora-alpha",
        type=float,
        default=None,
        help="LoRA alpha for online backbone. Default: read from the adapter's metadata/sidecar; "
        "an explicit value must match it.",
    )
    parser.add_argument(
        "--train-backbone-lora",
        action="store_true",
        help="Enable joint end-to-end LoRA/DoRA training of the online Cosmos 3 backbone alongside SmolExpert.",
    )
    parser.add_argument(
        "--backbone-lora-dora",
        "--dora",
        dest="backbone_lora_dora",
        action="store_true",
        help="Use DoRA (Weight-Decomposed Low-Rank Adaptation) when training Cosmos 3 backbone adapters.",
    )
    parser.add_argument(
        "--backbone-lora-lr",
        type=float,
        default=1e-4,
        help="Learning rate for trainable online backbone LoRA/DoRA adapters (default: 1e-4).",
    )
    parser.add_argument(
        "--backbone-lora-pathway",
        type=str,
        choices=("gen_only", "dual", "und_only"),
        default="gen_only",
        help="Target pathway for trainable Cosmos 3 LoRA/DoRA: 'gen_only' (visual DiT blocks, fastest), 'dual', or 'und_only'.",
    )
    parser.add_argument(
        "--backbone-grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use activation checkpointing across Cosmos 3 blocks when --train-backbone-lora is enabled (default: True).",
    )
    parser.add_argument(
        "--backbone-layer",
        type=int,
        default=20,
        help="Hidden layer index to tap from backbone (default: 20).",
    )
    parser.add_argument(
        "--backbone-prompt",
        type=str,
        default="take cube out of box",
        help="Conditioning prompt text for online backbone.",
    )
    parser.add_argument(
        "--backbone-fps",
        type=float,
        default=None,
        help="Temporal FPS passed to Cosmos 3 mRoPE modulation (default: dataset FPS in online mode, or 10.0 for cached Protocol 1.0).",
    )
    parser.add_argument(
        "--backbone-tokenizer",
        type=Path,
        default=None,
        help="Path to tokenizer for online backbone (e.g. Cosmos 2B tokenizer.pth).",
    )
    parser.add_argument(
        "--backbone-vae",
        type=Path,
        default=None,
        help="Native VAE directory for online FLUX.2 klein (required for --online-backbone flux2_klein).",
    )
    parser.add_argument(
        "--backbone-prompt-embedding",
        type=Path,
        default=None,
        help="Path to precomputed prompt embedding for online backbone (e.g. Cosmos 2B t5-11b.safetensors).",
    )
    parser.add_argument(
        "--high-noise-sigma",
        type=float,
        default=80.0,
        help="High noise sigma for Cosmos 2B video backbone (default: 80.0).",
    )
    parser.add_argument(
        "--compile-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile flow_matching_loss with mode='reduce-overhead' (CUDA graphs, default: True, 2.05x faster).",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coherent visual data augmentations on 5-frame temporal window during training.",
    )
    parser.add_argument(
        "--aug-strategy",
        type=str,
        choices=("physics", "legacy"),
        default="physics",
        help="Augmentation strategy when --augment is enabled: 'physics' (Planckian jitter + smooth cast shadow + gamma) or 'legacy' (naive color jitter + blur).",
    )
    parser.add_argument(
        "--spatial-crop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply spatial translation/crop during visual augmentations (default: False, photometric only).",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default=None,
        help="LeRobot dataset repository ID for online mode (e.g. Orellius/cube_out_of_box_v2).",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional local path to dataset root.",
    )
    parser.add_argument(
        "--dataset-revision",
        type=str,
        default=None,
        help="Pinned dataset commit sha. Required for online mode on datasets other than the canonical "
        "cube_out_of_box v1 (whose revision is fixed by the contract).",
    )
    parser.add_argument(
        "--train-stride",
        type=int,
        default=1,
        help="Frame sampling stride for training episodes in online mode (default: 1).",
    )
    parser.add_argument(
        "--train-episodes",
        type=str,
        default=None,
        help="Optional comma-separated or range for train episodes (e.g. '0:68' or '0,1,2').",
    )
    parser.add_argument(
        "--val-episodes",
        type=str,
        default=None,
        help="Optional comma-separated or range for val episodes (e.g. '68:77' or '32:40').",
    )
    parser.add_argument(
        "--use-cartesian-actions",
        action="store_true",
        help="Train policy on Cartesian end-effector targets and evaluate via inverse kinematics.",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path("assets/so101_kinematics.urdf"),
        help="Path to robot URDF for kinematics.",
    )
    parser.add_argument(
        "--normalizer-path",
        type=Path,
        default=None,
        help="Optional path to pre-existing normalizer.safetensors to avoid recomputing.",
    )
    parser.add_argument(
        "--eval2-manifest",
        type=Path,
        default=None,
        help="Optional path to Eval-Set 2 manifest.json (New Benchmark episodes).",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        choices=("protocol1", "scale100"),
        default="protocol1",
        help="Benchmark protocol (protocol1 or scale100).",
    )
    parser.add_argument(
        "--require-identity",
        action="store_true",
        help="Enforce exact manifest dataset repo_id and revision identity.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only without training, loading trained model from --model-checkpoint or --checkpoint-path.",
    )
    parser.add_argument(
        "--eval-run-dir", type=Path, help="Reevaluate a completed online Cosmos3 run without modifying it."
    )
    parser.add_argument(
        "--eval-report", type=Path, help="New, separate JSON report; existing paths are never overwritten."
    )
    parser.add_argument("--eval-checkpoint", choices=("best", "last"), default="best")
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=None,
        help="Optional path to trained model safetensors checkpoint for evaluation.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=None,
        help="Optional path to strict Protocol 1.0 VAMSplit JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: fresh repository outputs/train/smolexpert-* run).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=SMOLVLA_CHECKPOINT,
        help="SmolVLA pretrained checkpoint name or path.",
    )
    parser.add_argument(
        "--context-transform",
        choices=("auto", "none", "pool2"),
        default="auto",
        help="Context transform (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW optimizer (default: 1e-4).",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("cosine", "constant"),
        default="cosine",
        help="Learning rate schedule type (default: cosine).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps for LR scheduler (default: 1000).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-10,
        help="Weight decay for AdamW (default: 1e-10).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
        help="Max gradient norm for clipping (default: 10.0).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Maximum training optimizer steps (default: 50000).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=0,
        help="Minimum training steps before early stopping can trigger (default: 0).",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=1000,
        help="Evaluation interval in optimizer steps (default: 1000).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in evaluations (default: 15).",
    )
    parser.add_argument(
        "--select-metric",
        type=str,
        choices=("val_rmse", "val_flow_loss"),
        default="val_rmse",
        help="Validation metric used for best-checkpoint selection and early stopping (default: val_rmse).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Flow-matching action sampling steps for validation (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run CPU dry-run using architecture-matched synthetic tiny decoder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite own artifacts in output directory if non-empty.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="video-vam-world2action",
        help="W&B project name.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--save-last-every",
        type=int,
        default=1000,
        help="Last-weight save interval in optimizer steps; <=0 saves only at end.",
    )
    parser.add_argument(
        "--max-train-seconds",
        type=float,
        default=None,
        help="Optional wall-clock training time limit in seconds; triggers clean final eval and save when reached.",
    )
    args = parser.parse_args(argv)
    if args.eval_run_dir is not None:
        allowed = {
            "--eval-run-dir",
            "--eval-report",
            "--eval-checkpoint",
            "--eval-only",
            "--val-manifest",
            "--eval2-manifest",
            "--device",
            "--batch-size",
        }
        explicit = {
            arg.split("=", 1)[0] for arg in (sys.argv[1:] if argv is None else argv) if arg.startswith("--")
        }
        if explicit - allowed:
            parser.error(f"--eval-run-dir does not permit overrides: {sorted(explicit - allowed)}")
        if args.eval_report is None:
            parser.error("--eval-run-dir requires --eval-report")
        args.eval_only = True
        args.eval_batch_size_override = args.batch_size if "--batch-size" in explicit else None
        return args
    if args.eval_report is not None or args.eval_checkpoint != "best":
        parser.error("--eval-report and --eval-checkpoint require --eval-run-dir")
    if args.output_dir is None:
        if args.eval_only:
            parser.error("--eval-only requires --output-dir")
        args.output_dir = default_run_dir(Path(__file__).resolve().parents[2], "smolexpert")
    try:
        resolve_backbone_lora_hyperparameters(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    return args


def build_run_manifest(
    args: argparse.Namespace,
    train_dataset: Any,
    val_dataset: Any,
    eval2_dataset: Any | None,
    protocol: str,
) -> dict[str, Any]:
    """Preserve source metadata without inventing missing configuration or revisions."""
    datasets = {}
    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("eval2", eval2_dataset)):
        if dataset is None:
            continue
        if isinstance(dataset, OnlineVideoDataset):
            datasets[name] = {
                "manifest_sha256": "online_video_dataset",
                "source_path_hint": str(dataset.manifest_path),
                "metadata": dataset.payload,
                "episodes": list(dataset.episodes),
                "context_shape": [600, 2048],
            }
        else:
            datasets[name] = {
                "manifest_sha256": sha256_file(dataset.manifest_path),
                "source_path_hint": str(dataset.manifest_path),
                "metadata": {k: v for k, v in dataset.payload.items() if k != "entries"},
                "episodes": sorted({int(e["episode_index"]) for e in dataset.entries}),
                "context_shape": list(dataset[0].context.shape),
            }
    norm_path = args.output_dir / "normalizer.safetensors"
    norm_meta = (
        json.loads((args.output_dir / "normalizer.json").read_text())
        if (args.output_dir / "normalizer.json").exists()
        else {}
    )
    action_dim = (
        len(norm_meta["action_mean"])
        if "action_mean" in norm_meta and isinstance(norm_meta["action_mean"], list)
        else ACTION_DIM
    )
    return {
        "schema_version": 1,
        "status": "running",
        "trainer": "scripts/video_vam/train_smolexpert.py",
        "code": code_identity(Path(__file__).resolve().parents[2]),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "policy": {
            "type": "synthetic_tiny_smolexpert" if args.dry_run else "SmolExpertActionDecoder",
            "pretrained_source": args.checkpoint_path,
            "pretrained_revision": None,
            "action_dim": action_dim,
            "action_horizon": ACTION_HORIZON,
            "num_steps": args.num_steps,
            "backbone": args.backbone,
            "context_transform_requested": args.context_transform,
        },
        "protocol": protocol,
        "datasets": datasets,
        "normalizer": {
            "path": norm_path.name,
            "sha256": sha256_file(norm_path),
            "metadata": norm_meta,
        },
        "selection": {"metric": "val_rmse", "split": "val", "direction": "min"},
        "checkpoints": {"best": None, "last": None},
        "limitations": [
            "Weights only: no optimizer, scheduler, RNG or data-order state; no training resume.",
            "External datasets/backbones/LoRA are not bundled; source paths are hints only.",
            "Missing source metadata/revisions are unknown; dirty code is not reproduced by commit alone.",
            "Individual files are atomic, not the bundle; verify hashes after interruption.",
        ],
    }


def _restore_eval_run_args(manifest: dict[str, Any]) -> argparse.Namespace:
    """Restore only head, feature, and evaluation settings; never training/output controls."""
    saved = manifest.get("arguments")
    if not isinstance(saved, dict):
        raise ValueError("Run manifest is missing original arguments")
    values: dict[str, Any] = {}
    fields = {
        "online_backbone": str,
        "checkpoint_path": str,
        "backbone_prompt": str,
        "protocol": str,
        "context_transform": str,
        "backbone_layer": int,
        "backbone_lora_rank": int,
        "num_steps": int,
        "seed": int,
        "batch_size": int,
    }
    for name, expected_type in fields.items():
        if type(saved.get(name)) is not expected_type:
            raise ValueError(f"Missing or invalid saved run argument: {name}")
        values[name] = saved[name]
    alpha = cast(int | float, saved.get("backbone_lora_alpha"))
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Missing or invalid saved run argument: backbone_lora_alpha")
    values["backbone_lora_alpha"] = float(alpha)
    repo_root = Path(__file__).resolve().parents[2]
    for name in ("backbone_checkpoint", "backbone_lora_weights", "val_manifest", "eval2_manifest"):
        if name not in saved or (
            saved[name] is not None and (not isinstance(saved[name], str) or not saved[name])
        ):
            raise ValueError(f"Missing or invalid saved run argument: {name}")
        path = Path(saved[name]).expanduser() if saved[name] is not None else None
        values[name] = repo_root / path if path is not None and not path.is_absolute() else path
    if (
        not is_cosmos3(values["online_backbone"])
        or values["protocol"] not in ("protocol1", "scale100")
        or values["context_transform"] not in ("auto", "none", "pool2")
        or values["num_steps"] <= 0
        or values["batch_size"] <= 0
    ):
        raise ValueError("Isolated reevaluation requires a valid online Cosmos3 run configuration")
    policy = manifest.get("policy", {})
    if (
        policy.get("type") != "SmolExpertActionDecoder"
        or policy.get("action_dim") != ACTION_DIM
        or policy.get("action_horizon") != ACTION_HORIZON
        or policy.get("pretrained_source") != values["checkpoint_path"]
        or policy.get("num_steps") != values["num_steps"]
    ):
        raise ValueError("Saved SmolExpert head architecture/source does not match run arguments")
    checkpoint_source = Path(values["checkpoint_path"]).expanduser()
    if (repo_root / checkpoint_source).exists():
        values["checkpoint_path"] = str((repo_root / checkpoint_source).resolve())
    return argparse.Namespace(**values)


def _verified_run_artifact(run_dir: Path, record: Any, filename: str) -> tuple[Path, str]:
    if not isinstance(record, dict) or record.get("path") != filename:
        raise ValueError(f"Run manifest must record its own {filename}")
    digest = _require_sha256(record.get("sha256"), filename)
    path = run_dir / filename
    if path.resolve().parent != run_dir:
        raise ValueError(f"Run artifact must remain inside the source run: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Recorded run artifact not found: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"SHA-256 mismatch for recorded run artifact: {path}")
    return path, digest


def verify_reevaluation_samples(
    run_manifest: dict[str, Any], datasets: dict[str, UnifiedFeatureCacheDataset]
) -> dict[str, Any]:
    """Bind replacement data to the run's original, hash-verified ordered benchmarks."""
    records = run_manifest.get("datasets")
    if not isinstance(records, dict):
        raise ValueError("Run manifest lacks original evaluation dataset records")
    summary = {}
    anchor_fields = ("sample_id", "episode_index", "frame_index", "window_indices")
    for split in ("val", "eval2"):
        record = records.get(split)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("source_path_hint"), str)
            or not record["source_path_hint"]
        ):
            raise ValueError(f"Missing original {split} manifest source_path_hint")
        path = Path(record["source_path_hint"]).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        path = path.resolve(strict=True)
        digest = _require_sha256(record.get("manifest_sha256"), f"original {split} manifest")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch for original {split} manifest: {path}")
        original_payload = json.loads(contents)
        if not isinstance(original_payload, dict) or record.get("metadata") != {
            key: value for key, value in original_payload.items() if key != "entries"
        }:
            raise ValueError(f"Original {split} manifest metadata does not match the run record")
        replacement = datasets[split]
        old_data = original_payload.get("dataset")
        new_data = replacement.payload.get("dataset")
        if (
            not isinstance(old_data, dict)
            or not isinstance(new_data, dict)
            or not all(
                isinstance(data.get(key), str) and data[key]
                for data in (old_data, new_data)
                for key in ("repo_id", "revision")
            )
        ):
            raise ValueError(f"Missing original/replacement {split} dataset identity")
        if {key: value for key, value in old_data.items() if key != "revision"} != {
            key: value for key, value in new_data.items() if key != "revision"
        } or replacement.dataset_id != old_data["repo_id"]:
            raise ValueError(
                f"Replacement {split} dataset identity changed; only revision metadata may differ"
            )
        if original_payload.get("subset") != replacement.payload.get("subset"):
            raise ValueError(f"Replacement {split} subset metadata changed")
        entries = original_payload.get("entries")
        if not isinstance(entries, list) or not entries or len(entries) != len(replacement):
            raise ValueError(f"Original/replacement {split} anchor counts differ or are missing")
        original_file_hashes: dict[str, str] = {}
        for index, (old_entry, new_entry) in enumerate(zip(entries, replacement.entries, strict=True)):
            for entry in (old_entry, new_entry):
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("sample_id"), str)
                    or not entry["sample_id"]
                    or type(entry.get("episode_index")) is not int
                    or type(entry.get("frame_index")) is not int
                    or not isinstance(entry.get("window_indices"), list)
                    or len(entry["window_indices"]) != 5
                    or any(type(frame) is not int for frame in entry["window_indices"])
                ):
                    raise ValueError(f"Missing or invalid {split} anchor identity at position {index}")
            if any(old_entry[key] != new_entry[key] for key in anchor_fields):
                raise ValueError(f"Replacement {split} ordered anchor mismatch at position {index}")
            artifact_hash = _require_sha256(
                old_entry.get("safetensors_sha256"), f"original {split} artifact {index}"
            )
            filename = old_entry.get("safetensors")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Missing original {split} artifact filename at position {index}")
            if filename in original_file_hashes and original_file_hashes[filename] != artifact_hash:
                raise ValueError(f"Conflicting original {split} artifact hashes: {filename}")
            original_file_hashes[filename] = artifact_hash
        original = UnifiedFeatureCacheDataset(path, context_transform=replacement.context_transform)
        if original.payload != original_payload:
            raise ValueError(f"Original {split} manifest changed during verification: {path}")
        for index in range(len(original)):
            old_item, new_item = original[index], replacement[index]
            for field in ("state", "action", "action_is_pad"):
                if not torch.equal(getattr(old_item, field), getattr(new_item, field)):
                    raise ValueError(f"Replacement {split} {field} mismatch for {old_item.sample_id}")
        summary[split] = {
            "original_manifest": {"path": str(path), "sha256": digest},
            "replacement_manifest": {
                "path": str(replacement.manifest_path),
                "sha256": sha256_file(replacement.manifest_path),
            },
            "dataset_id": old_data["repo_id"],
            "samples_verified": len(original),
            "ordered_anchor_fields": list(anchor_fields),
            "ordered_anchors_equal": True,
            "original_metadata_and_artifact_hashes_verified": True,
            "labels_equal": dict.fromkeys(("state", "action", "action_is_pad"), True),
            "revision_correction": {
                "old": old_data["revision"],
                "new": new_data["revision"],
                "changed": old_data["revision"] != new_data["revision"],
                "independently_verified": False,
            },
        }
    return summary


def write_new_eval_report(path: Path, payload: dict[str, Any]) -> None:
    """Publish a complete JSON file atomically, without replacing even a racing writer."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard linking is atomic and fails if the destination exists,
        # unlike rename/replace. It also refuses existing or dangling symlinks.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_saved_run(cli: argparse.Namespace) -> int:
    """Evaluate verified caches with the original head/normalizer; write only a new report."""
    run_dir = cli.eval_run_dir.expanduser().resolve(strict=True)
    requested_report = cli.eval_report.expanduser().absolute()
    if requested_report.exists() or requested_report.is_symlink():
        raise FileExistsError(f"Evaluation report already exists: {requested_report}")
    report = requested_report.parent.resolve(strict=True) / requested_report.name
    if report.is_relative_to(run_dir):
        raise ValueError("Evaluation report must be outside the source run directory")
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json_object(manifest_path)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("status") != "completed"
        or manifest.get("trainer") != "scripts/video_vam/train_smolexpert.py"
    ):
        raise ValueError("Isolated reevaluation requires a completed train_smolexpert run manifest")
    manifest_sha = sha256_file(manifest_path)
    args = _restore_eval_run_args(manifest)
    for name in ("val_manifest", "eval2_manifest"):
        override = getattr(cli, name)
        if override is not None:
            setattr(args, name, override.expanduser().resolve())
        if getattr(args, name) is None:
            raise ValueError(f"Isolated dual reevaluation requires an explicit {name} cache")
    if cli.eval_batch_size_override is not None:
        args.batch_size = cli.eval_batch_size_override
    if args.batch_size <= 0:
        raise ValueError("Evaluation batch size must be positive")
    checkpoint_record = manifest.get("checkpoints", {}).get(cli.eval_checkpoint)
    checkpoint, checkpoint_sha = _verified_run_artifact(
        run_dir, checkpoint_record, f"{cli.eval_checkpoint}.safetensors"
    )
    normalizer_record = manifest.get("normalizer")
    normalizer_path, normalizer_sha = _verified_run_artifact(
        run_dir, normalizer_record, "normalizer.safetensors"
    )
    normalizer_record = cast(dict[str, Any], normalizer_record)
    if normalizer_record.get("metadata", {}).get("source_split") != "train":
        raise ValueError("Run normalizer must have recorded train-split provenance")
    identity, datasets = preflight_cosmos3_eval_caches(args)
    identity = cast(dict[str, Any], identity)
    recorded_identity = manifest.get("online_feature_identity")
    if recorded_identity is not None and recorded_identity != identity:
        raise ValueError(
            "Current Cosmos3 files do not match the online feature identity recorded by this run"
        )
    sample_verification = verify_reevaluation_samples(manifest, datasets)
    norm_sd = load_file(str(normalizer_path), device="cpu")
    normalizer = SmolVLANormalizer(
        state_mean=norm_sd["state_mean"],
        state_std=norm_sd["state_std"],
        action_mean=norm_sd["action_mean"],
        action_std=norm_sd["action_std"],
        eps=1e-8,
        source_split="train",
    )
    source_manifests = {
        split: {
            "path": str(dataset.manifest_path),
            "sha256": sha256_file(dataset.manifest_path),
            "dataset": dataset.payload["dataset"],
            "dataset_revision_verified": False,
            "samples": len(dataset),
            "sample_ids": [entry["sample_id"] for entry in dataset.entries],
            "episodes": sorted({entry["episode_index"] for entry in dataset.entries}),
            "provenance": dataset.payload["provenance"],
        }
        for split, dataset in datasets.items()
    }
    device = torch.device(cli.device)
    set_seed(args.seed)
    decoder = SmolExpertActionDecoder.from_training_checkpoint(
        checkpoint,
        normalizer=normalizer,
        expert_checkpoint=args.checkpoint_path,
        device=device,
        num_steps=args.num_steps,
        input_channels=2048,
    )
    results = {}
    for split, seed in (("val", args.seed), ("eval2", args.seed + 1000)):
        results[split] = evaluate_validation(
            decoder,
            datasets[split],
            device=device,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            seed=seed,
        )
    original_val = manifest.get("datasets", {}).get("val", {}).get("metadata", {})
    try:
        validate_cosmos3_cache_identity(original_val, identity, source="original validation cache")
        original_selection_matches = True
    except ValueError:
        original_selection_matches = False
    limits = [
        "Original checkpoint selection is unchanged; reevaluation does not reselect the best head or repair early stopping.",
        "Revision corrections are reported, not independently authenticated against dataset snapshots. Ordered cached anchors and labels were verified against original manifests; RGB/video contents were not compared.",
        "Original dirty training code cannot be reconstructed from its commit alone.",
    ]
    if recorded_identity is None:
        limits.append(
            "Historical online adapter digest was not recorded by the original trainer; current file hashes do not prove training-time identity."
        )
    if not original_selection_matches:
        limits.append(
            "Original best was selected using cached validation features whose identity does not match or cannot be verified against the current online backbone."
        )
    payload = {
        "schema_version": 1,
        "mode": "isolated_smolexpert_reevaluation",
        "source_run": {"path": str(run_dir), "manifest_sha256": manifest_sha, "code": manifest.get("code")},
        "code": code_identity(Path(__file__).resolve().parents[2]),
        "checkpoint": {
            **checkpoint_record,
            "path": str(checkpoint),
            "sha256": checkpoint_sha,
            "role": cli.eval_checkpoint,
        },
        "normalizer": {
            "path": str(normalizer_path),
            "sha256": normalizer_sha,
            "metadata": normalizer_record["metadata"],
        },
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "device": str(device),
        "backbone": args.online_backbone,
        "protocol": args.protocol,
        "current_backbone_identity": identity,
        "evaluation_augmented": False,
        "seeds": {"val": args.seed, "eval2": args.seed + 1000},
        "source_manifests": source_manifests,
        "sample_verification": sample_verification,
        "original_selection_cache_matches_current_identity": original_selection_matches,
        "eval1_historical": results["val"],
        "eval2_new_benchmark": results["eval2"],
        "limitations": limits,
    }
    write_new_eval_report(report, payload)
    print(f"Reevaluation report written without modifying the source run: {report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eval_run_dir is not None:
        return evaluate_saved_run(args)
    if args.eval_only and is_cosmos3(args.online_backbone):
        raise ValueError("Online Cosmos3 reevaluation requires --eval-run-dir and --eval-report")
    cosmos3_identity, cosmos3_datasets = preflight_cosmos3_eval_caches(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    set_seed(args.seed)

    effective_protocol = args.protocol
    is_online = args.online_backbone is not None
    if is_online:
        args.backbone = args.online_backbone

    print("=== Unified SmolExpert Action Policy Trainer ===")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Protocol: {effective_protocol}")
    print(f"Online Mode: {is_online} (Augment: {args.augment})")
    print(f"Output Directory: {args.output_dir}")

    kinematics: Any | None = None
    if getattr(args, "use_cartesian_actions", False):
        from lerobot.model.kinematics import RobotKinematics

        urdf_file = args.urdf_path if args.urdf_path.is_absolute() else (REPO_ROOT / args.urdf_path)
        kinematics = RobotKinematics(str(urdf_file), target_frame_name="gripper_frame_link")
        kinematics.solver.enable_joint_limits(False)
        print(f"Enabled Cartesian end-effector actions + IK (URDF: {urdf_file})")

    # Output directory safety check
    if not args.eval_only:
        if args.train_manifest is None and not is_online:
            raise ValueError("--train-manifest is required for offline training.")
        if args.output_dir.exists():
            existing_files = list(args.output_dir.iterdir())
            if existing_files:
                if not args.overwrite:
                    raise FileExistsError(
                        f"Output directory {args.output_dir} is not empty (contains {len(existing_files)} files). "
                        f"Pass --overwrite to overwrite own artifacts."
                    )
                clean_own_artifacts(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation & Split Enforcement Guard
    train_eps: tuple[int, ...] = ()
    val_eps: tuple[int, ...] = ()
    eval2_eps: tuple[int, ...] = ()
    require_identity = args.require_identity or (not args.dry_run)

    if is_online:
        if effective_protocol == "scale100":
            train_eps = tuple(sorted(list(range(32)) + list(range(40, 90))))
            val_eps = tuple(range(32, 40))
            eval2_eps = tuple(range(90, 100))
            if args.dataset_repo_id is None:
                args.dataset_repo_id = "Orellius/cube_out_of_box_v2"
            if args.dataset_repo_id == "Orellius/cube_out_of_box_v2" and args.dataset_revision is None:
                args.dataset_revision = SCALE100_V2_REVISION
            if args.dataset_root is None:
                args.dataset_root = Path(
                    f"/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/{SCALE100_V2_REVISION}"
                )
        else:
            train_eps = tuple(range(32))
            val_eps = tuple(range(32, 40))

        if args.train_episodes is not None:
            train_eps = parse_episodes(args.train_episodes)
        if args.val_episodes is not None:
            val_eps = parse_episodes(args.val_episodes)
        overlap = set(train_eps) & (set(val_eps) | set(eval2_eps))
        if overlap:
            raise ValueError(f"Train/eval episode overlap is forbidden: {sorted(overlap)}")

        dataset_contract = CUBE_OUT_OF_BOX_CONTRACT
        if args.dataset_repo_id and args.dataset_repo_id != CUBE_OUT_OF_BOX_CONTRACT.repo_id:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.datasets.vam.config import VideoVAMDatasetConfig

            if args.dataset_revision is None:
                raise ValueError(
                    f"--dataset-revision (a commit sha) is required for non-canonical dataset "
                    f"{args.dataset_repo_id}; unpinned 'main' makes results irreproducible."
                )
            meta = LeRobotDatasetMetadata(args.dataset_repo_id, revision=args.dataset_revision)
            task_str = args.backbone_prompt
            if getattr(meta, "tasks", None) is not None and len(meta.tasks) > 0:
                task_str = meta.tasks.index[0] if hasattr(meta.tasks, "index") else str(meta.tasks[0])
            dataset_contract = VideoVAMDatasetConfig(
                repo_id=args.dataset_repo_id,
                revision=args.dataset_revision,
                fps=meta.fps,
                total_episodes=meta.total_episodes,
                total_frames=meta.total_frames,
                task=task_str,
            )
        elif args.dataset_repo_id is None:
            args.dataset_repo_id = "hubnemo/cube_out_of_box_dataset"
            if args.dataset_root is None:
                args.dataset_root = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")

        print(f"Online Backbone ({args.online_backbone}) Split Protocol ({effective_protocol}) Verified:")
        print(f"  Train episodes ({len(train_eps)}): {list(train_eps)}")
        print(f"  Val episodes   ({len(val_eps)}): {list(val_eps)}")
        print(f"  Dataset Repo ID: {args.dataset_repo_id}")
        print(f"  Train Stride: {args.train_stride}")
        print(f"  Augmentation: {'ENABLED' if args.augment else 'DISABLED'}")

        # No cross-run auto-discovery of evaluation caches: a cache built with a different LoRA,
        # alpha or sigma silently evaluates a different feature space. Without explicit manifests the
        # online extractor itself is used for validation (see OnlineVideoDataset below).
        if not is_cosmos3(args.online_backbone):
            for name in ("val_manifest", "eval2_manifest"):
                path = getattr(args, name)
                if path is None:
                    continue
                if args.dry_run:
                    print(f"[dry-run] Skipping online feature-identity check for {name}: {path}")
                else:
                    validate_online_eval_cache_identity(Path(path), args, source=name)

    elif args.train_manifest is not None:
        with open(args.val_manifest, encoding="utf-8") as f:
            val_manifest_data = json.load(f)
        print("Loading manifests:")
        print(f"  Train: {args.train_manifest}")
        print(f"  Val:   {args.val_manifest}")
        with open(args.train_manifest, encoding="utf-8") as f:
            train_manifest_data = json.load(f)

        loaded_split = None
        if args.split is not None:
            from lerobot.policies.vam.vam_split import load_vam_split

            loaded_split = load_vam_split(args.split, args.train_manifest, allow_partial=True)

        # Strictly enforce Split Protocol (Prevent Data Leakage!)
        train_eps, val_eps = validate_manifests_for_protocol(
            train_manifest_data,
            val_manifest_data,
            split=loaded_split,
            allow_subset=True,
            protocol=effective_protocol,
            require_identity=require_identity,
        )
        print(f"Protocol ({effective_protocol}) Verification PASSED.")
        print(f"  Train episodes ({len(train_eps)}): {list(train_eps)}")
        print(f"  Val episodes   ({len(val_eps)}): {list(val_eps)}")
    else:
        with open(args.val_manifest, encoding="utf-8") as f:
            val_manifest_data = json.load(f)
        from lerobot.policies.vam.base.split_guard import extract_episodes_from_manifest

        val_eps = tuple(sorted(extract_episodes_from_manifest(val_manifest_data)))
        if not set(val_eps) <= set(range(32, 40)):
            raise ValueError("Historical evaluation requires episodes 32..39")
        print(f"Eval-only Val episodes ({len(val_eps)}): {list(val_eps)}")

    # Auto-discover Eval-Set 2 (V2 held-out) if not explicitly supplied (offline mode)
    if not is_online and args.eval2_manifest is None and args.val_manifest is not None:
        val_p = Path(args.val_manifest).resolve()
        candidate1 = val_p.parent.parent / "eval2/manifest.json"
        # Only the sibling split of the same cache build is safe to auto-discover.
        if candidate1.is_file():
            args.eval2_manifest = str(candidate1)
            print(f"Auto-discovered Eval-Set 2 manifest: {args.eval2_manifest}")

    eval2_dataset: Any = None
    if args.eval2_manifest is not None and Path(args.eval2_manifest).is_file():
        print(f"  Eval-2 manifest: {args.eval2_manifest}")
        eval2_dataset = cosmos3_datasets.get("eval2")
        if eval2_dataset is None:
            eval2_dataset = UnifiedFeatureCacheDataset(
                args.eval2_manifest, context_transform=args.context_transform
            )
        if not eval2_eps:
            from lerobot.policies.vam.base.split_guard import extract_episodes_from_manifest

            eval2_eps = tuple(
                sorted(extract_episodes_from_manifest(_read_json_object(Path(args.eval2_manifest))))
            )
        print(f"Eval-Set 2 Verification PASSED ({len(eval2_dataset)} samples).")
    elif is_online and effective_protocol == "scale100":
        eval2_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=eval2_eps,
            stride=20,
            contract=dataset_contract,
        )
        print(f"Eval-Set 2 Online Dataset Initialized ({len(eval2_dataset)} samples).")

    # 2. Build Datasets
    val_dataset: Any = None
    if args.val_manifest is not None and Path(args.val_manifest).is_file():
        val_dataset = cosmos3_datasets.get("val")
        if val_dataset is None:
            val_dataset = UnifiedFeatureCacheDataset(
                args.val_manifest, context_transform=args.context_transform
            )
    elif is_online:
        val_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=val_eps,
            stride=20,
            contract=dataset_contract,
        )

    train_dataset: Any = None
    if is_online:
        train_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=train_eps,
            stride=args.train_stride,
            contract=dataset_contract,
            ram_headroom_gb=12 if args.augment else 26,
        )
        print(
            f"Loaded online video training dataset with {len(train_dataset)} windows (stride {args.train_stride}, augment={args.augment})"
        )
    elif args.train_manifest is not None:
        train_dataset = UnifiedFeatureCacheDataset(
            args.train_manifest, context_transform=args.context_transform
        )
        print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    else:
        print(f"Loaded {len(val_dataset)} val samples.")

    # 4. Resolve Normalizer
    if is_online:
        if args.normalizer_path is not None and Path(args.normalizer_path).is_file():
            print(f"Loading normalizer from pre-existing checkpoint {args.normalizer_path}...")
            norm_sd = load_file(str(args.normalizer_path))
            normalizer = SmolVLANormalizer(
                state_mean=norm_sd["state_mean"],
                state_std=norm_sd["state_std"],
                action_mean=norm_sd["action_mean"],
                action_std=norm_sd["action_std"],
                eps=float(norm_sd.get("eps", 1e-8)),
                source_split=str(norm_sd.get("source_split", "train")),
            )
            save_file(
                normalizer.state_dict(),
                str(args.output_dir / "normalizer.safetensors"),
                metadata={"artifact": "smolvla_mean_std_normalizer", "source_split": "train"},
            )
            meta_payload = {
                "artifact": "smolvla_mean_std_normalizer",
                "source_split": "train",
                "train_episodes": list(train_eps),
                "padded_actions_excluded": True,
                "state_mean": normalizer.state_mean.tolist(),
                "state_std": normalizer.state_std.tolist(),
                "action_mean": normalizer.action_mean.tolist(),
                "action_std": normalizer.action_std.tolist(),
            }
            (args.output_dir / "normalizer.json").write_text(json.dumps(meta_payload, indent=2) + "\n")
        else:
            print("Computing SmolVLA normalizer from online train split...")
            normalizer = compute_online_training_normalizer(
                train_dataset, train_eps, args.output_dir, kinematics=kinematics
            )
            print(f"Normalizer computed and saved to {args.output_dir / 'normalizer.safetensors'}")
    elif args.eval_only:
        norm_candidates = []
        if args.model_checkpoint is not None:
            norm_candidates.append(args.model_checkpoint.parent / "normalizer.safetensors")
        norm_candidates.append(args.output_dir / "normalizer.safetensors")
        norm_candidates.append(Path(args.checkpoint_path) / "normalizer.safetensors")

        norm_path = None
        for cand in norm_candidates:
            if cand.is_file():
                norm_path = cand
                break

        if norm_path is None:
            raise FileNotFoundError(
                "Normalizer not found. In --eval-only mode, 'normalizer.safetensors' must exist in output_dir "
                "or alongside --model-checkpoint."
            )
        print(f"Loading normalizer from existing checkpoint {norm_path} (eval-only mode; no overwrite)...")
        norm_sd = load_file(str(norm_path))
        normalizer = SmolVLANormalizer(
            state_mean=norm_sd["state_mean"],
            state_std=norm_sd["state_std"],
            action_mean=norm_sd["action_mean"],
            action_std=norm_sd["action_std"],
            eps=float(norm_sd.get("eps", 1e-8)),
            source_split=str(norm_sd.get("source_split", "train")),
        )
    else:
        print("Computing SmolVLA normalizer from train split...")
        assert train_dataset is not None
        normalizer = compute_training_normalizer(train_dataset, train_eps, args.output_dir)
        print(f"Normalizer computed and saved to {args.output_dir / 'normalizer.safetensors'}")

    backbone_extractor: Any = None
    if val_dataset is not None and getattr(val_dataset[0], "context", None) is not None:
        sample_context = val_dataset[0].context
        input_channels = sample_context.shape[-1]
        num_tokens = sample_context.shape[0]
    elif is_online:
        b_key = args.online_backbone.replace("-", "_").lower()
        if b_key == "cosmos3_nano":
            input_channels = 4096
            num_tokens = 600
        elif b_key in ("cosmos3_edge", "cosmos3"):
            input_channels = 2048
            num_tokens = 600
        elif any(k in b_key for k in ("cosmos2b", "cosmos_2b", "cosmos_t2", "cosmos")):
            input_channels = 2048
            num_tokens = 2400
        elif any(k in b_key for k in ("flux2", "flux")):
            input_channels = 6144
            num_tokens = 256
        else:
            raise ValueError(f"Unsupported online backbone: {args.online_backbone}")
    else:
        sample_context = val_dataset[0].context
        input_channels = sample_context.shape[-1]
        num_tokens = sample_context.shape[0]

    if is_online and not args.dry_run:
        b_key = args.online_backbone.replace("-", "_").lower()
        if b_key in ("cosmos3_edge", "cosmos3", "cosmos3_nano"):
            from lerobot.policies.vam.cosmos3_features import (
                Cosmos3ExtractorConfig,
                Cosmos3FeatureExtractor,
            )

            is_nano = b_key == "cosmos3_nano"
            default_ckpt = Path(
                "/home/anton/.cache/video-vam/cosmos3-nano"
                if is_nano
                else "/home/anton/.cache/video-vam/cosmos3-edge"
            )
            ckpt_dir = args.backbone_checkpoint or default_ckpt
            train_bb_lora = bool(getattr(args, "train_backbone_lora", False))
            if getattr(args, "backbone_fps", None) is not None:
                resolved_cosmos3_fps = float(args.backbone_fps)
            elif args.val_manifest is not None or args.eval2_manifest is not None:
                resolved_cosmos3_fps = 10.0
            else:
                resolved_cosmos3_fps = float(dataset_contract.fps)
            b_cfg = Cosmos3ExtractorConfig(
                backbone_name="cosmos3-nano" if is_nano else "cosmos3-edge",
                checkpoint_path=ckpt_dir,
                hidden_layers=(args.backbone_layer,),
                device=str(device),
                dtype="bfloat16",
                fps=resolved_cosmos3_fps,
                base_fps=24.0,
                prompt=args.backbone_prompt,
                lora_checkpoint=args.backbone_lora_weights,
                lora_rank=args.backbone_lora_rank,
                lora_alpha=args.backbone_lora_alpha,
                trainable_lora=train_bb_lora,
                lora_use_dora=bool(getattr(args, "backbone_lora_dora", False)),
                lora_pathway=getattr(args, "backbone_lora_pathway", "gen_only"),
                gradient_checkpointing=bool(getattr(args, "backbone_grad_checkpointing", True)) and train_bb_lora,
                compile_dit=not train_bb_lora,
                compile_vae=True,
                vae_compile_mode="default",
            )
            adapter_mode_str = (
                f"trainable {'DoRA' if b_cfg.lora_use_dora else 'LoRA'} ({b_cfg.lora_pathway}, r={b_cfg.lora_rank}, alpha={b_cfg.lora_alpha})"
                if train_bb_lora
                else f"frozen (lora: {args.backbone_lora_weights})"
            )
            print(
                f"Loading online backbone {args.online_backbone} (layer {args.backbone_layer}, {adapter_mode_str})..."
            )
            backbone_extractor = Cosmos3FeatureExtractor(b_cfg)
            backbone_extractor._smolexpert_eval_identity = cosmos3_identity
            backbone_extractor.eval()
            if not train_bb_lora:
                for p in backbone_extractor.parameters():
                    p.requires_grad_(False)
        elif any(k in b_key for k in ("cosmos2b", "cosmos_2b", "cosmos_t2", "cosmos")):
            from types import SimpleNamespace

            from lerobot.policies.vam.cosmos_predict2_extractor import (
                CosmosPredict2Extractor,
                CosmosPredict2ExtractorConfig,
            )
            from lerobot.policies.vam.cosmos_prompt_embedding import load_prompt_embedding

            ckpt_pt = args.backbone_checkpoint or Path(
                "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
            )
            tok_pth = args.backbone_tokenizer or Path(
                "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
            )
            p_embed_path = args.backbone_prompt_embedding or Path(
                "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
            )
            prompt_art = load_prompt_embedding(p_embed_path)
            prompt_embed_tensor = prompt_art.embedding.to(device=device, dtype=torch.bfloat16)

            b_cfg = CosmosPredict2ExtractorConfig(
                checkpoint_path=ckpt_pt,
                tokenizer_path=tok_pth,
                device=str(device),
                dtype="bfloat16",
                high_noise_sigma=args.high_noise_sigma,
                seed=args.seed,
                state_t=2,
                hidden_layer=args.backbone_layer if args.backbone_layer is not None else 20,
                stop_after_step=0,
                vae_input_mode="observed_prefix",
                compile_friendly=True,
                torch_compile=False,
                compile_mode="default",
            )
            print(
                f"Loading online backbone Cosmos 2B (T=2 mode, checkpoint: {ckpt_pt}, lora: {args.backbone_lora_weights})..."
            )
            raw_c2b_ext = CosmosPredict2Extractor(b_cfg)
            if args.backbone_lora_weights is not None:
                from lerobot.policies.vam.cosmos_lora import inject_and_load_lora_file

                # Rank/alpha come from the adapter sidecar; explicit CLI values must agree with it.
                lora_info = inject_and_load_lora_file(
                    raw_c2b_ext.backbone,
                    args.backbone_lora_weights,
                    rank=args.backbone_lora_rank,
                    alpha=args.backbone_lora_alpha,
                )
                args.backbone_lora_rank = lora_info["rank"]
                args.backbone_lora_alpha = lora_info["alpha"]
                print(
                    f"Loaded Cosmos 2B LoRA {lora_info['path']} "
                    f"(rank={lora_info['rank']}, alpha={lora_info['alpha']})"
                )
            if device.type == "cuda":
                raw_c2b_ext.backbone = torch.compile(raw_c2b_ext.backbone, mode="default")
                if hasattr(raw_c2b_ext.tokenizer, "model") and hasattr(raw_c2b_ext.tokenizer.model, "model"):
                    inner_vae = raw_c2b_ext.tokenizer.model.model
                    if hasattr(inner_vae, "encoder") and not isinstance(
                        inner_vae.encoder, torch._dynamo.eval_frame.OptimizedModule
                    ):
                        inner_vae.encoder = torch.compile(inner_vae.encoder, mode="default")
                        print("Compiled Cosmos 2B VAE encoder with torch.compile (mode='default')")
            raw_c2b_ext.backbone.eval()

            class OnlineCosmos2BWrapper:
                def __init__(self, extractor: Any, prompt: Tensor) -> None:
                    self.extractor = extractor
                    self.prompt = prompt

                def extract(self, rgb_frames: Tensor) -> Any:
                    b = rgb_frames.shape[0]
                    feats = []
                    for i in range(b):
                        out = self.extractor.extract(
                            images=rgb_frames[i : i + 1], prompt_embedding=self.prompt
                        )
                        feats.append(out.tokens)
                    return SimpleNamespace(features=torch.cat(feats, dim=0))

                def parameters(self):
                    return self.extractor.parameters()

            backbone_extractor = OnlineCosmos2BWrapper(raw_c2b_ext, prompt_embed_tensor)
        elif any(k in b_key for k in ("flux2", "flux")):
            from types import SimpleNamespace

            from lerobot.policies.vam.base.native_video import load_and_validate_prompt_embedding
            from lerobot.policies.vam.flux2_klein_extractor import (
                Flux2KleinExtractor,
                Flux2KleinExtractorConfig,
                prepare_multi_reference_conditioning,
            )

            if args.backbone_vae is None or not Path(args.backbone_vae).exists():
                raise ValueError(
                    "Online FLUX.2 klein requires --backbone-vae (the native VAE). The previous "
                    "bilinear 32x32 pseudo-latent path was removed as invalid (see correctness audit)."
                )
            b_cfg = Flux2KleinExtractorConfig(
                checkpoint_path=args.backbone_checkpoint,
                vae_path=args.backbone_vae,
                device=str(device),
                dtype="bfloat16",
                num_layers=8,
                num_single_layers=0,
                tap_location="junction",
            )
            print(
                f"Loading online backbone FLUX.2 klein (tap_location: junction, lora: {args.backbone_lora_weights})..."
            )
            raw_flux_ext = Flux2KleinExtractor(config=b_cfg)
            flux_text_embedding = None
            if args.backbone_prompt_embedding is not None:
                flux_text_embedding, _ = load_and_validate_prompt_embedding(
                    args.backbone_prompt_embedding,
                    args.backbone_prompt,
                    b_cfg.joint_attention_dim,
                    device=device,
                    dtype=torch.bfloat16,
                )
            if args.backbone_lora_weights is not None:
                from lerobot.policies.vam.flux2_klein_lora import (
                    Flux2KleinLoRAConfig,
                    inject_flux2_klein_lora,
                    load_flux2_klein_lora,
                )

                lora_cfg = Flux2KleinLoRAConfig(
                    rank=args.backbone_lora_rank,
                    alpha=args.backbone_lora_alpha,
                    target_double_blocks=tuple(range(b_cfg.num_layers)),
                    target_single_blocks=(),
                )
                inject_flux2_klein_lora(raw_flux_ext.transformer, lora_cfg)
                load_flux2_klein_lora(
                    raw_flux_ext.transformer,
                    Path(args.backbone_lora_weights).expanduser().resolve(),
                    expected_rank=args.backbone_lora_rank,
                    expected_alpha=args.backbone_lora_alpha,
                )
                print(f"Loaded FLUX.2 klein LoRA weights from {args.backbone_lora_weights}")

            class OnlineFlux2KleinWrapper:
                """Same contract as build_vam_feature_cache.py: native VAE, history = frames t-4..t-2, target = t."""

                def __init__(self, extractor: Any, dev: torch.device, text: Tensor | None) -> None:
                    self.extractor = extractor
                    self.dev = dev
                    self.text = text

                def extract(self, rgb_frames: Tensor) -> Any:
                    # rgb_frames: [B, 3, T, H, W] in [0, 1]
                    lat = self.extractor.encode_latents(rgb_frames.to(device=self.dev, dtype=torch.bfloat16))
                    feats = []
                    for i in range(lat.shape[0]):
                        target = lat[i : i + 1, :, -1]
                        hist = [lat[i : i + 1, :, k] for k in range(min(3, lat.shape[2] - 1))]
                        cond = prepare_multi_reference_conditioning(
                            target_latent=target,
                            observation_history=hist,
                            joint_attention_dim=self.extractor.config.joint_attention_dim,
                            encoder_hidden_states=self.text,
                            device=self.dev,
                            dtype=torch.bfloat16,
                        )
                        feats.append(self.extractor.extract(cond_inputs=cond).features)
                    return SimpleNamespace(features=torch.cat(feats, dim=0))

                def parameters(self):
                    return self.extractor.parameters()

            backbone_extractor = OnlineFlux2KleinWrapper(raw_flux_ext, device, flux_text_embedding)
        else:
            raise ValueError(f"Unsupported online backbone: {args.online_backbone}")

    print(f"Context feature shape: [{num_tokens} tokens, {input_channels} channels]")

    # 5. Build or Load SmolExpert Action Decoder (Fail-Closed: no silent toy fallback)
    if args.dry_run:
        print("Using architecture-matched synthetic tiny SmolExpert decoder (explicit --dry-run)...")
        decoder = build_synthetic_tiny_decoder(
            input_channels=input_channels,
            normalizer=normalizer,
            device=device,
        )
    else:
        print(f"Loading pretrained SmolExpert from {args.checkpoint_path}...")
        decoder = SmolExpertActionDecoder.from_pretrained(
            args.checkpoint_path,
            normalizer=normalizer,
            device=device,
            num_steps=args.num_steps,
            input_channels=input_channels,
        )

    decoder_trainable_params = [p for p in decoder.parameters() if p.requires_grad]
    backbone_trainable_params = (
        [p for p in backbone_extractor.parameters() if p.requires_grad]
        if (backbone_extractor is not None and getattr(args, "train_backbone_lora", False))
        else []
    )
    trainable_params = decoder_trainable_params + backbone_trainable_params
    total_params = sum(p.numel() for p in trainable_params)
    if backbone_trainable_params:
        bb_params = sum(p.numel() for p in backbone_trainable_params)
        dec_params = sum(p.numel() for p in decoder_trainable_params)
        print(
            f"Trainable parameters: {total_params:,} (SmolExpert: {dec_params:,}, "
            f"Backbone {'DoRA' if getattr(args, 'backbone_lora_dora', False) else 'LoRA'}: {bb_params:,})"
        )
    else:
        print(f"Trainable parameters: {total_params:,}")

    if not args.eval_only and args.model_checkpoint and args.model_checkpoint.is_file():
        print(f"Warm-starting SmolExpert weights from {args.model_checkpoint}...")
        warm_sd = load_file(str(args.model_checkpoint))
        warm_sd = {k.replace("model.", ""): v for k, v in warm_sd.items()}
        decoder.load_state_dict(warm_sd)

    # Eval-only execution
    if args.eval_only:
        ckpt_to_load = None
        if args.model_checkpoint and args.model_checkpoint.is_file():
            ckpt_to_load = args.model_checkpoint
        elif (args.output_dir / "best.safetensors").is_file():
            ckpt_to_load = args.output_dir / "best.safetensors"
        elif Path(args.checkpoint_path).is_file():
            ckpt_to_load = Path(args.checkpoint_path)

        if ckpt_to_load:
            print(f"Loading weights from {ckpt_to_load}...")
            state_dict = load_file(str(ckpt_to_load))
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            decoder.load_state_dict(state_dict)

        print("Evaluating on Eval-Set 1 (Historical Benchmark)...")
        eval1_res = evaluate_validation(
            decoder,
            val_dataset,
            device=device,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            seed=args.seed,
        )
        eval2_res = None
        if eval2_dataset is not None:
            print("Evaluating on Eval-Set 2 (New Benchmark)...")
            eval2_res = evaluate_validation(
                decoder,
                eval2_dataset,
                device=device,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                seed=args.seed + 1000,
            )

        eval_out = {
            "backbone": args.backbone,
            "protocol": effective_protocol,
            "eval1_historical": eval1_res,
            "eval2_new_benchmark": eval2_res,
        }
        (args.output_dir / "eval_metrics.json").write_text(json.dumps(eval_out, indent=2) + "\n")

        print("\n" + "=" * 80)
        print(f"EVALUATION SUMMARY: {args.backbone.upper()} ({effective_protocol})")
        print("=" * 80)
        print(f"Eval-Set 1 (Historical Protocol 1.0, eps {val_eps[0]}..{val_eps[-1]}):")
        print(
            f"  Trajectory RMSE: {eval1_res['val_rmse']:.3f} | Arm RMSE: {eval1_res['arm_rmse_deg']:.3f}° | "
            f"Gripper RMSE: {eval1_res['gripper_rmse']:.3f} | H1: {eval1_res['val_h1']:.3f} | "
            f"First-5 (Mean): {eval1_res['val_first5']:.3f} | First-5 (Pooled): {eval1_res['val_first5_pooled_rmse']:.3f}"
        )
        if eval2_res is not None:
            print(f"Eval-Set 2 (New Benchmark, eps {eval2_eps[0]}..{eval2_eps[-1]}):")
            print(
                f"  Trajectory RMSE: {eval2_res['val_rmse']:.3f} | Arm RMSE: {eval2_res['arm_rmse_deg']:.3f}° | "
                f"Gripper RMSE: {eval2_res['gripper_rmse']:.3f} | H1: {eval2_res['val_h1']:.3f} | "
                f"First-5 (Mean): {eval2_res['val_first5']:.3f} | First-5 (Pooled): {eval2_res['val_first5_pooled_rmse']:.3f}"
            )
        print("=" * 80 + "\n")
        return 0

    assert train_dataset is not None

    run_manifest = build_run_manifest(args, train_dataset, val_dataset, eval2_dataset, effective_protocol)
    if cosmos3_identity is not None:
        run_manifest["online_feature_identity"] = cosmos3_identity
    atomic_write_json(args.output_dir / "run_manifest.json", run_manifest)

    # 6. Optimizer & Scheduler
    if backbone_trainable_params:
        param_groups: list[dict[str, Any]] = [
            {"params": decoder_trainable_params, "lr": args.lr},
            {"params": backbone_trainable_params, "lr": getattr(args, "backbone_lora_lr", args.lr)},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )
    early_stopping = EarlyStoppingTracker(patience=args.patience, min_steps=args.min_steps)

    # Initial Validation check at step 0
    print("Running initial validation check...")
    val_metrics = evaluate_validation(
        decoder,
        val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_steps=min(args.num_steps, 5 if args.dry_run else args.num_steps),
        seed=args.seed,
        extractor=backbone_extractor,
        kinematics=kinematics,
    )
    print(
        f"Initial Val -> RMSE: {val_metrics['val_rmse']:.3f}, "
        f"Arm RMSE: {val_metrics['arm_rmse_deg']:.3f}°, "
        f"Gripper RMSE: {val_metrics['gripper_rmse']:.3f}, "
        f"H1: {val_metrics['val_h1']:.3f}, "
        f"Flow Loss: {val_metrics['val_flow_loss']:.4f} "
        f"(select_metric={args.select_metric})"
    )
    if args.model_checkpoint and args.model_checkpoint.is_file():
        init_select_val = float(val_metrics[args.select_metric])
        early_stopping.update(init_select_val, 0)
        save_weights_artifact(
            args.output_dir, "best", decoder.state_dict(), 0, run_manifest, metric=init_select_val
        )
        if backbone_trainable_params and hasattr(backbone_extractor, "transformer"):
            from lerobot.policies.vam.cosmos3_lora import save_cosmos3_lora

            save_cosmos3_lora(
                backbone_extractor.transformer,
                args.output_dir / "best_backbone_lora.safetensors",
                rank=args.backbone_lora_rank,
                alpha=args.backbone_lora_alpha,
                use_dora=bool(getattr(args, "backbone_lora_dora", False)),
            )
        init_best_meta = {
            "step": 0,
            "select_metric": args.select_metric,
            "val_rmse": float(val_metrics["val_rmse"]),
            "val_h1": float(val_metrics["val_h1"]),
            "val_first5": float(val_metrics["val_first5"]),
            "val_first5_pooled_rmse": float(val_metrics["val_first5_pooled_rmse"]),
            "arm_rmse_deg": float(val_metrics["arm_rmse_deg"]),
            "gripper_rmse": float(val_metrics["gripper_rmse"]),
            "val_flow_loss": float(val_metrics["val_flow_loss"]),
            "eval1_rmse": float(val_metrics["val_rmse"]),
            "eval1_h1": float(val_metrics["val_h1"]),
            "eval1_first5": float(val_metrics["val_first5"]),
            "eval1_flow_loss": float(val_metrics["val_flow_loss"]),
            "backbone": args.backbone,
            "protocol": effective_protocol,
            "train_episodes": list(train_eps),
            "val_episodes": list(val_eps),
        }
        (args.output_dir / "best_metrics.json").write_text(json.dumps(init_best_meta, indent=2) + "\n")

    # 7. Training Loop with Exact Gradient Accumulation
    if args.compile_loss and device.type == "cuda" and not args.dry_run:
        compile_mode = "default"
        print(f"Compiling flow_matching_loss with mode='{compile_mode}'...", flush=True)
        flow_matching_loss_fn = torch.compile(decoder.flow_matching_loss, mode=compile_mode)
    else:
        flow_matching_loss_fn = decoder.flow_matching_loss
    decoder.train()
    opt_step = 0
    indices = list(range(len(train_dataset)))
    random.shuffle(indices)
    data_idx = 0

    best_checkpoint_path = args.output_dir / "best.safetensors"

    print(
        f"Starting training loop (max_steps={args.max_steps} optimizer updates, "
        f"grad_accum_steps={args.grad_accum_steps}, val_every={args.val_every})..."
    )
    start_time = time.time()

    from concurrent.futures import Future, ThreadPoolExecutor

    def _fetch_batch(batch_indices: list[int]) -> list[Any]:
        return [train_dataset[i] for i in batch_indices]

    def _next_batch_indices() -> list[int]:
        nonlocal data_idx
        out_idxs: list[int] = []
        for _ in range(args.batch_size):
            out_idxs.append(indices[data_idx])
            data_idx += 1
            if data_idx >= len(indices):
                random.shuffle(indices)
                data_idx = 0
        return out_idxs

    prefetch_pool = ThreadPoolExecutor(max_workers=1) if (is_online and not args.dry_run) else None
    prefetch_queue: list[Future[list[Any]]] = []
    if prefetch_pool is not None:
        for _ in range(2):
            prefetch_queue.append(prefetch_pool.submit(_fetch_batch, _next_batch_indices()))

    if is_online and not args.dry_run and backbone_extractor is not None and not hasattr(backbone_extractor, "_online_ram_cache"):
        backbone_extractor._online_ram_cache = {}
        backbone_extractor._ram_cache_full = False

    while opt_step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _micro_step in range(args.grad_accum_steps):
            if prefetch_pool is not None:
                batch_items = prefetch_queue.pop(0).result()
                prefetch_queue.append(prefetch_pool.submit(_fetch_batch, _next_batch_indices()))
            else:
                batch_items = _fetch_batch(_next_batch_indices())

            states = torch.stack([item.state for item in batch_items]).to(device=device, dtype=torch.float32)
            actions = torch.stack([item.action for item in batch_items]).to(
                device=device, dtype=torch.float32
            )
            if is_online:
                if args.dry_run:
                    contexts = torch.randn(len(batch_items), num_tokens, input_channels, device=device)
                else:
                    assert backbone_extractor is not None
                    train_bb = bool(backbone_trainable_params)
                    ram_cache: dict[str, Tensor] = getattr(backbone_extractor, "_online_ram_cache", {})
                    if not train_bb and not args.augment and all(item.sample_id in ram_cache for item in batch_items):
                        contexts = torch.stack([ram_cache[item.sample_id] for item in batch_items]).to(
                            device=device
                        )
                    else:
                        raw_rgb = torch.stack(
                            [item.rgb.to(device=device, non_blocking=True) for item in batch_items]
                        )
                        if args.augment:
                            if getattr(args, "aug_strategy", "physics") == "physics":
                                rgb_in = physics_augment_temporal_window_gpu(raw_rgb)
                            else:
                                rgb_in = augment_temporal_window_gpu(
                                    raw_rgb, spatial_crop=getattr(args, "spatial_crop", False)
                                )
                        else:
                            rgb_in = unaugmented_temporal_window_gpu(raw_rgb)

                        if train_bb and hasattr(backbone_extractor, "encode_latents") and hasattr(
                            backbone_extractor, "forward_transformer_blocks"
                        ):
                            if not hasattr(backbone_extractor, "_online_latent_cache"):
                                backbone_extractor._online_latent_cache = {}
                            lat_cache: dict[str, Tensor] = backbone_extractor._online_latent_cache
                            if not args.augment and all(item.sample_id in lat_cache for item in batch_items):
                                latents_batch = torch.stack(
                                    [lat_cache[item.sample_id] for item in batch_items]
                                ).to(device=device, non_blocking=True)
                            else:
                                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                    latents_batch = backbone_extractor.encode_latents(rgb_in)
                                    if (
                                        not args.augment
                                        and not getattr(backbone_extractor, "_ram_cache_full", False)
                                        and len(lat_cache) < 16384
                                    ):
                                        if (
                                            len(lat_cache) % 128 == 0
                                            and OnlineVideoDataset._get_available_ram_bytes() < 12 * (1024**3)
                                        ):
                                            backbone_extractor._ram_cache_full = True
                                        else:
                                            for b_item, lat_t in zip(batch_items, latents_batch, strict=False):
                                                lat_cache[b_item.sample_id] = lat_t.detach().cpu()
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                tapped = backbone_extractor.forward_transformer_blocks(latents_batch)
                                contexts = tapped[args.backbone_layer].to(device=device)
                        else:
                            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                                if hasattr(backbone_extractor, "encode_latents") and hasattr(
                                    backbone_extractor, "forward_transformer_blocks"
                                ):
                                    latents_batch = backbone_extractor.encode_latents(rgb_in)
                                    tapped = backbone_extractor.forward_transformer_blocks(latents_batch)
                                    contexts = tapped[args.backbone_layer].to(device=device)
                                else:
                                    extracted_list = []
                                    for b_i in range(rgb_in.shape[0]):
                                        out_f = backbone_extractor.extract(rgb_frames=rgb_in[b_i : b_i + 1])
                                        extracted_list.append(out_f.features)
                                    contexts = torch.cat(extracted_list, dim=0).to(device=device)
                                if (
                                    not args.augment
                                    and not getattr(backbone_extractor, "_ram_cache_full", False)
                                    and len(ram_cache) < 4096
                                ):
                                    if len(ram_cache) % 128 == 0 and OnlineVideoDataset._get_available_ram_bytes() < 14 * (1024**3):
                                        backbone_extractor._ram_cache_full = True
                                    else:
                                        for b_item, ctx_t in zip(batch_items, contexts, strict=False):
                                            ram_cache[b_item.sample_id] = ctx_t.detach().cpu()
            else:
                contexts = torch.stack([item.context for item in batch_items]).to(device=device)
            action_is_pad = torch.stack([item.action_is_pad for item in batch_items]).to(
                device=device, dtype=torch.bool
            )

            if kinematics is not None:
                states = joints_to_cartesian(states, kinematics)
                actions = joints_to_cartesian(actions, kinematics)

            # Flow matching loss with strict padding masking
            loss = flow_matching_loss_fn(
                state=states,
                action=actions,
                context=contexts,
                action_is_pad=action_is_pad,
            )

            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()
            accum_loss += float(loss.item()) / args.grad_accum_steps

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)

        optimizer.step()
        scheduler.step()
        opt_step += 1
        if args.save_last_every > 0 and opt_step % args.save_last_every == 0:
            save_weights_artifact(args.output_dir, "last", decoder.state_dict(), opt_step, run_manifest)
            if backbone_trainable_params and hasattr(backbone_extractor, "transformer"):
                from lerobot.policies.vam.cosmos3_lora import save_cosmos3_lora

                save_cosmos3_lora(
                    backbone_extractor.transformer,
                    args.output_dir / "last_backbone_lora.safetensors",
                    rank=args.backbone_lora_rank,
                    alpha=args.backbone_lora_alpha,
                    use_dora=bool(getattr(args, "backbone_lora_dora", False)),
                )

        log_interval = 20 if is_online else max(1, args.val_every // 5)
        if opt_step % log_interval == 0 or opt_step == args.max_steps:
            current_lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            print(
                f"Step {opt_step}/{args.max_steps} | Loss: {accum_loss:.4f} | "
                f"LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s",
                flush=True,
            )

        # Periodic Validation & Early Stopping Check on exact optimizer step boundaries
        max_sec = getattr(args, "max_train_seconds", None)
        time_limit_reached = max_sec is not None and (time.time() - start_time) >= float(max_sec)
        if opt_step % args.val_every == 0 or opt_step == args.max_steps or time_limit_reached:
            val_metrics = evaluate_validation(
                decoder,
                val_dataset,
                device=device,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                seed=args.seed,
                extractor=backbone_extractor,
                kinematics=kinematics,
            )
            rmse = float(val_metrics["val_rmse"])
            arm_rmse = float(val_metrics["arm_rmse_deg"])
            gripper_rmse = float(val_metrics["gripper_rmse"])
            h1 = float(val_metrics["val_h1"])
            first5 = float(val_metrics["val_first5"])
            first5_pooled = float(val_metrics["val_first5_pooled_rmse"])
            flow = float(val_metrics["val_flow_loss"])

            eval2_metrics = None
            eval2_rmse, eval2_arm, eval2_grip, eval2_h1, eval2_first5, eval2_flow = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
            if eval2_dataset is not None:
                eval2_metrics = evaluate_validation(
                    decoder,
                    eval2_dataset,
                    device=device,
                    batch_size=args.batch_size,
                    num_steps=args.num_steps,
                    seed=args.seed + 1000,
                    extractor=backbone_extractor,
                )
                eval2_rmse = float(eval2_metrics["val_rmse"])
                eval2_arm = float(eval2_metrics["arm_rmse_deg"])
                eval2_grip = float(eval2_metrics["gripper_rmse"])
                eval2_h1 = float(eval2_metrics["val_h1"])
                eval2_first5 = float(eval2_metrics["val_first5"])
                eval2_flow = float(eval2_metrics["val_flow_loss"])

            select_val = float(val_metrics[args.select_metric])
            is_best = early_stopping.update(select_val, opt_step)
            marker = f" *** BEST ({args.select_metric}) ***" if is_best else ""
            if eval2_dataset is not None:
                print(
                    f"[Eval @ {opt_step}] Eval-1 Val Loss: {flow:.4f} | RMSE: {rmse:.3f} (Arm: {arm_rmse:.3f}°, Grip: {gripper_rmse:.3f}) | H1: {h1:.3f} | First-5: {first5:.3f} "
                    f"|| Eval-2 Val Loss: {eval2_flow:.4f} | RMSE: {eval2_rmse:.3f} (Arm: {eval2_arm:.3f}°, Grip: {eval2_grip:.3f}) | H1: {eval2_h1:.3f} | First-5: {eval2_first5:.3f} "
                    f"(Patience: {early_stopping.bad_evaluations}/{early_stopping.patience}){marker}",
                    flush=True,
                )
            else:
                cart_info = (
                    f" | Cart: {val_metrics['cart_pos_mm']:.2f}mm" if "cart_pos_mm" in val_metrics else ""
                )
                print(
                    f"[Eval @ {opt_step}] Val Loss: {flow:.4f} | RMSE: {rmse:.3f} (Arm: {arm_rmse:.3f}°, Grip: {gripper_rmse:.3f}){cart_info} | "
                    f"H1: {h1:.3f} | First-5: {first5:.3f} "
                    f"(Patience: {early_stopping.bad_evaluations}/{early_stopping.patience}){marker}",
                    flush=True,
                )

            eval_record = {
                "step": opt_step,
                "val_rmse": rmse,
                "val_h1": h1,
                "val_first5": first5,
                "val_first5_pooled_rmse": first5_pooled,
                "arm_rmse_deg": arm_rmse,
                "gripper_rmse": gripper_rmse,
                "val_flow_loss": flow,
                "eval1_rmse": rmse,
                "eval1_h1": h1,
                "eval1_first5": first5,
                "eval1_flow_loss": flow,
                "select_metric": args.select_metric,
                "is_best": is_best,
                "bad_evaluations": early_stopping.bad_evaluations,
                "protocol": effective_protocol,
            }
            if eval2_dataset is not None:
                eval_record.update(
                    {
                        "eval2_rmse": eval2_rmse,
                        "eval2_arm_rmse_deg": eval2_arm,
                        "eval2_gripper_rmse": eval2_grip,
                        "eval2_h1": eval2_h1,
                        "eval2_first5": eval2_first5,
                        "eval2_flow_loss": eval2_flow,
                    }
                )
            with open(args.output_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(eval_record) + "\n")

            if is_best:
                save_weights_artifact(
                    args.output_dir, "best", decoder.state_dict(), opt_step, run_manifest, metric=select_val
                )
                if backbone_trainable_params and hasattr(backbone_extractor, "transformer"):
                    from lerobot.policies.vam.cosmos3_lora import save_cosmos3_lora

                    save_cosmos3_lora(
                        backbone_extractor.transformer,
                        args.output_dir / "best_backbone_lora.safetensors",
                        rank=args.backbone_lora_rank,
                        alpha=args.backbone_lora_alpha,
                        use_dora=bool(getattr(args, "backbone_lora_dora", False)),
                    )
                best_meta = {
                    "step": opt_step,
                    "select_metric": args.select_metric,
                    "val_rmse": rmse,
                    "val_h1": h1,
                    "val_first5": first5,
                    "val_first5_pooled_rmse": first5_pooled,
                    "arm_rmse_deg": arm_rmse,
                    "gripper_rmse": gripper_rmse,
                    "val_flow_loss": flow,
                    "eval1_rmse": rmse,
                    "eval1_h1": h1,
                    "eval1_first5": first5,
                    "eval1_flow_loss": flow,
                    "backbone": args.backbone,
                    "protocol": effective_protocol,
                    "train_episodes": list(train_eps),
                    "val_episodes": list(val_eps),
                }
                if eval2_dataset is not None:
                    best_meta.update(
                        {
                            "eval2_rmse": eval2_rmse,
                            "eval2_arm_rmse_deg": eval2_arm,
                            "eval2_gripper_rmse": eval2_grip,
                            "eval2_h1": eval2_h1,
                            "eval2_first5": eval2_first5,
                            "eval2_flow_loss": eval2_flow,
                            "eval2_episodes": list(eval2_eps),
                        }
                    )
                (args.output_dir / "best_metrics.json").write_text(json.dumps(best_meta, indent=2) + "\n")

            decoder.train()

            if early_stopping.should_stop:
                best_fmt = (
                    f"Best Val Flow Loss: {early_stopping.best_metric:.4f}"
                    if args.select_metric == "val_flow_loss"
                    else f"Best RMSE: {early_stopping.best_metric:.3f}"
                )
                print(
                    f"Early stopping triggered at step {opt_step}: no {args.select_metric} improvement "
                    f"for {early_stopping.patience} evaluations. {best_fmt} "
                    f"at step {early_stopping.best_step}."
                )
                break
            if time_limit_reached:
                print(
                    f"Wall-clock time limit ({max_sec}s) reached at step {opt_step} "
                    f"(elapsed {time.time() - start_time:.1f}s). Finishing cleanly...",
                    flush=True,
                )
                break

    total_time = time.time() - start_time
    if prefetch_pool is not None:
        prefetch_pool.shutdown(wait=False, cancel_futures=True)

    # Save live final weights before loading best for final evaluation.
    save_weights_artifact(args.output_dir, "last", decoder.state_dict(), opt_step, run_manifest)
    if backbone_trainable_params and hasattr(backbone_extractor, "transformer"):
        from lerobot.policies.vam.cosmos3_lora import save_cosmos3_lora

        save_cosmos3_lora(
            backbone_extractor.transformer,
            args.output_dir / "last_backbone_lora.safetensors",
            rank=args.backbone_lora_rank,
            alpha=args.backbone_lora_alpha,
            use_dora=bool(getattr(args, "backbone_lora_dora", False)),
        )

    # Final evaluation of best checkpoint on both Eval-1 and Eval-2
    if best_checkpoint_path.is_file():
        print("Loading best checkpoint for final dual evaluation...")
        best_state = load_file(str(best_checkpoint_path))
        decoder.load_state_dict(best_state)
        best_bb_lora_path = args.output_dir / "best_backbone_lora.safetensors"
        if backbone_trainable_params and best_bb_lora_path.is_file() and hasattr(backbone_extractor, "transformer"):
            from lerobot.policies.vam.cosmos3_lora import load_cosmos3_lora

            load_cosmos3_lora(
                backbone_extractor.transformer,
                best_bb_lora_path,
                rank=args.backbone_lora_rank,
                alpha=args.backbone_lora_alpha,
                use_dora=bool(getattr(args, "backbone_lora_dora", False)),
            )

    final_eval1 = evaluate_validation(
        decoder,
        val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        seed=args.seed,
        extractor=backbone_extractor,
        kinematics=kinematics,
    )
    final_eval2 = None
    if eval2_dataset is not None:
        final_eval2 = evaluate_validation(
            decoder,
            eval2_dataset,
            device=device,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            seed=args.seed + 1000,
            extractor=backbone_extractor,
        )

    dual_summary = {
        "backbone": args.backbone,
        "protocol": effective_protocol,
        "best_step": early_stopping.best_step,
        "total_steps": opt_step,
        "total_time_seconds": total_time,
        "eval1_historical": {
            "val_rmse": float(final_eval1["val_rmse"]),
            "val_h1": float(final_eval1["val_h1"]),
            "val_first5": float(final_eval1["val_first5"]),
            "val_first5_pooled_rmse": float(final_eval1["val_first5_pooled_rmse"]),
            "arm_rmse_deg": float(final_eval1["arm_rmse_deg"]),
            "gripper_rmse": float(final_eval1["gripper_rmse"]),
            "val_flow_loss": float(final_eval1["val_flow_loss"]),
            "per_joint_rmse_deg": final_eval1["per_joint_rmse_deg"],
            "episodes": list(val_eps),
            "samples": len(val_dataset),
        },
    }
    if final_eval2 is not None:
        dual_summary["eval2_new_benchmark"] = {
            "val_rmse": float(final_eval2["val_rmse"]),
            "val_h1": float(final_eval2["val_h1"]),
            "val_first5": float(final_eval2["val_first5"]),
            "val_first5_pooled_rmse": float(final_eval2["val_first5_pooled_rmse"]),
            "arm_rmse_deg": float(final_eval2["arm_rmse_deg"]),
            "gripper_rmse": float(final_eval2["gripper_rmse"]),
            "val_flow_loss": float(final_eval2["val_flow_loss"]),
            "per_joint_rmse_deg": final_eval2["per_joint_rmse_deg"],
            "episodes": list(eval2_eps),
            "samples": len(eval2_dataset),
        }

    (args.output_dir / "final_dual_eval_metrics.json").write_text(json.dumps(dual_summary, indent=2) + "\n")

    summary = {
        "status": "completed",
        "total_steps": opt_step,
        "best_step": early_stopping.best_step,
        "select_metric": args.select_metric,
        "best_select_metric": early_stopping.best_metric,
        "best_val_rmse": float(final_eval1["val_rmse"]),
        "best_val_flow_loss": float(final_eval1["val_flow_loss"]),
        "total_time_seconds": total_time,
        "backbone": args.backbone,
        "protocol": effective_protocol,
        "final_dual_eval": dual_summary,
    }
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 80)
    print(
        f"FINAL DUAL-EVALUATION BENCHMARK SUMMARY: {args.backbone.upper()} ({effective_protocol}, Best Step: {early_stopping.best_step})"
    )
    print("=" * 80)
    print(f"Eval-Set 1 (Historical Protocol 1.0, eps {val_eps[0]}..{val_eps[-1]}):")
    print(
        f"  Trajectory RMSE: {float(final_eval1['val_rmse']):.3f} | Arm RMSE: {float(final_eval1['arm_rmse_deg']):.3f}° | "
        f"Gripper RMSE: {float(final_eval1['gripper_rmse']):.3f} | H1: {float(final_eval1['val_h1']):.3f} | "
        f"First-5 (Mean): {float(final_eval1['val_first5']):.3f} | First-5 (Pooled): {float(final_eval1['val_first5_pooled_rmse']):.3f}"
    )
    if final_eval2 is not None:
        print(f"Eval-Set 2 (New Benchmark, eps {eval2_eps[0]}..{eval2_eps[-1]}):")
        print(
            f"  Trajectory RMSE: {float(final_eval2['val_rmse']):.3f} | Arm RMSE: {float(final_eval2['arm_rmse_deg']):.3f}° | "
            f"Gripper RMSE: {float(final_eval2['gripper_rmse']):.3f} | H1: {float(final_eval2['val_h1']):.3f} | "
            f"First-5 (Mean): {float(final_eval2['val_first5']):.3f} | First-5 (Pooled): {float(final_eval2['val_first5_pooled_rmse']):.3f}"
        )
    print("=" * 80 + "\n")
    run_manifest["status"] = "completed"
    atomic_write_json(args.output_dir / "run_manifest.json", run_manifest)

    backend_val = args.backbone
    if is_online:
        b_k = args.online_backbone.replace("-", "_").lower()
        if b_k == "cosmos3_nano":
            backend_val = "cosmos3_nano"
        elif "cosmos3" in b_k:
            backend_val = "cosmos3_edge"
        elif any(k in b_k for k in ("cosmos2b", "cosmos_2b", "cosmos_t2", "cosmos")):
            backend_val = "cosmos"
        elif any(k in b_k for k in ("flux2", "flux")):
            backend_val = "flux2_klein"

    deployment_cfg = {
        "type": "video_vam",
        "backend": backend_val,
        "device": "cuda",
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [6]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "camera_key": "observation.images.front",
        "action_seed": 0,
        "online_trained": is_online,
        "online_backbone": args.online_backbone if is_online else None,
        "augmented": args.augment,
        "train_stride": args.train_stride if is_online else None,
    }
    if "cosmos3" in backend_val:
        trained_bb_lora = args.output_dir / "best_backbone_lora.safetensors"
        if not trained_bb_lora.is_file():
            trained_bb_lora = args.output_dir / "last_backbone_lora.safetensors"
        resolved_lora_weights = (
            str(trained_bb_lora)
            if (backbone_trainable_params and trained_bb_lora.is_file())
            else (str(args.backbone_lora_weights) if args.backbone_lora_weights else None)
        )
        resolved_cfg_fps = (
            float(backbone_extractor.config.fps)
            if (backbone_extractor is not None and hasattr(backbone_extractor, "config"))
            else (
                float(args.backbone_fps)
                if getattr(args, "backbone_fps", None) is not None
                else (float(dataset_contract.fps) if is_online else 10.0)
            )
        )
        default_c3_ckpt = (
            "/home/anton/.cache/video-vam/cosmos3-nano"
            if backend_val == "cosmos3_nano"
            else "/home/anton/.cache/video-vam/cosmos3-edge"
        )
        deployment_cfg.update(
            {
                "cosmos3_checkpoint": str(args.backbone_checkpoint or default_c3_ckpt),
                "cosmos3_layer": args.backbone_layer,
                "cosmos3_lora_weights": resolved_lora_weights,
                "cosmos3_lora_rank": args.backbone_lora_rank,
                "cosmos3_lora_alpha": args.backbone_lora_alpha,
                "cosmos3_lora_use_dora": bool(getattr(args, "backbone_lora_dora", False)),
                "cosmos3_prompt": args.backbone_prompt,
                "cosmos3_fps": resolved_cfg_fps,
            }
        )
    elif backend_val == "cosmos":
        deployment_cfg.update(
            {
                "cosmos_state_t": 2,
                "cosmos_context_transform": "none",
                "cosmos_checkpoint": str(
                    args.backbone_checkpoint
                    or "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
                ),
                "cosmos_lora_weights": str(args.backbone_lora_weights)
                if args.backbone_lora_weights
                else None,
                "cosmos_lora_rank": args.backbone_lora_rank,
                "cosmos_lora_alpha": args.backbone_lora_alpha,
            }
        )
    elif backend_val == "flux2_klein":
        deployment_cfg.update(
            {
                "flux2_tap_location": "junction",
                "flux2_lora_weights": str(args.backbone_lora_weights) if args.backbone_lora_weights else None,
                "flux2_lora_rank": args.backbone_lora_rank,
                "flux2_lora_alpha": args.backbone_lora_alpha,
            }
        )
    (args.output_dir / "config.json").write_text(json.dumps(deployment_cfg, indent=2) + "\n")

    print(f"Training finished in {total_time:.1f}s. Summary written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
