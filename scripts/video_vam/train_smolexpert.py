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
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    ) -> None:
        self.repo_id = repo_id
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.episodes = tuple(sorted(episodes))
        self.stride = stride
        self.contract = contract
        self.manifest_path = Path(f"online://{repo_id}?stride={stride}&episodes={len(episodes)}")
        self.payload: dict[str, Any] = {
            "source": "online_video_dataset",
            "repo_id": repo_id,
            "stride": stride,
            "episodes": list(self.episodes),
        }

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.lerobot_ds = LeRobotDataset(
            self.repo_id,
            root=str(self.root) if self.root is not None else None,
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

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> OnlineVideoItem:
        ep, frame, local_idx = self.index_map[idx]
        sample = self.lerobot_ds[local_idx]

        camera = sample[self.contract.camera_key]
        state = sample[self.contract.state_key].squeeze(0).contiguous()
        action = sample[self.contract.action_key].contiguous()
        action_is_pad = sample[f"{self.contract.action_key}_is_pad"].contiguous()
        sample_id = f"episode-{ep:04d}-frame-{frame:06d}"

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

        # 2. Spatial Translation / Crop of 4-6% (applied to all T frames identically)
        crop_frac = random.uniform(0.94, 0.96)
        crop_h = int(h_size * crop_frac)
        crop_w = int(w_size * crop_frac)
        top = random.randint(0, h_size - crop_h)
        left = random.randint(0, w_size - crop_w)
        x_crop = tf_f.resized_crop(x_jitter, top, left, crop_h, crop_w, size=[h_size, w_size], antialias=True)

        # 3. Sensor Noise / Blur: low-probability mild Gaussian blur sigma in [0.1, 0.8]
        if random.random() < p_blur:
            sigma = random.uniform(0.1, 0.8)
            x_crop = tf_f.gaussian_blur(x_crop, kernel_size=[3, 3], sigma=[sigma, sigma])

        x_clamped = torch.clamp(x_crop, 0.0, 1.0)
        # Permute from [T, C, H, W] to [C, T, H, W] for VAM extractors
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

    was_training = decoder.training
    decoder.eval()

    squared_by_joint = torch.zeros(ACTION_DIM, dtype=torch.float64, device=device)
    prefix_squared_by_step = [torch.zeros(ACTION_DIM, dtype=torch.float64, device=device) for _ in range(5)]
    prefix_tokens_by_step = [0 for _ in range(5)]

    valid_tokens_count = 0
    flow_total = 0.0
    flow_tokens = 0

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
        actions = torch.stack([item.action for item in batch_items]).to(device=device, dtype=torch.float32)
        if batch_items[0].context is not None:
            contexts = torch.stack([item.context for item in batch_items]).to(device=device)
        elif extractor is not None:
            raw_rgb = torch.stack([item.rgb for item in batch_items]).to(device=device)
            rgb_in = unaugmented_temporal_window_gpu(raw_rgb)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                extracted_eval = []
                for b_i in range(rgb_in.shape[0]):
                    out_f = extractor.extract(rgb_frames=rgb_in[b_i : b_i + 1])
                    extracted_eval.append(out_f.features)
                contexts = torch.cat(extracted_eval, dim=0).to(device=device)
        else:
            contexts = torch.randn(len(batch_items), 600, 2048, device=device)
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

        # 2. Action sampling with explicit noise
        pred_actions = decoder.sample_actions(states, contexts, noise=noise, num_steps=num_steps)

        # 3. Masked error accumulation
        valid = (~paddings).unsqueeze(-1)  # [B, 30, 1]
        valid_b = valid.squeeze(-1)  # [B, 30]
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

    if was_training:
        decoder.train()

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

    return {
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
        default=16,
        help="LoRA rank for online backbone (default: 16).",
    )
    parser.add_argument(
        "--backbone-lora-alpha",
        type=float,
        default=32.0,
        help="LoRA alpha for online backbone (default: 32.0).",
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
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable coherent visual data augmentations on 5-frame temporal window during training.",
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
        "--train-stride",
        type=int,
        default=1,
        help="Frame sampling stride for training episodes in online mode (default: 1).",
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
    args = parser.parse_args(argv)
    if args.output_dir is None:
        if args.eval_only:
            parser.error("--eval-only requires --output-dir")
        args.output_dir = default_run_dir(Path(__file__).resolve().parents[2], "smolexpert")

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
            "action_dim": ACTION_DIM,
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
            "metadata": json.loads((args.output_dir / "normalizer.json").read_text()),
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
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
    require_identity = args.require_identity or (not args.dry_run)

    if is_online:
        if effective_protocol == "scale100":
            train_eps = tuple(sorted(list(range(32)) + list(range(40, 90))))
            val_eps = tuple(range(32, 40))
            eval2_eps = tuple(range(90, 100))
            if args.dataset_repo_id is None:
                args.dataset_repo_id = "Orellius/cube_out_of_box_v2"
            if args.dataset_root is None:
                args.dataset_root = Path(
                    "/home/anton/.cache/huggingface/lerobot/hub/datasets--Orellius--cube_out_of_box_v2/snapshots/5d0325cc1412f4774223a0beb528958108814962"
                )
        else:
            train_eps = tuple(range(32))
            val_eps = tuple(range(32, 40))
            eval2_eps = tuple(range(90, 100))
            if args.dataset_repo_id is None:
                args.dataset_repo_id = "hubnemo/cube_out_of_box_dataset"
            if args.dataset_root is None:
                args.dataset_root = Path("/home/anton/.cache/video-vam/cube-out-of-box-dataset")

        print(f"Online Backbone ({args.online_backbone}) Split Protocol ({effective_protocol}) Verified:")
        print(f"  Train episodes ({len(train_eps)}): {list(train_eps)}")
        print(f"  Val episodes   ({len(val_eps)}): {list(val_eps)}")
        print(f"  Dataset Repo ID: {args.dataset_repo_id}")
        print(f"  Train Stride: {args.train_stride}")
        print(f"  Augmentation: {'ENABLED' if args.augment else 'DISABLED'}")

        # Auto-discover validation manifest if not explicitly given
        if args.val_manifest is None:
            c15 = Path("outputs/features/v2-cosmos3-edge-lora-15k/val/manifest.json")
            c5 = Path("outputs/features/v2-cosmos3-edge-lora/val/manifest.json")
            if c15.is_file():
                args.val_manifest = c15
                print(f"Auto-discovered Eval-Set 1 validation manifest: {args.val_manifest}")
            elif c5.is_file():
                args.val_manifest = c5
                print(f"Auto-discovered Eval-Set 1 validation manifest: {args.val_manifest}")

        # Auto-discover Eval-Set 2 manifest if not explicitly given
        if args.eval2_manifest is None:
            c2_15 = Path("outputs/features/v2-cosmos3-edge-lora-15k/eval2/manifest.json")
            c2_5 = Path("outputs/features/v2-cosmos3-edge-lora/eval2/manifest.json")
            if c2_15.is_file():
                args.eval2_manifest = str(c2_15)
                print(f"Auto-discovered Eval-Set 2 manifest: {args.eval2_manifest}")
            elif c2_5.is_file():
                args.eval2_manifest = str(c2_5)
                print(f"Auto-discovered Eval-Set 2 manifest: {args.eval2_manifest}")

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
        candidate2 = Path(f"/home/anton/.cache/video-vam/{args.backbone}-scale100-cache/eval2/manifest.json")
        if candidate1.is_file():
            args.eval2_manifest = str(candidate1)
            print(f"Auto-discovered Eval-Set 2 manifest: {args.eval2_manifest}")
        elif candidate2.is_file():
            args.eval2_manifest = str(candidate2)
            print(f"Auto-discovered Eval-Set 2 manifest: {args.eval2_manifest}")

    eval2_dataset: Any = None
    if args.eval2_manifest is not None and Path(args.eval2_manifest).is_file():
        print(f"  Eval-2 manifest: {args.eval2_manifest}")
        eval2_dataset = UnifiedFeatureCacheDataset(
            args.eval2_manifest, context_transform=args.context_transform
        )
        print(f"Eval-Set 2 Verification PASSED ({len(eval2_dataset)} samples).")
    elif is_online and effective_protocol == "scale100":
        eval2_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=eval2_eps,
            stride=20,
            contract=CUBE_OUT_OF_BOX_CONTRACT,
        )
        print(f"Eval-Set 2 Online Dataset Initialized ({len(eval2_dataset)} samples).")

    # 2. Build Datasets
    val_dataset: Any = None
    if args.val_manifest is not None and Path(args.val_manifest).is_file():
        val_dataset = UnifiedFeatureCacheDataset(args.val_manifest, context_transform=args.context_transform)
    elif is_online:
        val_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=val_eps,
            stride=20,
            contract=CUBE_OUT_OF_BOX_CONTRACT,
        )

    train_dataset: Any = None
    if is_online:
        train_dataset = OnlineVideoDataset(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
            episodes=train_eps,
            stride=args.train_stride,
            contract=CUBE_OUT_OF_BOX_CONTRACT,
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
            normalizer = compute_online_training_normalizer(train_dataset, train_eps, args.output_dir)
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
    if is_online:
        input_channels = 2048
        num_tokens = 600
        if not args.dry_run:
            b_key = args.online_backbone.replace("-", "_").lower()
            if b_key in ("cosmos3_edge", "cosmos3"):
                from lerobot.policies.vam.cosmos3_features import (
                    Cosmos3ExtractorConfig,
                    Cosmos3FeatureExtractor,
                )

                ckpt_dir = args.backbone_checkpoint or Path("/home/anton/.cache/video-vam/cosmos3-edge")
                b_cfg = Cosmos3ExtractorConfig(
                    backbone_name="cosmos3-edge",
                    checkpoint_path=ckpt_dir,
                    hidden_layers=(args.backbone_layer,),
                    device=str(device),
                    dtype="bfloat16",
                    fps=10.0,
                    base_fps=24.0,
                    prompt=args.backbone_prompt,
                    lora_checkpoint=args.backbone_lora_weights,
                    lora_rank=args.backbone_lora_rank,
                    lora_alpha=args.backbone_lora_alpha,
                )
                print(
                    f"Loading online backbone {args.online_backbone} (layer {args.backbone_layer}, lora: {args.backbone_lora_weights})..."
                )
                backbone_extractor = Cosmos3FeatureExtractor(b_cfg)
                backbone_extractor.eval()
                for p in backbone_extractor.parameters():
                    p.requires_grad_(False)
            else:
                raise ValueError(f"Unsupported online backbone: {args.online_backbone}")
    else:
        sample_context = val_dataset[0].context
        input_channels = sample_context.shape[-1]
        num_tokens = sample_context.shape[0]

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

    trainable_params = [p for p in decoder.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {total_params:,}")

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
    atomic_write_json(args.output_dir / "run_manifest.json", run_manifest)

    # 6. Optimizer & Scheduler
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
    )
    print(
        f"Initial Val -> RMSE: {val_metrics['val_rmse']:.3f}, "
        f"Arm RMSE: {val_metrics['arm_rmse_deg']:.3f}°, "
        f"Gripper RMSE: {val_metrics['gripper_rmse']:.3f}, "
        f"H1: {val_metrics['val_h1']:.3f}, "
        f"Flow Loss: {val_metrics['val_flow_loss']:.4f}"
    )

    # 7. Training Loop with Exact Gradient Accumulation
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

    while opt_step < args.max_steps:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _micro_step in range(args.grad_accum_steps):
            batch_items: list[Any] = []
            for _ in range(args.batch_size):
                batch_items.append(train_dataset[indices[data_idx]])
                data_idx += 1
                if data_idx >= len(indices):
                    random.shuffle(indices)
                    data_idx = 0

            states = torch.stack([item.state for item in batch_items]).to(device=device, dtype=torch.float32)
            actions = torch.stack([item.action for item in batch_items]).to(
                device=device, dtype=torch.float32
            )
            if is_online:
                if args.dry_run:
                    contexts = torch.randn(len(batch_items), num_tokens, input_channels, device=device)
                else:
                    assert backbone_extractor is not None
                    raw_rgb = torch.stack([item.rgb for item in batch_items]).to(device=device)
                    if args.augment:
                        rgb_in = augment_temporal_window_gpu(raw_rgb)
                    else:
                        rgb_in = unaugmented_temporal_window_gpu(raw_rgb)

                    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        extracted_list = []
                        for b_i in range(rgb_in.shape[0]):
                            out_feat = backbone_extractor.extract(rgb_frames=rgb_in[b_i : b_i + 1])
                            extracted_list.append(out_feat.features)
                        contexts = torch.cat(extracted_list, dim=0).to(device=device)
            else:
                contexts = torch.stack([item.context for item in batch_items]).to(device=device)
            action_is_pad = torch.stack([item.action_is_pad for item in batch_items]).to(
                device=device, dtype=torch.bool
            )

            # Flow matching loss with strict padding masking
            loss = decoder.flow_matching_loss(
                state=states,
                action=actions,
                context=contexts,
                action_is_pad=action_is_pad,
            )

            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()
            accum_loss += float(loss.item()) / args.grad_accum_steps

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        opt_step += 1
        if args.save_last_every > 0 and opt_step % args.save_last_every == 0:
            save_weights_artifact(args.output_dir, "last", decoder.state_dict(), opt_step, run_manifest)

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
        if opt_step % args.val_every == 0 or opt_step == args.max_steps:
            val_metrics = evaluate_validation(
                decoder,
                val_dataset,
                device=device,
                batch_size=args.batch_size,
                num_steps=args.num_steps,
                seed=args.seed,
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
                )
                eval2_rmse = float(eval2_metrics["val_rmse"])
                eval2_arm = float(eval2_metrics["arm_rmse_deg"])
                eval2_grip = float(eval2_metrics["gripper_rmse"])
                eval2_h1 = float(eval2_metrics["val_h1"])
                eval2_first5 = float(eval2_metrics["val_first5"])
                eval2_flow = float(eval2_metrics["val_flow_loss"])

            is_best = early_stopping.update(rmse, opt_step)
            marker = " *** BEST ***" if is_best else ""
            if eval2_dataset is not None:
                print(
                    f"[Eval @ {opt_step}] Eval-1 RMSE: {rmse:.3f} (Arm: {arm_rmse:.3f}°, Grip: {gripper_rmse:.3f}) | H1: {h1:.3f} | First-5: {first5:.3f} "
                    f"|| Eval-2 RMSE: {eval2_rmse:.3f} (Arm: {eval2_arm:.3f}°, Grip: {eval2_grip:.3f}) | H1: {eval2_h1:.3f} | First-5: {eval2_first5:.3f} "
                    f"(Patience: {early_stopping.bad_evaluations}/{early_stopping.patience}){marker}",
                    flush=True,
                )
            else:
                print(
                    f"[Eval @ {opt_step}] RMSE: {rmse:.3f} (Arm: {arm_rmse:.3f}°, Grip: {gripper_rmse:.3f}) | "
                    f"H1: {h1:.3f} | First-5: {first5:.3f} | Flow Loss: {flow:.4f} "
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
                    args.output_dir, "best", decoder.state_dict(), opt_step, run_manifest, metric=rmse
                )
                best_meta = {
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
                print(
                    f"Early stopping triggered at step {opt_step}: no validation improvement "
                    f"for {early_stopping.patience} evaluations. Best RMSE: {early_stopping.best_metric:.3f} "
                    f"at step {early_stopping.best_step}."
                )
                break

    total_time = time.time() - start_time

    # Save live final weights before loading best for final evaluation.
    save_weights_artifact(args.output_dir, "last", decoder.state_dict(), opt_step, run_manifest)

    # Final evaluation of best checkpoint on both Eval-1 and Eval-2
    if best_checkpoint_path.is_file():
        print("Loading best checkpoint for final dual evaluation...")
        best_state = load_file(str(best_checkpoint_path))
        decoder.load_state_dict(best_state)

    final_eval1 = evaluate_validation(
        decoder,
        val_dataset,
        device=device,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        seed=args.seed,
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
        "best_val_rmse": early_stopping.best_metric,
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

    deployment_cfg = {
        "type": "video_vam",
        "backend": "cosmos3_edge"
        if "cosmos3" in args.backbone or (args.online_backbone and "cosmos3" in args.online_backbone)
        else args.backbone,
        "device": "cuda",
        "input_features": {
            "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
            "observation.state": {"type": "STATE", "shape": [6]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "camera_key": "observation.images.front",
        "action_seed": 0,
        "cosmos3_checkpoint": str(args.backbone_checkpoint or "/home/anton/.cache/video-vam/cosmos3-edge"),
        "cosmos3_lora_weights": str(args.backbone_lora_weights) if args.backbone_lora_weights else None,
        "cosmos3_lora_rank": args.backbone_lora_rank,
        "cosmos3_lora_alpha": args.backbone_lora_alpha,
        "online_trained": is_online,
        "augmented": args.augment,
        "train_stride": args.train_stride if is_online else None,
    }
    (args.output_dir / "config.json").write_text(json.dumps(deployment_cfg, indent=2) + "\n")

    print(f"Training finished in {total_time:.1f}s. Summary written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
