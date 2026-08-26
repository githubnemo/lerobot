"""Strict cache schema for the six-depth LTX layer-selection probe."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from .cosmos_cache_dataset import CacheManifest, CacheManifestEntry
from .ltx_layer_mix import (
    LTX_CONTEXT_TRANSFORMS,
    LTX_HIDDEN_WIDTH,
    LTX_LAYER_PROBE_DEPTHS,
    ltx_context_grid,
    ltx_layer_probe_tokens,
)

CACHE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
STATE_KEY = "state"
TARGET_ACTION_KEY = "target_action"
ACTION_IS_PAD_KEY = "action_is_pad"
STATE_SHAPE = (1, 1, 6)
ACTION_SHAPE = (1, 30, 6)
PADDING_SHAPE = (1, 30)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "cache_schema_version",
        "artifact",
        "dataset",
        "subset",
        "provenance",
        "global_seed",
        "entries",
        "total_bytes",
        "runtime",
    }
)
_ENTRY_KEYS = frozenset(
    {
        "sample_id",
        "episode_index",
        "frame_index",
        "window_indices",
        "noise_seed",
        "safetensors",
        "sidecar",
        "safetensors_sha256",
        "sidecar_sha256",
        "bytes",
    }
)
_SIDECAR_KEYS = frozenset(
    {
        "schema_version",
        "artifact",
        "dataset",
        "split",
        "prompt",
        "backbone",
        "extraction",
        "temporal_contract",
        "output",
        "tensors",
        "safetensors_sha256",
    }
)


class LTXLayerMixCacheError(ValueError):
    """Raised when a multi-depth artifact violates the probe contract."""


def layer_mix_artifact_name(context_transform: str, *, manifest: bool = False) -> str:
    """Return a transform-specific artifact identity."""
    if context_transform not in LTX_CONTEXT_TRANSFORMS:
        raise LTXLayerMixCacheError(f"unsupported LTX context transform: {context_transform!r}")
    suffix = "feature_manifest" if manifest else "feature_cache"
    return f"ltx25_multidepth_{context_transform}_{suffix}"


@dataclass(frozen=True, slots=True)
class LTXLayerMixCacheItem:
    contexts: dict[int, torch.Tensor]
    state: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    provenance: dict[str, Any]
    entry: CacheManifestEntry | None = None


def context_key(layer: int) -> str:
    if layer not in LTX_LAYER_PROBE_DEPTHS:
        raise ValueError(f"unsupported probe layer: {layer}")
    return f"context_layer_{layer:02d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LTXLayerMixCacheError(
            f"{label} keys mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def validate_layer_mix_provenance(payload: Mapping[str, Any], *, manifest: bool = False) -> None:
    """Validate the immutable producer facts needed for scientific comparison."""

    producer = payload["provenance"] if manifest else payload
    context_transform = producer.get("context_transform")
    try:
        context_grid = ltx_context_grid(context_transform)
        context_tokens = ltx_layer_probe_tokens(context_transform)
    except ValueError as exc:
        raise LTXLayerMixCacheError("multi-depth LTX producer contract is invalid") from exc
    if (
        producer.get("backbone") != "LTX-2.5-22B-distilled"
        or producer.get("tapped_layers") != list(LTX_LAYER_PROBE_DEPTHS)
        or producer.get("deepest_layer") != LTX_LAYER_PROBE_DEPTHS[-1]
        or producer.get("num_blocks") != 48
        or producer.get("high_noise_sigma") != 1.0
        or producer.get("context_tokens_per_layer") != context_tokens
        or producer.get("context_channels") != LTX_HIDDEN_WIDTH
        or producer.get("context_grid") != list(context_grid)
        or producer.get("context_dtype") != "bfloat16"
        or producer.get("one_forward_pass") is not True
    ):
        raise LTXLayerMixCacheError("multi-depth LTX producer contract is invalid")


def validate_tensors(tensors: Mapping[str, torch.Tensor], context_transform: str) -> None:
    context_shape = (1, ltx_layer_probe_tokens(context_transform), LTX_HIDDEN_WIDTH)
    expected_keys = {
        *(context_key(layer) for layer in LTX_LAYER_PROBE_DEPTHS),
        STATE_KEY,
        TARGET_ACTION_KEY,
        ACTION_IS_PAD_KEY,
    }
    if set(tensors) != expected_keys:
        raise LTXLayerMixCacheError(
            f"cache tensor keys mismatch: expected={sorted(expected_keys)}, got={sorted(tensors)}"
        )
    for layer in LTX_LAYER_PROBE_DEPTHS:
        tensor = tensors[context_key(layer)]
        if tuple(tensor.shape) != context_shape or tensor.dtype != torch.bfloat16:
            raise LTXLayerMixCacheError(
                f"layer {layer} context must be BF16 {context_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all().item():
            raise LTXLayerMixCacheError(f"layer {layer} context contains non-finite values")
    expected_shapes = {
        STATE_KEY: STATE_SHAPE,
        TARGET_ACTION_KEY: ACTION_SHAPE,
        ACTION_IS_PAD_KEY: PADDING_SHAPE,
    }
    for key, shape in expected_shapes.items():
        if tuple(tensors[key].shape) != shape:
            raise LTXLayerMixCacheError(f"{key} shape must be {shape}, got {tuple(tensors[key].shape)}")
    if tensors[STATE_KEY].dtype != torch.float32 or tensors[TARGET_ACTION_KEY].dtype != torch.float32:
        raise LTXLayerMixCacheError("state and target_action must be float32")
    if tensors[ACTION_IS_PAD_KEY].dtype != torch.bool:
        raise LTXLayerMixCacheError("action_is_pad must be bool")
    if (
        not torch.isfinite(tensors[STATE_KEY]).all().item()
        or not torch.isfinite(tensors[TARGET_ACTION_KEY]).all().item()
    ):
        raise LTXLayerMixCacheError("state and target_action must be finite")


def save_layer_mix_feature_cache(
    item: LTXLayerMixCacheItem,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    context_transform = item.provenance["output"]["context_transform"]
    tensors = {
        **{
            context_key(layer): item.contexts[layer].detach().cpu().contiguous()
            for layer in LTX_LAYER_PROBE_DEPTHS
        },
        STATE_KEY: item.state.detach().cpu().contiguous(),
        TARGET_ACTION_KEY: item.target_action.detach().cpu().contiguous(),
        ACTION_IS_PAD_KEY: item.action_is_pad.detach().cpu().contiguous(),
    }
    validate_tensors(tensors, context_transform)
    _strict_keys(item.provenance, _SIDECAR_KEYS - {"safetensors_sha256"}, "provenance")
    validate_layer_mix_provenance(item.provenance["output"])
    if item.provenance["artifact"] != layer_mix_artifact_name(context_transform):
        raise LTXLayerMixCacheError("sidecar artifact identity does not match context transform")
    if not overwrite and (path.exists() or sidecar_path.exists()):
        raise FileExistsError(f"refusing existing cache artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    payload = dict(item.provenance)
    payload["safetensors_sha256"] = sha256_file(path)
    _atomic_json(sidecar_path, payload)


def verify_layer_mix_feature_cache(
    path: Path,
    *,
    expected_entry: CacheManifestEntry | None = None,
) -> LTXLayerMixCacheItem:
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"cache tensor and sidecar are required: {path}")
    try:
        payload = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise LTXLayerMixCacheError(f"invalid sidecar JSON: {sidecar_path}") from exc
    if not isinstance(payload, Mapping):
        raise LTXLayerMixCacheError("sidecar must be a JSON object")
    _strict_keys(payload, _SIDECAR_KEYS, "sidecar")
    context_transform = payload["output"].get("context_transform")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION or payload["artifact"] != layer_mix_artifact_name(
        context_transform
    ):
        raise LTXLayerMixCacheError("sidecar schema or artifact identity is invalid")
    validate_layer_mix_provenance(payload["output"])
    actual_hash = sha256_file(path)
    if payload["safetensors_sha256"] != actual_hash:
        raise LTXLayerMixCacheError(f"safetensors hash mismatch: {path}")
    tensors = load_file(str(path), device="cpu")
    validate_tensors(tensors, context_transform)
    dataset = payload["dataset"]
    if expected_entry is not None and (
        dataset.get("episode_index") != expected_entry.episode_index
        or dataset.get("frame_index") != expected_entry.frame_index
        or dataset.get("window_indices") != list(expected_entry.window_indices)
        or payload["extraction"].get("noise_seed") != expected_entry.noise_seed
        or actual_hash != expected_entry.safetensors_sha256
        or sha256_file(sidecar_path) != expected_entry.sidecar_sha256
    ):
        raise LTXLayerMixCacheError(f"artifact does not match manifest entry: {expected_entry.sample_id}")
    return LTXLayerMixCacheItem(
        {layer: tensors[context_key(layer)] for layer in LTX_LAYER_PROBE_DEPTHS},
        tensors[STATE_KEY],
        tensors[TARGET_ACTION_KEY],
        tensors[ACTION_IS_PAD_KEY],
        dict(payload),
        expected_entry,
    )


def write_layer_mix_manifest(payload: Mapping[str, Any], path: Path, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing existing manifest: {path}")
    _atomic_json(path, payload)
    load_layer_mix_manifest(path)


def load_layer_mix_manifest(path: Path) -> CacheManifest:
    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LTXLayerMixCacheError(f"could not load layer-mix manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LTXLayerMixCacheError("manifest must be a JSON object")
    _strict_keys(payload, _MANIFEST_KEYS, "manifest")
    context_transform = payload["provenance"].get("context_transform")
    if (
        payload["schema_version"] != MANIFEST_SCHEMA_VERSION
        or payload["cache_schema_version"] != CACHE_SCHEMA_VERSION
        or payload["artifact"] != layer_mix_artifact_name(context_transform, manifest=True)
    ):
        raise LTXLayerMixCacheError("manifest schema or artifact identity is invalid")
    validate_layer_mix_provenance(payload, manifest=True)
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LTXLayerMixCacheError("manifest entries must be non-empty")
    entries: list[CacheManifestEntry] = []
    seen: set[str] = set()
    previous: tuple[int, int] | None = None
    total_bytes = 0
    for index, value in enumerate(raw_entries):
        if not isinstance(value, Mapping):
            raise LTXLayerMixCacheError(f"entry {index} must be an object")
        _strict_keys(value, _ENTRY_KEYS, f"entry {index}")
        episode = value["episode_index"]
        frame = value["frame_index"]
        sample_id = value["sample_id"]
        pair = (episode, frame)
        if sample_id != f"episode-{episode:04d}-frame-{frame:06d}" or sample_id in seen:
            raise LTXLayerMixCacheError(f"entry {index} sample identity is invalid or duplicated")
        if previous is not None and pair <= previous:
            raise LTXLayerMixCacheError("manifest entries must be episode-major and strictly ordered")
        if value["window_indices"] != list(range(frame - 4, frame + 1)):
            raise LTXLayerMixCacheError(f"entry {sample_id} does not contain five causal frames")
        for key in ("safetensors", "sidecar"):
            relative = Path(value[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise LTXLayerMixCacheError(f"entry {sample_id} contains an unsafe path")
        entry = CacheManifestEntry(
            sample_id,
            episode,
            frame,
            tuple(value["window_indices"]),
            value["noise_seed"],
            value["safetensors"],
            value["sidecar"],
            value["safetensors_sha256"],
            value["sidecar_sha256"],
            value["bytes"],
        )
        entries.append(entry)
        seen.add(sample_id)
        previous = pair
        total_bytes += entry.bytes
    if total_bytes != payload["total_bytes"]:
        raise LTXLayerMixCacheError("manifest total_bytes does not equal entry sizes")
    return CacheManifest(path, dict(payload), tuple(entries))


class LTXLayerMixCacheDataset(Dataset[LTXLayerMixCacheItem]):
    """Random-access verified dataset for aligned six-layer LTX contexts."""

    def __init__(self, manifest: CacheManifest) -> None:
        self.manifest = manifest
        self._verified: set[str] = set()

    def __len__(self) -> int:
        return len(self.manifest.entries)

    def load_entry(self, entry: CacheManifestEntry) -> LTXLayerMixCacheItem:
        path = self.manifest.root / entry.safetensors
        if entry.sample_id not in self._verified:
            item = verify_layer_mix_feature_cache(path, expected_entry=entry)
            self._verified.add(entry.sample_id)
            return item
        tensors = load_file(str(path), device="cpu")
        context_transform = self.manifest.payload["provenance"]["context_transform"]
        validate_tensors(tensors, context_transform)
        return LTXLayerMixCacheItem(
            {layer: tensors[context_key(layer)] for layer in LTX_LAYER_PROBE_DEPTHS},
            tensors[STATE_KEY],
            tensors[TARGET_ACTION_KEY],
            tensors[ACTION_IS_PAD_KEY],
            json.loads(path.with_suffix(".json").read_text()),
            entry,
        )

    def __getitem__(self, index: int) -> LTXLayerMixCacheItem:
        return self.load_entry(self.manifest.entries[index])
