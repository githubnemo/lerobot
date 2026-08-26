"""Strict cache artifacts for frozen LTX-2.5 action features."""

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

CACHE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
CONTEXT_KEY = "context"
STATE_KEY = "state"
TARGET_ACTION_KEY = "target_action"
ACTION_IS_PAD_KEY = "action_is_pad"
TENSOR_KEYS = frozenset({CONTEXT_KEY, STATE_KEY, TARGET_ACTION_KEY, ACTION_IS_PAD_KEY})
CONTEXT_SPECS = {
    "pool2": ((1, 640, 4096), [8, 8, 10]),
    "none": ((1, 2400, 4096), [8, 15, 20]),
}
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


class LTXFeatureCacheError(ValueError):
    """Raised when an LTX cache artifact or manifest is inconsistent."""


@dataclass(frozen=True, slots=True)
class LTXFeatureCacheItem:
    context: torch.Tensor
    state: torch.Tensor
    target_action: torch.Tensor
    action_is_pad: torch.Tensor
    provenance: dict[str, Any]
    entry: CacheManifestEntry | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise LTXFeatureCacheError(
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


def validate_tensors(tensors: Mapping[str, torch.Tensor]) -> None:
    _strict_keys(tensors, TENSOR_KEYS, "cache tensors")
    expected = {
        STATE_KEY: STATE_SHAPE,
        TARGET_ACTION_KEY: ACTION_SHAPE,
        ACTION_IS_PAD_KEY: PADDING_SHAPE,
    }
    context_shape = tuple(tensors[CONTEXT_KEY].shape)
    allowed_context_shapes = {spec[0] for spec in CONTEXT_SPECS.values()}
    if context_shape not in allowed_context_shapes:
        raise LTXFeatureCacheError(
            f"context shape must be one of {sorted(allowed_context_shapes)}, got {context_shape}"
        )
    for key, shape in expected.items():
        if tuple(tensors[key].shape) != shape:
            raise LTXFeatureCacheError(f"{key} shape must be {shape}, got {tuple(tensors[key].shape)}")
    if tensors[CONTEXT_KEY].dtype != torch.bfloat16:
        raise LTXFeatureCacheError("context dtype must be bfloat16")
    if tensors[STATE_KEY].dtype != torch.float32 or tensors[TARGET_ACTION_KEY].dtype != torch.float32:
        raise LTXFeatureCacheError("state and target_action dtypes must be float32")
    if tensors[ACTION_IS_PAD_KEY].dtype != torch.bool:
        raise LTXFeatureCacheError("action_is_pad dtype must be bool")
    for key in (CONTEXT_KEY, STATE_KEY, TARGET_ACTION_KEY):
        if not torch.isfinite(tensors[key]).all().item():
            raise LTXFeatureCacheError(f"{key} contains non-finite values")


def save_ltx_feature_cache(
    item: LTXFeatureCacheItem,
    path: Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    tensors = {
        CONTEXT_KEY: item.context.detach().cpu().contiguous(),
        STATE_KEY: item.state.detach().cpu().contiguous(),
        TARGET_ACTION_KEY: item.target_action.detach().cpu().contiguous(),
        ACTION_IS_PAD_KEY: item.action_is_pad.detach().cpu().contiguous(),
    }
    validate_tensors(tensors)
    _strict_keys(item.provenance, _SIDECAR_KEYS - {"safetensors_sha256"}, "provenance")
    if not overwrite and (path.exists() or sidecar_path.exists()):
        raise FileExistsError(f"refusing existing cache artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    payload = dict(item.provenance)
    payload["safetensors_sha256"] = sha256_file(path)
    _atomic_json(sidecar_path, payload)


def verify_ltx_feature_cache(
    path: Path,
    *,
    expected_entry: CacheManifestEntry | None = None,
) -> LTXFeatureCacheItem:
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"cache tensor and sidecar are required: {path}")
    try:
        payload = json.loads(sidecar_path.read_text())
    except json.JSONDecodeError as exc:
        raise LTXFeatureCacheError(f"invalid sidecar JSON: {sidecar_path}") from exc
    if not isinstance(payload, Mapping):
        raise LTXFeatureCacheError("sidecar must be a JSON object")
    _strict_keys(payload, _SIDECAR_KEYS, "sidecar")
    if (
        payload["schema_version"] != CACHE_SCHEMA_VERSION
        or payload["artifact"] != "ltx25_frozen_feature_cache"
    ):
        raise LTXFeatureCacheError("sidecar schema or artifact identity is invalid")
    actual_hash = sha256_file(path)
    if payload["safetensors_sha256"] != actual_hash:
        raise LTXFeatureCacheError(f"safetensors hash mismatch: {path}")
    tensors = load_file(str(path), device="cpu")
    validate_tensors(tensors)
    dataset = payload["dataset"]
    output = payload["output"]
    transform = output.get("context_transform")
    if transform not in CONTEXT_SPECS:
        raise LTXFeatureCacheError(f"unsupported LTX context transform: {transform!r}")
    expected_context_shape, _grid = CONTEXT_SPECS[transform]
    if (
        output.get("context_shape") != list(expected_context_shape)
        or output.get("context_dtype") != "bfloat16"
        or tuple(tensors[CONTEXT_KEY].shape) != expected_context_shape
    ):
        raise LTXFeatureCacheError("LTX context tensor does not match its sidecar representation")
    if expected_entry is not None and (
        dataset.get("episode_index") != expected_entry.episode_index
        or dataset.get("frame_index") != expected_entry.frame_index
        or dataset.get("window_indices") != list(expected_entry.window_indices)
        or payload["extraction"].get("noise_seed") != expected_entry.noise_seed
        or actual_hash != expected_entry.safetensors_sha256
        or sha256_file(sidecar_path) != expected_entry.sidecar_sha256
    ):
        raise LTXFeatureCacheError(f"artifact does not match manifest entry: {expected_entry.sample_id}")
    return LTXFeatureCacheItem(
        tensors[CONTEXT_KEY],
        tensors[STATE_KEY],
        tensors[TARGET_ACTION_KEY],
        tensors[ACTION_IS_PAD_KEY],
        dict(payload),
        expected_entry,
    )


def write_ltx_cache_manifest(payload: Mapping[str, Any], path: Path, *, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing existing manifest: {path}")
    _atomic_json(path, payload)
    load_ltx_cache_manifest(path)


def load_ltx_cache_manifest(path: Path) -> CacheManifest:
    path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LTXFeatureCacheError(f"could not load LTX cache manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LTXFeatureCacheError("manifest must be a JSON object")
    _strict_keys(payload, _MANIFEST_KEYS, "manifest")
    if (
        payload["schema_version"] != MANIFEST_SCHEMA_VERSION
        or payload["cache_schema_version"] != CACHE_SCHEMA_VERSION
        or payload["artifact"] != "ltx25_frozen_feature_manifest"
    ):
        raise LTXFeatureCacheError("manifest schema or artifact identity is invalid")
    provenance = payload["provenance"]
    transform = provenance.get("context_transform")
    expected = CONTEXT_SPECS.get(transform)
    if (
        provenance.get("backbone") != "LTX-2.5-22B-distilled"
        or provenance.get("hidden_layer") != 34
        or provenance.get("high_noise_sigma") != 1.0
        or expected is None
        or provenance.get("context_tokens") != expected[0][1]
        or provenance.get("context_channels") != 4096
        or provenance.get("context_grid") != expected[1]
        or provenance.get("context_dtype") != "bfloat16"
    ):
        raise LTXFeatureCacheError("manifest LTX producer contract is invalid")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise LTXFeatureCacheError("manifest entries must be non-empty")
    entries: list[CacheManifestEntry] = []
    seen: set[str] = set()
    total_bytes = 0
    previous: tuple[int, int] | None = None
    for index, value in enumerate(raw_entries):
        if not isinstance(value, Mapping):
            raise LTXFeatureCacheError(f"entry {index} must be an object")
        _strict_keys(value, _ENTRY_KEYS, f"entry {index}")
        sample_id = value["sample_id"]
        episode = value["episode_index"]
        frame = value["frame_index"]
        pair = (episode, frame)
        if sample_id != f"episode-{episode:04d}-frame-{frame:06d}" or sample_id in seen:
            raise LTXFeatureCacheError(f"entry {index} sample identity is invalid or duplicated")
        if previous is not None and pair <= previous:
            raise LTXFeatureCacheError("manifest entries must be episode-major and strictly ordered")
        previous = pair
        seen.add(sample_id)
        window = value["window_indices"]
        if window != list(range(frame - 4, frame + 1)):
            raise LTXFeatureCacheError(f"entry {sample_id} does not contain five causal frames")
        for key in ("safetensors", "sidecar"):
            relative = Path(value[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise LTXFeatureCacheError(f"entry {sample_id} contains an unsafe path")
        entry = CacheManifestEntry(
            sample_id,
            episode,
            frame,
            tuple(window),
            value["noise_seed"],
            value["safetensors"],
            value["sidecar"],
            value["safetensors_sha256"],
            value["sidecar_sha256"],
            value["bytes"],
        )
        entries.append(entry)
        total_bytes += entry.bytes
    if total_bytes != payload["total_bytes"]:
        raise LTXFeatureCacheError("manifest total_bytes does not equal entry sizes")
    return CacheManifest(path, dict(payload), tuple(entries))


class LTXFeatureCacheDataset(Dataset[LTXFeatureCacheItem]):
    """Random-access verified LTX cache dataset."""

    def __init__(self, manifest: CacheManifest) -> None:
        self.manifest = manifest
        self._order = list(range(len(manifest.entries)))
        self._verified: set[str] = set()

    def __len__(self) -> int:
        return len(self._order)

    def load_entry(self, entry: CacheManifestEntry) -> LTXFeatureCacheItem:
        path = self.manifest.root / entry.safetensors
        if entry.sample_id in self._verified:
            tensors = load_file(str(path), device="cpu")
            validate_tensors(tensors)
            payload = json.loads(path.with_suffix(".json").read_text())
            item = LTXFeatureCacheItem(
                tensors[CONTEXT_KEY],
                tensors[STATE_KEY],
                tensors[TARGET_ACTION_KEY],
                tensors[ACTION_IS_PAD_KEY],
                payload,
            )
        else:
            item = verify_ltx_feature_cache(path, expected_entry=entry)
            self._verified.add(entry.sample_id)
        return LTXFeatureCacheItem(
            item.context, item.state, item.target_action, item.action_is_pad, item.provenance, entry
        )

    def __getitem__(self, index: int) -> LTXFeatureCacheItem:
        return self.load_entry(self.manifest.entries[self._order[index]])
