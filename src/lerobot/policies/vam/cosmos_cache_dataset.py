from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context_transform import CONTEXT_TRANSFORMS, context_transform_spec
from .cosmos_feature_cache import (
    CACHE_SCHEMA_VERSION,
    CosmosFeatureCacheArtifact,
    CosmosFeatureCacheValidationError,
    load_feature_cache,
)
from .cosmos_predict2_extractor import VAE_INPUT_MODE_LEGACY_PADDED, VAE_INPUT_MODES

MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "cache_schema_version",
        "dataset",
        "subset",
        "provenance",
        "global_seed",
        "entries",
        "total_bytes",
        "runtime",
    }
)
_DATASET_KEYS = frozenset({"repo_id", "revision"})
_SUBSET_KEYS_V1 = frozenset({"episodes", "frame_start", "frame_end", "max_samples"})
_SUBSET_KEYS_V2 = _SUBSET_KEYS_V1 | {"stride"}
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


class CosmosFeatureCacheManifestError(ValueError):
    # Raised when a cache manifest is malformed or inconsistent.
    pass


@dataclass(frozen=True, slots=True)
class CacheManifestEntry:
    sample_id: str
    episode_index: int
    frame_index: int
    window_indices: tuple[int, ...]
    noise_seed: int
    safetensors: str
    sidecar: str
    safetensors_sha256: str
    sidecar_sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class CacheManifest:
    path: Path
    payload: dict[str, Any]
    entries: tuple[CacheManifestEntry, ...]

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def global_seed(self) -> int:
        return int(self.payload["global_seed"])

    @property
    def dataset_repo_id(self) -> str:
        return str(self.payload["dataset"]["repo_id"])

    @property
    def stride(self) -> int:
        return int(self.payload["subset"].get("stride", 1))

    @property
    def ordered_pairs(self) -> tuple[tuple[int, int], ...]:
        selection = self.payload["provenance"].get("selection")
        if isinstance(selection, Mapping) and "ordered_pairs" in selection:
            return tuple((int(pair[0]), int(pair[1])) for pair in selection["ordered_pairs"])
        return tuple((entry.episode_index, entry.frame_index) for entry in self.entries)

    @property
    def dataset_revision(self) -> str:
        return str(self.payload["dataset"]["revision"])

    @property
    def context_transform(self) -> str:
        """Return the transform already present in stored context tensors."""
        transform = self.payload["provenance"].get("context_transform", "none")
        return str(transform)

    @property
    def vae_input_mode(self) -> str:
        """Return the VAE mode, treating pre-mode manifests as legacy caches."""
        mode = self.payload["provenance"].get("vae_input_mode", VAE_INPUT_MODE_LEGACY_PADDED)
        return str(mode)

    @property
    def context_tokens(self) -> int | None:
        """Return the declared stored token count, if this is a new manifest."""
        value = self.payload["provenance"].get("context_tokens")
        return None if value is None else int(value)


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CosmosFeatureCacheManifestError(f"{name} must be a lowercase SHA256")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CosmosFeatureCacheManifestError(f"Cannot hash cache file {path}: {exc}") from exc
    return digest.hexdigest()


def manifest_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def _require_exact_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise CosmosFeatureCacheManifestError(
            f"{name} keys must match exactly; missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CosmosFeatureCacheManifestError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name)


def _validate_relative_path(value: Any, name: str, suffix: str) -> str:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise CosmosFeatureCacheManifestError(f"{name} must be a relative path")
    path = Path(value)
    if (suffix and path.suffix != suffix) or any(part in {"", ".", ".."} for part in path.parts):
        raise CosmosFeatureCacheManifestError(f"{name} is not a safe {suffix} path")
    return value


def _parse_entry(value: Mapping[str, Any], position: int) -> CacheManifestEntry:
    _require_exact_keys(value, _ENTRY_KEYS, f"entries[{position}]")
    episode = _nonnegative_int(value["episode_index"], f"entries[{position}].episode_index")
    frame = _nonnegative_int(value["frame_index"], f"entries[{position}].frame_index")
    window = value["window_indices"]
    if (
        not isinstance(window, list)
        or any(type(index) is not int or index < 0 for index in window)
        or tuple(window) != tuple(range(frame - 4, frame + 1))
    ):
        raise CosmosFeatureCacheManifestError(
            f"entries[{position}].window_indices must be the causal five-frame window"
        )
    noise_seed = _nonnegative_int(value["noise_seed"], f"entries[{position}].noise_seed")
    sample_id = value["sample_id"]
    expected_id = f"episode-{episode:04d}-frame-{frame:06d}"
    if sample_id != expected_id:
        raise CosmosFeatureCacheManifestError(f"entries[{position}].sample_id must be {expected_id!r}")
    safetensors = _validate_relative_path(
        value["safetensors"], f"entries[{position}].safetensors", ".safetensors"
    )
    sidecar = _validate_relative_path(value["sidecar"], f"entries[{position}].sidecar", ".json")
    if Path(sidecar).with_suffix(".safetensors").as_posix() != Path(safetensors).as_posix():
        raise CosmosFeatureCacheManifestError(f"entries[{position}] sidecar does not match safetensors path")
    size = _nonnegative_int(value["bytes"], f"entries[{position}].bytes")
    if size <= 0:
        raise CosmosFeatureCacheManifestError(f"entries[{position}].bytes must be positive")
    return CacheManifestEntry(
        sample_id=sample_id,
        episode_index=episode,
        frame_index=frame,
        window_indices=tuple(window),
        noise_seed=noise_seed,
        safetensors=safetensors,
        sidecar=sidecar,
        safetensors_sha256=_sha256(value["safetensors_sha256"], f"entries[{position}].safetensors_sha256"),
        sidecar_sha256=_sha256(value["sidecar_sha256"], f"entries[{position}].sidecar_sha256"),
        bytes=size,
    )


def _validate_payload(
    payload: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], tuple[CacheManifestEntry, ...]]:
    if not isinstance(payload, Mapping):
        raise CosmosFeatureCacheManifestError("manifest must contain a JSON object")
    _require_exact_keys(payload, _MANIFEST_KEYS, "manifest")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise CosmosFeatureCacheManifestError(
            f"unsupported manifest schema_version {payload['schema_version']!r}"
        )
    if payload["cache_schema_version"] != CACHE_SCHEMA_VERSION:
        raise CosmosFeatureCacheManifestError(
            f"manifest cache_schema_version={payload['cache_schema_version']!r} is obsolete; expected {CACHE_SCHEMA_VERSION}"
        )
    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise CosmosFeatureCacheManifestError("manifest dataset must be an object")
    _require_exact_keys(dataset, _DATASET_KEYS, "dataset")
    if (
        not isinstance(dataset["repo_id"], str)
        or not dataset["repo_id"]
        or not isinstance(dataset["revision"], str)
        or not dataset["revision"]
    ):
        raise CosmosFeatureCacheManifestError("manifest dataset repo_id/revision must be non-empty strings")
    subset = payload["subset"]
    if not isinstance(subset, Mapping):
        raise CosmosFeatureCacheManifestError("manifest subset must be an object")
    subset_keys = set(subset)
    if subset_keys == _SUBSET_KEYS_V1:
        stride = 1
    elif subset_keys == _SUBSET_KEYS_V2:
        stride = subset["stride"]
        if type(stride) is not int or stride <= 0:
            raise CosmosFeatureCacheManifestError("manifest subset stride must be a positive integer")
    else:
        raise CosmosFeatureCacheManifestError(
            f"subset keys must match exactly; missing={sorted(_SUBSET_KEYS_V2 - subset_keys)}, "
            f"extra={sorted(subset_keys - _SUBSET_KEYS_V2)}"
        )
    episodes = subset["episodes"]
    if (
        not isinstance(episodes, list)
        or any(type(item) is not int or item < 0 for item in episodes)
        or len(set(episodes)) != len(episodes)
    ):
        raise CosmosFeatureCacheManifestError("manifest subset episodes must be unique non-negative integers")
    frame_start = _optional_nonnegative_int(subset["frame_start"], "subset.frame_start")
    frame_end = _optional_nonnegative_int(subset["frame_end"], "subset.frame_end")
    if frame_start is not None and frame_end is not None and frame_end <= frame_start:
        raise CosmosFeatureCacheManifestError("manifest subset frame_end must be greater than frame_start")
    _optional_nonnegative_int(subset["max_samples"], "subset.max_samples")
    max_samples = subset["max_samples"]
    if type(payload["global_seed"]) is not int or payload["global_seed"] < 0:
        raise CosmosFeatureCacheManifestError("manifest global_seed must be a non-negative integer")
    if not isinstance(payload["provenance"], Mapping) or not payload["provenance"]:
        raise CosmosFeatureCacheManifestError("manifest provenance must be a non-empty object")
    provenance = payload["provenance"]
    transform = provenance.get("context_transform")
    if transform is not None:
        if transform not in CONTEXT_TRANSFORMS:
            raise CosmosFeatureCacheManifestError("manifest provenance.context_transform is unsupported")
        if type(provenance.get("context_tokens")) is not int or provenance["context_tokens"] <= 0:
            raise CosmosFeatureCacheManifestError(
                "manifest provenance.context_tokens must be a positive integer"
            )
        grid = provenance.get("context_grid")
        if not isinstance(grid, Mapping) or set(grid) != {
            "temporal",
            "height",
            "width",
            "flatten_order",
        }:
            raise CosmosFeatureCacheManifestError("manifest provenance.context_grid is malformed")
        if (
            any(type(grid[name]) is not int or grid[name] <= 0 for name in ("temporal", "height", "width"))
            or grid["flatten_order"] != "T,H,W"
        ):
            raise CosmosFeatureCacheManifestError(
                "manifest provenance.context_grid has invalid dimensions or order"
            )
        input_grid = provenance.get("context_input_grid")
        if isinstance(input_grid, Mapping) and type(input_grid.get("temporal")) is int:
            temporal_frames = input_grid["temporal"]
        elif transform == "gen_frames_pool2":
            temporal_frames = grid["temporal"] + 2
        else:
            temporal_frames = grid["temporal"]
        try:
            spec = context_transform_spec(transform, temporal_frames=temporal_frames)
        except ValueError as exc:
            raise CosmosFeatureCacheManifestError(
                "manifest provenance context transform has an invalid input grid"
            ) from exc
        if (grid["temporal"], grid["height"], grid["width"]) != spec.output_grid or provenance[
            "context_tokens"
        ] != spec.output_tokens:
            raise CosmosFeatureCacheManifestError(
                "manifest context transform, grid, and token count are inconsistent"
            )
    vae_input_mode = provenance.get("vae_input_mode", VAE_INPUT_MODE_LEGACY_PADDED)
    if vae_input_mode not in VAE_INPUT_MODES:
        raise CosmosFeatureCacheManifestError("manifest provenance.vae_input_mode is unsupported")
    random_init_seed = provenance.get("random_init_seed")
    if random_init_seed is not None and (type(random_init_seed) is not int or random_init_seed < 0):
        raise CosmosFeatureCacheManifestError(
            "manifest provenance.random_init_seed must be null or non-negative"
        )
    if not isinstance(payload["runtime"], Mapping):
        raise CosmosFeatureCacheManifestError("manifest runtime must be an object")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise CosmosFeatureCacheManifestError("manifest entries must be a list")
    entries = tuple(_parse_entry(value, position) for position, value in enumerate(raw_entries))
    selection = payload["provenance"].get("selection")
    if selection is not None:
        if not isinstance(selection, Mapping) or set(selection) != {"stride", "ordered_pairs"}:
            raise CosmosFeatureCacheManifestError("manifest provenance.selection keys are malformed")
        if selection["stride"] != stride:
            raise CosmosFeatureCacheManifestError("manifest selection stride does not match subset stride")
        pairs = selection["ordered_pairs"]
        if (
            not isinstance(pairs, list)
            or len(pairs) != len(entries)
            or any(
                not isinstance(pair, list)
                or len(pair) != 2
                or any(type(item) is not int or item < 0 for item in pair)
                for pair in pairs
            )
            or [(entry.episode_index, entry.frame_index) for entry in entries]
            != [tuple(pair) for pair in pairs]
        ):
            raise CosmosFeatureCacheManifestError(
                "manifest provenance.selection.ordered_pairs must exactly match entries"
            )
    if max_samples is not None and len(entries) > max_samples:
        raise CosmosFeatureCacheManifestError("manifest entries exceed subset.max_samples")
    previous = None
    seen = set()
    for entry in entries:
        if entry.episode_index not in episodes:
            raise CosmosFeatureCacheManifestError(
                f"manifest entry {entry.sample_id} is outside the selected episodes"
            )
        if frame_start is not None and entry.frame_index < frame_start:
            raise CosmosFeatureCacheManifestError(
                f"manifest entry {entry.sample_id} is before subset.frame_start"
            )
        if frame_end is not None and entry.frame_index >= frame_end:
            raise CosmosFeatureCacheManifestError(
                f"manifest entry {entry.sample_id} is at or after subset.frame_end"
            )
        key = (entry.episode_index, entry.frame_index)
        if key in seen or (previous is not None and key <= previous):
            raise CosmosFeatureCacheManifestError(
                "manifest entries must be unique and ordered by episode/frame"
            )
        seen.add(key)
        previous = key
        tensor_path = path.parent / entry.safetensors
        sidecar_path = path.parent / entry.sidecar
        if not tensor_path.is_file() or not sidecar_path.is_file():
            raise CosmosFeatureCacheManifestError(f"manifest entry files are missing for {entry.sample_id}")
        actual_size = tensor_path.stat().st_size + sidecar_path.stat().st_size
        if actual_size != entry.bytes:
            raise CosmosFeatureCacheManifestError(
                f"manifest byte count mismatch for {entry.sample_id}: {actual_size} != {entry.bytes}"
            )
    total_bytes = _nonnegative_int(payload["total_bytes"], "manifest total_bytes")
    if total_bytes != sum(entry.bytes for entry in entries):
        raise CosmosFeatureCacheManifestError("manifest total_bytes does not equal entry byte sum")
    return json.loads(json.dumps(payload)), entries


def load_cache_manifest(path: str | Path) -> CacheManifest:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CosmosFeatureCacheManifestError(f"Could not read cache manifest {path}: {exc}") from exc
    validated, entries = _validate_payload(payload, path)
    return CacheManifest(path, validated, entries)


def write_cache_manifest(payload: Mapping[str, Any], path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path).expanduser()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite cache manifest; pass explicit overwrite: {path}")
    validated, _ = _validate_payload(payload, path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(validated, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)
        raise
    return path


@dataclass(frozen=True, slots=True)
class CacheDatasetItem:
    sample_id: str
    episode_index: int
    frame_index: int
    noise_seed: int
    artifact: CosmosFeatureCacheArtifact

    @property
    def context(self):
        return self.artifact.context

    @property
    def state(self):
        return self.artifact.state

    @property
    def target_action(self):
        return self.artifact.target_action

    @property
    def action_is_pad(self):
        return self.artifact.action_is_pad


class CosmosFeatureCacheDataset:
    # Lazy indexed cache dataset with deterministic seeded sample order.

    def __init__(
        self,
        manifest: str | Path | CacheManifest,
        *,
        shuffle_seed: int | None = None,
        verify_every_access: bool = False,
    ) -> None:
        self.manifest = manifest if isinstance(manifest, CacheManifest) else load_cache_manifest(manifest)
        seed = (
            self.manifest.global_seed
            if shuffle_seed is None
            else _nonnegative_int(shuffle_seed, "shuffle_seed")
        )
        if not isinstance(verify_every_access, bool):
            raise TypeError("verify_every_access must be boolean")
        self.verify_every_access = verify_every_access
        self._verified_sample_ids: set[str] = set()
        self._order = list(range(len(self.manifest.entries)))
        random.Random(seed).shuffle(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def ordered_entry(self, index: int) -> CacheManifestEntry:
        if not isinstance(index, int) or index < 0 or index >= len(self):
            raise IndexError(index)
        return self.manifest.entries[self._order[index]]

    def __iter__(self) -> Iterator[CacheDatasetItem]:
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index: int) -> CacheDatasetItem:
        return self.load_entry(self.ordered_entry(index))

    def load_entry(self, entry: CacheManifestEntry) -> CacheDatasetItem:
        # Hashes are verified once per dataset process by default.
        tensor_path = self.manifest.root / entry.safetensors
        sidecar_path = self.manifest.root / entry.sidecar
        verify_hashes = self.verify_every_access or entry.sample_id not in self._verified_sample_ids
        if verify_hashes and (
            sha256_file(tensor_path) != entry.safetensors_sha256
            or sha256_file(sidecar_path) != entry.sidecar_sha256
        ):
            raise CosmosFeatureCacheManifestError(f"cache artifact hash mismatch for {entry.sample_id}")
        try:
            artifact = load_feature_cache(tensor_path, verify_hashes=verify_hashes)
        except (OSError, ValueError, CosmosFeatureCacheValidationError) as exc:
            raise CosmosFeatureCacheManifestError(
                f"cache artifact failed strict validation for {entry.sample_id}: {exc}"
            ) from exc
        dataset = artifact.provenance.payload["dataset"]
        extractor = artifact.provenance.payload["extractor"]
        if (
            dataset["episode_index"] != entry.episode_index
            or dataset["frame_index"] != entry.frame_index
            or dataset["window_indices"] != list(entry.window_indices)
            or extractor["noise_seed"] != entry.noise_seed
        ):
            raise CosmosFeatureCacheManifestError(
                f"cache artifact provenance does not match manifest entry {entry.sample_id}"
            )
        artifact_mode = artifact.provenance.payload["extractor"].get(
            "vae_input_mode", VAE_INPUT_MODE_LEGACY_PADDED
        )
        if artifact_mode != self.manifest.vae_input_mode:
            raise CosmosFeatureCacheManifestError(
                f"cache artifact VAE input mode does not match manifest for {entry.sample_id}"
            )
        output = artifact.provenance.payload["output"]
        artifact_transform = output.get("context_transform", "none")
        if artifact_transform != self.manifest.context_transform:
            raise CosmosFeatureCacheManifestError(
                f"cache artifact context transform does not match manifest for {entry.sample_id}"
            )
        declared_tokens = self.manifest.context_tokens
        if declared_tokens is not None and (
            artifact.context.shape[1] != declared_tokens
            or output.get("context_tokens", artifact.context.shape[1]) != declared_tokens
        ):
            raise CosmosFeatureCacheManifestError(
                f"cache artifact token count does not match manifest for {entry.sample_id}"
            )
        if verify_hashes:
            self._verified_sample_ids.add(entry.sample_id)
        return CacheDatasetItem(
            entry.sample_id, entry.episode_index, entry.frame_index, entry.noise_seed, artifact
        )

    def sample_for_step(self, step: int) -> CacheDatasetItem:
        if not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if not self:
            raise ValueError("cache manifest contains no samples")
        return self[step % len(self)]
