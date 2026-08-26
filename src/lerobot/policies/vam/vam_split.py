"""Strict episode-level splits and persisted validation probes for VAM."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cosmos_cache_dataset import CacheManifest, CacheManifestEntry, load_cache_manifest

SPLIT_SCHEMA_VERSION = 1
PROBE_SCHEMA_VERSION = 1
DEFAULT_TRAIN_EPISODES = tuple(range(32))
DEFAULT_VAL_EPISODES = tuple(range(32, 40))
_PROBE_ALGORITHM = "python_random_box_muller_v1"
_SPLIT_KEYS = frozenset(
    {"schema_version", "split_name", "dataset", "train_episodes", "val_episodes", "validation_probe"}
)
_DATASET_KEYS = frozenset({"repo_id", "revision"})
_PROBE_KEYS = frozenset({"schema_version", "seed", "algorithm", "entries"})
_PROBE_ENTRY_KEYS = frozenset(
    {"sample_id", "episode_index", "frame_index", "tau_a", "epsilon", "sample_seed"}
)


class VAMSplitError(ValueError):
    """Raised when a persisted split or validation probe is invalid."""


@dataclass(frozen=True, slots=True)
class ValidationProbe:
    """One deterministic validation input record."""

    sample_id: str
    episode_index: int
    frame_index: int
    tau_a: float
    epsilon: tuple[tuple[float, ...], ...]
    sample_seed: int

    def epsilon_tensor(self):
        """Return the persisted noise draw as a CPU float32 tensor."""
        import torch

        return torch.tensor(self.epsilon, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class VAMSplit:
    """Validated episode split and its fixed validation probe."""

    path: Path
    payload: dict[str, Any]
    validation_probe: tuple[ValidationProbe, ...]

    @property
    def split_name(self) -> str:
        return str(self.payload["split_name"])

    @property
    def dataset_repo_id(self) -> str:
        return str(self.payload["dataset"]["repo_id"])

    @property
    def dataset_revision(self) -> str:
        return str(self.payload["dataset"]["revision"])

    @property
    def train_episodes(self) -> tuple[int, ...]:
        return tuple(self.payload["train_episodes"])

    @property
    def val_episodes(self) -> tuple[int, ...]:
        return tuple(self.payload["val_episodes"])

    @property
    def probe_seed(self) -> int:
        return int(self.payload["validation_probe"]["seed"])

    def probe_for(self, sample_id: str) -> ValidationProbe:
        """Return the fixed probe record for one validation sample."""
        for probe in self.validation_probe:
            if probe.sample_id == sample_id:
                return probe
        raise VAMSplitError(f"validation probe has no record for {sample_id!r}")


def _strict_keys(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise VAMSplitError(
            f"{name} keys must match exactly; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _episode_list(value: Any, name: str) -> list[int]:
    if (
        not isinstance(value, list)
        or any(type(episode) is not int or episode < 0 for episode in value)
        or value != sorted(set(value))
    ):
        raise VAMSplitError(f"{name} must be a sorted list of unique non-negative integers")
    return list(value)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise VAMSplitError(f"{name} must be finite")
    return float(value)


def _probe_entry(value: Mapping[str, Any], position: int) -> ValidationProbe:
    _strict_keys(value, _PROBE_ENTRY_KEYS, f"validation_probe.entries[{position}]")
    sample_id = value["sample_id"]
    if not isinstance(sample_id, str) or not sample_id:
        raise VAMSplitError(f"validation_probe.entries[{position}].sample_id must be non-empty")
    episode = value["episode_index"]
    frame = value["frame_index"]
    sample_seed = value["sample_seed"]
    if any(type(item) is not int or item < 0 for item in (episode, frame, sample_seed)):
        raise VAMSplitError(f"validation_probe.entries[{position}] integer fields are invalid")
    tau = _finite_float(value["tau_a"], f"validation_probe.entries[{position}].tau_a")
    if not 0.001 <= tau <= 1.0:
        raise VAMSplitError(f"validation_probe.entries[{position}].tau_a must be in [0.001, 1.0]")
    epsilon = value["epsilon"]
    if (
        not isinstance(epsilon, list)
        or len(epsilon) != 30
        or any(not isinstance(row, list) or len(row) != 6 for row in epsilon)
    ):
        raise VAMSplitError(f"validation_probe.entries[{position}].epsilon must have shape [30, 6]")
    parsed_epsilon = tuple(
        tuple(_finite_float(number, f"validation_probe.entries[{position}].epsilon") for number in row)
        for row in epsilon
    )
    return ValidationProbe(sample_id, episode, frame, tau, parsed_epsilon, sample_seed)


def _manifest_value(manifest: CacheManifest | str | Path) -> CacheManifest:
    return manifest if isinstance(manifest, CacheManifest) else load_cache_manifest(manifest)


def _manifest_entries_for_split(
    manifest: CacheManifest, split: VAMSplit, *, allow_partial: bool = False
) -> tuple[tuple[CacheManifestEntry, ...], tuple[CacheManifestEntry, ...]]:
    dataset = manifest.payload["dataset"]
    if dataset["repo_id"] != split.dataset_repo_id or dataset["revision"] != split.dataset_revision:
        raise VAMSplitError(
            "split dataset identity does not match manifest: "
            f"split={split.dataset_repo_id}@{split.dataset_revision}, "
            f"manifest={dataset['repo_id']}@{dataset['revision']}"
        )
    covered = set(split.train_episodes) | set(split.val_episodes)
    manifest_episodes = {entry.episode_index for entry in manifest.entries}
    uncovered = sorted(manifest_episodes - covered)
    if uncovered:
        raise VAMSplitError(f"manifest contains episodes not covered by split: {uncovered}")
    missing = sorted(covered - manifest_episodes)
    if missing and not allow_partial:
        raise VAMSplitError(f"split references episodes absent from manifest: {missing}")
    if allow_partial and not manifest_episodes <= covered:
        raise VAMSplitError(f"manifest contains episodes not covered by split: {uncovered}")
    train = tuple(entry for entry in manifest.entries if entry.episode_index in split.train_episodes)
    val = tuple(entry for entry in manifest.entries if entry.episode_index in split.val_episodes)
    expected_probe_ids = [entry.sample_id for entry in val]
    actual_probe_ids = [probe.sample_id for probe in split.validation_probe]
    if actual_probe_ids != expected_probe_ids:
        raise VAMSplitError(
            "validation probe entries must exactly match val manifest entries in manifest order"
        )
    for entry, probe in zip(val, split.validation_probe, strict=True):
        if (probe.episode_index, probe.frame_index) != (entry.episode_index, entry.frame_index):
            raise VAMSplitError(f"validation probe identity does not match {entry.sample_id}")
    return train, val


def _validate_payload(
    payload: Mapping[str, Any], manifest: CacheManifest, *, allow_partial: bool = False
) -> tuple[dict[str, Any], tuple[ValidationProbe, ...]]:
    if not isinstance(payload, Mapping):
        raise VAMSplitError("split must contain a JSON object")
    _strict_keys(payload, _SPLIT_KEYS, "split")
    if payload["schema_version"] != SPLIT_SCHEMA_VERSION:
        raise VAMSplitError(f"unsupported split schema_version {payload['schema_version']!r}")
    if not isinstance(payload["split_name"], str) or not payload["split_name"]:
        raise VAMSplitError("split_name must be a non-empty string")
    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise VAMSplitError("split dataset must be an object")
    _strict_keys(dataset, _DATASET_KEYS, "split.dataset")
    if any(not isinstance(dataset[key], str) or not dataset[key] for key in _DATASET_KEYS):
        raise VAMSplitError("split dataset repo_id/revision must be non-empty strings")
    train = _episode_list(payload["train_episodes"], "train_episodes")
    val = _episode_list(payload["val_episodes"], "val_episodes")
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise VAMSplitError(f"train_episodes and val_episodes overlap: {overlap}")
    probe_payload = payload["validation_probe"]
    if not isinstance(probe_payload, Mapping):
        raise VAMSplitError("validation_probe must be an object")
    _strict_keys(probe_payload, _PROBE_KEYS, "validation_probe")
    if probe_payload["schema_version"] != PROBE_SCHEMA_VERSION:
        raise VAMSplitError("unsupported validation probe schema_version")
    if type(probe_payload["seed"]) is not int or probe_payload["seed"] < 0:
        raise VAMSplitError("validation_probe.seed must be a non-negative integer")
    if probe_payload["algorithm"] != _PROBE_ALGORITHM:
        raise VAMSplitError(f"validation_probe.algorithm must be {_PROBE_ALGORITHM!r}")
    raw_entries = probe_payload["entries"]
    if not isinstance(raw_entries, list):
        raise VAMSplitError("validation_probe.entries must be a list")
    probes = tuple(_probe_entry(entry, index) for index, entry in enumerate(raw_entries))
    if len({probe.sample_id for probe in probes}) != len(probes):
        raise VAMSplitError("validation_probe.entries sample_id values must be unique")
    normalized = json.loads(json.dumps(payload))
    split = VAMSplit(Path("<memory>"), normalized, probes)
    _manifest_entries_for_split(manifest, split, allow_partial=allow_partial)
    return normalized, probes


def load_vam_split(
    path: str | Path,
    manifest: CacheManifest | str | Path,
    *,
    allow_partial: bool = False,
) -> VAMSplit:
    """Load and validate a strict split against the cache manifest it will serve."""
    path = Path(path).expanduser().resolve()
    manifest_value = _manifest_value(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"VAM split not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise VAMSplitError(f"Could not read VAM split {path}: {exc}") from exc
    normalized, probes = _validate_payload(payload, manifest_value, allow_partial=allow_partial)
    return VAMSplit(path, normalized, probes)


def _normal_draws(rng: random.Random, count: int) -> list[float]:
    values: list[float] = []
    for _ in range(count // 2 + count % 2):
        u1 = max(rng.random(), 1e-15)
        u2 = rng.random()
        radius = math.sqrt(-2.0 * math.log(u1))
        values.extend((radius * math.cos(2.0 * math.pi * u2), radius * math.sin(2.0 * math.pi * u2)))
    return values[:count]


def _make_probe(entries: Sequence[CacheManifestEntry], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    records = []
    for entry in entries:
        tau = 0.001 + 0.999 * rng.random()
        draws = _normal_draws(rng, 30 * 6)
        epsilon = [draws[offset : offset + 6] for offset in range(0, 30 * 6, 6)]
        records.append(
            {
                "sample_id": entry.sample_id,
                "episode_index": entry.episode_index,
                "frame_index": entry.frame_index,
                "tau_a": tau,
                "epsilon": epsilon,
                "sample_seed": rng.randrange(2**63),
            }
        )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "seed": seed,
        "algorithm": _PROBE_ALGORITHM,
        "entries": records,
    }


def create_vam_split(
    manifest: CacheManifest | str | Path,
    output_path: str | Path,
    *,
    train_episodes: Sequence[int] = DEFAULT_TRAIN_EPISODES,
    val_episodes: Sequence[int] = DEFAULT_VAL_EPISODES,
    split_name: str = "cube_out_of_box_episode_32_val",
    probe_seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Create, persist, and validate an episode split plus fixed val probe."""
    manifest_value = _manifest_value(manifest)
    train = sorted(set(train_episodes))
    val = sorted(set(val_episodes))
    if any(type(episode) is not int or episode < 0 for episode in train + val):
        raise VAMSplitError("train_episodes and val_episodes must contain non-negative integers")
    if set(train) & set(val):
        raise VAMSplitError("train_episodes and val_episodes must be disjoint")
    manifest_episodes = {entry.episode_index for entry in manifest_value.entries}
    uncovered = sorted(manifest_episodes - (set(train) | set(val)))
    if uncovered:
        raise VAMSplitError(f"manifest contains episodes not covered by requested split: {uncovered}")
    if type(probe_seed) is not int or probe_seed < 0:
        raise VAMSplitError("probe_seed must be a non-negative integer")
    if not split_name:
        raise VAMSplitError("split_name must be non-empty")
    val_entries = tuple(entry for entry in manifest_value.entries if entry.episode_index in val)
    payload = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_name": split_name,
        "dataset": dict(manifest_value.payload["dataset"]),
        "train_episodes": train,
        "val_episodes": val,
        "validation_probe": _make_probe(val_entries, probe_seed),
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite VAM split; pass overwrite=True: {output}")
    normalized, probes = _validate_payload(payload, manifest_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(normalized, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    # Make sure the bytes written remain a valid strict artifact.
    loaded = load_vam_split(output, manifest_value)
    if loaded.validation_probe != probes:
        raise VAMSplitError("persisted validation probe failed its round-trip check")
    return output


def get_train_entries(
    manifest: CacheManifest | str | Path, split: VAMSplit | str | Path
) -> tuple[CacheManifestEntry, ...]:
    """Return manifest entries belonging to the train episodes."""
    manifest_value = _manifest_value(manifest)
    split_value = split if isinstance(split, VAMSplit) else load_vam_split(split, manifest_value)
    return _manifest_entries_for_split(manifest_value, split_value)[0]


def get_val_entries(
    manifest: CacheManifest | str | Path, split: VAMSplit | str | Path
) -> tuple[CacheManifestEntry, ...]:
    """Return manifest entries belonging to the validation episodes."""
    manifest_value = _manifest_value(manifest)
    split_value = split if isinstance(split, VAMSplit) else load_vam_split(split, manifest_value)
    return _manifest_entries_for_split(manifest_value, split_value)[1]
