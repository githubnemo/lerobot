"""Protocol 1.0 Split Enforcement Guard for Video Action Models.

Enforces strict episode-level train/validation isolation to prevent
frame-level data leakage and ensure fair, reproducible benchmarking.

Under Protocol 1.0:
- Training episodes: 0 to 31 (inclusive)
- Validation episodes: 32 to 39 (inclusive)
- Frame-level random splits (such as torch.utils.data.random_split) across
  continuous trajectories are strictly forbidden.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class ProtocolViolationError(RuntimeError):
    """Raised when Protocol 1.0 split enforcement or data isolation is violated."""


# Protocol 1.0 canonical constants
PROTOCOL_1_0_TRAIN_EPISODES: tuple[int, ...] = tuple(range(32))  # Episodes 0..31
PROTOCOL_1_0_VAL_EPISODES: tuple[int, ...] = tuple(range(32, 40))  # Episodes 32..39
CANONICAL_DATASET_REPO: str = "hubnemo/cube_out_of_box_dataset"
CANONICAL_DATASET_REVISION: str = "243370c3c08bcbd860133c4a0d658ea7c1d2e77e"


def enforce_protocol_1_0_split(
    train_episodes: Sequence[int],
    val_episodes: Sequence[int],
    *,
    allow_subset: bool = True,
    protocol: str = "protocol1",
) -> None:
    """Validate that train and validation episode sets strictly adhere to Protocol 1.0.

    Args:
        train_episodes: Iterable of episode indices for training.
        val_episodes: Iterable of episode indices for validation.
        allow_subset: If True, allows a subset of 0..31 for train and 32..39 for val.
            If False, requires exactly 0..31 and 32..39.

    Raises:
        ProtocolViolationError: If there is overlap (data leakage) or if episodes
            cross the regime boundary.
    """
    train_set = set(train_episodes)
    val_set = set(val_episodes)

    if not train_set:
        raise ProtocolViolationError("Train episode set must not be empty.")
    if not val_set:
        raise ProtocolViolationError("Validation episode set must not be empty.")

    # 1. Critical Disjointness Check: Prevents Data Leakage
    overlap = sorted(train_set & val_set)
    if overlap:
        raise ProtocolViolationError(
            f"DATA LEAKAGE DETECTED: Episodes {overlap} appear in BOTH train and validation sets! "
            f"Train and validation episode sets must be strictly disjoint."
        )

    if protocol not in {"protocol1", "scale100"}:
        raise ProtocolViolationError(f"Unknown protocol: {protocol}")
    if any(type(ep) is not int or ep < 0 for ep in (*train_episodes, *val_episodes)):
        raise ProtocolViolationError("Episode identifiers must be non-negative integers")
    canonical_train = set(PROTOCOL_1_0_TRAIN_EPISODES)
    if protocol == "scale100":
        canonical_train.update(range(40, 90))
    canonical_val = set(PROTOCOL_1_0_VAL_EPISODES)

    # 2. Check that train episodes do not contain held-out validation episodes
    leaked_val = sorted(train_set & canonical_val)
    if leaked_val:
        raise ProtocolViolationError(
            f"PROTOCOL VIOLATION: Training set contains held-out validation episodes {leaked_val}. "
            f"Protocol 1.0 reserves episodes 32-39 exclusively for validation."
        )

    # 3. Check that validation episodes do not contain training episodes
    leaked_train = sorted(val_set & canonical_train)
    if leaked_train:
        raise ProtocolViolationError(
            f"PROTOCOL VIOLATION: Validation set contains training episodes {leaked_train}. "
            f"Validation must evaluate unseen episodes (32-39 under Protocol 1.0)."
        )

    if not train_set <= canonical_train or not val_set <= canonical_val:
        raise ProtocolViolationError(
            f"Episodes outside {protocol}: train={sorted(train_set - canonical_train)}, val={sorted(val_set - canonical_val)}"
        )
    if not allow_subset:
        if train_set != canonical_train:
            missing_train = sorted(canonical_train - train_set)
            extra_train = sorted(train_set - canonical_train)
            raise ProtocolViolationError(
                f"Protocol 1.0 requires exact train episodes 0..31; missing={missing_train}, extra={extra_train}"
            )
        if val_set != canonical_val:
            missing_val = sorted(canonical_val - val_set)
            extra_val = sorted(val_set - canonical_val)
            raise ProtocolViolationError(
                f"Protocol 1.0 requires exact val episodes 32..39; missing={missing_val}, extra={extra_val}"
            )


def extract_episodes_from_manifest(manifest: Any) -> set[int]:
    """Extract all episode indices present in a cache manifest or manifest dictionary."""
    episodes: set[int] = set()

    # If it's a CacheManifest instance with entries
    if hasattr(manifest, "entries"):
        for entry in manifest.entries:
            if hasattr(entry, "episode_index"):
                episodes.add(int(entry.episode_index))
            elif isinstance(entry, Mapping) and "episode_index" in entry:
                episodes.add(int(entry["episode_index"]))
        return episodes

    # If it's a dict / json payload
    if isinstance(manifest, Mapping):
        entries = manifest.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, Mapping) or type(entry.get("episode_index")) is not int:
                    raise ProtocolViolationError("Every entry requires an integer episode_index")
                episodes.add(entry["episode_index"])
            if not entries:
                raise ProtocolViolationError("Manifest entries must not be empty")
            if episodes:
                return episodes
        subset = manifest.get("subset")
        if isinstance(subset, Mapping) and "episodes" in subset:
            return set(subset["episodes"])

    raise ValueError(
        f"Could not extract episode indices from manifest object of type {type(manifest).__name__}"
    )


def validate_manifests_for_protocol(
    train_manifest: Any,
    val_manifest: Any,
    split: Any | None = None,
    *,
    allow_subset: bool = True,
    protocol: str = "protocol1",
    require_identity: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate that train and validation cache manifests comply with Protocol 1.0.

    Returns:
        tuple of (sorted_train_episodes, sorted_val_episodes).
    """
    train_eps = extract_episodes_from_manifest(train_manifest)
    val_eps = extract_episodes_from_manifest(val_manifest)

    enforce_protocol_1_0_split(
        sorted(train_eps), sorted(val_eps), allow_subset=allow_subset, protocol=protocol
    )
    train_data = getattr(train_manifest, "payload", train_manifest).get("dataset")
    val_data = getattr(val_manifest, "payload", val_manifest).get("dataset")
    if require_identity or train_data is not None or val_data is not None:
        if not isinstance(train_data, Mapping) or not isinstance(val_data, Mapping):
            raise ProtocolViolationError("Both manifests require dataset repo_id and revision")
        expected_repo = CANONICAL_DATASET_REPO if protocol == "protocol1" else "Orellius/cube_out_of_box_v2"
        if train_data.get("repo_id") != expected_repo or val_data.get("repo_id") != expected_repo:
            raise ProtocolViolationError("Dataset does not match selected protocol")
        revision = train_data.get("revision")
        if not revision or revision != val_data.get("revision"):
            raise ProtocolViolationError("Dataset revisions must be present and identical")
        if protocol == "protocol1" and revision != CANONICAL_DATASET_REVISION:
            raise ProtocolViolationError("Canonical dataset revision mismatch")

    if split is not None:
        split_train = set(getattr(split, "train_episodes", []))
        split_val = set(getattr(split, "val_episodes", []))
        if split_train and not (train_eps <= split_train):
            raise ProtocolViolationError(
                f"Train manifest episodes {sorted(train_eps - split_train)} not in split train episodes."
            )
        if split_val and not (val_eps <= split_val):
            raise ProtocolViolationError(
                f"Val manifest episodes {sorted(val_eps - split_val)} not in split val episodes."
            )

    return tuple(sorted(train_eps)), tuple(sorted(val_eps))


def forbid_frame_level_random_split(split_object: Any) -> None:
    """Reject in-memory random splits that break episode boundaries.

    Raises:
        ProtocolViolationError: If an object indicates an in-memory frame-level random split.
    """
    class_name = type(split_object).__name__
    if class_name in ("Subset", "_RandomSplit"):
        raise ProtocolViolationError(
            f"Frame-level {class_name} detected! In-memory random splits across frames of continuous "
            f"trajectories cause severe data leakage. Use disjoint episode manifests (Protocol 1.0)."
        )


@dataclass(frozen=True, slots=True)
class ProtocolSplitGuard:
    """Convenience guard object for verifying Protocol 1.0 adherence."""

    train_episodes: tuple[int, ...]
    val_episodes: tuple[int, ...]

    @classmethod
    def from_episodes(
        cls,
        train_episodes: Sequence[int],
        val_episodes: Sequence[int],
        *,
        allow_subset: bool = True,
    ) -> ProtocolSplitGuard:
        enforce_protocol_1_0_split(train_episodes, val_episodes, allow_subset=allow_subset)
        return cls(tuple(sorted(train_episodes)), tuple(sorted(val_episodes)))

    @classmethod
    def from_manifests(
        cls,
        train_manifest: Any,
        val_manifest: Any,
        *,
        allow_subset: bool = True,
    ) -> ProtocolSplitGuard:
        train_eps, val_eps = validate_manifests_for_protocol(
            train_manifest, val_manifest, allow_subset=allow_subset
        )
        return cls(train_eps, val_eps)
