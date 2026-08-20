"""Metadata and sample validation for reusable video-action data contracts."""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

import numpy as np

from .config import CUBE_OUT_OF_BOX_CONTRACT, VideoVAMDatasetConfig

_MISSING = object()


class DatasetContractError(ValueError):
    """Raised when one or more dataset contract checks fail."""

    def __init__(self, violations: Iterable[str]):
        self.violations = tuple(violations)
        super().__init__("Dataset contract violations:\n- " + "\n- ".join(self.violations))


@dataclass
class ValidationReport:
    """Collect validation violations without coupling callers to a dataset type."""

    violations: list[str] = field(default_factory=list)
    samples_checked: int = 0

    @property
    def is_valid(self) -> bool:
        """Return whether every check passed."""
        return not self.violations

    @property
    def ok(self) -> bool:
        """Alias for ``is_valid`` suitable for concise CLI checks."""
        return self.is_valid

    def add(self, violation: str) -> None:
        """Add one human-readable violation."""
        self.violations.append(violation)

    def extend(self, other: "ValidationReport") -> None:
        """Merge another report into this report."""
        self.violations.extend(other.violations)
        self.samples_checked += other.samples_checked

    def raise_if_invalid(self) -> None:
        """Raise ``DatasetContractError`` when this report is not valid."""
        if not self.is_valid:
            raise DatasetContractError(self.violations)


def _field(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _metadata_field(metadata: Any, key: str, default: Any = None) -> Any:
    value = _field(metadata, key, _MISSING)
    if value is not _MISSING:
        return value
    return _field(_field(metadata, "info", None), key, default)


def _tuple(value: Any) -> tuple[Any, ...]:
    try:
        return tuple(value) if value is not None else ()
    except TypeError:
        return ()


def _shape(value: Any) -> tuple[int, ...] | None:
    value_shape = getattr(value, "shape", None)
    if value_shape is not None:
        return tuple(int(dimension) for dimension in value_shape)
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shape = _shape(value[0])
        return (len(value),) + (child_shape or ())
    return ()


def _flat_values(value: Any) -> list[Any]:
    if hasattr(value, "detach") and hasattr(value, "reshape"):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for child in value:
            flattened.extend(_flat_values(child))
        return flattened
    return [value]


def _is_finite(value: Any) -> bool:
    if hasattr(value, "detach") and hasattr(value, "is_floating_point"):
        if value.is_floating_point() or getattr(value, "is_complex", lambda: False)():
            import torch

            return bool(torch.isfinite(value).all().item())
        return True
    if isinstance(value, np.ndarray):
        try:
            return bool(np.isfinite(value).all())
        except TypeError:
            return False
    if isinstance(value, (list, tuple)):
        return all(_is_finite(child) for child in value)
    if isinstance(value, Number):
        return math.isfinite(float(value))
    return False


def _is_bool_tensor(value: Any) -> bool:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        if str(dtype) in {"torch.bool", "bool", "bool_"}:
            return True
        return getattr(dtype, "kind", None) == "b"
    values = _flat_values(value)
    return bool(values) and all(isinstance(item, bool) for item in values)


def _task_names(tasks: Any) -> list[str]:
    if tasks is None:
        return []
    index = _field(tasks, "index", _MISSING)
    if index is not _MISSING:
        return [str(task) for task in index]
    if isinstance(tasks, Mapping):
        for key in ("task", "tasks"):
            if key in tasks:
                return [str(task) for task in _flat_values(tasks[key])]
        return [str(task) for task in tasks]
    if isinstance(tasks, (list, tuple, set)):
        names = []
        for task in tasks:
            names.append(str(_field(task, "task", task)))
        return names
    return []


def _episode_rows(episodes: Any) -> list[Mapping[str, Any]]:
    if episodes is None:
        return []
    if hasattr(episodes, "to_pylist"):
        return list(episodes.to_pylist())
    if hasattr(episodes, "to_dict"):
        try:
            rows = episodes.to_dict(orient="records")
        except TypeError:
            rows = None
        if rows is not None:
            return list(rows)
    if isinstance(episodes, Mapping):
        return []
    try:
        return [row for row in episodes if isinstance(row, Mapping)]
    except TypeError:
        return []


def _number(value: Any) -> float | int | None:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Number):
        return value
    return None


def _check_shape(report: ValidationReport, name: str, value: Any, expected: tuple[int, ...]) -> None:
    actual = _shape(value)
    if actual != expected:
        report.add(f"{name} shape is {actual}, expected {expected}")


def _check_finite(report: ValidationReport, name: str, value: Any) -> None:
    if not _is_finite(value):
        report.add(f"{name} contains non-finite or non-numeric values")


def _check_mask(
    report: ValidationReport,
    sample: Mapping[str, Any],
    key: str,
    expected_length: int,
    mode: str,
) -> None:
    value = sample.get(key, _MISSING)
    if value is _MISSING:
        report.add(f"missing required padding mask {key}")
        return
    _check_shape(report, key, value, (expected_length,))
    if not _is_bool_tensor(value):
        report.add(f"{key} must have boolean dtype")
        return
    values = [bool(item) for item in _flat_values(value)]
    if len(values) != expected_length:
        return
    if mode == "history":
        if values[-1]:
            report.add(f"{key} marks the current frame as padded")
        seen_real = False
        for item in values:
            if not item:
                seen_real = True
            elif seen_real:
                report.add(f"{key} must be true only before the causal history enters the episode")
                break
    else:
        if values[0]:
            report.add(f"{key} marks the current action as padded")
        seen_pad = False
        for item in values:
            if item:
                seen_pad = True
            elif seen_pad:
                report.add(f"{key} must be false for real actions followed only by true padding")
                break


def _validate_episode_metadata(report: ValidationReport, episodes: Any) -> None:
    rows = _episode_rows(episodes)
    if not rows:
        return
    previous_episode = None
    previous_to_index = None
    for position, row in enumerate(rows):
        episode_index = _number(row.get("episode_index", _MISSING))
        if episode_index is not None:
            if previous_episode is not None and episode_index <= previous_episode:
                report.add(f"episode metadata index is not strictly increasing at row {position}")
            previous_episode = episode_index
        from_index = _number(row.get("dataset_from_index", _MISSING))
        to_index = _number(row.get("dataset_to_index", _MISSING))
        if from_index is not None and to_index is not None:
            if to_index <= from_index:
                report.add(f"episode {episode_index} has non-positive index span")
            if previous_to_index is not None and from_index < previous_to_index:
                report.add(f"episode {episode_index} starts before the previous episode ends")
            previous_to_index = to_index
            length = _number(row.get("length", _MISSING))
            if length is not None and int(length) != int(to_index - from_index):
                report.add(f"episode {episode_index} length does not match its index span")
        from_timestamp = _number(row.get("from_timestamp", _MISSING))
        to_timestamp = _number(row.get("to_timestamp", _MISSING))
        if from_timestamp is not None and to_timestamp is not None:
            if not math.isfinite(float(from_timestamp)) or not math.isfinite(float(to_timestamp)):
                report.add(f"episode {episode_index} has non-finite timing metadata")
            elif to_timestamp < from_timestamp:
                report.add(f"episode {episode_index} has decreasing timing metadata")


def validate_metadata(
    metadata: Any,
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> ValidationReport:
    """Validate a LeRobot metadata object or a metadata-shaped fake object."""
    report = ValidationReport()
    repo_id = _metadata_field(metadata, "repo_id", _MISSING)
    if repo_id != config.repo_id:
        report.add(f"repo_id is {repo_id!r}, expected {config.repo_id!r}")
    revision = _metadata_field(metadata, "revision", _MISSING)
    if revision != config.revision:
        report.add(f"revision is {revision!r}, expected pinned {config.revision!r}")
    codebase_version = _metadata_field(metadata, "codebase_version", _MISSING)
    if codebase_version != config.codebase_version:
        report.add(f"codebase_version is {codebase_version!r}, expected {config.codebase_version!r}")
    fps = _metadata_field(metadata, "fps", _MISSING)
    if fps != config.fps:
        report.add(f"fps is {fps!r}, expected {config.fps}")
    total_episodes = _metadata_field(metadata, "total_episodes", _MISSING)
    if total_episodes != config.total_episodes:
        report.add(f"total_episodes is {total_episodes!r}, expected {config.total_episodes}")
    total_frames = _metadata_field(metadata, "total_frames", _MISSING)
    if total_frames != config.total_frames:
        report.add(f"total_frames is {total_frames!r}, expected {config.total_frames}")

    features = _metadata_field(metadata, "features", {})
    if not isinstance(features, Mapping):
        report.add("metadata features are not mapping-like")
        features = {}
    camera = features.get(config.camera_key)
    if camera is None:
        report.add(f"missing camera feature {config.camera_key}")
    else:
        if _field(camera, "dtype") not in {"image", "video"}:
            report.add(f"{config.camera_key} dtype must be image or video")
        camera_shape = _tuple(_field(camera, "shape", ()))
        if camera_shape != config.camera_shape:
            report.add(
                f"{config.camera_key} metadata shape is {camera_shape!r}, expected {config.camera_shape}"
            )
        if int(config.camera_shape[-1]) != 3:
            report.add("camera contract must describe RGB with three channels")

    for key, expected_shape, expected_names in (
        (config.state_key, config.state_shape, config.state_names),
        (config.action_key, config.action_shape, config.action_names),
    ):
        feature = features.get(key)
        if feature is None:
            report.add(f"missing feature {key}")
            continue
        feature_shape = _tuple(_field(feature, "shape", ()))
        if feature_shape != expected_shape:
            report.add(f"{key} metadata shape is {feature_shape!r}, expected {expected_shape}")
        names = tuple(_field(feature, "names", ()) or ())
        if names != expected_names:
            report.add(f"{key} names/order are {names!r}, expected {expected_names!r}")

    task_names = _task_names(_metadata_field(metadata, "tasks", None))
    if config.task not in task_names:
        report.add(f"task {config.task!r} is not declared; declared tasks are {task_names!r}")
    _validate_episode_metadata(report, _metadata_field(metadata, "episodes", None))
    return report


def validate_sample(
    sample: Mapping[str, Any],
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> ValidationReport:
    """Validate one expanded LeRobot sample without normalizing its values."""
    report = ValidationReport(samples_checked=1)
    if not isinstance(sample, Mapping):
        report.add(f"sample is not mapping-like: {type(sample).__name__}")
        return report
    for key, expected_shape in (
        (config.camera_key, config.sample_camera_shape),
        (config.state_key, config.sample_state_shape),
        (config.action_key, config.sample_action_shape),
    ):
        value = sample.get(key, _MISSING)
        if value is _MISSING:
            report.add(f"missing sample feature {key}")
            continue
        _check_shape(report, key, value, expected_shape)
        _check_finite(report, key, value)

    _check_mask(report, sample, f"{config.camera_key}_is_pad", config.history_length, "history")
    # The pinned LeRobot reader emits action_is_pad exactly; padded action values are clamped, not discarded.
    _check_mask(report, sample, f"{config.state_key}_is_pad", config.state_history_length, "history")
    _check_mask(report, sample, f"{config.action_key}_is_pad", config.action_chunk_size, "action")

    task = sample.get("task", _MISSING)
    if task is _MISSING:
        report.add("sample is missing task")
    elif str(task) != config.task:
        report.add(f"sample task is {task!r}, expected {config.task!r}")

    for key in ("timestamp", "episode_index", "index"):
        value = sample.get(key, _MISSING)
        if value is not _MISSING:
            scalar = _number(value)
            if scalar is None or not math.isfinite(float(scalar)):
                report.add(f"sample {key} is not finite numeric metadata")
    return report


def _validate_sample_timing(report: ValidationReport, samples: list[Mapping[str, Any]]) -> None:
    previous_episode = None
    previous_timestamp = None
    previous_index = None
    for position, sample in enumerate(samples):
        episode = _number(sample.get("episode_index", _MISSING))
        timestamp = _number(sample.get("timestamp", _MISSING))
        index = _number(sample.get("index", _MISSING))
        if episode is not None and previous_episode is not None and episode < previous_episode:
            report.add(f"sample episode_index decreases at sample {position}")
        if (
            episode is not None
            and timestamp is not None
            and episode == previous_episode
            and previous_timestamp is not None
            and timestamp < previous_timestamp
        ):
            report.add(f"sample timestamp decreases inside episode {episode}")
        if index is not None and previous_index is not None and index < previous_index:
            report.add(f"sample index decreases at sample {position}")
        if episode is not None:
            previous_episode = episode
        if timestamp is not None:
            previous_timestamp = timestamp
        if index is not None:
            previous_index = index


def validate_samples(
    samples: Iterable[Mapping[str, Any]],
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> ValidationReport:
    """Validate expanded samples and monotonic timing when those fields exist."""
    sample_list = list(samples)
    report = ValidationReport()
    for sample in sample_list:
        report.extend(validate_sample(sample, config))
    _validate_sample_timing(report, [sample for sample in sample_list if isinstance(sample, Mapping)])
    return report


def validate_dataset(
    metadata: Any,
    samples: Iterable[Mapping[str, Any]] = (),
    config: VideoVAMDatasetConfig = CUBE_OUT_OF_BOX_CONTRACT,
) -> ValidationReport:
    """Validate metadata and optional expanded samples as one report."""
    report = validate_metadata(metadata, config)
    report.extend(validate_samples(samples, config))
    return report
