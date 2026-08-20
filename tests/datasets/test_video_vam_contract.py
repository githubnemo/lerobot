"""Offline tests for the Video VAM dataset contract."""

from types import SimpleNamespace

import torch

from lerobot.datasets.vam import (
    CUBE_OUT_OF_BOX_CONTRACT,
    validate_metadata,
    validate_sample,
    validate_samples,
)


def make_metadata(**overrides):
    """Build metadata-shaped objects without touching the Hub."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    features = {
        config.camera_key: {
            "dtype": "video",
            "shape": config.camera_shape,
            "names": ["height", "width", "channels"],
        },
        config.state_key: {
            "dtype": "float32",
            "shape": config.state_shape,
            "names": list(config.state_names),
        },
        config.action_key: {
            "dtype": "float32",
            "shape": config.action_shape,
            "names": list(config.action_names),
        },
    }
    info = SimpleNamespace(
        codebase_version=config.codebase_version,
        fps=config.fps,
        features=features,
        total_episodes=config.total_episodes,
        total_frames=config.total_frames,
    )
    values = {
        "repo_id": config.repo_id,
        "revision": config.revision,
        "info": info,
        "features": features,
        "tasks": SimpleNamespace(index=[config.task]),
        "episodes": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_sample(*, history_mask=None, state_mask=None, action_mask=None, task=None, nan_state=False):
    """Build one expanded ``LeRobotDataset`` sample using decoded TCHW layout."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    if history_mask is None:
        history_mask = torch.zeros(config.history_length, dtype=torch.bool)
    if state_mask is None:
        state_mask = torch.zeros(config.state_history_length, dtype=torch.bool)
    if action_mask is None:
        action_mask = torch.tensor([False] * 27 + [True] * 3, dtype=torch.bool)
    state = torch.zeros(config.sample_state_shape, dtype=torch.float32)
    if nan_state:
        state[0, 0] = torch.nan
    return {
        config.camera_key: torch.zeros(config.sample_camera_shape, dtype=torch.uint8),
        config.state_key: state,
        config.action_key: torch.ones(config.sample_action_shape, dtype=torch.float32),
        f"{config.camera_key}_is_pad": history_mask,
        f"{config.state_key}_is_pad": state_mask,
        f"{config.action_key}_is_pad": action_mask,
        "task": config.task if task is None else task,
        "episode_index": torch.tensor(0),
        "timestamp": torch.tensor(0.0),
        "index": torch.tensor(0),
    }


def test_delta_timestamps_match_causal_image_and_action_contract():
    """Only images use the five-frame history; state is one current token."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    assert config.camera_key == "observation.images.front"
    assert config.state_key == "observation.state"
    assert config.action_key == "action"
    assert config.state_names == (
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    )
    assert config.action_names == config.state_names
    timestamps = config.delta_timestamps()
    assert timestamps[config.camera_key] == [-0.4, -0.3, -0.2, -0.1, 0.0]
    assert timestamps[config.state_key] == [0.0]
    assert timestamps[config.action_key] == [index / config.fps for index in range(30)]


def test_metadata_contract_checks_pinned_identity_and_shapes():
    """The synthetic metadata satisfies the pinned contract."""
    report = validate_metadata(make_metadata())
    assert report.is_valid, report.violations


def test_sample_shapes_and_finite_values_pass():
    """Expanded camera, state, action, and masks use expected dimensions."""
    report = validate_sample(make_sample())
    assert report.is_valid, report.violations


def test_state_and_action_names_are_ordered():
    """Names are validated as an ordered tuple rather than an unordered set."""
    metadata = make_metadata()
    metadata.features["action"]["names"] = list(reversed(CUBE_OUT_OF_BOX_CONTRACT.action_names))
    report = validate_metadata(metadata)
    assert not report.is_valid
    assert any("action names/order" in violation for violation in report.violations)


def test_current_state_padding_mask_keeps_current_frame_real():
    """The one-token state mask must never mark the current state as padded."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    image_mask = torch.zeros(config.history_length, dtype=torch.bool)
    valid = make_sample(history_mask=image_mask, state_mask=torch.tensor([False]))
    assert validate_sample(valid, config).is_valid

    invalid = make_sample(history_mask=image_mask, state_mask=torch.tensor([True]))
    report = validate_sample(invalid, config)
    assert not report.is_valid
    assert any("current frame" in violation for violation in report.violations)


def test_image_history_has_five_causal_padding_positions():
    """The image mask remains five positions even though state has one token."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    sample = make_sample(history_mask=torch.tensor([True, True, True, True, False]))
    report = validate_sample(sample, config)
    assert report.is_valid, report.violations


def test_action_padding_is_trailing_and_padded_values_stay_finite():
    """``action_is_pad`` is trailing padding, and padded actions stay finite."""
    config = CUBE_OUT_OF_BOX_CONTRACT
    valid = make_sample(action_mask=torch.tensor([False] * 10 + [True] * 20))
    assert validate_sample(valid, config).is_valid

    invalid = make_sample(action_mask=torch.tensor([False] * 10 + [True, False] + [True] * 18))
    report = validate_sample(invalid, config)
    assert not report.is_valid
    assert any("action_is_pad" in violation for violation in report.violations)


def test_non_finite_state_is_rejected():
    """A NaN in the state is a hard contract violation."""
    report = validate_sample(make_sample(nan_state=True))
    assert not report.is_valid
    assert any("non-finite" in violation for violation in report.violations)


def test_episode_boundaries_and_sample_timing_are_monotonic():
    """Available episode and sample timing metadata are checked without requiring them."""
    episodes = [
        {
            "episode_index": 0,
            "dataset_from_index": 0,
            "dataset_to_index": 3,
            "length": 3,
            "from_timestamp": 0.0,
            "to_timestamp": 0.2,
        },
        {
            "episode_index": 1,
            "dataset_from_index": 3,
            "dataset_to_index": 6,
            "length": 3,
            "from_timestamp": 0.0,
            "to_timestamp": 0.2,
        },
    ]
    metadata_report = validate_metadata(make_metadata(episodes=episodes))
    assert metadata_report.is_valid, metadata_report.violations

    first = make_sample()
    second = make_sample()
    second["timestamp"] = torch.tensor(0.1)
    second["index"] = torch.tensor(1)
    sample_report = validate_samples([first, second])
    assert sample_report.is_valid, sample_report.violations


def test_wrong_action_shape_is_rejected():
    """A malformed action chunk fails instead of being silently reshaped."""
    sample = make_sample()
    sample["action"] = torch.zeros(29, 6)
    report = validate_sample(sample)
    assert not report.is_valid
    assert any("action shape" in violation for violation in report.violations)
