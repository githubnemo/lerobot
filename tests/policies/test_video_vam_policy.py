from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.rtc import RTCConfig
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.cosmos_feature_cache import derive_window_seed
from lerobot.policies.vam.modeling_video_vam import (
    HISTORY_FRAME_INDEX_KEY,
    VideoVAMPolicy,
    VideoVAMSafetyError,
    restore_uint8_history,
)
from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.rollout.inference import RTCInferenceConfig, SyncInferenceConfig


class _FakeExtractor:
    def __init__(self) -> None:
        self.images = None
        self.noise_seed = None
        self.closed = False

    def extract(self, images, prompt, *, noise_seed):
        self.images = images.clone()
        self.noise_seed = noise_seed
        assert prompt.shape == (1, 2, 3)
        return SimpleNamespace(tokens=torch.zeros(1, 1, 1))

    def close(self) -> None:
        self.closed = True


class _FakeDecoder(nn.Module):
    def __init__(self, actions: torch.Tensor) -> None:
        super().__init__()
        self.actions = actions
        self.call = None

    def sample_actions(self, state, context, **kwargs):
        self.call = (state.clone(), context.clone(), kwargs)
        return self.actions.clone()


def _config(**kwargs) -> VideoVAMConfig:
    return VideoVAMConfig(
        device="cpu",
        backend="cosmos",
        feature_seed_episode_index=32,
        joint_limits_min=[-180.0, -180.0, -180.0, -180.0, -180.0, 0.0],
        joint_limits_max=[180.0, 180.0, 180.0, 180.0, 180.0, 100.0],
        **kwargs,
    )


def test_restore_uint8_history_is_pixel_exact() -> None:
    raw = torch.randint(0, 256, (1, 3, 5, 480, 640), dtype=torch.uint8)
    restored = restore_uint8_history(raw.float() / 255.0)
    assert torch.equal(restored, raw)


def test_restore_uint8_history_rejects_non_uint8_grid() -> None:
    history = torch.zeros(1, 3, 5, 480, 640)
    history[0, 0, 0, 0, 0] = 0.1
    with pytest.raises(ValueError, match="lossless uint8/255"):
        restore_uint8_history(history)


def test_wrapper_matches_live_history_seed_and_internal_physical_actions(monkeypatch) -> None:
    actions = torch.zeros(1, 30, 6)
    actions[..., 5] = 50.0
    decoder = _FakeDecoder(actions)
    extractor = _FakeExtractor()
    config = _config(feature_seed_frame_offset=100)
    config.rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    policy = VideoVAMPolicy(
        config,
        decoder=decoder,
        extractor=extractor,
        prompt_embedding=torch.zeros(1, 2, 3),
    )
    monkeypatch.setattr(
        "lerobot.policies.vam.modeling_video_vam.apply_context_transform",
        lambda tokens, transform: torch.zeros(1, 4800, 2048, dtype=torch.bfloat16),
    )
    raw = torch.randint(0, 256, (1, 3, 5, 480, 640), dtype=torch.uint8)
    batch = {
        "observation.images.front.history": raw.float() / 255.0,
        "observation.state": torch.arange(6, dtype=torch.float32).unsqueeze(0),
        HISTORY_FRAME_INDEX_KEY: torch.tensor([4]),
    }

    prediction = policy.predict_action_chunk(batch)

    assert torch.equal(prediction, actions)
    assert torch.equal(extractor.images, raw)
    assert extractor.noise_seed == derive_window_seed(config.dataset_revision, 32, 104, 0)
    assert decoder.call is not None
    state, context, call_kwargs = decoder.call
    assert torch.equal(state, batch["observation.state"])
    assert context.shape == (1, 4800, 2048)
    assert call_kwargs["seed"] == 0
    assert call_kwargs["rtc_processor"] is None


def test_wrapper_fails_loudly_before_out_of_range_chunk_is_returned(monkeypatch) -> None:
    actions = torch.zeros(1, 30, 6)
    actions[0, 3, 5] = 101.0
    policy = VideoVAMPolicy(
        _config(),
        decoder=_FakeDecoder(actions),
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )
    monkeypatch.setattr(
        "lerobot.policies.vam.modeling_video_vam.apply_context_transform",
        lambda tokens, transform: torch.zeros(1, 4800, 2048, dtype=torch.bfloat16),
    )
    batch = {
        "observation.images.front.history": torch.zeros(1, 3, 5, 480, 640),
        "observation.state": torch.zeros(1, 6),
    }
    with pytest.raises(VideoVAMSafetyError, match="gripper.pos=101.0 range_0_100"):
        policy.predict_action_chunk(batch, feature_noise_seed=1)


def test_rollout_safety_requires_limits_degrees_relative_cap_and_seed() -> None:
    config = VideoVAMConfig(device="cpu")
    relative_limits = {
        "shoulder_pan": 5.0,
        "shoulder_lift": 5.0,
        "elbow_flex": 5.0,
        "wrist_flex": 5.0,
        "wrist_roll": 5.0,
        "gripper": 10.0,
    }
    robot = SimpleNamespace(use_degrees=True, max_relative_target=relative_limits)

    def rollout(robot_config=robot, *, fps=10, inference=None):
        return SimpleNamespace(
            robot=robot_config,
            fps=fps,
            inference=inference or RTCInferenceConfig(observation_history_size=5),
        )

    with pytest.raises(ValueError, match="joint_limits"):
        config.validate_rollout_safety(rollout())

    config = _config()
    with pytest.raises(ValueError, match="use_degrees"):
        config.validate_rollout_safety(
            rollout(SimpleNamespace(use_degrees=False, max_relative_target=relative_limits))
        )
    with pytest.raises(ValueError, match="max_relative_target"):
        config.validate_rollout_safety(rollout(SimpleNamespace(use_degrees=True, max_relative_target=None)))
    with pytest.raises(ValueError, match="uniform degree/range_0_100"):
        config.validate_rollout_safety(rollout(SimpleNamespace(use_degrees=True, max_relative_target=5.0)))
    with pytest.raises(ValueError, match="fps=10"):
        config.validate_rollout_safety(rollout(fps=30))
    with pytest.raises(ValueError, match="inference.type=rtc"):
        config.validate_rollout_safety(rollout(inference=SyncInferenceConfig()))
    with pytest.raises(ValueError, match="observation_history_size=5"):
        config.validate_rollout_safety(rollout(inference=RTCInferenceConfig(observation_history_size=4)))


def test_processors_do_not_double_normalize() -> None:
    preprocessor, postprocessor = make_pre_post_processors(_config(), pretrained_path="custom-run")
    steps = [*preprocessor.steps, *postprocessor.steps]
    assert not any(isinstance(step, (NormalizerProcessorStep, UnnormalizerProcessorStep)) for step in steps)
