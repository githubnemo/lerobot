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
        return SimpleNamespace(
            tokens=torch.zeros(1, 1, 1),
            hidden_grid=torch.zeros(1, 8, 15, 20, 4096),
        )

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
    backend = kwargs.pop("backend", "cosmos")
    return VideoVAMConfig(
        device="cpu",
        backend=backend,
        feature_seed_episode_index=32,
        joint_limits_min=[-180.0, -180.0, -180.0, -180.0, -180.0, 0.0],
        joint_limits_max=[180.0, 180.0, 180.0, 180.0, 180.0, 100.0],
        **kwargs,
    )


def test_cosmos_torch_compile_defaults_on_and_can_be_disabled() -> None:
    assert _config().cosmos_torch_compile is True
    assert _config().cosmos_compile_friendly is True
    disabled = _config(cosmos_torch_compile=False, cosmos_compile_friendly=False)
    assert disabled.cosmos_torch_compile is False
    assert disabled.cosmos_compile_friendly is False
    with pytest.raises(ValueError, match="incompatible"):
        _config(cosmos_fp8_linear=True)


def test_ltx_checkpoint_contract_selects_pool2_or_unpooled_context() -> None:
    action_semantics = {"action_dim": 6, "horizon": 30, "internal_action_dim": 32, "euler_steps": 10}
    for transform, input_shape, expected_shape in (
        ("pool2", ["B", 640, 4096], (1, 640, 4096)),
        ("none", ["B", 2400, 4096], (1, 2400, 4096)),
    ):
        extractor = _FakeExtractor()
        policy = VideoVAMPolicy(
            _config(backend="ltx"),
            decoder=_FakeDecoder(torch.zeros(1, 30, 6)),
            extractor=extractor,
            prompt_embedding=torch.zeros(1, 2, 3),
        )
        policy._validate_checkpoint_contract(
            {
                "action_semantics": action_semantics,
                "injection": {
                    "stored_context_transform": transform,
                    "input_shape": input_shape,
                },
            }
        )
        context = policy._extract_context(torch.zeros(1, 3, 5, 480, 640, dtype=torch.uint8), 1)
        assert policy._ltx_context_transform == transform
        assert tuple(context.shape) == expected_shape


def test_cosmos_observed_only_cached_context_checkpoint_contract() -> None:
    policy = VideoVAMPolicy(
        _config(cosmos_state_t=2),
        decoder=_FakeDecoder(torch.zeros(1, 30, 6)),
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )

    policy._validate_checkpoint_contract(
        {
            "artifact": "smolexpert_on_cached_context_training_checkpoint",
            "action_semantics": {
                "action_dim": 6,
                "horizon": 30,
                "internal_action_dim": 32,
                "euler_steps": 10,
            },
            "injection": {
                "stored_context_transform": "pool2",
                "input_shape": ["B", 600, 2048],
            },
        }
    )


def test_cosmos_cached_context_checkpoint_contract() -> None:
    policy = VideoVAMPolicy(
        _config(),
        decoder=_FakeDecoder(torch.zeros(1, 30, 6)),
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )

    policy._validate_checkpoint_contract(
        {
            "artifact": "smolexpert_on_cached_context_training_checkpoint",
            "action_semantics": {
                "action_dim": 6,
                "horizon": 30,
                "internal_action_dim": 32,
                "euler_steps": 10,
            },
            "injection": {
                "stored_context_transform": "pool2",
                "input_shape": ["B", 4800, 2048],
            },
        }
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


def test_wrapper_forwards_rtc_leftover_and_delay(monkeypatch) -> None:
    actions = torch.zeros(1, 30, 6)
    actions[..., 5] = 50.0
    decoder = _FakeDecoder(actions)
    config = _config()
    config.rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    policy = VideoVAMPolicy(
        config,
        decoder=decoder,
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )
    policy.init_rtc_processor()
    monkeypatch.setattr(
        "lerobot.policies.vam.modeling_video_vam.apply_context_transform",
        lambda tokens, transform: torch.zeros(1, 4800, 2048, dtype=torch.bfloat16),
    )
    leftover = torch.zeros(10, 6)
    leftover[..., 5] = 50.0
    policy.predict_action_chunk(
        {
            "observation.images.front.history": torch.zeros(1, 3, 5, 480, 640),
            "observation.state": torch.zeros(1, 6),
            HISTORY_FRAME_INDEX_KEY: torch.tensor([4]),
        },
        inference_delay=3,
        prev_chunk_left_over=leftover,
    )
    assert decoder.call is not None
    call_kwargs = decoder.call[2]
    assert call_kwargs["inference_delay"] == 3
    assert call_kwargs["execution_horizon"] == 10
    assert call_kwargs["rtc_processor"] is policy.rtc_processor
    assert torch.equal(call_kwargs["prev_chunk_left_over"], leftover)


def test_wrapper_rejects_leftover_without_rtc_processor(monkeypatch) -> None:
    policy = VideoVAMPolicy(
        _config(),
        decoder=_FakeDecoder(torch.zeros(1, 30, 6)),
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )
    monkeypatch.setattr(
        "lerobot.policies.vam.modeling_video_vam.apply_context_transform",
        lambda tokens, transform: torch.zeros(1, 4800, 2048, dtype=torch.bfloat16),
    )
    with pytest.raises(ValueError, match="init_rtc_processor"):
        policy.predict_action_chunk(
            {
                "observation.images.front.history": torch.zeros(1, 3, 5, 480, 640),
                "observation.state": torch.zeros(1, 6),
                HISTORY_FRAME_INDEX_KEY: torch.tensor([4]),
            },
            prev_chunk_left_over=torch.zeros(10, 6),
        )


def test_init_rtc_processor_requires_rtc_config() -> None:
    policy = VideoVAMPolicy(
        _config(),
        decoder=_FakeDecoder(torch.zeros(1, 30, 6)),
        extractor=_FakeExtractor(),
        prompt_embedding=torch.zeros(1, 2, 3),
    )
    with pytest.raises(ValueError, match="rtc_config"):
        policy.init_rtc_processor()


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
