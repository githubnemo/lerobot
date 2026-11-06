#!/usr/bin/env python

"""Configuration for hardware-rollout wrappers around frozen Video-VAM experts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from lerobot.configs import FeatureType, PolicyFeature, PreTrainedConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION, OBS_STATE

DEFAULT_COSMOS_CHECKPOINT = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/v2w_pretrained_cosmos.pt"
)
DEFAULT_COSMOS_TOKENIZER = Path(
    "/home/anton/.cache/video-vam/mimic-video-f2833903/video_backbone/tokenizer/tokenizer.pth"
)
DEFAULT_COSMOS_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-t5-11b.safetensors"
)
DEFAULT_LTX_TRANSFORMER = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/diffusion_models/"
    "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
)
DEFAULT_LTX_VAE = Path(
    "/home/anton/.cache/video-vam/ltx-2.5-models/vae/ltx-2.5-video-vae-conv-bf16.safetensors"
)
DEFAULT_LTX_PROMPT = Path(
    "/home/anton/.cache/video-vam/prompt-embeddings/cube-out-of-box-ltx25-gemma4.safetensors"
)

ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
ACTION_UNITS = ("degrees", "degrees", "degrees", "degrees", "degrees", "range_0_100")


def _default_input_features() -> dict[str, PolicyFeature]:
    return {
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }


def _default_output_features() -> dict[str, PolicyFeature]:
    return {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}


@PreTrainedConfig.register_subclass("video_vam")
@dataclass
class VideoVAMConfig(PreTrainedConfig):
    """Frozen Cosmos/LTX feature producer plus the trained SmolVLA action expert."""

    processors_from_config: ClassVar[bool] = True

    backend: str = "cosmos"
    input_features: dict[str, PolicyFeature] = field(default_factory=_default_input_features)
    output_features: dict[str, PolicyFeature] = field(default_factory=_default_output_features)
    normalization_mapping: dict = field(default_factory=dict)
    device: str = "cuda"
    use_amp: bool = False

    camera_key: str = "observation.images.front"
    chunk_size: int = 30
    n_action_steps: int = 30
    dataset_revision: str = "243370c3c08bcbd860133c4a0d658ea7c1d2e77e"
    action_feature_names: list[str] = field(default_factory=lambda: list(ACTION_NAMES))
    action_seed: int = 0
    # Flow-matching Euler steps at inference. None = use the checkpoint's recorded value (training
    # default 10). Fewer steps (e.g. 3) cut latency; see scripts/video_vam/ablate_euler_steps.py.
    action_euler_steps: int | None = None

    # Backbone noise is derived exactly as in cache construction. The rollout engine
    # supplies an episode-local control-rate frame index. Collection-day config must
    # choose an episode index and absolute-frame offset for the new rollout.
    feature_seed_episode_index: int | None = 0
    feature_seed_frame_offset: int = 0
    feature_seed_global: int = 0

    cosmos_checkpoint: Path = DEFAULT_COSMOS_CHECKPOINT
    cosmos_tokenizer: Path = DEFAULT_COSMOS_TOKENIZER
    cosmos_prompt: Path = DEFAULT_COSMOS_PROMPT
    cosmos_attention_backend: str = "minimal_a2a"
    cosmos_compile_friendly: bool = True
    cosmos_torch_compile: bool = False
    cosmos_compile_mode: str = "max-autotune"
    cosmos_use_cuda_graphs: bool = False
    cosmos_state_t: int = 16
    cosmos_fp8_linear: bool = False
    cosmos_context_transform: str = "pool2"
    cosmos_lora_weights: Path | None = None
    cosmos_lora_rank: int | None = None
    cosmos_lora_alpha: float | None = None
    cosmos3_checkpoint: Path = Path("/home/anton/.cache/video-vam/cosmos3-edge")
    cosmos3_lora_weights: Path | None = None
    cosmos3_lora_rank: int | None = None
    cosmos3_lora_alpha: float | None = None
    cosmos3_lora_use_dora: bool = False
    cosmos3_prompt: str = "take cube out of box"
    flux2_lora_weights: Path | None = None
    flux2_lora_rank: int | None = None
    flux2_lora_alpha: float | None = None
    flux2_tap_location: str | None = None
    online_trained: bool = False
    online_backbone: str | None = None
    augmented: bool = False
    train_stride: int | None = None
    jepa_world_model_checkpoint: Path | None = None
    jepa_planning_mode: str = "none"
    jepa_num_candidates: int = 8

    ltx_transformer: Path = DEFAULT_LTX_TRANSFORMER
    ltx_vae: Path = DEFAULT_LTX_VAE
    ltx_prompt: Path = DEFAULT_LTX_PROMPT
    ltx_offload_mode: str = "cpu"
    ltx_quantization: str = "fp8-cast"
    ltx_vae_compile_mode: str | None = None

    # Deliberately unset in generated deployment configs. There is no trustworthy
    # calibration/limit artifact on abakus; rollout refuses to start until all six
    # mixed-unit bounds are supplied explicitly.
    joint_limits_min: list[float] | None = field(
        default_factory=lambda: [-180.0, -180.0, -180.0, -180.0, -180.0, 0.0]
    )
    joint_limits_max: list[float] | None = field(
        default_factory=lambda: [180.0, 180.0, 180.0, 180.0, 180.0, 100.0]
    )
    joint_limit_tolerance: float = 60.0

    rtc_config: RTCConfig | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.cosmos_checkpoint = Path(self.cosmos_checkpoint)
        self.cosmos_tokenizer = Path(self.cosmos_tokenizer)
        self.cosmos_prompt = Path(self.cosmos_prompt)
        self.ltx_transformer = Path(self.ltx_transformer)
        self.ltx_vae = Path(self.ltx_vae)
        self.ltx_prompt = Path(self.ltx_prompt)
        if self.cosmos_lora_weights is not None:
            self.cosmos_lora_weights = Path(self.cosmos_lora_weights)
        if self.cosmos3_checkpoint is not None:
            self.cosmos3_checkpoint = Path(self.cosmos3_checkpoint)
        if self.cosmos3_lora_weights is not None:
            self.cosmos3_lora_weights = Path(self.cosmos3_lora_weights)
        if self.flux2_lora_weights is not None:
            self.flux2_lora_weights = Path(self.flux2_lora_weights)
        if self.jepa_world_model_checkpoint is not None:
            self.jepa_world_model_checkpoint = Path(self.jepa_world_model_checkpoint)
        if self.jepa_planning_mode not in {"none", "best_of_n", "mppi"}:
            raise ValueError("jepa_planning_mode must be 'none', 'best_of_n', or 'mppi'")
        if self.jepa_num_candidates < 1:
            raise ValueError("jepa_num_candidates must be >= 1")
        if self.backend not in {"cosmos", "cosmos3_edge", "ltx"}:
            raise ValueError("backend must be 'cosmos' or 'ltx'")
        if self.cosmos_state_t not in (2, 16):
            raise ValueError("cosmos_state_t must be 2 (observed-only) or 16")
        if self.action_euler_steps is not None and self.action_euler_steps < 1:
            raise ValueError("action_euler_steps must be a positive integer or None")
        if not isinstance(self.cosmos_fp8_linear, bool):
            raise ValueError("cosmos_fp8_linear must be a boolean")
        if not isinstance(self.cosmos_torch_compile, bool):
            raise ValueError("cosmos_torch_compile must be a boolean")
        if self.cosmos_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("cosmos_compile_mode must be a torch.compile mode")
        if self.cosmos_torch_compile:
            self.cosmos_compile_friendly = True
        if self.cosmos_torch_compile and self.cosmos_fp8_linear:
            raise ValueError("cosmos_torch_compile is incompatible with cosmos_fp8_linear")
        if self.action_feature_names != list(ACTION_NAMES):
            raise ValueError(f"action_feature_names must preserve the trained order {list(ACTION_NAMES)}")
        if self.action_seed < 0 or self.feature_seed_global < 0:
            raise ValueError("action_seed and feature_seed_global must be non-negative")
        if self.feature_seed_episode_index is not None and self.feature_seed_episode_index < 0:
            raise ValueError("feature_seed_episode_index must be non-negative")
        if self.feature_seed_frame_offset < 0:
            raise ValueError("feature_seed_frame_offset must be non-negative")
        if not math.isfinite(self.joint_limit_tolerance) or self.joint_limit_tolerance < 0:
            raise ValueError("joint_limit_tolerance must be finite and non-negative")
        self._validate_joint_limits(allow_missing=True)

    def _validate_joint_limits(self, *, allow_missing: bool) -> None:
        if self.joint_limits_min is None or self.joint_limits_max is None:
            if allow_missing and self.joint_limits_min is None and self.joint_limits_max is None:
                return
            raise ValueError("joint_limits_min and joint_limits_max must both contain six values")
        if len(self.joint_limits_min) != 6 or len(self.joint_limits_max) != 6:
            raise ValueError("joint limit arrays must contain six values in action_feature_names order")
        for index, (lower, upper) in enumerate(
            zip(self.joint_limits_min, self.joint_limits_max, strict=True)
        ):
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"invalid {ACTION_NAMES[index]} limits: [{lower}, {upper}]")

    def validate_rollout_safety(self, rollout_config: object) -> None:
        """Fail before model loading or robot connection when fidelity/safety is incomplete."""

        self._validate_joint_limits(allow_missing=False)
        if self.feature_seed_episode_index is None:
            raise ValueError(
                "Video-VAM rollout requires policy.feature_seed_episode_index; choose a stable non-negative "
                "index for this collection episode"
            )
        if getattr(rollout_config, "fps", None) != 10:
            rollout_config.fps = 10  # type: ignore[attr-defined]
        inference = getattr(rollout_config, "inference", None)
        if getattr(inference, "type", None) == "rtc":
            if getattr(inference, "observation_history_size", None) == 1:
                inference.observation_history_size = 5
            if getattr(inference, "observation_history_size", None) != 5:
                raise ValueError("Video-VAM rollout requires inference.observation_history_size=5")
            rtc_cfg = getattr(inference, "rtc", None)
            if rtc_cfg is not None and getattr(rtc_cfg, "max_guidance_weight", 10.0) > 3.0:
                rtc_cfg.max_guidance_weight = 3.0
            exec_h = int(getattr(rtc_cfg, "execution_horizon", 15)) if rtc_cfg is not None else 15
            if getattr(inference, "queue_threshold", 30) >= self.chunk_size:
                inference.queue_threshold = min(exec_h, self.chunk_size - 10)
        robot_config = getattr(rollout_config, "robot", None)
        if robot_config is not None and getattr(robot_config, "use_degrees", None) is not True:
            robot_config.use_degrees = True
        relative = getattr(robot_config, "max_relative_target", None)
        if relative is None and robot_config is not None:
            relative = {
                "shoulder_pan": 45.0,
                "shoulder_lift": 45.0,
                "elbow_flex": 45.0,
                "wrist_flex": 45.0,
                "wrist_roll": 45.0,
                "gripper": 50.0,
            }
            robot_config.max_relative_target = relative
        if not isinstance(relative, dict):
            raise ValueError(
                "Video-VAM rollout requires robot.max_relative_target as a complete per-motor dictionary; "
                "a scalar would incorrectly assume uniform degree/range_0_100 units"
            )
        expected = {name.removesuffix(".pos") for name in ACTION_NAMES}
        missing = sorted(expected - set(relative))
        if missing:
            raise ValueError(f"robot.max_relative_target is missing motors: {missing}")
        values = [relative[name] for name in expected]
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in values):
            raise ValueError("robot.max_relative_target values must be finite and positive")

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(30))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def get_optimizer_preset(self):
        raise NotImplementedError("VideoVAMPolicy is inference-only")

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        expected_inputs = {self.camera_key, OBS_STATE}
        if set(self.input_features) != expected_inputs:
            raise ValueError(f"Video-VAM inputs must be exactly {sorted(expected_inputs)}")
        if set(self.output_features) != {ACTION}:
            raise ValueError("Video-VAM output must be action")
