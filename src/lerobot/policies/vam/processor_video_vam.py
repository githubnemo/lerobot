"""No-normalization processor pair for Video-VAM rollout.

The SmolExpert decoder owns train-split state normalization and physical-action
denormalization. Adding LeRobot normalizer steps here would be a safety bug.
"""

from __future__ import annotations

from typing import Any

from lerobot.lerobot_types import PolicyAction
from lerobot.processor import (
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    make_policy_processor_pipelines,
)

from .configuration_video_vam import VideoVAMConfig


def make_video_vam_pre_post_processors(
    config: VideoVAMConfig,
    dataset_stats: dict | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    del dataset_stats
    return make_policy_processor_pipelines(
        input_steps=[
            RenameObservationsProcessorStep(rename_map={}),
            DeviceProcessorStep(device=config.device),
        ],
        output_steps=[DeviceProcessorStep(device="cpu")],
    )
