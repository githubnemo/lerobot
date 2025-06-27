#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from ..config import RobotConfig
from .config_so101_follower import SO101FollowerConfig


@RobotConfig.register_subclass("so101_follower_end_effector")
@dataclass
class SO101FollowerEndEffectorConfig(SO101FollowerConfig):
    """Configuration for the SO101FollowerEndEffector robot."""

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(  # bounds for the end-effector in x,y,z direction
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50  # maximum gripper position that the gripper will be open at

    end_effector_step_sizes: dict[str, float] = (
        field(  # maximum step size for the end-effector in x,y,z direction
            default_factory=lambda: {
                "x": 0.02,
                "y": 0.02,
                "z": 0.02,
            }
        )
    )
