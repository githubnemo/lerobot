# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
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

"""
XQC (Cross-Q with Corrections) configuration.

XQC is SAC with three architectural modifications:
1. Batch Normalization (BN) on inputs and after every linear layer (BN → ReLU ordering).
   Uses joined forward pass concatenating (s,a) and (s',a') for shared BN statistics.
2. Weight Normalization (WN) — after each gradient step, project all dense layer weights
   onto the unit sphere.
3. Distributional Cross-Entropy (CE) Loss — replace MSE Bellman error with C51-style
   categorical critic. Reward normalization via running std of returns.

Reference: XQC paper (Bhatt et al.)
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.sac.configuration_sac import (
    ActorNetworkConfig,
    CriticNetworkConfig,
    SACConfig,
)


@PreTrainedConfig.register_subclass("xqc")
@dataclass
class XQCConfig(SACConfig):
    """XQC (Cross-Q with Corrections) configuration.

    Extends SAC with:
    - C51 distributional critic (categorical cross-entropy loss)
    - Batch normalization in critic and actor networks
    - Weight normalization after gradient steps
    - Reward normalization via running std of returns

    All SAC fields are inherited (encoder, optimizer, discount, etc.).
    """

    # ── C51 distributional parameters ──
    # Number of atoms in the categorical distribution
    num_atoms: int = 101
    # Lower bound of the categorical support
    v_min: float = -5.0
    # Upper bound of the categorical support
    v_max: float = 5.0

    # ── Weight normalization ──
    # Whether to project Linear weights onto unit sphere after each gradient step
    use_weight_normalization: bool = True

    # ── Reward normalization ──
    # Whether to normalize rewards by running std of returns: r̂ = r / σ(R)
    use_reward_normalization: bool = True

    # ── XQC defaults (override SAC defaults) ──
    # XQC uses 4-block networks with larger hidden dims
    critic_network_kwargs: CriticNetworkConfig = field(
        default_factory=lambda: CriticNetworkConfig(
            hidden_dims=[512, 512, 512, 512],
            activate_final=True,
        )
    )
    actor_network_kwargs: ActorNetworkConfig = field(
        default_factory=lambda: ActorNetworkConfig(
            hidden_dims=[256, 256, 256, 256],
            activate_final=True,
        )
    )

    # Policy update delay: update actor every N critic updates (paper uses 3)
    policy_update_freq: int = 3

    # XQC paper uses lower initial temperature
    temperature_init: float = 0.01

    # UTD ratio of 2 (paper default)
    utd_ratio: int = 2

    # Disable torch.compile by default (BN + joined forward can cause issues)
    use_torch_compile: bool = False
