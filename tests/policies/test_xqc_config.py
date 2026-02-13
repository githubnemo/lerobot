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

import pytest

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.xqc.configuration_xqc import XQCConfig
from lerobot.utils.constants import ACTION, OBS_STATE


class TestXQCConfigDefaults:
    """Test that XQC config defaults match the paper's hyperparameters."""

    def test_default_initialization(self):
        config = XQCConfig()

        # C51 distributional parameters
        assert config.num_atoms == 101
        assert config.v_min == -5.0
        assert config.v_max == 5.0

        # Weight normalization
        assert config.use_weight_normalization is True

        # Reward normalization
        assert config.use_reward_normalization is True

        # torch.compile disabled by default for XQC
        assert config.use_torch_compile is False

    def test_xqc_overridden_sac_defaults(self):
        config = XQCConfig()

        # XQC uses 4-block networks with larger hidden dims
        assert config.critic_network_kwargs.hidden_dims == [512, 512, 512, 512]
        assert config.actor_network_kwargs.hidden_dims == [256, 256, 256, 256]

        # Policy update delay: every 3 critic updates
        assert config.policy_update_freq == 3

        # Lower initial temperature
        assert config.temperature_init == 0.01

        # UTD ratio of 2
        assert config.utd_ratio == 2

    def test_inherits_from_sac_config(self):
        config = XQCConfig()
        assert isinstance(config, SACConfig)

    def test_inherited_sac_defaults_unchanged(self):
        config = XQCConfig()

        # These should still be SAC defaults
        assert config.discount == 0.99
        assert config.num_critics == 2
        assert config.critic_lr == 3e-4
        assert config.actor_lr == 3e-4
        assert config.temperature_lr == 3e-4
        assert config.critic_target_update_weight == 0.005
        assert config.use_backup_entropy is True
        assert config.grad_clip_norm == 40.0
        assert config.shared_encoder is True


class TestXQCConfigCustomization:
    """Test that XQC config can be customized."""

    def test_custom_distributional_params(self):
        config = XQCConfig(num_atoms=51, v_min=-10.0, v_max=10.0)
        assert config.num_atoms == 51
        assert config.v_min == -10.0
        assert config.v_max == 10.0

    def test_disable_weight_normalization(self):
        config = XQCConfig(use_weight_normalization=False)
        assert config.use_weight_normalization is False

    def test_disable_reward_normalization(self):
        config = XQCConfig(use_reward_normalization=False)
        assert config.use_reward_normalization is False

    def test_custom_network_dims(self):
        from lerobot.policies.sac.configuration_sac import ActorNetworkConfig, CriticNetworkConfig

        config = XQCConfig(
            critic_network_kwargs=CriticNetworkConfig(hidden_dims=[1024, 1024]),
            actor_network_kwargs=ActorNetworkConfig(hidden_dims=[512, 512]),
        )
        assert config.critic_network_kwargs.hidden_dims == [1024, 1024]
        assert config.actor_network_kwargs.hidden_dims == [512, 512]


class TestXQCConfigValidation:
    """Test feature validation (inherited from SACConfig)."""

    def test_validate_features_success(self):
        config = XQCConfig(
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        )
        config.validate_features()

    def test_validate_features_missing_observation(self):
        config = XQCConfig(
            input_features={"wrong_key": PolicyFeature(type=FeatureType.STATE, shape=(10,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        )
        with pytest.raises(ValueError, match="You must provide either 'observation.state' or an image"):
            config.validate_features()

    def test_validate_features_missing_action(self):
        config = XQCConfig(
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(10,))},
            output_features={"wrong_key": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        )
        with pytest.raises(ValueError, match="You must provide 'action' in the output features"):
            config.validate_features()


class TestXQCConfigFactory:
    """Test factory registration."""

    def test_make_policy_config(self):
        from lerobot.policies.factory import make_policy_config

        config = make_policy_config("xqc")
        assert isinstance(config, XQCConfig)

    def test_get_policy_class(self):
        from lerobot.policies.factory import get_policy_class

        cls = get_policy_class("xqc")
        assert cls.__name__ == "XQCPolicy"

    def test_isinstance_ordering_xqc_before_sac(self):
        """XQCConfig is a subclass of SACConfig, so isinstance checks must be ordered correctly."""
        config = XQCConfig()
        assert isinstance(config, XQCConfig)
        assert isinstance(config, SACConfig)
        # The factory should check XQC first
