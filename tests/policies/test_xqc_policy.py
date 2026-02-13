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

"""
Tests for XQC (Cross-Q with Corrections) policy.

Covers:
- C51 distributional utilities (projection, cross-entropy loss)
- RunningMeanStd reward normalization
- Weight normalization
- XQC network architecture (BN-based MLP, distributional critic)
- Full training loop (critic, actor, temperature)
- Inference (select_action with batch_size=1, eval mode)
- Save/load round-trip
- Gradient flow
"""

import math

import pytest
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.xqc.configuration_xqc import XQCConfig
from lerobot.policies.xqc.modeling_xqc import (
    RunningMeanStd,
    XQCActorMlp,
    XQCCriticEnsemble,
    XQCCriticHead,
    XQCMlpBlock,
    XQCPolicy,
    c51_target_projection,
    categorical_cross_entropy_loss,
    weight_normalize_,
)
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE
from lerobot.utils.random_utils import seeded_context, set_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures and helpers
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def set_random_seed():
    set_seed(42)


def create_xqc_config(
    state_dim: int = 10,
    action_dim: int = 6,
) -> XQCConfig:
    """Create a minimal XQC config for testing (small networks for speed)."""
    from lerobot.policies.sac.configuration_sac import ActorNetworkConfig, CriticNetworkConfig

    return XQCConfig(
        input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))},
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        dataset_stats={
            OBS_STATE: {"min": [0.0] * state_dim, "max": [1.0] * state_dim},
            ACTION: {"min": [0.0] * action_dim, "max": [1.0] * action_dim},
        },
        # Use small networks for fast tests
        critic_network_kwargs=CriticNetworkConfig(hidden_dims=[64, 64], activate_final=True),
        actor_network_kwargs=ActorNetworkConfig(hidden_dims=[64, 64], activate_final=True),
        use_torch_compile=False,
    )


def create_xqc_config_with_visual(
    state_dim: int = 10,
    action_dim: int = 6,
) -> XQCConfig:
    """Create XQC config with visual input for testing."""
    config = create_xqc_config(state_dim=state_dim, action_dim=action_dim)
    config.input_features[OBS_IMAGE] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84))
    config.dataset_stats[OBS_IMAGE] = {
        "mean": torch.randn(3, 1, 1),
        "std": torch.randn(3, 1, 1),
    }
    config.state_encoder_hidden_dim = 32
    config.latent_dim = 32
    config.validate_features()
    return config


def create_dummy_state(batch_size: int, state_dim: int = 10) -> dict[str, Tensor]:
    return {OBS_STATE: torch.randn(batch_size, state_dim)}


def create_dummy_with_visual(batch_size: int, state_dim: int = 10) -> dict[str, Tensor]:
    return {
        OBS_IMAGE: torch.randn(batch_size, 3, 84, 84),
        OBS_STATE: torch.randn(batch_size, state_dim),
    }


def create_train_batch(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 6
) -> dict[str, Tensor]:
    return {
        ACTION: torch.randn(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_state(batch_size, state_dim),
        "next_state": create_dummy_state(batch_size, state_dim),
        "done": torch.zeros(batch_size),
    }


def create_train_batch_with_visual(
    batch_size: int = 8, state_dim: int = 10, action_dim: int = 6
) -> dict[str, Tensor]:
    return {
        ACTION: torch.randn(batch_size, action_dim),
        "reward": torch.randn(batch_size),
        "state": create_dummy_with_visual(batch_size, state_dim),
        "next_state": create_dummy_with_visual(batch_size, state_dim),
        "done": torch.zeros(batch_size),
    }


def make_optimizers(policy: XQCPolicy) -> dict[str, torch.optim.Optimizer]:
    """Create optimizers for the XQC policy."""
    return {
        "actor": torch.optim.Adam(
            params=[
                p
                for n, p in policy.actor.named_parameters()
                if not policy.config.shared_encoder or not n.startswith("encoder")
            ],
            lr=policy.config.actor_lr,
        ),
        "critic": torch.optim.Adam(
            params=policy.critic_ensemble.parameters(),
            lr=policy.config.critic_lr,
        ),
        "temperature": torch.optim.Adam(
            params=[policy.log_alpha],
            lr=policy.config.temperature_lr,
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# C51 Distributional Utilities
# ═══════════════════════════════════════════════════════════════════════════════


class TestC51TargetProjection:
    """Test the C51 categorical projection."""

    def test_output_shape(self):
        support = torch.linspace(-5, 5, 101)
        target = c51_target_projection(
            rewards=torch.randn(8),
            dones=torch.zeros(8),
            gamma=0.99,
            next_logits=torch.randn(8, 101),
            support=support,
            num_atoms=101,
        )
        assert target.shape == (8, 101)

    def test_probabilities_sum_to_one(self):
        support = torch.linspace(-5, 5, 101)
        target = c51_target_projection(
            rewards=torch.randn(8),
            dones=torch.zeros(8),
            gamma=0.99,
            next_logits=torch.randn(8, 101),
            support=support,
            num_atoms=101,
        )
        sums = target.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_terminal_states_sum_to_one(self):
        """When done=1, all mass should be at the reward location."""
        support = torch.linspace(-5, 5, 101)
        target = c51_target_projection(
            rewards=torch.tensor([0.0, 1.0, -1.0]),
            dones=torch.ones(3),
            gamma=0.99,
            next_logits=torch.randn(3, 101),
            support=support,
            num_atoms=101,
        )
        sums = target.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_terminal_state_concentrates_at_reward(self):
        """Terminal state should place mass around the reward atom."""
        support = torch.linspace(-5, 5, 101)
        reward = 2.0
        target = c51_target_projection(
            rewards=torch.tensor([reward]),
            dones=torch.ones(1),
            gamma=0.99,
            next_logits=torch.randn(1, 101),
            support=support,
            num_atoms=101,
        )
        # Find the atom closest to the reward
        expected_atom = ((reward - (-5)) / 0.1)  # delta_z = 0.1
        expected_idx = int(expected_atom)
        # Most mass should be around this atom
        assert target[0, expected_idx].item() > 0.0 or target[0, min(expected_idx + 1, 100)].item() > 0.0

    def test_non_negative_probabilities(self):
        support = torch.linspace(-5, 5, 101)
        target = c51_target_projection(
            rewards=torch.randn(16),
            dones=torch.bernoulli(torch.full((16,), 0.5)),
            gamma=0.99,
            next_logits=torch.randn(16, 101),
            support=support,
            num_atoms=101,
        )
        assert (target >= -1e-7).all()

    def test_different_num_atoms(self):
        """Test with non-default number of atoms."""
        num_atoms = 51
        support = torch.linspace(-10, 10, num_atoms)
        target = c51_target_projection(
            rewards=torch.randn(4),
            dones=torch.zeros(4),
            gamma=0.99,
            next_logits=torch.randn(4, num_atoms),
            support=support,
            num_atoms=num_atoms,
        )
        assert target.shape == (4, num_atoms)
        assert torch.allclose(target.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestCategoricalCrossEntropyLoss:
    """Test the cross-entropy loss for distributional critic."""

    def test_output_is_scalar(self):
        logits = torch.randn(8, 101)
        target_probs = torch.softmax(torch.randn(8, 101), dim=-1)
        loss = categorical_cross_entropy_loss(logits, target_probs)
        assert loss.shape == ()

    def test_loss_is_positive(self):
        logits = torch.randn(8, 101)
        target_probs = torch.softmax(torch.randn(8, 101), dim=-1)
        loss = categorical_cross_entropy_loss(logits, target_probs)
        assert loss.item() > 0

    def test_loss_is_zero_for_perfect_prediction(self):
        """When logits perfectly match target, loss should be minimal."""
        target_probs = torch.softmax(torch.randn(8, 101), dim=-1)
        # Make logits match target_probs exactly (log of target = logits for softmax)
        logits = torch.log(target_probs + 1e-10)
        loss = categorical_cross_entropy_loss(logits, target_probs)
        # Not exactly zero due to numerical precision, but should be very small
        # compared to random logits
        random_loss = categorical_cross_entropy_loss(torch.randn(8, 101), target_probs)
        assert loss.item() < random_loss.item()

    def test_gradient_exists(self):
        logits = torch.randn(8, 101, requires_grad=True)
        target_probs = torch.softmax(torch.randn(8, 101), dim=-1)
        loss = categorical_cross_entropy_loss(logits, target_probs)
        loss.backward()
        assert logits.grad is not None
        assert logits.grad.shape == (8, 101)


# ═══════════════════════════════════════════════════════════════════════════════
# RunningMeanStd
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunningMeanStd:
    """Test the running statistics tracker for reward normalization."""

    def test_initial_values(self):
        rms = RunningMeanStd()
        assert rms.mean.item() == 0.0
        assert rms.var.item() == 1.0
        assert rms.count.item() == 0.0

    def test_single_batch_update(self):
        rms = RunningMeanStd()
        x = torch.randn(100)
        rms.update(x)
        assert rms.count.item() == 100
        assert abs(rms.mean.item() - x.mean().item()) < 0.01

    def test_multiple_batch_updates(self):
        rms = RunningMeanStd()
        all_data = []
        for _ in range(10):
            x = torch.randn(50)
            all_data.append(x)
            rms.update(x)

        all_data = torch.cat(all_data)
        assert rms.count.item() == 500
        assert abs(rms.mean.item() - all_data.mean().item()) < 0.05

    def test_normalize(self):
        rms = RunningMeanStd()
        x = torch.randn(1000) * 5 + 3  # mean=3, std=5
        rms.update(x)
        normalized = rms.normalize(x)
        # After normalization by std, the std of normalized should be ~1
        assert abs(normalized.std().item() - 1.0) < 0.2

    def test_std_never_zero(self):
        """Std should never be exactly zero due to epsilon."""
        rms = RunningMeanStd()
        rms.update(torch.zeros(10))
        assert rms.std.item() > 0

    def test_state_dict_persistence(self):
        """RunningMeanStd state should be saved/loaded with model."""
        rms = RunningMeanStd()
        rms.update(torch.randn(100) * 3)

        state = rms.state_dict()
        rms2 = RunningMeanStd()
        rms2.load_state_dict(state)

        assert torch.allclose(rms.mean, rms2.mean)
        assert torch.allclose(rms.var, rms2.var)
        assert torch.allclose(rms.count, rms2.count)


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Normalization
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeightNormalization:
    """Test weight normalization (projecting weights onto unit sphere)."""

    def test_linear_weights_normalized(self):
        linear = nn.Linear(10, 5)
        weight_normalize_(linear)
        norms = linear.weight.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_preserves_bias(self):
        linear = nn.Linear(10, 5)
        original_bias = linear.bias.data.clone()
        weight_normalize_(linear)
        assert torch.allclose(linear.bias.data, original_bias)

    def test_nested_module(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )
        weight_normalize_(model)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                norms = m.weight.data.norm(dim=1)
                assert torch.allclose(norms, torch.ones(norms.shape[0]), atol=1e-5)

    def test_skips_frozen_params(self):
        linear = nn.Linear(10, 5)
        linear.weight.requires_grad = False
        original_weight = linear.weight.data.clone()
        weight_normalize_(linear)
        # Frozen weights should not be modified
        assert torch.allclose(linear.weight.data, original_weight)

    def test_idempotent(self):
        """Applying weight normalization twice should give the same result."""
        linear = nn.Linear(10, 5)
        weight_normalize_(linear)
        weights_after_first = linear.weight.data.clone()
        weight_normalize_(linear)
        assert torch.allclose(linear.weight.data, weights_after_first)

    def test_batchnorm_untouched(self):
        """Weight normalization should not affect BatchNorm layers."""
        bn = nn.BatchNorm1d(10)
        original_weight = bn.weight.data.clone()
        weight_normalize_(bn)
        assert torch.allclose(bn.weight.data, original_weight)


# ═══════════════════════════════════════════════════════════════════════════════
# XQC Network Architecture
# ═══════════════════════════════════════════════════════════════════════════════


class TestXQCMlpBlock:
    """Test the BN-based MLP block."""

    def test_output_shape(self):
        block = XQCMlpBlock(input_dim=10, hidden_dims=[64, 64])
        x = torch.randn(4, 10)
        assert block(x).shape == (4, 64)

    def test_contains_batchnorm(self):
        block = XQCMlpBlock(input_dim=10, hidden_dims=[64, 64])
        bn_count = sum(1 for m in block.modules() if isinstance(m, nn.BatchNorm1d))
        # Input BN + one BN per hidden layer
        assert bn_count >= 2

    def test_contains_relu(self):
        block = XQCMlpBlock(input_dim=10, hidden_dims=[64, 64])
        relu_count = sum(1 for m in block.modules() if isinstance(m, nn.ReLU))
        assert relu_count >= 1

    def test_no_activation_on_final_when_disabled(self):
        block = XQCMlpBlock(input_dim=10, hidden_dims=[64, 64], activate_final=False)
        # Last layer should be Linear (no BN/ReLU after it)
        last_layer = list(block.net.children())[-1]
        assert isinstance(last_layer, nn.Linear)


class TestXQCCriticHead:
    """Test the distributional critic head."""

    def test_output_shape(self):
        head = XQCCriticHead(input_dim=16, hidden_dims=[64, 64], num_atoms=101)
        x = torch.randn(4, 16)
        logits = head(x)
        assert logits.shape == (4, 101)

    def test_custom_num_atoms(self):
        head = XQCCriticHead(input_dim=16, hidden_dims=[64, 64], num_atoms=51)
        x = torch.randn(4, 16)
        logits = head(x)
        assert logits.shape == (4, 51)


class TestXQCCriticEnsemble:
    """Test the critic ensemble with joined BN forward pass."""

    @pytest.fixture
    def ensemble_setup(self):
        from lerobot.policies.sac.modeling_sac import SACObservationEncoder

        config = create_xqc_config(state_dim=10, action_dim=6)
        encoder = SACObservationEncoder(config)
        support = torch.linspace(-5, 5, 101)

        heads = [
            XQCCriticHead(input_dim=encoder.output_dim + 6, hidden_dims=[64, 64], num_atoms=101)
            for _ in range(2)
        ]
        ensemble = XQCCriticEnsemble(encoder=encoder, ensemble=heads, support=support)
        return ensemble

    def test_forward_returns_q_values(self, ensemble_setup):
        obs = create_dummy_state(4, 10)
        actions = torch.randn(4, 6)
        q_vals = ensemble_setup(obs, actions)
        assert q_vals.shape == (2, 4)  # [num_critics, batch_size]

    def test_forward_logits_returns_list(self, ensemble_setup):
        obs = create_dummy_state(4, 10)
        actions = torch.randn(4, 6)
        logits_list = ensemble_setup.forward_logits(obs, actions)
        assert len(logits_list) == 2
        assert all(l.shape == (4, 101) for l in logits_list)

    def test_joined_forward_logits(self, ensemble_setup):
        """Test the key XQC/CrossQ trick: joined BN forward pass."""
        from lerobot.policies.sac.modeling_sac import SACObservationEncoder

        obs = create_dummy_state(4, 10)
        next_obs = create_dummy_state(4, 10)
        actions = torch.randn(4, 6)
        next_actions = torch.randn(4, 6)

        # Encode observations manually
        obs_enc = ensemble_setup.encoder(obs)
        next_obs_enc = ensemble_setup.encoder(next_obs)

        current_logits, next_logits = ensemble_setup.joined_forward_logits(
            obs_enc, actions, next_obs_enc, next_actions
        )

        assert len(current_logits) == 2
        assert len(next_logits) == 2
        assert all(l.shape == (4, 101) for l in current_logits)
        assert all(l.shape == (4, 101) for l in next_logits)


class TestXQCActorMlp:
    """Test the BN-based actor MLP."""

    def test_output_shape(self):
        mlp = XQCActorMlp(input_dim=10, hidden_dims=[64, 64])
        x = torch.randn(4, 10)
        assert mlp(x).shape == (4, 64)

    def test_has_net_attribute(self):
        """Policy class introspects network.net to find output dim."""
        mlp = XQCActorMlp(input_dim=10, hidden_dims=[64, 64])
        assert hasattr(mlp, "net")
        # Should be able to find a Linear layer by iterating reversed
        found_linear = False
        for layer in reversed(mlp.net):
            if isinstance(layer, nn.Linear):
                found_linear = True
                break
        assert found_linear


# ═══════════════════════════════════════════════════════════════════════════════
# XQCPolicy Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestXQCPolicyInit:
    """Test XQC policy initialization."""

    def test_instantiation(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        assert policy.name == "xqc"

    def test_has_support(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        assert policy.support.shape == (101,)
        assert policy.support[0].item() == -5.0
        assert policy.support[-1].item() == 5.0

    def test_has_reward_running_stats(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        assert policy.reward_running_stats is not None

    def test_no_reward_stats_when_disabled(self):
        config = create_xqc_config()
        config.use_reward_normalization = False
        policy = XQCPolicy(config=config)
        assert policy.reward_running_stats is None

    def test_critic_ensemble_is_distributional(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        assert isinstance(policy.critic_ensemble, XQCCriticEnsemble)
        assert isinstance(policy.critic_target, XQCCriticEnsemble)

    def test_target_entropy(self):
        config = create_xqc_config(action_dim=6)
        policy = XQCPolicy(config=config)
        assert policy.target_entropy == -3.0  # -action_dim / 2

    def test_custom_target_entropy(self):
        config = create_xqc_config()
        config.target_entropy = -2.5
        policy = XQCPolicy(config=config)
        assert policy.target_entropy == -2.5


class TestXQCPolicyForward:
    """Test forward passes and loss computation."""

    @pytest.mark.parametrize("batch_size,state_dim,action_dim", [(4, 6, 6), (8, 10, 4)])
    def test_critic_loss(self, batch_size, state_dim, action_dim):
        config = create_xqc_config(state_dim=state_dim, action_dim=action_dim)
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch(batch_size=batch_size, state_dim=state_dim, action_dim=action_dim)
        output = policy.forward(batch, model="critic")

        assert "loss_critic" in output
        assert output["loss_critic"].shape == ()
        assert not torch.isnan(output["loss_critic"])

    @pytest.mark.parametrize("batch_size,state_dim,action_dim", [(4, 6, 6), (8, 10, 4)])
    def test_actor_loss(self, batch_size, state_dim, action_dim):
        config = create_xqc_config(state_dim=state_dim, action_dim=action_dim)
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch(batch_size=batch_size, state_dim=state_dim, action_dim=action_dim)
        output = policy.forward(batch, model="actor")

        assert "loss_actor" in output
        assert output["loss_actor"].shape == ()
        assert not torch.isnan(output["loss_actor"])

    def test_temperature_loss(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch()
        output = policy.forward(batch, model="temperature")

        assert "loss_temperature" in output
        assert output["loss_temperature"].shape == ()
        assert not torch.isnan(output["loss_temperature"])

    def test_critic_forward_returns_q_values(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        obs = create_dummy_state(4)
        actions = torch.randn(4, 6)
        q_vals = policy.critic_forward(obs, actions)
        assert q_vals.shape == (2, 4)  # [num_critics, batch_size]


class TestXQCPolicyInference:
    """Test inference (select_action)."""

    def test_select_action_single(self):
        config = create_xqc_config(action_dim=6)
        policy = XQCPolicy(config=config)
        policy.train()  # Even in train mode, select_action should work

        with torch.no_grad():
            action = policy.select_action(create_dummy_state(1))
        assert action.shape == (1, 6)

    def test_select_action_batch(self):
        config = create_xqc_config(action_dim=6)
        policy = XQCPolicy(config=config)
        policy.eval()

        with torch.no_grad():
            action = policy.select_action(create_dummy_state(4))
        assert action.shape == (4, 6)

    def test_select_action_eval_mode(self):
        """select_action should use BN running stats, not batch stats."""
        config = create_xqc_config(action_dim=6)
        policy = XQCPolicy(config=config)
        policy.eval()

        with torch.no_grad():
            action = policy.select_action(create_dummy_state(1))
        assert action.shape == (1, 6)
        assert not torch.isnan(action).any()


class TestXQCPolicyTrainingLoop:
    """Test a complete training step (critic → actor → temperature)."""

    @pytest.mark.parametrize("batch_size,state_dim,action_dim", [(4, 6, 6), (2, 10, 10)])
    def test_full_training_step(self, batch_size, state_dim, action_dim):
        config = create_xqc_config(state_dim=state_dim, action_dim=action_dim)
        policy = XQCPolicy(config=config)
        policy.train()
        optimizers = make_optimizers(policy)

        batch = create_train_batch(batch_size=batch_size, state_dim=state_dim, action_dim=action_dim)

        # Critic step
        critic_output = policy.forward(batch, model="critic")
        loss_critic = critic_output["loss_critic"]
        assert loss_critic.shape == ()
        optimizers["critic"].zero_grad()
        loss_critic.backward()
        optimizers["critic"].step()
        policy.on_optimizer_step("critic")

        # Actor step
        actor_output = policy.forward(batch, model="actor")
        loss_actor = actor_output["loss_actor"]
        assert loss_actor.shape == ()
        optimizers["actor"].zero_grad()
        loss_actor.backward()
        optimizers["actor"].step()
        policy.on_optimizer_step("actor")

        # Temperature step
        temp_output = policy.forward(batch, model="temperature")
        loss_temp = temp_output["loss_temperature"]
        assert loss_temp.shape == ()
        optimizers["temperature"].zero_grad()
        loss_temp.backward()
        optimizers["temperature"].step()

        # Inference after training
        policy.eval()
        with torch.no_grad():
            action = policy.select_action(create_dummy_state(batch_size, state_dim))
            assert action.shape == (batch_size, action_dim)

    def test_training_with_visual_input(self):
        config = create_xqc_config_with_visual(state_dim=10, action_dim=6)
        policy = XQCPolicy(config=config)
        policy.train()
        optimizers = make_optimizers(policy)

        batch = create_train_batch_with_visual(batch_size=4, state_dim=10, action_dim=6)

        # Critic step
        critic_output = policy.forward(batch, model="critic")
        optimizers["critic"].zero_grad()
        critic_output["loss_critic"].backward()
        optimizers["critic"].step()
        policy.on_optimizer_step("critic")

        # Actor step
        actor_output = policy.forward(batch, model="actor")
        optimizers["actor"].zero_grad()
        actor_output["loss_actor"].backward()
        optimizers["actor"].step()
        policy.on_optimizer_step("actor")

    def test_multiple_training_steps(self):
        """Test multiple training iterations don't crash."""
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()
        optimizers = make_optimizers(policy)

        for _ in range(5):
            batch = create_train_batch()

            # Critic
            optimizers["critic"].zero_grad()
            loss = policy.forward(batch, model="critic")["loss_critic"]
            loss.backward()
            optimizers["critic"].step()
            policy.on_optimizer_step("critic")

            # Actor
            optimizers["actor"].zero_grad()
            loss = policy.forward(batch, model="actor")["loss_actor"]
            loss.backward()
            optimizers["actor"].step()
            policy.on_optimizer_step("actor")


class TestXQCWeightNormIntegration:
    """Test weight normalization integration with the policy."""

    def test_critic_weights_normalized_after_step(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch()
        loss = policy.forward(batch, model="critic")["loss_critic"]
        optimizer = torch.optim.Adam(policy.critic_ensemble.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy.on_optimizer_step("critic")

        # Check all Linear weights in critic are unit-norm
        for m in policy.critic_ensemble.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                norms = m.weight.data.norm(dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_actor_weights_normalized_after_step(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch()
        loss = policy.forward(batch, model="actor")["loss_actor"]
        optimizer = torch.optim.Adam(
            [p for n, p in policy.actor.named_parameters() if not n.startswith("encoder")],
            lr=1e-3,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy.on_optimizer_step("actor")

        # Check all Linear weights in actor are unit-norm
        for m in policy.actor.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                norms = m.weight.data.norm(dim=1)
                assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_weight_norm_when_disabled(self):
        config = create_xqc_config()
        config.use_weight_normalization = False
        policy = XQCPolicy(config=config)
        policy.train()

        # Store original norms
        original_norms = {}
        for name, m in policy.critic_ensemble.named_modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                original_norms[name] = m.weight.data.norm(dim=1).clone()

        # Train and call on_optimizer_step (should be a no-op)
        batch = create_train_batch()
        loss = policy.forward(batch, model="critic")["loss_critic"]
        optimizer = torch.optim.Adam(policy.critic_ensemble.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        policy.on_optimizer_step("critic")

        # Weights should NOT be unit-norm (optimizer changed them freely)
        any_not_unit = False
        for name, m in policy.critic_ensemble.named_modules():
            if isinstance(m, nn.Linear) and name in original_norms and m.weight.requires_grad:
                norms = m.weight.data.norm(dim=1)
                if not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
                    any_not_unit = True
                    break
        # At least some weights should have changed away from unit norm
        # (unless they happened to stay close by chance, but that's extremely unlikely)
        assert any_not_unit


class TestXQCRewardNormalization:
    """Test reward normalization integration."""

    def test_reward_stats_updated_during_critic_loss(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        assert policy.reward_running_stats.count.item() == 0

        batch = create_train_batch()
        policy.forward(batch, model="critic")

        assert policy.reward_running_stats.count.item() > 0

    def test_no_reward_stats_when_disabled(self):
        config = create_xqc_config()
        config.use_reward_normalization = False
        policy = XQCPolicy(config=config)
        policy.train()

        assert policy.reward_running_stats is None

        # Training should still work
        batch = create_train_batch()
        output = policy.forward(batch, model="critic")
        assert not torch.isnan(output["loss_critic"])


class TestXQCGradientFlow:
    """Test that gradients flow through the distributional critic."""

    def test_critic_gradients(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch()
        loss = policy.forward(batch, model="critic")["loss_critic"]
        loss.backward()

        # Critic parameters should have gradients
        grad_count = sum(
            1
            for p in policy.critic_ensemble.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0

    def test_actor_gradients(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        batch = create_train_batch()
        loss = policy.forward(batch, model="actor")["loss_actor"]
        loss.backward()

        grad_count = sum(
            1
            for p in policy.actor.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert grad_count > 0


class TestXQCSaveLoad:
    """Test saving and loading the XQC policy."""

    def test_save_and_load_roundtrip(self, tmp_path):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)

        # Run a forward pass to populate BN running stats and reward stats
        policy.train()
        batch = create_train_batch()
        policy.forward(batch, model="critic")

        policy.eval()
        save_path = tmp_path / "xqc_test"
        policy.save_pretrained(save_path)

        loaded_policy = XQCPolicy.from_pretrained(save_path, config=config)
        loaded_policy.eval()

        # Check state dicts match
        assert policy.state_dict().keys() == loaded_policy.state_dict().keys()
        for k in policy.state_dict():
            assert torch.allclose(policy.state_dict()[k], loaded_policy.state_dict()[k], atol=1e-6), (
                f"Mismatch in {k}"
            )

    def test_save_load_produces_same_outputs(self, tmp_path):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.eval()

        save_path = tmp_path / "xqc_test"
        policy.save_pretrained(save_path)

        loaded_policy = XQCPolicy.from_pretrained(save_path, config=config)
        loaded_policy.eval()

        batch = create_train_batch(batch_size=4)

        with torch.no_grad():
            with seeded_context(42):
                critic_loss = policy.forward(batch, model="critic")["loss_critic"]
                actor_loss = policy.forward(batch, model="actor")["loss_actor"]
                obs = create_dummy_state(2)
                actions = policy.select_action(obs)

            with seeded_context(42):
                loaded_critic_loss = loaded_policy.forward(batch, model="critic")["loss_critic"]
                loaded_actor_loss = loaded_policy.forward(batch, model="actor")["loss_actor"]
                loaded_obs = create_dummy_state(2)
                loaded_actions = loaded_policy.select_action(loaded_obs)

        assert torch.allclose(critic_loss, loaded_critic_loss)
        assert torch.allclose(actor_loss, loaded_actor_loss)
        assert torch.allclose(actions, loaded_actions)

    def test_reward_stats_persisted(self, tmp_path):
        """RunningMeanStd buffers should be saved and restored."""
        config = create_xqc_config()
        policy = XQCPolicy(config=config)
        policy.train()

        # Update reward stats
        batch = create_train_batch()
        policy.forward(batch, model="critic")
        original_mean = policy.reward_running_stats.mean.clone()
        original_count = policy.reward_running_stats.count.clone()

        save_path = tmp_path / "xqc_reward_stats"
        policy.save_pretrained(save_path)

        loaded_policy = XQCPolicy.from_pretrained(save_path, config=config)
        assert torch.allclose(loaded_policy.reward_running_stats.mean, original_mean)
        assert torch.allclose(loaded_policy.reward_running_stats.count, original_count)


class TestXQCTargetNetworks:
    """Test target network behavior."""

    def test_target_network_initialized_same(self):
        config = create_xqc_config()
        policy = XQCPolicy(config=config)

        for p1, p2 in zip(
            policy.critic_ensemble.parameters(), policy.critic_target.parameters()
        ):
            assert torch.allclose(p1.data, p2.data)

    def test_target_network_update(self):
        config = create_xqc_config()
        config.critic_target_update_weight = 1.0  # Hard update
        policy = XQCPolicy(config=config)
        policy.train()

        # Modify critic ensemble weights
        for p in policy.critic_ensemble.parameters():
            p.data = torch.ones_like(p.data)

        policy.update_target_networks()

        for p in policy.critic_target.parameters():
            assert torch.allclose(p.data, torch.ones_like(p.data))
