#!/usr/bin/env python

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
XQC (Cross-Q with Corrections) policy implementation.

XQC is SAC with three architectural modifications:
1. Batch Normalization (BN) — applied on raw input and after every linear layer,
   before ReLU. Uses joined forward pass for (s,a) and (s',a') BN statistics.
2. Weight Normalization (WN) — after each gradient step, project all dense layer
   weights onto the unit sphere.
3. Distributional Cross-Entropy (CE) Loss — C51-style categorical critic
   (101 atoms, support [-5, 5]) replacing MSE Bellman error.

Reference: Bhatt et al., CrossQ / XQC.
"""

import math
from dataclasses import asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.sac.configuration_sac import is_image_feature
from lerobot.policies.sac.modeling_sac import (
    MLP,
    CriticEnsemble,
    CriticHead,
    Policy,
    SACObservationEncoder,
    SACPolicy,
    orthogonal_init,
)
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.constants import ACTION

from .configuration_xqc import XQCConfig

DISCRETE_DIMENSION_INDEX = -1  # Gripper is always the last dimension


# ═══════════════════════════════════════════════════════════════════════════════
# Running statistics for reward normalization
# ═══════════════════════════════════════════════════════════════════════════════


class RunningMeanStd(nn.Module):
    """Welford online algorithm for tracking running mean and variance.

    Used for reward normalization: r̂ = r / σ(R).
    Registered as a module so state is saved/loaded with the model.
    """

    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))
        self.epsilon = epsilon

    def update(self, x: Tensor) -> None:
        """Update running statistics with a batch of values."""
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False) if x.numel() > 1 else torch.zeros_like(batch_mean)
        batch_count = x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count.clamp(min=1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count.clamp(min=1)

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count.clamp(min=1))
        self.count.copy_(total_count)

    @property
    def std(self) -> Tensor:
        return (self.var + self.epsilon).sqrt()

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize values by running std: x̂ = x / σ."""
        return x / self.std


# ═══════════════════════════════════════════════════════════════════════════════
# XQC Network Components
# ═══════════════════════════════════════════════════════════════════════════════


class XQCMlpBlock(nn.Module):
    """MLP block with Batch Normalization for XQC.

    Architecture: BN(input) → [Linear → BN → ReLU] × N

    Key design choice: BN is applied BEFORE activation (BN → ReLU),
    which the XQC paper found to outperform ReLU → BN (contrary to CrossQ).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activate_final: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # BN on raw input
        layers.append(nn.BatchNorm1d(input_dim))

        in_dim = input_dim
        for idx, out_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(hidden_dims) - 1
            if not is_last or activate_final:
                layers.append(nn.BatchNorm1d(out_dim))
                layers.append(nn.ReLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class XQCCriticHead(nn.Module):
    """Distributional critic head for XQC.

    Uses XQCMlpBlock backbone and outputs logits over a categorical distribution
    (C51-style) instead of a single Q-value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_atoms: int = 101,
        activate_final: bool = True,
        # Accept but ignore SAC-specific kwargs for compatibility with asdict()
        final_activation: str | None = None,
    ):
        super().__init__()
        self.num_atoms = num_atoms

        self.net = XQCMlpBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activate_final=activate_final,
        )
        self.output_layer = nn.Linear(hidden_dims[-1], num_atoms)
        orthogonal_init()(self.output_layer.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Returns raw logits of shape [batch_size, num_atoms]."""
        return self.output_layer(self.net(x))


class XQCCriticEnsemble(nn.Module):
    """Critic ensemble for XQC with support for joined BN forward pass.

    The key trick from CrossQ/XQC: concatenate current (s,a) and next (s',a')
    into one batch before the forward pass so that BatchNorm computes statistics
    over both, then split the outputs back.

    For standard forward (inference, actor loss), it behaves like CriticEnsemble
    but returns logits instead of scalar Q-values.
    """

    def __init__(
        self,
        encoder: SACObservationEncoder,
        ensemble: list[XQCCriticHead],
        support: Tensor,
    ):
        super().__init__()
        self.encoder = encoder
        self.critics = nn.ModuleList(ensemble)
        self.register_buffer("support", support)

    def forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Standard forward returning expected Q-values [num_critics, batch_size].

        Used for actor loss and inference.
        """
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)

        q_values = []
        for critic in self.critics:
            logits = critic(inputs)
            probs = F.softmax(logits, dim=-1)
            q = (probs * self.support).sum(-1)
            q_values.append(q)

        return torch.stack(q_values, dim=0)

    def forward_logits(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        observation_features: Tensor | None = None,
    ) -> list[Tensor]:
        """Forward returning raw logits for each critic.

        Returns list of [batch_size, num_atoms] tensors, one per critic.
        Used internally for loss computation.
        """
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}

        obs_enc = self.encoder(observations, cache=observation_features)
        inputs = torch.cat([obs_enc, actions], dim=-1)

        return [critic(inputs) for critic in self.critics]

    def joined_forward_logits(
        self,
        obs_enc_current: Tensor,
        actions_current: Tensor,
        obs_enc_next: Tensor,
        actions_next: Tensor,
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Joined BN forward pass — the key XQC/CrossQ trick.

        Concatenates current and next (obs, action) pairs into one batch
        for shared BatchNorm statistics, then splits the results back.

        Args:
            obs_enc_current: Encoded current observations [B, D]
            actions_current: Current actions [B, A]
            obs_enc_next: Encoded next observations [B, D]
            actions_next: Next actions [B, A]

        Returns:
            Tuple of (current_logits, next_logits) where each is a list of
            [B, num_atoms] tensors, one per critic.
        """
        inputs_current = torch.cat([obs_enc_current, actions_current], dim=-1)
        inputs_next = torch.cat([obs_enc_next, actions_next], dim=-1)

        # Concatenate into doubled batch for joined BN statistics
        joined = torch.cat([inputs_current, inputs_next], dim=0)
        batch_size = inputs_current.shape[0]

        current_logits = []
        next_logits = []
        for critic in self.critics:
            logits = critic(joined)
            current_logits.append(logits[:batch_size])
            next_logits.append(logits[batch_size:])

        return current_logits, next_logits


class XQCActorMlp(nn.Module):
    """Actor MLP with Batch Normalization for XQC.

    Same BN architecture as the critic: BN(input) → [Linear → BN → ReLU] × N.
    This is wrapped in an nn.Module with a `.net` attribute so that the existing
    Policy class can introspect it to find the last Linear layer's output dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activate_final: bool = True,
    ):
        super().__init__()
        self.block = XQCMlpBlock(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activate_final=activate_final,
        )
        # Expose the inner Sequential as `.net` for compatibility with Policy.__init__
        # which iterates reversed(network.net) to find the last Linear layer
        self.net = self.block.net

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ═══════════════════════════════════════════════════════════════════════════════
# C51 Distributional RL Utilities
# ═══════════════════════════════════════════════════════════════════════════════


def c51_target_projection(
    rewards: Tensor,
    dones: Tensor,
    gamma: float,
    next_logits: Tensor,
    support: Tensor,
    num_atoms: int,
) -> Tensor:
    """Standard C51 categorical projection.

    Projects the distributional Bellman target Tz = r + γz onto the fixed support.

    Args:
        rewards: [B] reward tensor
        dones: [B] done mask (1.0 = terminal)
        gamma: discount factor
        next_logits: [B, num_atoms] logits from target critic
        support: [num_atoms] support vector
        num_atoms: number of atoms

    Returns:
        target_probs: [B, num_atoms] projected target distribution (no grad)
    """
    next_probs = F.softmax(next_logits, dim=-1)
    delta_z = (support[-1] - support[0]) / (num_atoms - 1)

    # Tz = r + γ * z (for non-terminal states)
    Tz = rewards.unsqueeze(-1) + gamma * (1.0 - dones.unsqueeze(-1)) * support.unsqueeze(0)  # noqa: N806
    Tz = Tz.clamp(support[0].item(), support[-1].item())  # noqa: N806

    # Compute projection indices
    b = (Tz - support[0]) / delta_z
    lower = b.floor().long()
    upper = b.ceil().long()

    lower = lower.clamp(0, num_atoms - 1)
    upper = upper.clamp(0, num_atoms - 1)

    # When b is exactly an integer, lower == upper and both scatter terms are 0.
    # Fix: add 1.0 to the lower scatter weight so full mass lands on that atom.
    eq_mask = (lower == upper).float()

    # Distribute probability mass
    target_probs = torch.zeros_like(next_probs)
    target_probs.scatter_add_(1, lower, next_probs * (upper.float() - b + eq_mask))
    target_probs.scatter_add_(1, upper, next_probs * (b - lower.float()))

    return target_probs


def categorical_cross_entropy_loss(logits: Tensor, target_probs: Tensor) -> Tensor:
    """Cross-entropy loss for distributional critic.

    Has bounded gradients (‖∇ŷ l‖ ≤ √2), unlike MSE which is unbounded.

    Args:
        logits: [B, num_atoms] predicted logits
        target_probs: [B, num_atoms] target distribution (detached)

    Returns:
        Scalar loss
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(-1).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# Weight Normalization Utility
# ═══════════════════════════════════════════════════════════════════════════════


def weight_normalize_(module: nn.Module) -> None:
    """Project all Linear layer weights to unit sphere (per output neuron).

    This is safe due to BatchNorm's scale invariance — normalizing weights
    doesn't change the function output. Keeps the denominator of the
    effective learning rate (ELR = η/‖θ‖²) constant.

    Only normalizes parameters with requires_grad=True to skip frozen encoders.
    """
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                m.weight.data = F.normalize(m.weight.data, dim=1)


# ═══════════════════════════════════════════════════════════════════════════════
# XQC Policy
# ═══════════════════════════════════════════════════════════════════════════════


class XQCPolicy(SACPolicy):
    """XQC (Cross-Q with Corrections) policy.

    Extends SACPolicy with:
    - Distributional (C51) critic with categorical cross-entropy loss
    - Batch normalization in critic and actor networks
    - Weight normalization after gradient steps
    - Reward normalization via running std of returns
    - Joined BN forward pass for critic training

    Inherits from SACPolicy: encoder initialization, temperature,
    target network updates, select_action, forward dispatcher, etc.
    """

    config_class = XQCConfig
    name = "xqc"

    def __init__(self, config: XQCConfig | None = None):
        # Don't call SACPolicy.__init__ directly — we need to intercept
        # _init_critics and _init_actor before they're called.
        # Instead, call PreTrainedPolicy.__init__ and set up manually.
        from lerobot.policies.pretrained import PreTrainedPolicy

        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config

        # Build the C51 support vector
        self.register_buffer(
            "support",
            torch.linspace(config.v_min, config.v_max, config.num_atoms),
        )

        # Reward normalization tracker
        self.reward_running_stats = RunningMeanStd() if config.use_reward_normalization else None

        # Initialize all components (encoder is inherited, critics/actor are overridden)
        continuous_action_dim = config.output_features[ACTION].shape[0]
        self._init_encoders()
        self._init_critics(continuous_action_dim)
        self._init_actor(continuous_action_dim)
        self._init_temperature()

    def _init_critics(self, continuous_action_dim: int) -> None:
        """Build distributional critic ensemble with XQC architecture."""
        input_dim = self.encoder_critic.output_dim + continuous_action_dim

        critic_kwargs = asdict(self.config.critic_network_kwargs)
        # Remove SAC-specific kwargs that XQCCriticHead doesn't use
        critic_kwargs.pop("final_activation", None)

        heads = [
            XQCCriticHead(
                input_dim=input_dim,
                num_atoms=self.config.num_atoms,
                **critic_kwargs,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = XQCCriticEnsemble(
            encoder=self.encoder_critic,
            ensemble=heads,
            support=self.support,
        )

        target_heads = [
            XQCCriticHead(
                input_dim=input_dim,
                num_atoms=self.config.num_atoms,
                **critic_kwargs,
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = XQCCriticEnsemble(
            encoder=self.encoder_critic,
            ensemble=target_heads,
            support=self.support,
        )
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        # torch.compile is disabled by default for XQC (BN compatibility)
        if self.config.use_torch_compile:
            self.critic_ensemble = torch.compile(self.critic_ensemble)
            self.critic_target = torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    def _init_actor(self, continuous_action_dim: int) -> None:
        """Initialize actor with XQC BN architecture."""
        self.actor = Policy(
            encoder=self.encoder_actor,
            network=XQCActorMlp(
                input_dim=self.encoder_actor.output_dim,
                **asdict(self.config.actor_network_kwargs),
            ),
            action_dim=continuous_action_dim,
            encoder_is_shared=self.shared_encoder,
            **asdict(self.config.policy_kwargs),
        )

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            dim = continuous_action_dim + (1 if self.config.num_discrete_actions is not None else 0)
            self.target_entropy = -np.prod(dim) / 2

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action for inference/evaluation.

        Overrides SACPolicy to temporarily set the actor to eval mode
        so that BatchNorm uses running statistics (not batch statistics),
        which also avoids the batch_size=1 error during inference.
        """
        was_training = self.actor.training
        self.actor.eval()
        try:
            return super().select_action(batch)
        finally:
            if was_training:
                self.actor.train()

    def critic_forward(
        self,
        observations: dict[str, Tensor],
        actions: Tensor,
        use_target: bool = False,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Forward pass returning expected Q-values [num_critics, batch_size].

        For XQC, this computes Q = E[Z] = Σ p_i * z_i from the categorical distribution.
        Compatible with SAC's actor loss and temperature loss which expect scalar Q-values.
        """
        critics = self.critic_target if use_target else self.critic_ensemble
        return critics(observations, actions, observation_features)

    def compute_loss_critic(
        self,
        observations,
        actions,
        rewards,
        next_observations,
        done,
        observation_features: Tensor | None = None,
        next_observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute distributional critic loss with joined BN forward pass.

        Key differences from SAC:
        1. Reward normalization: r̂ = r / σ(R) to keep Q-values within [-v_min, v_max]
        2. Joined BN forward: concat (s,a) and (s',a') for shared BN statistics
        3. C51 categorical projection instead of scalar TD target
        4. Cross-entropy loss instead of MSE
        """
        # ── 1. Reward normalization ──
        if self.reward_running_stats is not None:
            self.reward_running_stats.update(rewards)
            rewards_normalized = self.reward_running_stats.normalize(rewards)
        else:
            rewards_normalized = rewards

        # ── 2. Get next actions from actor ──
        with torch.no_grad():
            next_action_preds, next_log_probs, _ = self.actor(
                next_observations, next_observation_features
            )

        # ── 3. Encode observations ──
        device = get_device_from_parameters(self)
        observations = {k: v.to(device) for k, v in observations.items()}
        next_observations = {k: v.to(device) for k, v in next_observations.items()}

        obs_enc = self.encoder_critic(observations, cache=observation_features)
        next_obs_enc = self.encoder_critic(next_observations, cache=next_observation_features)

        # Handle discrete actions (strip gripper dimension for continuous critic)
        if self.config.num_discrete_actions is not None:
            actions = actions[:, :DISCRETE_DIMENSION_INDEX]

        # ── 4. Joined BN forward pass ──
        current_logits_list, next_logits_list = self.critic_ensemble.joined_forward_logits(
            obs_enc_current=obs_enc,
            actions_current=actions,
            obs_enc_next=next_obs_enc.detach(),
            actions_next=next_action_preds.detach(),
        )

        # ── 5. C51 target from target network ──
        with torch.no_grad():
            target_logits_list = self.critic_target.forward_logits(
                next_observations, next_action_preds, next_observation_features
            )

            # Min over target critics (on expected Q-values)
            target_q_values = []
            for logits in target_logits_list:
                probs = F.softmax(logits, dim=-1)
                q = (probs * self.support).sum(-1)
                target_q_values.append(q)
            target_q_stack = torch.stack(target_q_values, dim=0)

            # Subsample critics if configured
            if self.config.num_subsample_critics is not None:
                indices = torch.randperm(self.config.num_critics)[:self.config.num_subsample_critics]
                target_q_stack = target_q_stack[indices]
                # Use the corresponding logits for projection
                min_idx = target_q_stack.min(dim=0)[1]
                # Select logits from the critic with minimum Q-value per sample
                selected_target_logits = []
                subset_logits = [target_logits_list[i] for i in indices]
                for b in range(min_idx.shape[0]):
                    selected_target_logits.append(subset_logits[min_idx[b]][b])
                target_logits_for_projection = torch.stack(selected_target_logits, dim=0)
            else:
                # Use the critic with minimum expected Q-value for each sample
                min_idx = target_q_stack.min(dim=0)[1]  # [batch_size]
                selected_target_logits = []
                for b in range(min_idx.shape[0]):
                    selected_target_logits.append(target_logits_list[min_idx[b]][b])
                target_logits_for_projection = torch.stack(selected_target_logits, dim=0)

            # Backup entropy: subtract temperature * log_pi from reward
            effective_rewards = rewards_normalized
            if self.config.use_backup_entropy:
                effective_rewards = rewards_normalized - self.temperature * next_log_probs

            # C51 categorical projection
            target_probs = c51_target_projection(
                rewards=effective_rewards,
                dones=done,
                gamma=self.config.discount,
                next_logits=target_logits_for_projection,
                support=self.support,
                num_atoms=self.config.num_atoms,
            )

        # ── 6. Cross-entropy loss for each critic ──
        total_loss = torch.tensor(0.0, device=device)
        for logits in current_logits_list:
            total_loss = total_loss + categorical_cross_entropy_loss(logits, target_probs.detach())

        return total_loss

    def compute_loss_actor(
        self,
        observations,
        observation_features: Tensor | None = None,
    ) -> Tensor:
        """Compute actor loss using expected Q-values from distributional critic.

        Same as SAC's actor loss but Q-values come from E[Z] of the categorical distribution.
        """
        actions_pi, log_probs, _ = self.actor(observations, observation_features)

        # critic_forward returns expected Q-values via the support
        q_preds = self.critic_forward(
            observations=observations,
            actions=actions_pi,
            use_target=False,
            observation_features=observation_features,
        )
        min_q_preds = q_preds.min(dim=0)[0]

        actor_loss = ((self.temperature * log_probs) - min_q_preds).mean()
        return actor_loss

    def on_optimizer_step(self, component: str) -> None:
        """Called by the learner after each optimizer step.

        Applies weight normalization to the specified component.

        Args:
            component: One of "critic", "actor", "temperature", "discrete_critic"
        """
        if not self.config.use_weight_normalization:
            return

        if component == "critic":
            weight_normalize_(self.critic_ensemble)
        elif component == "actor":
            weight_normalize_(self.actor)
        # Temperature and discrete_critic don't need weight normalization
