# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""EXPO Residual Action-Edit Actor + Concave Value Critic with Dr. GRPO Score Centering.

Designed for sparse-reward and learned-detector RL post-training on top of frozen Video-VAM
policies (`Cosmos 3 Edge LoRA` + `SmolExpert` flow-matching decoder).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor


def compute_dr_grpo_advantages(
    rewards: list[float] | Tensor,
    group_ids: list[int] | Tensor | None = None,
) -> Tensor:
    """Compute Dr. GRPO / Score-Centered advantages `A_i = R_i - mean_{j in group(i)} R_j` without dividing by std.

    Why we do NOT divide by std (`sigma_g`):
    - Under binary sparse rewards (`0/1`), when a group has 5 successes and 1 failure (`std = 0.37`),
      dividing by `std` inflates gradients by 2.7x, and when all rollouts in a group succeed or fail
      (`std = 0`), `1 / std` explodes.
    - Pure mean subtraction (`R_i - R_mean`) strictly zeroes out failed rollouts (`A_i <= 0`) from
      positive actor imitation while automatically assigning higher positive advantages (`1 - R_mean`)
      to harder initial states where `R_mean` is low.
    """
    r_t = torch.as_tensor(rewards, dtype=torch.float32)
    if group_ids is None:
        return r_t - r_t.mean()
    g_t = torch.as_tensor(group_ids, dtype=torch.long, device=r_t.device)
    adv = torch.zeros_like(r_t)
    for g_val in torch.unique(g_t):
        mask = g_t == g_val
        adv[mask] = r_t[mask] - r_t[mask].mean()
    return adv


def compute_marek_score_centered_loss(
    pred_actions: Tensor,
    exec_actions: Tensor,
    sampler_mean_actions: Tensor,
    advantages: Tensor,
    *,
    cispo_clip: float = 2.0,
    sigma_deg: float = 50.0,
    trust_beta: float = 1.0,
) -> Tensor:
    """Marek & Ryabinin Off-Policy Score Centering + CISPO (`arXiv:2609.20807` / Tinker `stable_reinforce`)
    adapted to continuous-action EXPO residual policies.

    Why Dr. GRPO (`A_i = R_i - R_mean`) alone does not eliminate off-policy drift:
    - Dr. GRPO centers rewards only at `t = 0`. Conditioned on an intermediate state `s_t` along a
      promising trajectory, `E_q[A_i | s_t] > 0`, so the uncentered policy score
      `s_phi(a_t | s_t) = grad_phi log pi_phi(a_t | s_t)` has non-zero expectation under the behavior
      sampler `q(. | s_t)`: `s_bar_t = E_{a' ~ q}[grad_phi log pi_phi(a' | s_t)] = -grad_phi KL(q || pi_phi) != 0`.
    - Subtracting the sampler's expected score `s_bar_t` over the K candidate proposals (with empirical
      sampler mean `a_bar_q = sampler_mean_actions`):
        `s_tilde_t = grad_phi log pi_phi(a_exec | s_t) - E_{a' ~ q}[grad_phi log pi_phi(a' | s_t)]`
      forces `E_q[s_tilde_t | s_t] = 0` identically at every state `s_t`, eliminating off-policy sampler
      drift while preserving the exact covariance `Cov_q(A_i, s_phi | s_t)`.
    """
    logp_exec = -0.5 * ((pred_actions - exec_actions) / sigma_deg).pow(2).mean(dim=(1, 2))
    logp_sampler = -0.5 * ((pred_actions - sampler_mean_actions) / sigma_deg).pow(2).mean(dim=(1, 2))
    with torch.no_grad():
        log_ratio = (logp_exec - logp_sampler).clamp(-4.0, 4.0)
        w_cispo = torch.exp(log_ratio).clamp(max=cispo_clip)

    # Stop-gradient sampler drift cancellation:
    # In continuous action space, subtracting `logp_sampler` without stop-gradient on `mu_phi`
    # would cancel the quadratic restoring force `-0.5 * ||mu_phi||^2`. Instead, we attenuate
    # transitions that merely coast along the behavior sampler mean (`a_exec == a_bar_q`, pure drift)
    # while upweighting transitions with high covariance `||a_exec - a_bar_q||` (true RL signal).
    with torch.no_grad():
        innovation = (exec_actions - sampler_mean_actions).abs().mean(dim=(1, 2))
        cov_weight = (innovation / (innovation.mean().clamp(min=1e-4))).clamp(0.5, 1.5)
        w_sc = w_cispo.clamp(0.5, cispo_clip) * cov_weight

    w_pos = torch.clamp(advantages, min=0.0) * w_sc
    l1_err = ((pred_actions - exec_actions).abs() / sigma_deg).mean(dim=(1, 2))
    return (trust_beta * w_pos * l1_err).mean()


class ExpoResidualCritic(nn.Module):
    """Lightweight EXPO Residual Actor (`delta_a_phi`) + Concave Value Critic (`Q_psi`).

    Features:
      - 3x4 adaptive spatial token pooling (`[B, 600, 2048] -> [B, 24, 2048]`).
      - Optional episode-initial visual anchoring (`use_init_anchor=True`) so the 2D object
        spawn position observed by the wrist camera at `t=0` remains drift-free throughout
        closed-loop execution.
      - Batched K-candidate sampling + Score-Centered critic selection (`select_best_edited_chunk`).
    """

    def __init__(
        self,
        ctx_dim: int = 2048,
        state_dim: int = 6,
        chunk_len: int = 30,
        use_init_anchor: bool = True,
        base_alpha: float = 0.06,
        curr_scale: float = 0.06,
        max_delta_deg: float = 100.0,
        zero_init_edit: bool = False,
    ) -> None:
        super().__init__()
        self.ctx_dim = ctx_dim
        self.state_dim = state_dim
        self.chunk_len = chunk_len
        self.use_init_anchor = use_init_anchor
        self.base_alpha = base_alpha if use_init_anchor else 1.0
        self.curr_scale = curr_scale if use_init_anchor else 1.0
        self.max_delta_deg = max_delta_deg if use_init_anchor else 60.0
        self.zero_init_edit = zero_init_edit

        self.register_buffer("init_mean", torch.zeros(1, 24, ctx_dim))
        self.register_buffer("init_std", torch.ones(1, 24, ctx_dim))
        self._cached_init_grid: Tensor | None = None

        self.cell_proj = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )
        if use_init_anchor:
            self.init_proj = nn.Sequential(
                nn.Linear(ctx_dim, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
            )
            vis_dim = 2 * 24 * 64
        else:
            self.init_proj = None
            vis_dim = 24 * 64

        self.vis_mlp = nn.Sequential(
            nn.Linear(vis_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.edit_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16 + chunk_len * state_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, chunk_len * state_dim),
        )
        self.v_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.prec_net = nn.Sequential(
            nn.Linear(256 + state_dim + 16, 256),
            nn.SiLU(),
            nn.Linear(256, chunk_len * state_dim),
            nn.Softplus(),
        )
        if zero_init_edit:
            last_linear = self.edit_net[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)

    def reset_episode(self) -> None:
        """Clear the cached episode-initial visual token grid at `policy.reset()`."""
        self._cached_init_grid = None

    @staticmethod
    def pool_ctx_tokens(ctx: Tensor) -> Tensor:
        """Pool `[B, 600, D]`, `[B, 160, D]`, or `[B, 64, D]` (SmolVLA 8x8) tokens to `[B, 24, D]` (`2` temporal x `3x4` spatial)."""
        if ctx.shape[1] == 24:
            return ctx.float()
        bsz, n_tok, dim = ctx.shape
        if n_tok == 64:
            x = ctx.float().reshape(bsz, 8, 8, dim).permute(0, 3, 1, 2)
            p12 = F.adaptive_avg_pool2d(x, (3, 4)).reshape(bsz, 1, dim, 12).permute(0, 1, 3, 2)
            return p12.expand(bsz, 2, 12, dim).reshape(bsz, 24, dim)
        h_g, w_g = (15, 20) if n_tok == 600 else (8, 10)
        x = ctx.float().reshape(bsz, 2, h_g, w_g, dim).permute(0, 1, 4, 2, 3).reshape(bsz * 2, dim, h_g, w_g)
        return F.adaptive_avg_pool2d(x, (3, 4)).reshape(bsz, 2, dim, 12).permute(0, 1, 3, 2).reshape(bsz, 24, dim)

    def _encode_ctx(self, ctx: Tensor, init_ctx: Tensor | None = None) -> Tensor:
        grid = self.pool_ctx_tokens(ctx)
        f_curr = self.cell_proj(grid).reshape(grid.shape[0], -1) * self.curr_scale
        if not self.use_init_anchor or self.init_proj is None:
            return self.vis_mlp(f_curr)
        if init_ctx is None:
            if self._cached_init_grid is None:
                self._cached_init_grid = grid[:1].detach().clone()
            init_grid = self._cached_init_grid.expand(grid.shape[0], -1, -1)
        else:
            init_grid = self.pool_ctx_tokens(init_ctx)
        z_init = ((init_grid - self.init_mean) / self.init_std).clamp(-5.0, 5.0)
        f_init = self.init_proj(z_init).reshape(grid.shape[0], -1)
        return self.vis_mlp(torch.cat([f_curr, f_init], dim=-1))

    def _time_emb(self, t_norm: Tensor) -> Tensor:
        freqs = torch.arange(1, 9, device=t_norm.device, dtype=torch.float32).unsqueeze(0) * 3.14159
        return torch.cat([torch.sin(t_norm * freqs), torch.cos(t_norm * freqs)], dim=-1)

    def forward_edit_with_feat(
        self,
        ctx: Tensor,
        state_deg: Tensor,
        t_norm: Tensor,
        base_actions_deg: Tensor,
        init_ctx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        ctx_feat = self._encode_ctx(ctx, init_ctx=init_ctx)
        state_scaled = (state_deg / 50.0) * self.curr_scale
        t_emb = self._time_emb(t_norm)
        base_scaled = (base_actions_deg * self.base_alpha).reshape(base_actions_deg.shape[0], -1) / 50.0
        inp = torch.cat([ctx_feat, state_scaled, t_emb, base_scaled], dim=-1)
        delta = self.max_delta_deg * torch.tanh(self.edit_net(inp).view(-1, self.chunk_len, self.state_dim))
        return base_actions_deg * self.base_alpha + delta, ctx_feat

    def forward_edit(
        self,
        ctx: Tensor,
        state_deg: Tensor,
        t_norm: Tensor,
        base_actions_deg: Tensor,
        init_ctx: Tensor | None = None,
    ) -> Tensor:
        edited, _ = self.forward_edit_with_feat(ctx, state_deg, t_norm, base_actions_deg, init_ctx=init_ctx)
        return edited

    def forward_q_from_feat(
        self,
        ctx_feat_det: Tensor,
        state_deg: Tensor,
        t_norm: Tensor,
        cand_actions_deg: Tensor,
        ref_edit_deg: Tensor,
    ) -> Tensor:
        state_scaled = (state_deg / 50.0) * self.curr_scale
        t_emb = self._time_emb(t_norm)
        s_inp = torch.cat([ctx_feat_det, state_scaled, t_emb], dim=-1)
        v_peak = self.v_net(s_inp).squeeze(-1)
        prec = self.prec_net(s_inp) + 0.1
        diff_abs = (cand_actions_deg - ref_edit_deg).abs().reshape(cand_actions_deg.shape[0], -1)
        advantage_penalty = (prec * diff_abs).mean(dim=-1)
        return v_peak * torch.exp(-advantage_penalty)

    def forward_q(
        self,
        ctx: Tensor,
        state_deg: Tensor,
        t_norm: Tensor,
        cand_actions_deg: Tensor,
        ref_edit_deg: Tensor,
        init_ctx: Tensor | None = None,
    ) -> Tensor:
        ctx_feat = self._encode_ctx(ctx, init_ctx=init_ctx).detach()
        return self.forward_q_from_feat(ctx_feat, state_deg, t_norm, cand_actions_deg, ref_edit_deg)

    @torch.no_grad()
    def select_best_edited_chunk(
        self,
        decoder: Any,
        state_t: Tensor,
        ctx: Tensor,
        t_norm: Tensor,
        base_seed: int = 0,
        k_candidates: int = 6,
    ) -> Tensor:
        """Sample `k_candidates` chunks from the frozen decoder in a single batched pass,
        apply residual edits, and select the highest score-centered Q candidate.
        """
        ctx_pooled_1 = self.pool_ctx_tokens(ctx)
        if float(t_norm.item()) < 1e-4 or self._cached_init_grid is None:
            self._cached_init_grid = ctx_pooled_1[:1].detach().clone()

        if hasattr(decoder, "_noise_for_seed"):
            eps_k = torch.cat(
                [decoder._noise_for_seed(1, state_t.device, base_seed + k * 97) for k in range(k_candidates)],
                dim=0,
            )
            state_k = state_t.expand(k_candidates, -1)
            ctx_k = ctx.expand(k_candidates, -1, -1)
            base_k = decoder.sample_actions(state_k, ctx_k, noise=eps_k, use_cuda_graph=False)
        else:
            cands = [
                decoder.sample_actions(state_t, ctx, seed=base_seed + k * 97, use_cuda_graph=False)
                for k in range(k_candidates)
            ]
            base_k = torch.cat(cands, dim=0)
            state_k = state_t.expand(k_candidates, -1)

        t_k = t_norm.expand(k_candidates, -1)
        ctx_pooled = ctx_pooled_1.expand(k_candidates, -1, -1)
        init_k = self._cached_init_grid.expand(k_candidates, -1, -1)
        cands_b, ctx_feat = self.forward_edit_with_feat(ctx_pooled, state_k, t_k, base_k, init_ctx=init_k)
        q_scores = self.forward_q_from_feat(
            ctx_feat,
            state_k,
            t_k,
            cands_b,
            cands_b[:1].expand(k_candidates, -1, -1),
        )
        q_centered = q_scores - q_scores.mean()
        best_idx = int(q_centered.argmax().item())
        return cands_b[best_idx : best_idx + 1]

    def update_from_episode_buffer(
        self,
        episodes: list[dict[str, Any]],
        steps: int = 600,
        lr: float = 3e-4,
        batch_size: int = 64,
    ) -> dict[str, float]:
        """Update `ExpoResidualCritic` in-place on an accumulated buffer of episode trajectories
        using Dr. GRPO score-centered advantages.

        Each episode dict in `episodes` contains:
          - `"steps"`: list of step dicts with `"ctx"` (`[24, 2048]`), `"state"` (`[6]`),
            `"t_norm"` (`[1]`), `"base_action"` (`[30, 6]`), and `"exec_action"` (`[30, 6]`).
          - `"reward"`: binary or scalar episode return in `[0, 1]`.
          - `"spawn_group"`: integer group ID for per-spawn mean-centering.
        """
        valid_eps = [ep for ep in episodes if ep.get("steps")]
        if not valid_eps:
            return {"loss": 0.0, "num_episodes": 0.0, "mean_reward": 0.0}

        device = next(self.parameters()).device
        rewards = [float(ep.get("reward", 0.0)) for ep in valid_eps]
        groups = [int(ep.get("spawn_group", 0)) for ep in valid_eps]
        adv_ep = compute_dr_grpo_advantages(rewards, groups)

        ctx_list: list[Tensor] = []
        init_list: list[Tensor] = []
        state_list: list[Tensor] = []
        tnorm_list: list[Tensor] = []
        base_list: list[Tensor] = []
        exec_list: list[Tensor] = []
        rew_list: list[ float ] = []
        adv_list: list[ float ] = []
        ep_init_grids: list[Tensor] = []

        for ep_idx, ep in enumerate(valid_eps):
            ep_steps = ep["steps"]
            init_g = self.pool_ctx_tokens(ep_steps[0]["ctx"].unsqueeze(0) if ep_steps[0]["ctx"].ndim == 2 else ep_steps[0]["ctx"])[0].cpu()
            ep_init_grids.append(init_g)
            r_val = rewards[ep_idx]
            a_val = float(adv_ep[ep_idx].item())
            for st in ep_steps:
                c_tok = self.pool_ctx_tokens(st["ctx"].unsqueeze(0) if st["ctx"].ndim == 2 else st["ctx"])[0].cpu()
                ctx_list.append(c_tok)
                init_list.append(init_g)
                state_list.append(st["state"].view(-1).cpu().float())
                tnorm_list.append(st["t_norm"].view(1).cpu().float())
                base_list.append(st["base_action"].view(self.chunk_len, self.state_dim).cpu().float())
                exec_list.append(st["exec_action"].view(self.chunk_len, self.state_dim).cpu().float())
                rew_list.append(r_val)
                adv_list.append(a_val)

        if len(ep_init_grids) >= 2:
            stacked_inits = torch.stack(ep_init_grids, dim=0).to(device)
            self.init_mean.copy_(stacked_inits.mean(dim=0, keepdim=True))
            self.init_std.copy_(stacked_inits.std(dim=0, keepdim=True).clamp(min=1e-2))

        ctx_all = torch.stack(ctx_list, dim=0).to(device)
        init_all = torch.stack(init_list, dim=0).to(device)
        state_all = torch.stack(state_list, dim=0).to(device)
        tnorm_all = torch.stack(tnorm_list, dim=0).to(device)
        base_all = torch.stack(base_list, dim=0).to(device)
        exec_all = torch.stack(exec_list, dim=0).to(device)
        rew_all = torch.tensor(rew_list, device=device, dtype=torch.float32)
        adv_all = torch.tensor(adv_list, device=device, dtype=torch.float32)

        n_total = ctx_all.shape[0]
        bsz = min(batch_size, n_total)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)

        self.train()
        last_loss = 0.0
        for _ in range(steps):
            idx = torch.randint(0, n_total, (bsz,), device=device)
            c_b = ctx_all[idx]
            i_b = init_all[idx]
            s_b = state_all[idx]
            t_b = tnorm_all[idx]
            ba_b = base_all[idx]
            ex_b = exec_all[idx]
            r_b = rew_all[idx]
            a_b = adv_all[idx]

            pred_edit, ctx_feat = self.forward_edit_with_feat(c_b, s_b, t_b, ba_b, init_ctx=i_b)
            q_exec = self.forward_q_from_feat(ctx_feat.detach(), s_b, t_b, ex_b, pred_edit.detach())
            critic_loss = F.mse_loss(q_exec, r_b)

            # Dr. GRPO group advantage + Marek & Ryabinin Off-Policy Score Centering (arXiv:2609.20807)
            w_pos = torch.where(a_b > 0.0, a_b, 0.25 * r_b)
            sc_loss = compute_marek_score_centered_loss(pred_edit, ex_b, ba_b, w_pos)
            trust_reg = ((pred_edit - ba_b * self.base_alpha) / 50.0).pow(2).mean(dim=(1, 2))
            actor_loss = sc_loss + (0.02 * (1.0 - r_b) * trust_reg).mean()

            loss = critic_loss + actor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())

        self.eval()
        self.reset_episode()
        return {
            "loss": last_loss,
            "num_episodes": float(len(valid_eps)),
            "mean_reward": float(sum(rewards) / len(rewards)),
        }
