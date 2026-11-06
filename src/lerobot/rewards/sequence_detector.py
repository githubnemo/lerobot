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

"""Multi-Frame Sequence + Proprioception Success Detector for wrist-camera robot tasks.

Why a sequence detector is required for wrist-only cameras (e.g. SO-101):
- A single-frame classifier (such as ResNet-50) only sees the current image I_t.
- At the end of a successful "put cube in cardboard box" episode, the wrist camera looks
  at the cardboard box with an open gripper while the dropped cube is often occluded or deep
  in shadow inside the box.
- Consequently, a single-frame ResNet-50 learns the shortcut "cardboard box in view + open gripper -> 1.0",
  causing false reward spikes whenever the robot sweeps over the box empty-handed.
- `MultiFrameSuccessDetector` observes a causal window of T frames (e.g. T=8 frames spaced by stride=6
  over the last ~1.5 seconds) TOGETHER with the 6-D joint state trajectory `agent_pos` (including gripper
  aperture transitions), explicitly verifying both pre-release transport/grasp and terminal release into the box.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision.models as tv_models
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def extract_sequence_window(
    frames: Tensor,
    states: Tensor,
    end_idx: int,
    seq_len: int = 8,
    stride: int = 6,
) -> tuple[Tensor, Tensor]:
    """Extract a causal window of `seq_len` frames and states ending at `end_idx`.

    Args:
        frames: Tensor of shape `[N, 3, H, W]` (float in [0, 1] or uint8).
        states: Tensor of shape `[N, state_dim]` (float32, degrees).
        end_idx: Inclusive terminal frame index in `[0, N - 1]`.
        seq_len: Number of frames in the sequence window.
        stride: Step stride between consecutive sequence frames.

    Returns:
        `(seq_frames, seq_states)` of shapes `[seq_len, 3, H, W]` and `[seq_len, state_dim]`.
    """
    indices = [max(0, end_idx - (seq_len - 1 - k) * stride) for k in range(seq_len)]
    idx_t = torch.tensor(indices, device=frames.device, dtype=torch.long)
    seq_f = frames.index_select(0, idx_t)
    if seq_f.dtype == torch.uint8:
        seq_f = seq_f.float() / 255.0
    seq_s = states.index_select(0, idx_t.to(states.device)).float()
    return seq_f, seq_s


class _ImageNetNormalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std


class SingleFrameResNet50Detector(nn.Module):
    """Baseline single-frame ResNet-50 reward detector (uses only the current frame I_t)."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.norm = _ImageNetNormalizer()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        if freeze_backbone:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        # Spatial pooling preserving coarse layout (2x2) + global average pool
        self.head = nn.Sequential(
            nn.Linear(2048 * 5, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def train(self, mode: bool = True) -> SingleFrameResNet50Detector:
        super().train(mode)
        if self.freeze_backbone:
            self.stem.eval()
        return self

    def encode_frames(self, x: Tensor) -> Tensor:
        """Encode `[B, 3, H, W]` images in `[0, 1]` to `[B, 2048 * 5]` features."""
        if x.shape[-2:] != (128, 128):
            x = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.norm(x)
        feat_map = self.stem(x)  # [B, 2048, 4, 4]
        gap = feat_map.mean(dim=(-2, -1))  # [B, 2048]
        quad = F.adaptive_avg_pool2d(feat_map, (2, 2)).flatten(1)  # [B, 2048 * 4]
        return torch.cat([gap, quad], dim=-1)

    def forward(self, seq_frames: Tensor, seq_states: Tensor | None = None) -> Tensor:
        """Accepts `[B, T, 3, H, W]` or `[B, 3, H, W]` and classifies ONLY the last frame."""
        last_frame = seq_frames[:, -1] if seq_frames.ndim == 5 else seq_frames
        feat = self.encode_frames(last_frame)
        return self.head(feat).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, seq_frames: Tensor, seq_states: Tensor | None = None) -> Tensor:
        return torch.sigmoid(self.forward(seq_frames, seq_states))


class MultiFrameSuccessDetector(nn.Module):
    """Multi-frame temporal sequence + proprioception reward detector.

    Processes `T` frames `[B, T, 3, H, W]` and `T` joint states `[B, T, state_dim]`.
    Combines:
      1. Spatial visual features (global + 2x2 quadrant + gripper-jaw ROI crop features).
      2. Explicit frame-to-frame visual difference features `df_t = f_t - f_{t-1}` to detect
         the physical cube leaving the gripper jaws into the cardboard box.
      3. Proprioceptive joint trajectory `[q_t, q_t - q_{t-1}]` (capturing the gripper opening
         transition over the box).
      4. Causal temporal self-attention + dual-phase verification gate (requiring both
         pre-release carry evidence in frames `0..T//2` and terminal release evidence at `T-1`).
    """

    def __init__(
        self,
        seq_len: int = 8,
        state_dim: int = 6,
        d_model: int = 192,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.norm = _ImageNetNormalizer()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        if freeze_backbone:
            self.stem.eval()
            for p in self.stem.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

        # Visual feature per frame: global (512) + 2x2 spatial grid (2048) + jaw-ROI (512) = 3072
        vis_raw_dim = 512 * 6
        self.vis_in_norm = nn.LayerNorm(vis_raw_dim)
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.diff_in_norm = nn.LayerNorm(vis_raw_dim)
        self.diff_proj = nn.Sequential(
            nn.Linear(vis_raw_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
        )
        # Proprioception: [q_t (6), dq_t (6), q_t - q_0 (6)] = 18 dims
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim * 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
        )

        token_dim = d_model + (d_model // 2) + (d_model // 2)  # 2 * d_model
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=token_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Dual-phase verification heads:
        # 1) Pre-release transport/grasp gate over early/mid frames (0 .. T-2)
        # 2) Terminal release/in-box head over the full sequence summary + last token
        self.pre_grasp_head = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.release_head = nn.Sequential(
            nn.Linear(token_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def train(self, mode: bool = True) -> MultiFrameSuccessDetector:
        super().train(mode)
        if self.freeze_backbone:
            self.stem.eval()
        return self

    def encode_raw_visual(self, frames_bt: Tensor) -> Tensor:
        """Encode `[B*T, 3, H, W]` images in `[0, 1]` into `[B*T, 3072]` spatial + jaw-ROI descriptors."""
        if frames_bt.shape[-2:] != (128, 128):
            frames_bt = F.interpolate(frames_bt, size=(128, 128), mode="bilinear", align_corners=False)
        x = self.norm(frames_bt)
        fmap = self.stem(x)  # [B*T, 512, 4, 4]
        gap = fmap.mean(dim=(-2, -1))  # [B*T, 512]
        quad = F.adaptive_avg_pool2d(fmap, (2, 2)).flatten(1)  # [B*T, 2048]
        # Bottom-center region where the wrist camera sees the gripper jaws & held cube
        jaw_roi = fmap[:, :, 2:, 1:3].mean(dim=(-2, -1))  # [B*T, 512]
        return torch.cat([gap, quad, jaw_roi], dim=-1)

    def forward_with_aux(
        self,
        seq_frames: Tensor,
        seq_states: Tensor,
        precomputed_vis: Tensor | None = None,
        proprio_dropout: float = 0.0,
        token_dropout: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning `(final_success_logit, pre_grasp_logit)`.

        Args:
            seq_frames: `[B, T, 3, H, W]` sequence of RGB images in `[0, 1]`.
            seq_states: `[B, T, state_dim]` sequence of joint states (degrees).
            precomputed_vis: Optional `[B, T, 3072]` cached visual features.
            proprio_dropout: Probability of dropping out the proprioceptive branch during training
                so the visual pathway cannot rely solely on a joint-angle shortcut.
            token_dropout: Probability of dropping intermediate temporal visual tokens during training.
        """
        bsz, seq_t = seq_states.shape[:2]
        if precomputed_vis is None:
            flat_frames = seq_frames.reshape(bsz * seq_t, *seq_frames.shape[2:])
            vis_raw = self.encode_raw_visual(flat_frames).view(bsz, seq_t, -1)
        else:
            vis_raw = precomputed_vis

        if self.training and token_dropout > 0.0 and seq_t > 2:
            keep_mask = (torch.rand(bsz, seq_t - 2, 1, device=vis_raw.device) >= token_dropout).float() / (
                1.0 - token_dropout
            )
            mid_vis = vis_raw[:, 1:-1] * keep_mask
            vis_raw = torch.cat([vis_raw[:, :1], mid_vis, vis_raw[:, -1:]], dim=1)

        vis_normed = self.vis_in_norm(vis_raw)
        vis_emb = self.vis_proj(vis_normed)  # [B, T, d_model]
        vis_diff_raw = torch.cat([torch.zeros_like(vis_raw[:, :1]), vis_raw[:, 1:] - vis_raw[:, :-1]], dim=1)
        vis_span_raw = vis_raw - vis_raw[:, :1]
        diff_emb = self.diff_proj(self.diff_in_norm(vis_diff_raw + 0.5 * vis_span_raw))  # [B, T, d_model // 2]

        s_scaled = seq_states / 50.0
        ds = torch.cat([torch.zeros_like(s_scaled[:, :1]), s_scaled[:, 1:] - s_scaled[:, :-1]], dim=1)
        ds_init = s_scaled - s_scaled[:, :1]
        state_emb = self.state_proj(torch.cat([s_scaled, ds, ds_init], dim=-1))  # [B, T, d_model // 2]

        if self.training and proprio_dropout > 0.0:
            p_keep = (torch.rand(bsz, 1, 1, device=state_emb.device) >= proprio_dropout).float()
            state_emb = state_emb * p_keep

        tokens = torch.cat([vis_emb, diff_emb, state_emb], dim=-1) + self.pos_emb[:, :seq_t]
        h = self.temporal_transformer(tokens)  # [B, T, token_dim]

        early_mid = h[:, : max(1, seq_t - 2)].mean(dim=1)
        late = h[:, -2:].mean(dim=1)
        last = h[:, -1]

        pre_grasp_logit = self.pre_grasp_head(early_mid).squeeze(-1)
        seq_summary = torch.cat([early_mid, late - early_mid, last], dim=-1)
        release_logit = self.release_head(seq_summary).squeeze(-1)

        # Dual-phase verification gate:
        # When `pre_grasp_logit` is negative (no cube carried in first half of sequence),
        # `-2.5 * F.softplus(-pre_grasp_logit)` strongly suppresses false positives on empty-box views
        # while maintaining clean linear gradients when `pre_grasp_logit > 0`.
        joint_logit = release_logit - 2.5 * F.softplus(-pre_grasp_logit)
        return joint_logit, pre_grasp_logit

    def forward(
        self,
        seq_frames: Tensor,
        seq_states: Tensor,
        precomputed_vis: Tensor | None = None,
    ) -> Tensor:
        joint_logit, _ = self.forward_with_aux(seq_frames, seq_states, precomputed_vis=precomputed_vis)
        return joint_logit

    @torch.no_grad()
    def predict_proba(
        self,
        seq_frames: Tensor,
        seq_states: Tensor,
        precomputed_vis: Tensor | None = None,
    ) -> Tensor:
        return torch.sigmoid(self.forward(seq_frames, seq_states, precomputed_vis=precomputed_vis))

    def update_from_episode_buffer(
        self,
        episodes: list[dict[str, object]],
        *,
        anchor_cache: dict[str, Tensor] | None = None,
        steps: int = 80,
        lr: float = 2e-4,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Fine-tune the Temporal Transformer & dual-phase heads online using rollout + human-annotated episodes."""
        device = next(self.parameters()).device
        seq_len = self.seq_len
        vis_list: list[Tensor] = []
        st_list: list[Tensor] = []
        y_list: list[float] = []
        pre_list: list[float] = []

        for ep in episodes:
            ep_steps = ep.get("steps")
            if not isinstance(ep_steps, list) or len(ep_steps) < 4 or "vis_raw" not in ep_steps[0]:
                continue
            n_st = len(ep_steps)
            rew = float(ep.get("reward", 0.0))  # type: ignore[arg-type]
            is_human = bool(ep.get("human_labeled", False))
            repeat = 3 if is_human else 1

            if rew >= 0.5:
                # Positive terminal release windows
                pos_ends = [n_st - 1, max(0, n_st - 2)]
                for _ in range(repeat):
                    for end_i in pos_ends:
                        for stride in (1, 2):
                            if stride == 2 and end_i < seq_len:
                                continue
                            idxs = [max(0, end_i - (seq_len - 1 - k) * stride) for k in range(seq_len)]
                            vis_list.append(torch.stack([ep_steps[j]["vis_raw"] for j in idxs], dim=0))
                            st_list.append(torch.stack([ep_steps[j]["state"] for j in idxs], dim=0))
                            y_list.append(1.0)
                            pre_list.append(1.0)
                # Early approach windows in the same episode are negatives (not released yet)
                for end_i in range(2, max(2, n_st - 6), 2):
                    idxs = [max(0, end_i - (seq_len - 1 - k)) for k in range(seq_len)]
                    vis_list.append(torch.stack([ep_steps[j]["vis_raw"] for j in idxs], dim=0))
                    st_list.append(torch.stack([ep_steps[j]["state"] for j in idxs], dim=0))
                    y_list.append(0.0)
                    pre_list.append(0.0)
            else:
                # Failed episode: all windows are negative; upweight any false-positive spikes
                for end_i in range(3, n_st):
                    det_p = float(ep_steps[end_i].get("det_prob", 0.0))
                    reps = (repeat * 3) if det_p >= 0.4 else (1 if end_i % 2 == 0 else 0)
                    for _ in range(reps):
                        for stride in (1, 2):
                            if stride == 2 and end_i < seq_len:
                                continue
                            idxs = [max(0, end_i - (seq_len - 1 - k) * stride) for k in range(seq_len)]
                            vis_list.append(torch.stack([ep_steps[j]["vis_raw"] for j in idxs], dim=0))
                            st_list.append(torch.stack([ep_steps[j]["state"] for j in idxs], dim=0))
                            y_list.append(0.0)
                            pre_list.append(0.0)

        if not vis_list:
            return {"loss": 0.0, "num_windows": 0.0}

        on_vis = torch.stack(vis_list, dim=0).to(device, dtype=torch.float32)
        on_st = torch.stack(st_list, dim=0).to(device, dtype=torch.float32)
        on_y = torch.tensor(y_list, device=device, dtype=torch.float32)
        on_pre = torch.tensor(pre_list, device=device, dtype=torch.float32)
        n_on = on_vis.shape[0]

        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        self.train()
        last_loss = 0.0

        has_anchor = (
            anchor_cache is not None
            and "vis_seq" in anchor_cache
            and anchor_cache["vis_seq"].shape[0] > 0
        )
        for _ in range(steps):
            b_on = min(batch_size // 2 if has_anchor else batch_size, n_on)
            idx_on = torch.randint(0, n_on, (b_on,), device=device)
            v_b = on_vis[idx_on]
            s_b = on_st[idx_on]
            y_b = on_y[idx_on]
            p_b = on_pre[idx_on]

            if has_anchor and anchor_cache is not None:
                n_anc = anchor_cache["vis_seq"].shape[0]
                b_anc = max(1, batch_size - b_on)
                idx_anc = torch.randint(0, n_anc, (b_anc,))
                v_anc = anchor_cache["vis_seq"][idx_anc].to(device, dtype=torch.float32)
                s_anc = anchor_cache["state_seq"][idx_anc].to(device, dtype=torch.float32)
                y_anc = anchor_cache["y"][idx_anc].to(device, dtype=torch.float32)
                p_anc = anchor_cache["pre"][idx_anc].to(device, dtype=torch.float32)
                v_b = torch.cat([v_b, v_anc], dim=0)
                s_b = torch.cat([s_b, s_anc], dim=0)
                y_b = torch.cat([y_b, y_anc], dim=0)
                p_b = torch.cat([p_b, p_anc], dim=0)

            dummy_f = torch.empty(v_b.shape[0], seq_len, 3, 1, 1, device=device)
            joint_logit, pre_logit = self.forward_with_aux(
                dummy_f, s_b, precomputed_vis=v_b, proprio_dropout=0.15, token_dropout=0.1
            )
            loss_joint = F.binary_cross_entropy_with_logits(joint_logit, y_b)
            loss_pre = F.binary_cross_entropy_with_logits(pre_logit, p_b)
            loss = loss_joint + 0.4 * loss_pre
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            last_loss = float(loss.item())

        self.eval()
        return {"loss": last_loss, "num_windows": float(n_on)}


class SharedFeatureSequenceDetector(nn.Module):
    """Zero-backbone-overhead Multi-Frame Sequence Detector operating directly on frozen
    Video-VAM (`Cosmos 3 Edge LoRA`) pooled latent tokens `[B, T, 24, ctx_dim]` + `seq_states` `[B, T, state_dim]`.

    Why share the policy's video-backbone features?
    - During closed-loop control, `VideoVAMPolicy` + `ExpoResidualCritic` already extract and pool
      `Cosmos 3 Edge LoRA` tokens `[1, 24, 2048]` (`2` temporal x `3x4` spatial grid) at every chunk step.
    - Reusing these exact latent tokens for multi-frame reward detection eliminates any separate ResNet
      inference pass (0 MB extra backbone VRAM, <0.15 ms forward pass per episode) while preserving the
      spatial jaw-ROI (`row=2, col=1..2`) and dual-phase pre-grasp verification gate.
    """

    def __init__(
        self,
        ctx_dim: int = 2048,
        seq_len: int = 12,
        state_dim: int = 6,
        cell_dim: int = 64,
        d_model: int = 192,
    ) -> None:
        super().__init__()
        self.ctx_dim = ctx_dim
        self.seq_len = seq_len
        self.state_dim = state_dim
        self.cell_dim = cell_dim

        self.cell_proj = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, cell_dim),
            nn.LayerNorm(cell_dim),
            nn.SiLU(),
        )
        # Per step visual descriptor:
        # 24 pooled cells * cell_dim + jaw-ROI (bottom-center cells, 2*2*cell_dim) + global mean (cell_dim)
        vis_raw_dim = (24 + 4 + 1) * cell_dim
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_raw_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.diff_proj = nn.Sequential(
            nn.Linear(vis_raw_dim, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim * 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.SiLU(),
        )

        token_dim = d_model + (d_model // 2) + (d_model // 2)
        self.pos_emb = nn.Parameter(torch.randn(1, max(seq_len, 32), token_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=token_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)

        self.pre_grasp_head = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.release_head = nn.Sequential(
            nn.Linear(token_dim * 3, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    @staticmethod
    def pool_ctx_sequence(seq_ctx: Tensor) -> Tensor:
        """Accepts `[B, T, 24, D]`, `[B, T, 600, D]`, `[B, T, 160, D]`, or `[B, T, 64, D]` (SmolVLA 8x8) and returns `[B, T, 24, D]` float32."""
        if seq_ctx.shape[2] == 24:
            return seq_ctx.float()
        bsz, seq_t, n_tok, dim = seq_ctx.shape
        if n_tok == 64:
            x = seq_ctx.float().reshape(bsz * seq_t, 8, 8, dim).permute(0, 3, 1, 2)
            p12 = F.adaptive_avg_pool2d(x, (3, 4)).reshape(bsz, seq_t, 1, dim, 12).permute(0, 1, 2, 4, 3)
            return p12.expand(bsz, seq_t, 2, 12, dim).reshape(bsz, seq_t, 24, dim)
        h_g, w_g = (15, 20) if n_tok == 600 else (8, 10)
        x = seq_ctx.float().reshape(bsz * seq_t, 2, h_g, w_g, dim).permute(0, 1, 4, 2, 3).reshape(bsz * seq_t * 2, dim, h_g, w_g)
        pooled = F.adaptive_avg_pool2d(x, (3, 4)).reshape(bsz, seq_t, 2, dim, 12).permute(0, 1, 2, 4, 3).reshape(bsz, seq_t, 24, dim)
        return pooled

    def _encode_pooled_tokens(self, pooled_ctx: Tensor) -> Tensor:
        """Convert `[B, T, 24, ctx_dim]` into `[B, T, vis_raw_dim]` spatial + jaw-ROI descriptors."""
        bsz, seq_t = pooled_ctx.shape[:2]
        cells = self.cell_proj(pooled_ctx)  # [B, T, 24, cell_dim]
        grid_2x3x4 = cells.view(bsz, seq_t, 2, 3, 4, self.cell_dim)
        # Bottom-center cells (row=2, cols=1..2 across both temporal sub-frames) capture the gripper jaws
        jaw_roi = grid_2x3x4[:, :, :, 2, 1:3, :].reshape(bsz, seq_t, 4 * self.cell_dim)
        gap = cells.mean(dim=2)  # [B, T, cell_dim]
        flat_cells = cells.reshape(bsz, seq_t, 24 * self.cell_dim)
        return torch.cat([flat_cells, jaw_roi, gap], dim=-1)

    def forward_with_aux(
        self,
        seq_ctx: Tensor,
        seq_states: Tensor,
        proprio_dropout: float = 0.0,
        token_dropout: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass on policy backbone tokens `[B, T, 24, ctx_dim]` and states `[B, T, state_dim]`."""
        bsz, seq_t = seq_states.shape[:2]
        pooled = self.pool_ctx_sequence(seq_ctx)
        vis_raw = self._encode_pooled_tokens(pooled)

        if self.training and token_dropout > 0.0 and seq_t > 2:
            keep_mask = (torch.rand(bsz, seq_t - 2, 1, device=vis_raw.device) >= token_dropout).float() / (
                1.0 - token_dropout
            )
            vis_raw = torch.cat([vis_raw[:, :1], vis_raw[:, 1:-1] * keep_mask, vis_raw[:, -1:]], dim=1)

        vis_emb = self.vis_proj(vis_raw)
        vis_diff = torch.cat([torch.zeros_like(vis_raw[:, :1]), vis_raw[:, 1:] - vis_raw[:, :-1]], dim=1)
        vis_span = vis_raw - vis_raw[:, :1]
        diff_emb = self.diff_proj(vis_diff + 0.5 * vis_span)

        s_scaled = seq_states / 50.0
        ds = torch.cat([torch.zeros_like(s_scaled[:, :1]), s_scaled[:, 1:] - s_scaled[:, :-1]], dim=1)
        ds_init = s_scaled - s_scaled[:, :1]
        state_emb = self.state_proj(torch.cat([s_scaled, ds, ds_init], dim=-1))

        if self.training and proprio_dropout > 0.0:
            p_keep = (torch.rand(bsz, 1, 1, device=state_emb.device) >= proprio_dropout).float()
            state_emb = state_emb * p_keep

        tokens = torch.cat([vis_emb, diff_emb, state_emb], dim=-1) + self.pos_emb[:, :seq_t]
        h = self.temporal_transformer(tokens)

        early_mid = h[:, : max(1, seq_t - 2)].mean(dim=1)
        late = h[:, -2:].mean(dim=1)
        last = h[:, -1]

        pre_grasp_logit = self.pre_grasp_head(early_mid).squeeze(-1)
        seq_summary = torch.cat([early_mid, late - early_mid, last], dim=-1)
        release_logit = self.release_head(seq_summary).squeeze(-1)
        joint_logit = release_logit - 2.5 * F.softplus(-pre_grasp_logit)
        return joint_logit, pre_grasp_logit

    def forward(self, seq_ctx: Tensor, seq_states: Tensor) -> Tensor:
        joint_logit, _ = self.forward_with_aux(seq_ctx, seq_states)
        return joint_logit

    @torch.no_grad()
    def predict_proba(self, seq_ctx: Tensor, seq_states: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(seq_ctx, seq_states))
