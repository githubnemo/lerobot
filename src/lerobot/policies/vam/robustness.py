"""Shared training/evaluation robustness primitives for Cosmos 3 Video-VAM and SmolVLA.

Implements first-principles fixes for the offline-to-closed-loop gap (arXiv:2611.00283v1):
  1. State feature group slicing (`pos` 6-D vs `pos_vel_load_curr` 24-D on 48-D telemetry datasets).
  2. Randomized variable-FPS history stride, per-frame timing jitter, and static-history rest-start aug.
  3. Critical-phase & contact-weighted loss weighting (upweight grasp/release/load; downweight idle).
  4. Delta arm actions, calibration-offset augmentation (+/-deg), and proprioception dropout.
"""

from __future__ import annotations

import torch
from torch import Tensor

STATE_FEATURE_GROUP_DIMS: dict[str, int | None] = {
    "pos": 6,
    "pos_vel_load_curr": 24,
    "all": None,
}


def resolve_state_dim(raw_dim: int, groups: str = "pos") -> int:
    if groups not in STATE_FEATURE_GROUP_DIMS:
        raise ValueError(
            f"Unknown state_feature_groups={groups!r}; expected one of {sorted(STATE_FEATURE_GROUP_DIMS)}"
        )
    target = STATE_FEATURE_GROUP_DIMS[groups]
    if target is None:
        return int(raw_dim)
    return min(int(raw_dim), int(target))


def slice_state_features(state: Tensor, groups: str = "pos") -> Tensor:
    """Slice trailing state dimension to the selected STS3215 telemetry subset."""
    if groups not in STATE_FEATURE_GROUP_DIMS:
        raise ValueError(
            f"Unknown state_feature_groups={groups!r}; expected one of {sorted(STATE_FEATURE_GROUP_DIMS)}"
        )
    target = STATE_FEATURE_GROUP_DIMS[groups]
    if target is None or state.shape[-1] <= target:
        return state
    return state[..., :target].contiguous()


def sample_history_window_indices(
    frame: int,
    ep_len: int,
    history_size: int,
    *,
    base_stride: int = 1,
    min_stride: int = 0,
    max_stride: int = 0,
    jitter_frames: int = 0,
    static_prob: float = 0.0,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Sample strictly causal, monotonic frame indices ending at `frame`.

    - Always ends at `curr = clamp(frame, 0, ep_len - 1)` (zero future leakage).
    - If `history_size == 1`, returns `[curr]`.
    - With probability `static_prob` (boosted near episode starts `frame < 15`), returns
      `[curr] * history_size` to train against rest-pose initiation failure.
    - Otherwise draws effective stride `s` in `[min_stride, max_stride]` (or `base_stride`)
      and adds independent `[-jitter_frames, +jitter_frames]` per past frame while enforcing
      monotonicity `w[0] <= w[1] <= ... <= curr`.
    """
    curr = max(0, min(ep_len - 1, int(frame)))
    if history_size <= 1:
        return [curr]

    if static_prob > 0.0:
        eff_prob = min(0.5, static_prob * 3.0) if curr < 15 else static_prob
        u = float(torch.rand(1, generator=generator).item())
        if u < eff_prob:
            return [curr] * history_size

    if min_stride > 0 and max_stride >= min_stride:
        stride = int(torch.randint(min_stride, max_stride + 1, (1,), generator=generator).item())
    else:
        stride = max(1, int(base_stride))

    nominal = [curr - (history_size - 1 - k) * stride for k in range(history_size)]
    if jitter_frames > 0:
        raw = []
        for idx_k in nominal[:-1]:
            j = int(torch.randint(-jitter_frames, jitter_frames + 1, (1,), generator=generator).item())
            raw.append(max(0, min(curr, idx_k + j)))
        raw.append(curr)
    else:
        raw = [max(0, min(curr, idx_k)) for idx_k in nominal]

    # Enforce monotonic causal ordering (past -> present)
    raw.sort()
    raw[-1] = curr
    return raw


def compute_episode_frame_weights(
    ep_states: Tensor,
    ep_actions: Tensor | None = None,
    *,
    critical_weight: float = 1.0,
    idle_weight: float = 1.0,
    fps: int = 30,
) -> Tensor:
    """Compute per-frame training weights for a single episode `[T, D]`.

    - Downweights static/idle frames (`idle_weight`, e.g. 0.1) to prevent idle-copying.
    - Upweights frames within +/-0.5s of gripper open/close transitions and high contact
      load/current spikes (when 24/48-D telemetry is present) to `critical_weight` (e.g. 4.0).
    """
    t_len = int(ep_states.shape[0])
    weights = torch.ones(t_len, dtype=torch.float32)
    if t_len < 3 or (critical_weight == 1.0 and idle_weight == 1.0):
        return weights

    pos = ep_states[:, :6].float()
    # 1. Detect idle frames: arm and gripper move < 0.25 deg over next 5 ticks
    horizon_check = min(5, t_len - 1)
    if ep_actions is not None and ep_actions.ndim == 3 and ep_actions.shape[0] == t_len:
        fut_delta = (ep_actions[:, :horizon_check, :6].float() - pos.unsqueeze(1)).abs().amax(dim=(1, 2))
    else:
        shifted = torch.cat([pos[horizon_check:], pos[-1:].expand(horizon_check, -1)], dim=0)
        fut_delta = (shifted - pos).abs().amax(dim=-1)
    idle_mask = fut_delta < 0.25
    if idle_weight != 1.0:
        weights[idle_mask] = float(idle_weight)

    # 2. Detect critical grasp / release windows (+/- 0.5s around gripper transitions)
    grip = pos[:, 5]
    dg = torch.zeros(t_len, dtype=torch.float32)
    dg[1:] = (grip[1:] - grip[:-1]).abs()
    # Look at cumulative gripper change over 5-frame windows (~0.17s at 30fps)
    win_g = min(5, t_len - 1)
    grip_move = torch.zeros(t_len, dtype=torch.float32)
    grip_move[:-win_g] = (grip[win_g:] - grip[:-win_g]).abs()
    crit_seed = grip_move > 2.5

    # Also mark high contact load/current frames if 24-D/48-D telemetry is present
    if ep_states.shape[-1] >= 24:
        load = ep_states[:, 12:18].float().abs()
        curr = ep_states[:, 18:24].float().abs()
        # Gripper load/current or strong arm contact
        contact_seed = (load[:, 5] > 180.0) | (curr[:, 5] > 45.0) | (load[:, :5].amax(dim=-1) > 450.0)
        crit_seed = crit_seed | contact_seed

    if crit_seed.any() and critical_weight != 1.0:
        radius = max(1, int(round(0.5 * fps)))
        # Dilate critical mask by +/- radius frames
        kernel = torch.ones((1, 1, 2 * radius + 1), dtype=torch.float32)
        dilated = torch.nn.functional.conv1d(
            crit_seed.float().view(1, 1, -1), kernel, padding=radius
        ).view(-1)
        weights[dilated > 0.5] = float(critical_weight)

    return weights


def apply_calibration_and_dropout(
    states: Tensor,
    actions: Tensor | None,
    *,
    delta_actions: bool = False,
    offset_aug_deg: float = 0.0,
    state_dropout_prob: float = 0.0,
    arm_dim: int = 5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Apply delta arm-action conversion, +/-deg calibration offset aug, and state dropout.

    Works identically for both 6-D and 24-D/48-D state vectors (calibration offset applies
    strictly to arm joint positions `0..arm_dim-1`).
    """
    out_states = states
    out_actions = actions

    if delta_actions and out_actions is not None:
        out_actions = out_actions.clone()
        out_actions[..., :arm_dim] = out_actions[..., :arm_dim] - out_states[:, None, :arm_dim]

    if offset_aug_deg > 0.0:
        b = out_states.shape[0]
        cal_off = (
            torch.rand((b, arm_dim), device=out_states.device, dtype=out_states.dtype, generator=generator)
            * 2.0
            - 1.0
        ) * float(offset_aug_deg)
        out_states = out_states.clone()
        out_states[:, :arm_dim] = out_states[:, :arm_dim] + cal_off
        if out_actions is not None and not delta_actions:
            out_actions = out_actions.clone()
            out_actions[..., :arm_dim] = out_actions[..., :arm_dim] + cal_off[:, None, :]

    if state_dropout_prob > 0.0:
        b = out_states.shape[0]
        drop = torch.rand((b, 1), device=out_states.device, generator=generator) < float(state_dropout_prob)
        if drop.any():
            out_states = out_states.clone()
            out_states = out_states.masked_fill(drop, 0.0)

    return out_states, out_actions
