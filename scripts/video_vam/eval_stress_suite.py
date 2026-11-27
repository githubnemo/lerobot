#!/usr/bin/env python3
"""5-Metric Offline Stress-Test Suite for SO-101 Imitation Policies (Cosmos 3 Video-VAM & SmolVLA).

Directly targets the offline-to-closed-loop blind spots identified in Cowling (arXiv:2611.00283v1)
and observed live on `shabby`:
  1. Critical-Interval Grasp/Release RMSE (+/-0.5s around gripper close/open transitions).
  2. Static-Start Initiation Score (% of episodes commanding >5 deg motion from repeated rest frames).
  3. Counterfactual Visual Sensitivity (CVS): ratio of action change under image swap vs. +/-5 deg state shift.
  4. Calibration (+/-5 deg) + Dim-Lighting (0.7x brightness) Perturbation Critical RMSE.
  5. Consecutive-Chunk Self-Consistency: L2/MAE agreement between overlapping predicted chunks.

Works with any callable `predict_fn(rgb_u8: [B,3,T,H,W], state: [B,D]) -> actions [B,H,6]`
so unit tests run on CPU in milliseconds and GPU eval runs on any VideoVAM or SmolVLA checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from lerobot.policies.vam.robustness import compute_episode_frame_weights

PredictFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class EpisodeData:
    """In-memory held-out episode used by the stress suite."""

    episode_index: int
    frames_u8: Tensor  # [N, 3, H, W] uint8 RGB
    states: Tensor  # [N, D] float32 physical units
    actions: Tensor  # [N, K, 6] float32 physical units (horizon K)
    action_is_pad: Tensor  # [N, K] bool


@dataclass
class StressSuiteReport:
    uniform_first5_rmse: float
    uniform_full_rmse: float
    critical_first5_rmse: float
    critical_full_rmse: float
    critical_anchor_count: int
    initiation_rate: float
    initiation_mean_move_deg: float
    cvs_visual_delta_deg: float
    cvs_state_delta_deg: float
    cvs_ratio: float
    perturbed_critical_first5_rmse: float
    perturbation_degradation_deg: float
    chunk_self_consistency_deg: float
    num_episodes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _masked_rmse(pred: Tensor, target: Tensor, valid: Tensor) -> float:
    """Compute root-mean-square error over valid (non-padded) [B, H, 6] entries."""
    v = valid.unsqueeze(-1).to(dtype=torch.float64)
    denom = float(v.sum().item()) * float(pred.shape[-1])
    if denom <= 0.0:
        return float("nan")
    sq = ((pred.double() - target.double()).square() * v).sum().item()
    return math.sqrt(sq / denom)


def _build_window(frames_u8: Tensor, idx: int, history_size: int, history_stride: int) -> Tensor:
    """Build [1, 3, T, H, W] uint8 causal window ending at `idx`."""
    n_frames = int(frames_u8.shape[0])
    curr = max(0, min(n_frames - 1, int(idx)))
    idxs = [max(0, min(curr, curr - (history_size - 1 - k) * history_stride)) for k in range(history_size)]
    # frames_u8[idxs]: [T, 3, H, W] -> [1, 3, T, H, W]
    return frames_u8[idxs].permute(1, 0, 2, 3).unsqueeze(0).contiguous()


def _detect_critical_indices(ep: EpisodeData, fps: int = 30) -> list[int]:
    """Identify frames within +/-0.5s of gripper transitions (or contact spikes)."""
    w = compute_episode_frame_weights(
        ep.states, ep.actions, critical_weight=4.0, idle_weight=0.1, fps=fps
    )
    idxs = torch.nonzero(w > 1.5, as_tuple=False).view(-1).tolist()
    return [int(i) for i in idxs]


@torch.no_grad()
def run_stress_suite(
    predict_fn: PredictFn,
    episodes: list[EpisodeData],
    *,
    history_size: int = 5,
    history_stride: int = 3,
    eval_stride: int = 10,
    fps: int = 30,
    cal_offset_deg: float = 5.0,
    brightness_scale: float = 0.70,
    initiation_threshold_deg: float = 5.0,
    consistency_step_gap: int = 3,
    seed: int = 42,
) -> StressSuiteReport:
    """Run all 5 closed-loop stress probes on held-out episodes."""
    if not episodes:
        raise ValueError("episodes must be non-empty")

    gen = torch.Generator().manual_seed(seed)
    uni_preds, uni_tgts, uni_valids = [], [], []
    crit_preds, crit_tgts, crit_valids = [], [], []
    pert_preds, pert_tgts, pert_valids = [], [], []

    init_moves: list[float] = []
    cvs_vis_deltas: list[float] = []
    cvs_state_deltas: list[float] = []
    consistency_errs: list[float] = []

    for ep_pos, ep in enumerate(episodes):
        n = int(ep.frames_u8.shape[0])
        if n < 2:
            continue
        other_ep = episodes[(ep_pos + 1) % len(episodes)]
        other_mid = int(other_ep.frames_u8.shape[0]) // 2

        # 1. Static-Start Initiation Probe: feed repeated rest frame [I_0, ..., I_0]
        static_rgb = ep.frames_u8[0:1].repeat(history_size, 1, 1, 1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()
        st0 = ep.states[0:1]
        pred0 = predict_fn(static_rgb, st0)  # [1, H, 6]
        # Max arm excursion over the chunk from initial pose (joints 0..4)
        move_deg = float((pred0[0, :, :5] - st0[0, None, :5]).abs().amax().item())
        init_moves.append(move_deg)

        # 2. Uniform anchor sweep + Consecutive-Chunk Self-Consistency
        anchors = list(range(0, n, max(1, eval_stride)))
        for idx in anchors:
            rgb = _build_window(ep.frames_u8, idx, history_size, history_stride)
            st = ep.states[idx : idx + 1]
            pred = predict_fn(rgb, st)
            h_len = min(int(pred.shape[1]), int(ep.actions.shape[1]))
            uni_preds.append(pred[:, :h_len, :])
            uni_tgts.append(ep.actions[idx : idx + 1, :h_len, :])
            uni_valids.append(~ep.action_is_pad[idx : idx + 1, :h_len])

            # Consecutive-chunk consistency at idx + consistency_step_gap
            nxt = idx + consistency_step_gap
            if nxt < n and h_len > consistency_step_gap + 2:
                rgb_n = _build_window(ep.frames_u8, nxt, history_size, history_stride)
                st_n = ep.states[nxt : nxt + 1]
                pred_n = predict_fn(rgb_n, st_n)
                overlap = min(12, h_len - consistency_step_gap, int(pred_n.shape[1]))
                if overlap > 0:
                    diff = (
                        pred[0, consistency_step_gap : consistency_step_gap + overlap, :5]
                        - pred_n[0, :overlap, :5]
                    ).abs().mean().item()
                    consistency_errs.append(float(diff))

        # 3. Critical-Interval Grasp/Release RMSE + CVS + Calibration/Lighting Perturbation Probe
        crit_idxs = _detect_critical_indices(ep, fps=fps)
        if not crit_idxs:
            crit_idxs = anchors
        sampled_crit = crit_idxs[:: max(1, len(crit_idxs) // 16)]

        for idx in sampled_crit:
            rgb = _build_window(ep.frames_u8, idx, history_size, history_stride)
            st = ep.states[idx : idx + 1]
            pred = predict_fn(rgb, st)
            h_len = min(int(pred.shape[1]), int(ep.actions.shape[1]))
            crit_preds.append(pred[:, :h_len, :])
            crit_tgts.append(ep.actions[idx : idx + 1, :h_len, :])
            crit_valids.append(~ep.action_is_pad[idx : idx + 1, :h_len])

            # Counterfactual Visual Sensitivity (swap image vs. perturb state by +cal_offset_deg)
            swap_rgb = _build_window(other_ep.frames_u8, other_mid, history_size, history_stride)
            pred_swap = predict_fn(swap_rgb, st)
            vis_delta = float((pred_swap[0, :5, :5] - pred[0, :5, :5]).abs().mean().item())
            cvs_vis_deltas.append(vis_delta)

            # Random +/- cal_offset_deg sign per arm joint
            signs = torch.where(torch.rand((1, 5), generator=gen) > 0.5, 1.0, -1.0)
            st_pert = st.clone()
            st_pert[:, :5] = st_pert[:, :5] + signs * float(cal_offset_deg)
            pred_st_pert = predict_fn(rgb, st_pert)
            # In delta-space ideal visual servoing, target shifts with physical frame (or stays invariant
            # in delta); measure raw sensitivity of output action to state shift:
            st_delta = float((pred_st_pert[0, :5, :5] - pred[0, :5, :5]).abs().mean().item())
            cvs_state_deltas.append(st_delta)

            # Combined Calibration + Dim-Lighting Perturbation Probe
            dim_rgb = (rgb.float() * float(brightness_scale)).round().clamp(0, 255).to(torch.uint8)
            pred_pert = predict_fn(dim_rgb, st_pert)
            # Note: under a physical calibration offset +delta, the physical servo target for the same
            # world position shifts by +delta on arm joints 0..4.
            tgt_pert = ep.actions[idx : idx + 1, :h_len, :].clone()
            tgt_pert[:, :, :5] = tgt_pert[:, :, :5] + signs.unsqueeze(1) * float(cal_offset_deg)
            pert_preds.append(pred_pert[:, :h_len, :])
            pert_tgts.append(tgt_pert)
            pert_valids.append(~ep.action_is_pad[idx : idx + 1, :h_len])

    def _cat_rmse(ps: list[Tensor], ts: list[Tensor], vs: list[Tensor], steps: int | None = None) -> float:
        if not ps:
            return float("nan")
        p = torch.cat(ps, dim=0)
        t = torch.cat(ts, dim=0)
        v = torch.cat(vs, dim=0)
        if steps is not None:
            k = min(steps, int(p.shape[1]))
            p, t, v = p[:, :k], t[:, :k], v[:, :k]
        return _masked_rmse(p, t, v)

    uni_f5 = _cat_rmse(uni_preds, uni_tgts, uni_valids, steps=5)
    uni_full = _cat_rmse(uni_preds, uni_tgts, uni_valids, steps=None)
    crit_f5 = _cat_rmse(crit_preds, crit_tgts, crit_valids, steps=5)
    crit_full = _cat_rmse(crit_preds, crit_tgts, crit_valids, steps=None)
    pert_f5 = _cat_rmse(pert_preds, pert_tgts, pert_valids, steps=5)

    init_rate = (
        sum(1 for m in init_moves if m >= initiation_threshold_deg) / max(1, len(init_moves))
    )
    init_mean = sum(init_moves) / max(1, len(init_moves))
    mean_vis = sum(cvs_vis_deltas) / max(1, len(cvs_vis_deltas))
    mean_st = sum(cvs_state_deltas) / max(1, len(cvs_state_deltas))
    cvs_ratio = mean_vis / max(mean_st, 1e-4)
    consistency_mean = sum(consistency_errs) / max(1, len(consistency_errs))

    return StressSuiteReport(
        uniform_first5_rmse=round(uni_f5, 4),
        uniform_full_rmse=round(uni_full, 4),
        critical_first5_rmse=round(crit_f5, 4),
        critical_full_rmse=round(crit_full, 4),
        critical_anchor_count=len(crit_preds),
        initiation_rate=round(init_rate, 4),
        initiation_mean_move_deg=round(init_mean, 3),
        cvs_visual_delta_deg=round(mean_vis, 3),
        cvs_state_delta_deg=round(mean_st, 3),
        cvs_ratio=round(cvs_ratio, 3),
        perturbed_critical_first5_rmse=round(pert_f5, 4),
        perturbation_degradation_deg=round(pert_f5 - crit_f5, 4),
        chunk_self_consistency_deg=round(consistency_mean, 3),
        num_episodes=len(episodes),
    )


def spearman_rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Exact tie-aware Spearman rank correlation (zero SciPy dependency)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def _ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_r = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_r
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run 5-Metric Offline Stress-Test Suite on a checkpoint.")
    p.add_argument("--policy-type", choices=("video_vam", "smolvla"), required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset-repo-id", type=str, default="Orellius/so101_sort_cubes_no_top")
    p.add_argument("--dataset-revision", type=str, default="a7c156de0e84a81e19457e6d783e4316e9f901e1")
    p.add_argument("--episodes", type=str, default="68:77", help="Slice start:end (default 68:77)")
    p.add_argument("--history-stride", type=int, default=3)
    p.add_argument("--eval-stride", type=int, default=15)
    p.add_argument("--task", type=str, default="Grab the purple cube and put it into the cardboard box")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    s_ep, e_ep = (int(x) for x in args.episodes.split(":"))
    ep_list = list(range(s_ep, e_ep))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(
        args.dataset_repo_id,
        revision=args.dataset_revision,
        episodes=ep_list,
        return_uint8=True,
    )
    fps = int(ds.fps)
    horizon = 30
    by_ep: dict[int, list[tuple[Tensor, Tensor, Tensor]]] = {}
    for i in range(len(ds)):
        row = ds[i]
        ep_i = int(row["episode_index"].item())
        by_ep.setdefault(ep_i, []).append(
            (row["observation.images.front"], row["observation.state"].float(), row["action"].float())
        )
    episodes: list[EpisodeData] = []
    for ep_i, rows in sorted(by_ep.items()):
        frames = torch.stack([r[0] for r in rows], dim=0)
        states = torch.stack([r[1] for r in rows], dim=0)
        acts_1 = torch.stack([r[2] for r in rows], dim=0)
        n = len(rows)
        act_chunks = torch.zeros((n, horizon, 6), dtype=torch.float32)
        pad_mask = torch.zeros((n, horizon), dtype=torch.bool)
        for t in range(n):
            for k in range(horizon):
                if t + k < n:
                    act_chunks[t, k] = acts_1[t + k, :6]
                else:
                    act_chunks[t, k] = acts_1[-1, :6]
                    pad_mask[t, k] = True
        episodes.append(EpisodeData(ep_i, frames, states, act_chunks, pad_mask))

    device = torch.device(args.device)
    if args.policy_type == "video_vam":
        from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy

        policy = VideoVAMPolicy.from_pretrained(args.checkpoint)
        policy.to(device).eval()
        hist_sz = int(getattr(policy.config, "history_size", 5))
        hist_key = f"{policy.config.camera_key}.history"

        def predict_fn(rgb_u8: Tensor, state: Tensor) -> Tensor:
            batch = {
                hist_key: rgb_u8.to(device=device, dtype=torch.float32).div(255.0),
                "observation.state": state.to(device=device, dtype=torch.float32),
            }
            return policy.predict_action_chunk(batch).cpu()
    else:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        policy = SmolVLAPolicy.from_pretrained(args.checkpoint)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)
        hist_sz = int(getattr(policy.config, "history_size", 1))
        use_delta = bool(getattr(policy.config, "delta_actions", False))

        def predict_fn(rgb_u8: Tensor, state: Tensor) -> Tensor:
            raw_obs: dict[str, Any] = {
                "observation.images.front": rgb_u8[:, :, -1].float().div(255.0),
                "observation.state": state.float(),
                "task": args.task,
            }
            batch = pre(raw_obs)
            if hist_sz > 1:
                batch["observation.images.front.history"] = rgb_u8.to(device=device, dtype=torch.float32).div(
                    255.0
                )
            chunk = policy.predict_action_chunk(batch)
            out = post(chunk).cpu()
            if out.ndim == 2:
                out = out.unsqueeze(0)
            if use_delta:
                out = out.clone()
                out[:, :, :5] = out[:, :, :5] + state[:, None, :5]
            return out

    report = run_stress_suite(
        predict_fn,
        episodes,
        history_size=hist_sz,
        history_stride=args.history_stride,
        eval_stride=args.eval_stride,
        fps=fps,
    )
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
