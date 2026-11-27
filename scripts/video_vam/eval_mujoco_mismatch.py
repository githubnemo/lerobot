#!/usr/bin/env python3
"""Controlled MuJoCo Closed-Loop Mismatch Eval Harness for SO-101 Policies.

Injects exact real-world deployment mismatches observed on `shabby` into `SO101CubeBoxEnv`
(or a lightweight kinematic mock when MuJoCo is unavailable on CPU CI):
  - `--cal-offset-deg`: constant per-episode calibration drift on arm joints 0..4
  - `--brightness-scale`: dimmer/brighter bench lighting (e.g. 0.70 = Cowling §9 shift)
  - `--eval-fps`: deployment control rate (10, 15, or 30 Hz) vs 30 Hz recording
  - `--sim-latency-ms`: simulated inference delay (e.g. 0, 300, 680 ms) with chunk discarding
  - Reports overall success, initiation rate, success-given-started (Cowling §11), and JSON stats.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

PredictFn = Callable[[Tensor, Tensor], Tensor]


@dataclass
class MismatchConfig:
    cal_offset_deg: float = 0.0
    brightness_scale: float = 1.0
    eval_fps: int = 10
    sim_latency_ms: float = 0.0
    execution_horizon: int = 15
    history_size: int = 5
    max_steps: int = 150
    initiation_threshold_deg: float = 5.0
    seed: int = 0


@dataclass
class MismatchEvalReport:
    num_episodes: int
    success_rate: float
    initiation_rate: float
    success_given_started: float
    mean_max_move_deg: float
    mean_steps: float
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MockSO101MismatchEnv:
    """Fast CPU-only 6-DOF SO-101 kinematic env for validating mismatch injection & eval loops."""

    def __init__(self, target_arm_deg: tuple[float, ...] = (15.0, -45.0, 50.0, 40.0, -30.0)) -> None:
        self._target = np.asarray(target_arm_deg, dtype=np.float32)
        self._state = np.asarray([-0.3, -86.7, 88.2, 71.1, -53.4, 17.7], dtype=np.float32)
        self._step = 0

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = np.random.default_rng(seed)
        self._state = np.asarray([-0.3, -86.7, 88.2, 71.1, -53.4, 17.7], dtype=np.float32)
        self._state[:5] += rng.uniform(-2.0, 2.0, size=5).astype(np.float32)
        self._step = 0
        return self._obs(), {}

    def _obs(self) -> dict[str, Any]:
        val = int(np.clip(120 + self._step * 2, 0, 255))
        pixels = np.full((480, 640, 3), val, dtype=np.uint8)
        return {"pixels": pixels, "agent_pos": self._state.copy()}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._step += 1
        cmd = np.asarray(action, dtype=np.float32)[:6]
        self._state = 0.6 * self._state + 0.4 * cmd
        dist = float(np.linalg.norm(self._state[:5] - self._target))
        success = dist < 8.0 and self._state[5] < 12.0
        return self._obs(), (1.0 if success else 0.0), success, False, {"is_success": success}

    def close(self) -> None:
        pass


def run_mismatch_eval(
    predict_fn: PredictFn,
    env_factory: Callable[[], Any],
    *,
    num_episodes: int = 10,
    cfg: MismatchConfig | None = None,
) -> MismatchEvalReport:
    """Evaluate a policy closed-loop under controlled calibration, lighting, and latency mismatches."""
    cfg = cfg or MismatchConfig()
    rng = np.random.default_rng(cfg.seed)
    delay_ticks = max(0, int(math.ceil((cfg.sim_latency_ms / 1000.0) * cfg.eval_fps)))
    exec_h = max(1, int(cfg.execution_horizon))

    successes = 0
    started_count = 0
    success_when_started = 0
    max_moves: list[float] = []
    ep_steps: list[int] = []

    env = env_factory()
    try:
        for ep_idx in range(num_episodes):
            obs, _ = env.reset(seed=cfg.seed + ep_idx)
            # Constant per-episode calibration drift on arm joints 0..4
            cal_offset = (
                rng.uniform(-cfg.cal_offset_deg, cfg.cal_offset_deg, size=5).astype(np.float32)
                if cfg.cal_offset_deg > 0.0
                else np.zeros(5, dtype=np.float32)
            )
            init_true_pos = np.asarray(obs["agent_pos"][:6], dtype=np.float32).copy()
            frames: deque[Tensor] = deque(maxlen=max(1, cfg.history_size))
            queue: deque[np.ndarray] = deque()
            ep_success = False
            max_move = 0.0
            steps_taken = 0

            for step_i in range(cfg.max_steps):
                steps_taken = step_i + 1
                raw_img = np.asarray(obs["pixels"], dtype=np.float32)
                if cfg.brightness_scale != 1.0:
                    raw_img = np.clip(raw_img * cfg.brightness_scale, 0.0, 255.0)
                frame_t = torch.from_numpy(raw_img.astype(np.uint8)).permute(2, 0, 1).contiguous()
                if not frames:
                    for _ in range(frames.maxlen or 1):
                        frames.append(frame_t)
                else:
                    frames.append(frame_t)

                true_state = np.asarray(obs["agent_pos"][:6], dtype=np.float32)
                max_move = max(max_move, float(np.abs(true_state[:5] - init_true_pos[:5]).max()))

                # Robot reports calibrated (shifted) state: q_obs = q_true + cal_offset
                obs_state = true_state.copy()
                obs_state[:5] = obs_state[:5] + cal_offset

                if not queue:
                    rgb_win = torch.stack(list(frames), dim=1).unsqueeze(0)  # [1, 3, T, H, W]
                    st_t = torch.from_numpy(obs_state).unsqueeze(0)
                    with torch.no_grad():
                        chunk = predict_fn(rgb_win, st_t)[0].detach().cpu().numpy()
                    # Simulate inference delay discarding early chunk actions
                    start_k = min(delay_ticks, max(0, len(chunk) - 1))
                    usable = chunk[start_k : start_k + exec_h]
                    for row in usable:
                        # Convert commanded calibrated target back to true physical joint angle
                        true_cmd = row[:6].copy()
                        true_cmd[:5] = true_cmd[:5] - cal_offset
                        queue.append(true_cmd)

                cmd = queue.popleft()
                obs, reward, terminated, truncated, info = env.step(cmd)
                if float(reward) >= 0.5 or bool(info.get("is_success", False)):
                    ep_success = True
                    break
                if terminated or truncated:
                    break

            started = max_move >= cfg.initiation_threshold_deg
            if started:
                started_count += 1
                if ep_success:
                    success_when_started += 1
            if ep_success:
                successes += 1
            max_moves.append(max_move)
            ep_steps.append(steps_taken)
    finally:
        env.close()

    return MismatchEvalReport(
        num_episodes=num_episodes,
        success_rate=round(successes / max(1, num_episodes), 4),
        initiation_rate=round(started_count / max(1, num_episodes), 4),
        success_given_started=round(
            success_when_started / max(1, started_count) if started_count > 0 else 0.0, 4
        ),
        mean_max_move_deg=round(float(np.mean(max_moves)) if max_moves else 0.0, 3),
        mean_steps=round(float(np.mean(ep_steps)) if ep_steps else 0.0, 2),
        config=asdict(cfg),
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Closed-loop MuJoCo mismatch eval harness for SO-101.")
    p.add_argument("--policy-type", choices=("video_vam", "smolvla"), required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--cal-offset-deg", type=float, default=5.0)
    p.add_argument("--brightness-scale", type=float, default=0.7)
    p.add_argument("--eval-fps", type=int, default=10)
    p.add_argument("--sim-latency-ms", type=float, default=300.0)
    p.add_argument("--execution-horizon", type=int, default=15)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--task", type=str, default="Grab the purple cube and put it into the cardboard box")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--mock-env", action="store_true", help="Use CPU kinematic mock instead of MuJoCo")
    p.add_argument("--output-json", type=Path, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
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

    def env_factory() -> Any:
        if args.mock_env:
            return MockSO101MismatchEnv()
        from lerobot.envs.so101_cube_box import SO101CubeBoxEnv

        return SO101CubeBoxEnv(
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=640,
            observation_height=480,
            randomize=True,
            reward_mode="sparse",
        )

    cfg = MismatchConfig(
        cal_offset_deg=args.cal_offset_deg,
        brightness_scale=args.brightness_scale,
        eval_fps=args.eval_fps,
        sim_latency_ms=args.sim_latency_ms,
        execution_horizon=args.execution_horizon,
        history_size=hist_sz,
        max_steps=args.max_steps,
    )
    report = run_mismatch_eval(predict_fn, env_factory, num_episodes=args.num_episodes, cfg=cfg)
    payload = report.to_dict()
    print(json.dumps(payload, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
