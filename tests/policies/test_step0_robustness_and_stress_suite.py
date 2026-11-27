"""CPU-only unit tests for Step 0 robustness primitives, Cosmos 3 / SmolVLA parity,
the 5-metric offline stress-test suite, and the MuJoCo mismatch eval harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
from lerobot.policies.vam.modeling_video_vam import VideoVAMPolicy, restore_uint8_history
from lerobot.policies.vam.robustness import (
    apply_calibration_and_dropout,
    compute_episode_frame_weights,
    resolve_state_dim,
    sample_history_window_indices,
    slice_state_features,
)
from lerobot.policies.vam.smol_expert import SmolExpertActionDecoder, SmolVLANormalizer


def _load_script(rel_path: str, mod_name: str):
    path = Path(__file__).resolve().parents[2] / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_state_feature_slicing_and_dims() -> None:
    raw48 = torch.arange(48, dtype=torch.float32).unsqueeze(0)
    assert resolve_state_dim(48, "pos") == 6
    assert resolve_state_dim(48, "pos_vel_load_curr") == 24
    assert resolve_state_dim(48, "all") == 48
    assert resolve_state_dim(6, "pos_vel_load_curr") == 6
    assert slice_state_features(raw48, "pos").shape == (1, 6)
    assert slice_state_features(raw48, "pos_vel_load_curr").shape == (1, 24)
    assert slice_state_features(raw48, "all").shape == (1, 48)


@pytest.mark.parametrize("history_size", [1, 3, 5])
def test_sample_history_window_causal_monotonic_and_static(history_size: int) -> None:
    gen = torch.Generator().manual_seed(123)
    for frame in (0, 2, 10, 50):
        win = sample_history_window_indices(
            frame=frame,
            ep_len=60,
            history_size=history_size,
            base_stride=3,
            min_stride=1,
            max_stride=4,
            jitter_frames=1,
            static_prob=0.0,
            generator=gen,
        )
        assert len(win) == history_size
        assert win[-1] == frame
        assert all(0 <= x <= frame for x in win)
        assert win == sorted(win)

    static_win = sample_history_window_indices(
        frame=25,
        ep_len=60,
        history_size=history_size,
        static_prob=1.0,
        generator=gen,
    )
    assert static_win == [25] * history_size


def test_critical_and_idle_frame_weights_with_contact_telemetry() -> None:
    t_len = 90
    states = torch.zeros((t_len, 24), dtype=torch.float32)
    # Frames 0..19: idle (zero motion, gripper steady at 30)
    states[:20, 5] = 30.0
    # Frames 20..59: transit (linear motion, gripper open=30)
    for t in range(20, 60):
        states[t, 0] = float(t - 20) * 1.5
        states[t, 5] = 30.0
    # Frame 60..66: gripper closes sharply (grasp event!)
    for t in range(60, 70):
        states[t, 0] = 60.0
        states[t, 5] = max(5.0, 30.0 - (t - 59) * 5.0)
        states[t, 17] = 260.0  # gripper load spike
    # Frames 70..89: hold/transit
    for t in range(70, t_len):
        states[t, 0] = 60.0 + (t - 70) * 1.0
        states[t, 5] = 5.0

    weights = compute_episode_frame_weights(states, critical_weight=4.0, idle_weight=0.1, fps=30)
    assert weights.shape == (t_len,)
    assert float(weights[5].item()) == pytest.approx(0.1)
    assert float(weights[35].item()) == pytest.approx(1.0)
    assert float(weights[62].item()) == pytest.approx(4.0)


def test_calibration_offset_delta_actions_and_dropout() -> None:
    gen = torch.Generator().manual_seed(7)
    states = torch.tensor([[10.0, -20.0, 30.0, 40.0, -50.0, 15.0]], dtype=torch.float32)
    actions = torch.tensor([[[15.0, -18.0, 28.0, 41.0, -48.0, 8.0]] * 30], dtype=torch.float32)

    # Delta actions: arm targets are q_{t+k} - q_t (invariant to calibration offset); gripper stays 8.0
    st_out, act_out = apply_calibration_and_dropout(
        states, actions, delta_actions=True, offset_aug_deg=5.0, state_dropout_prob=0.0, generator=gen
    )
    assert act_out is not None
    expected_arm_delta = torch.tensor([5.0, 2.0, -2.0, 1.0, 2.0])
    assert torch.allclose(act_out[0, 0, :5], expected_arm_delta, atol=1e-5)
    assert float(act_out[0, 0, 5].item()) == pytest.approx(8.0)
    assert not torch.allclose(st_out[0, :5], states[0, :5])
    assert float(st_out[0, 5].item()) == pytest.approx(15.0)

    # State dropout = 1.0 zeroes state cleanly
    st_drop, _ = apply_calibration_and_dropout(
        states, actions, delta_actions=True, state_dropout_prob=1.0, generator=gen
    )
    assert torch.all(st_drop == 0.0)


@pytest.mark.parametrize("history_size,expected_tokens", [(1, 300), (3, 600), (5, 600)])
def test_video_vam_policy_history_delta_rtc_and_48d_slicing(
    history_size: int, expected_tokens: int
) -> None:
    train_mod = _load_script("scripts/video_vam/train_smolexpert.py", "train_smolexpert_mod")
    state_dim = 24
    norm = SmolVLANormalizer(
        state_mean=torch.zeros(state_dim),
        state_std=torch.ones(state_dim),
        action_mean=torch.zeros(6),
        action_std=torch.ones(6),
    )
    decoder = train_mod.build_synthetic_tiny_decoder(
        input_channels=16,
        normalizer=norm,
        device=torch.device("cpu"),
    )
    # Test weighted flow_matching_loss on CPU
    s_b = torch.randn(2, state_dim)
    a_b = torch.randn(2, 30, 6)
    c_b = torch.randn(2, expected_tokens, 16)
    pad_b = torch.zeros(2, 30, dtype=torch.bool)
    w_b = torch.tensor([4.0, 0.1])
    loss_w = decoder.flow_matching_loss(s_b, a_b, c_b, action_is_pad=pad_b, sample_weight=w_b)
    assert torch.isfinite(loss_w) and float(loss_w.item()) > 0.0

    # Dummy extractor matching Cosmos 3 token count
    class DummyExtractor:
        def extract(self, rgb_frames: torch.Tensor):
            t_in = rgb_frames.shape[2]
            t_lat = 1 if t_in <= 1 else (1 + (t_in - 1 + 3) // 4)
            return SimpleNamespace(features=torch.zeros((1, t_lat * 300, 2048), dtype=torch.float32))

    # Build matching 2048-ch tiny decoder for VideoVAMPolicy forward
    decoder2 = train_mod.build_synthetic_tiny_decoder(
        input_channels=2048,
        normalizer=norm,
        device=torch.device("cpu"),
    )
    cfg = VideoVAMConfig(
        backend="cosmos3_edge",
        device="cpu",
        history_size=history_size,
        delta_actions=True,
        state_dim=state_dim,
        state_feature_groups="pos_vel_load_curr",
        rtc_config=RTCConfig(enabled=True, execution_horizon=10, max_guidance_weight=2.0),
    )
    policy = VideoVAMPolicy(
        cfg, decoder=decoder2, extractor=DummyExtractor(), prompt_embedding=torch.zeros((1, 1, 1))
    )
    policy.init_rtc_processor()

    # Feed 48-D telemetry state + [1, 3, T, 480, 640] uint8 history + RTC leftover
    hist_u8 = torch.randint(0, 255, (1, 3, history_size, 480, 640), dtype=torch.uint8)
    assert restore_uint8_history(hist_u8, expected_frames=history_size).shape == (1, 3, history_size, 480, 640)
    state48 = torch.zeros((1, 48), dtype=torch.float32)
    state48[0, :5] = torch.tensor([10.0, -30.0, 40.0, 20.0, -15.0])
    leftover = torch.zeros((15, 6), dtype=torch.float32)
    leftover[:, :5] = state48[0, :5]

    out = policy.predict_action_chunk(
        {
            "observation.images.front.history": hist_u8.float().div(255.0),
            "observation.state": state48,
            "observation.history_frame_index": torch.tensor([0]),
        },
        prev_chunk_left_over=leftover,
        inference_delay=4,
    )
    assert out.shape == (1, 30, 6)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("history_size", [1, 3, 5])
def test_smolvla_prepare_images_multiframe_and_state_slicing(history_size: int) -> None:
    cfg = SmolVLAConfig(
        history_size=history_size,
        delta_actions=True,
        state_feature_groups="pos_vel_load_curr",
        state_dropout_prob=0.25,
        resize_imgs_with_padding=(64, 64),
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(24,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    # Test prepare_images & prepare_state directly without downloading VLM weights
    stub = SimpleNamespace(config=cfg)
    raw_hist = torch.rand((2, 3, history_size, 64, 64), dtype=torch.float32)
    batch = {
        "observation.images.front": raw_hist[:, :, -1],
        "observation.images.front.history": raw_hist,
        "observation.state": torch.randn((2, 48), dtype=torch.float32),
    }
    imgs, masks = SmolVLAPolicy.prepare_images(stub, batch)
    assert len(imgs) == history_size
    assert len(masks) == history_size
    assert all(im.shape == (2, 3, 64, 64) for im in imgs)

    st = SmolVLAPolicy.prepare_state(stub, batch)
    assert st.shape == (2, cfg.max_state_dim)
    # Entries beyond 24 (padded zeros) must be exactly zero
    assert torch.all(st[:, 24:] == 0.0)


def test_stress_suite_separates_visual_servoing_from_stuck_copycat() -> None:
    stress = _load_script("scripts/video_vam/eval_stress_suite.py", "eval_stress_suite_mod")
    episodes = []
    for ep_i in range(3):
        n = 60
        # Encode target pan in image mean intensity so a visual policy can read it from the image
        target_pan = 20.0 * (ep_i - 1)  # -20, 0, +20
        img_val = int(100 + ep_i * 50)
        frames = torch.full((n, 3, 16, 16), img_val, dtype=torch.uint8)
        states = torch.zeros((n, 6), dtype=torch.float32)
        actions = torch.zeros((n, 30, 6), dtype=torch.float32)
        for t in range(n):
            frac = min(1.0, t / 30.0)
            states[t, 0] = frac * target_pan
            states[t, 5] = 30.0 if t < 35 else 5.0
            for k in range(30):
                fk = min(1.0, (t + k + 1) / 30.0)
                actions[t, k, 0] = fk * target_pan
                actions[t, k, 1] = 15.0  # clear movement away from rest pose
                actions[t, k, 5] = 30.0 if (t + k) < 35 else 5.0
        episodes.append(
            stress.EpisodeData(
                episode_index=ep_i,
                frames_u8=frames,
                states=states,
                actions=actions,
                action_is_pad=torch.zeros((n, 30), dtype=torch.bool),
            )
        )

    def visual_policy(rgb_u8: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Reads target_pan from image brightness, adds delta toward target
        mean_v = rgb_u8[:, :, -1].float().mean(dim=(1, 2, 3))  # [B]
        est_ep = (mean_v - 100.0) / 50.0
        target_pan = 20.0 * (est_ep - 1.0)
        out = state[:, None, :6].repeat(1, 30, 1).clone()
        out[:, :, 0] = target_pan.unsqueeze(1)
        out[:, :, 1] = state[:, 1:2] + 15.0
        return out

    def stuck_copycat_policy(rgb_u8: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del rgb_u8
        return state[:, None, :6].repeat(1, 30, 1)

    rep_good = stress.run_stress_suite(visual_policy, episodes, history_size=3, history_stride=2, eval_stride=10)
    rep_bad = stress.run_stress_suite(stuck_copycat_policy, episodes, history_size=3, history_stride=2, eval_stride=10)

    assert rep_good.initiation_rate == 1.0
    assert rep_bad.initiation_rate == 0.0
    assert rep_good.cvs_visual_delta_deg > 1.0
    assert rep_bad.cvs_visual_delta_deg == 0.0
    assert rep_good.cvs_ratio > rep_bad.cvs_ratio

    assert stress.spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)
    assert stress.spearman_rank_correlation([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_mujoco_mismatch_harness_detects_calibration_and_initiation_failures() -> None:
    harness = _load_script("scripts/video_vam/eval_mujoco_mismatch.py", "eval_mujoco_mismatch_mod")
    target = torch.tensor([15.0, -45.0, 50.0, 40.0, -30.0, 8.0], dtype=torch.float32)

    # Ideal closed-loop visual policy (immune to calibration drift because it servos to physical target)
    # Wait: in MockSO101MismatchEnv, q_obs = q_true + cal_offset, and true_cmd = cmd - cal_offset.
    # A policy that outputs `q_obs + delta_to_true_target` stays invariant to cal_offset!
    # Here we test a delta-style policy that moves smoothly toward target vs. a frozen policy.
    def moving_policy(rgb_u8: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del rgb_u8
        return target.view(1, 1, 6).repeat(state.shape[0], 30, 1)

    def frozen_policy(rgb_u8: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        del rgb_u8
        return state[:, None, :6].repeat(1, 30, 1)

    rep_clean = harness.run_mismatch_eval(
        moving_policy,
        harness.MockSO101MismatchEnv,
        num_episodes=5,
        cfg=harness.MismatchConfig(cal_offset_deg=0.0, brightness_scale=0.7, sim_latency_ms=300.0),
    )
    rep_shifted = harness.run_mismatch_eval(
        moving_policy,
        harness.MockSO101MismatchEnv,
        num_episodes=5,
        cfg=harness.MismatchConfig(cal_offset_deg=15.0, brightness_scale=0.7, sim_latency_ms=300.0),
    )
    rep_frozen = harness.run_mismatch_eval(
        frozen_policy,
        harness.MockSO101MismatchEnv,
        num_episodes=5,
        cfg=harness.MismatchConfig(cal_offset_deg=0.0, sim_latency_ms=300.0),
    )

    assert rep_clean.success_rate == 1.0
    assert rep_clean.initiation_rate == 1.0
    # Large calibration drift hurts a non-delta absolute policy!
    assert rep_shifted.success_rate < rep_clean.success_rate
    # Frozen policy has 0% initiation and 0% success
    assert rep_frozen.initiation_rate == 0.0
    assert rep_frozen.success_rate == 0.0
