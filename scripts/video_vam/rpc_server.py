#!/usr/bin/env python3
"""Minimal HTTP RPC server for SmolVLA or Video-VAM on abakus."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from PIL import Image

HOST = "127.0.0.1"
PORT = 8766
HISTORY = 5


def _frame(value: Any) -> torch.Tensor:
    if isinstance(value, str):
        try:
            raw = base64.b64decode(value, validate=True)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            return (
                torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                .reshape(image.height, image.width, 3)
                .clone()
            )
        except Exception as exc:
            raise ValueError(f"invalid PNG base64 frame: {exc}") from exc
    if isinstance(value, list):
        if len(value) == HISTORY and all(isinstance(item, str) for item in value):
            return torch.stack([_frame(item) for item in value])
        tensor = torch.tensor(value, dtype=torch.uint8)
        if tensor.ndim == 3:
            return tensor
        if tensor.ndim == 4 and tensor.shape[0] == HISTORY:
            return tensor
    raise ValueError("frame must be PNG base64 or an [H,W,3] uint8 list")


def _validate_frame(frame: torch.Tensor) -> torch.Tensor:
    if frame.ndim != 3 or tuple(frame.shape[-1:]) != (3,):
        raise ValueError(f"frame must have shape [H,W,3], got {tuple(frame.shape)}")
    if tuple(frame.shape[:2]) != (480, 640):
        raise ValueError(f"frame must have shape [480,640,3], got {tuple(frame.shape)}")
    return frame.contiguous()


PROTOCOL_VERSION = 2
MAX_REQUEST_BYTES = 24 * 1024 * 1024


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def validate_chunk(value: Any, *, horizon: int, name: str = "action_chunk") -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tuple(tensor.shape) != (horizon, 6) or not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite [{horizon}, 6]")
    return tensor


def _smooth_chunk_nyquist(chunk: torch.Tensor, passes: int = 2) -> torch.Tensor:
    if chunk.ndim != 2 or chunk.shape[0] < 3 or passes <= 0:
        return chunk
    out = chunk.clone()
    for _ in range(passes):
        prev = out.clone()
        out[1:-1] = 0.25 * prev[:-2] + 0.50 * prev[1:-1] + 0.25 * prev[2:]
    return out


HF_FALLBACK_REPOS = {
    "c3_sort_cubes_fast30m": "Orellius/so101_sort_cubes_no_top_cosmos3_edge_smolexpert",
    "c2b_t2_sort_cubes_fast30m": "Orellius/so101_sort_cubes_no_top_cosmos2b_t2_smolexpert",
    "pretrained_model": "Orellius/so101_sort_cubes_no_top_smolvla_base_100k",
}


def _has_complete_weights(directory: Path) -> bool:
    return (directory / "config.json").is_file() and any(directory.glob("*.safetensors"))


def _ensure_local_checkpoint(raw_path: Path) -> Path:
    expanded = raw_path.expanduser()
    if _has_complete_weights(expanded):
        return expanded.resolve(strict=True)
    # Remap /home/anton/lerobot-video-vam/... to current repo root if running on another machine
    repo_root = Path(__file__).resolve().parents[2]
    s = str(raw_path)
    if "/lerobot-video-vam/" in s:
        rel = s.split("/lerobot-video-vam/", 1)[1]
        candidate = repo_root / rel
        if _has_complete_weights(candidate):
            return candidate.resolve(strict=True)
        expanded = candidate
    hf_repo = None
    if "/" in s and not s.startswith("/") and len(s.split("/")) == 2:
        hf_repo = s
        expanded = repo_root / "outputs/train" / s.split("/")[1]
    elif "sort-cubes-smolvla-100k-exact" in s:
        hf_repo = "Orellius/so101_sort_cubes_no_top_smolvla_base_100k"
    elif expanded.name in HF_FALLBACK_REPOS:
        hf_repo = HF_FALLBACK_REPOS[expanded.name]
    if hf_repo is not None:
        from huggingface_hub import snapshot_download

        expanded.mkdir(parents=True, exist_ok=True)
        snapshot_download(hf_repo, local_dir=str(expanded))
        return expanded.resolve(strict=True)
    return expanded.resolve(strict=True)


def checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Content-address local checkpoint artifacts; never infer identity from a missing config."""
    checkpoint = _ensure_local_checkpoint(args.checkpoint)
    config_path = checkpoint / "config.json"
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict) or config.get("type") != args.policy:
        raise ValueError("checkpoint config.type does not match requested policy")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(checkpoint).parts)
    )
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("checkpoint has no safetensors weights")
    for path in files:
        digest.update(str(path.relative_to(checkpoint)).encode())
        before = path.stat()
        with path.open("rb") as stream:
            if before.st_size > 16 * 1024 * 1024:
                digest.update(str(before.st_size).encode())
                digest.update(stream.read(1024 * 1024))
                stream.seek(max(0, before.st_size - 1024 * 1024))
                digest.update(stream.read(1024 * 1024))
            else:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("checkpoint changed while fingerprinting")
    # A working-tree server update must also invalidate a running instance.
    code = hashlib.sha256()
    for filename in ("rpc_server.py", "run_rpc_server.sh"):
        code.update(Path(__file__).with_name(filename).read_bytes())
    return {
        "protocol": PROTOCOL_VERSION,
        "policy": args.policy,
        "checkpoint": str(checkpoint),
        "fingerprint": digest.hexdigest(),
        "server_code": code.hexdigest(),
        "config": {
            "device": args.device,
            "compile": args.compile,
            "rtc": args.rtc,
            "execution_horizon": args.execution_horizon,
            "max_guidance_weight": args.max_guidance_weight,
            "joint_limits_min": args.joint_limits_min,
            "joint_limits_max": args.joint_limits_max,
            "jepa_planning_mode": getattr(args, "jepa_planning_mode", "none"),
            "jepa_num_candidates": getattr(args, "jepa_num_candidates", 8),
            "expo_checkpoint": str(args.expo_checkpoint) if getattr(args, "expo_checkpoint", None) else None,
            **({"euler_steps": int(args.euler_steps)} if getattr(args, "euler_steps", None) is not None else {}),
            **(
                {"shared_detector_checkpoint": str(args.shared_detector_checkpoint)}
                if getattr(args, "shared_detector_checkpoint", None)
                else {}
            ),
        },
    }


def rtc_kwargs_from_payload(payload: dict[str, Any], *, horizon: int = 30) -> dict[str, Any]:
    leftover = payload.get("prev_chunk_left_over")
    delay = _integer(payload.get("inference_delay", 0), "inference_delay", 0, horizon)
    if leftover is None:
        if delay:
            raise ValueError("inference_delay requires prev_chunk_left_over")
        return {}
    tensor = torch.tensor(leftover, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[-1] != 6 or not 1 <= tensor.shape[0] <= horizon:
        raise ValueError(f"prev_chunk_left_over must have shape [T, 6], 1 <= T <= {horizon}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("prev_chunk_left_over must be finite")
    if delay > tensor.shape[0]:
        raise ValueError("inference delay exceeds available prefix")
    return {"prev_chunk_left_over": tensor, "inference_delay": delay}


class RPCApplication:
    def __init__(self, args: argparse.Namespace) -> None:
        self.name = args.policy
        self.identity = checkpoint_identity(args)
        self.checkpoint_path = self.identity["checkpoint"]
        args.checkpoint = Path(self.checkpoint_path)
        self.instance_id = args.instance_id
        self.predict_lock = threading.Lock()
        self.rtc_enabled = args.rtc
        self.execution_horizon = args.execution_horizon
        self.lower = args.joint_limits_min
        self.upper = args.joint_limits_max
        self.device = torch.device(args.device)
        self.frame_index = 0
        if args.policy == "smolvla":
            from lerobot.policies.vam.action_rmse import SmolVLABackend

            self.backend = SmolVLABackend.from_pretrained(str(args.checkpoint), device=self.device)
            self.policy = self.backend.policy
            self.policy._enable_compile = args.compile
            if self.policy.config.adapt_to_pi_aloha or self.policy.config.use_delta_joint_actions_aloha:
                raise ValueError("SO follower RPC does not support Aloha action transforms")
            if self.policy.config.action_feature.shape != (6,) and list(
                self.policy.config.action_feature.shape
            ) != [6]:
                raise ValueError("SO follower RPC requires six action dimensions")
            self.horizon = self.policy.config.chunk_size
        else:
            from lerobot.policies.vam.configuration_video_vam import VideoVAMConfig
            from lerobot.policies.vam.modeling_video_vam import HISTORY_FRAME_INDEX_KEY, VideoVAMPolicy

            config = VideoVAMConfig.from_pretrained(args.checkpoint)
            config.device = str(self.device)
            config.cosmos_torch_compile = args.compile
            config.action_euler_steps = getattr(args, "euler_steps", None)
            if getattr(args, "jepa_world_model", None) is not None:
                config.jepa_world_model_checkpoint = args.jepa_world_model
                config.jepa_planning_mode = args.jepa_planning_mode
                config.jepa_num_candidates = args.jepa_num_candidates
            if args.compile:
                config.cosmos_compile_friendly = True
            if args.joint_limits_min is not None:
                config.joint_limits_min = args.joint_limits_min
                config.joint_limits_max = args.joint_limits_max
            elif config.joint_limits_min is None or config.joint_limits_max is None:
                config.joint_limits_min = [-180.0, -180.0, -180.0, -180.0, -180.0, 0.0]
                config.joint_limits_max = [180.0, 180.0, 180.0, 180.0, 180.0, 100.0]
            config.joint_limit_tolerance = max(getattr(config, "joint_limit_tolerance", 0.0), 15.0)
            self.backend = VideoVAMPolicy.from_pretrained(args.checkpoint, config=config)
            if (
                getattr(args, "euler_steps", None) is None
                and getattr(self.backend, "decoder", None) is not None
                and int(getattr(self.backend.decoder, "num_steps", 10)) < 10
            ):
                self.backend.decoder.num_steps = 10
            self.policy = self.backend
            self.horizon = 30
            self.lower, self.upper = config.joint_limits_min, config.joint_limits_max
            self.history_key = f"{config.camera_key}.history"
            self.history_index_key = HISTORY_FRAME_INDEX_KEY
            self.config = config

        self.expo_critic = None
        self.expo_checkpoint_path = getattr(args, "expo_checkpoint", None)
        self.shared_detector = None
        self.multiframe_detector = None
        self.current_episode_steps: list[dict[str, Any]] = []
        self.online_rollout_buffer: list[dict[str, Any]] = []
        self.latest_frame_b64: str | None = None
        self.last_update_loss: float | None = None
        self.last_detector_loss: float | None = None
        self.detector_checkpoint_path = getattr(args, "shared_detector_checkpoint", None)
        self.detector_anchor_cache: dict[str, torch.Tensor] | None = None
        anchor_file = Path(__file__).resolve().parents[2] / "outputs/sparse_seq_rl/real_detector_features_v2.pt"
        if anchor_file.is_file():
            raw_anc = torch.load(anchor_file, map_location="cpu", weights_only=False)
            if isinstance(raw_anc, dict) and "train" in raw_anc:
                self.detector_anchor_cache = raw_anc["train"]
        if getattr(args, "shared_detector_checkpoint", None) is not None:
            det_ckpt = torch.load(
                args.shared_detector_checkpoint, map_location=self.device, weights_only=False
            )
            det_state = (
                det_ckpt["state_dict"]
                if isinstance(det_ckpt, dict) and "state_dict" in det_ckpt
                else det_ckpt
            )
            is_multiframe = (
                (isinstance(det_ckpt, dict) and det_ckpt.get("detector_type") == "multiframe")
                or any(k.startswith("stem.") or k.startswith("vis_in_norm.") for k in det_state)
            )
            if is_multiframe:
                from lerobot.rewards.sequence_detector import MultiFrameSuccessDetector

                seq_len = int(det_ckpt.get("seq_len", 8)) if isinstance(det_ckpt, dict) else 8
                self.multiframe_detector = MultiFrameSuccessDetector(
                    seq_len=seq_len, state_dim=6, pretrained=True
                ).to(self.device)
                self.multiframe_detector.load_state_dict(det_state, strict=False)
                self.multiframe_detector.eval()
            else:
                from lerobot.rewards.sequence_detector import SharedFeatureSequenceDetector

                seq_len = int(det_ckpt.get("seq_len", 12)) if isinstance(det_ckpt, dict) else 12
                self.shared_detector = SharedFeatureSequenceDetector(seq_len=seq_len).to(self.device)
                self.shared_detector.load_state_dict(det_state, strict=False)
                self.shared_detector.eval()

        if (
            self.expo_checkpoint_path is not None
            or self.shared_detector is not None
            or self.multiframe_detector is not None
        ):
            from lerobot.policies.vam.expo_residual import ExpoResidualCritic

            default_ctx_dim = 512 if self.name == "smolvla" else 2048
            if self.expo_checkpoint_path is not None and Path(self.expo_checkpoint_path).is_file():
                raw_ckpt = torch.load(
                    self.expo_checkpoint_path, map_location=self.device, weights_only=False
                )
                if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt:
                    state_dict = raw_ckpt["state_dict"]
                    expo_cfg = raw_ckpt.get("config", {})
                    self.expo_critic = ExpoResidualCritic(**expo_cfg).to(self.device)
                else:
                    state_dict = raw_ckpt
                    use_init_anchor = any(k.startswith("init_proj.") for k in state_dict)
                    self.expo_critic = ExpoResidualCritic(
                        ctx_dim=default_ctx_dim,
                        chunk_len=self.horizon,
                        use_init_anchor=use_init_anchor,
                    ).to(self.device)
                self.expo_critic.load_state_dict(state_dict, strict=False)
            elif self.name == "smolvla":
                if self.multiframe_detector is None:
                    from lerobot.rewards.sequence_detector import MultiFrameSuccessDetector

                    self.multiframe_detector = MultiFrameSuccessDetector(
                        seq_len=8, state_dim=6, pretrained=True
                    ).to(self.device)
                    self.multiframe_detector.eval()
                self.expo_critic = ExpoResidualCritic(
                    ctx_dim=512,
                    state_dim=6,
                    chunk_len=self.horizon,
                    use_init_anchor=True,
                    base_alpha=1.0,
                    curr_scale=1.0,
                    max_delta_deg=8.0,
                    zero_init_edit=True,
                ).to(self.device)
            else:
                self.expo_critic = ExpoResidualCritic(use_init_anchor=True).to(self.device)
            self.expo_critic.eval()
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if not 1 <= self.execution_horizon <= self.horizon:
            raise ValueError("execution horizon exceeds policy chunk size")
        self.original_space = "normalized" if self.name == "smolvla" else "physical"
        self.policy.config.rtc_config = RTCConfig(
            enabled=args.rtc,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
        )
        self.policy.init_rtc_processor()
        # Refuse a checkpoint changed during model loading.
        if checkpoint_identity(args) != self.identity:
            raise ValueError("checkpoint changed during model loading")

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "identity": self.identity,
            "instance_id": self.instance_id,
            "policy": self.name,
            "checkpoint": self.checkpoint_path,
            "rtc": self.rtc_enabled,
            "horizon": self.horizon,
            "original_space": self.original_space,
            "action_dim": 6,
            "history_size": HISTORY if self.name == "video_vam" else 1,
            "fps": 10,
            "action_units": ["degrees"] * 5 + ["range_0_100"],
        }

    @torch.no_grad()
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("identity") != self.identity or payload.get("instance_id") != self.instance_id:
            raise ValueError("server identity/instance mismatch")
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id is required")
        state = torch.tensor(payload.get("state"), dtype=torch.float32)
        if tuple(state.shape) != (6,) or not torch.isfinite(state).all().item():
            raise ValueError("state must contain six finite floats")
        frame_index = _integer(payload.get("frame_index", 0), "frame_index", 0, 2**63 - 1)
        kwargs = rtc_kwargs_from_payload(payload, horizon=self.horizon)
        if kwargs and not self.rtc_enabled:
            raise ValueError("RTC inputs require RTC-enabled server")
        if kwargs and payload.get("original_space") != self.original_space:
            raise ValueError("RTC original action space mismatch")
        if payload.get("execution_horizon", self.execution_horizon) != self.execution_horizon:
            raise ValueError("execution_horizon differs from server configuration")
        supplied_seed = payload.get("feature_seed")
        if supplied_seed is not None and (not isinstance(supplied_seed, int) or supplied_seed < 0):
            raise ValueError("feature_seed must be a non-negative integer")
        value = payload.get(
            "images_front_u8_png_b64", payload.get("images_front_u8_raw", payload.get("images_front_u8"))
        )
        if value is None:
            raise ValueError("missing images_front_u8_png_b64 or images_front_u8_raw")
        started = time.perf_counter()
        if self.name == "smolvla":
            frame = _validate_frame(_frame(value))
            observation = {
                "observation.images.front": frame.permute(2, 0, 1).float().div(255.0),
                "observation.state": state,
                "task": str(payload.get("task", "take cube out of box")),
            }
            processed = self.backend.preprocessor(observation)
            device, dtype = self.backend._device_and_dtype()
            processed = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for key, value in processed.items()
            }
            if "prev_chunk_left_over" in kwargs:
                kwargs["prev_chunk_left_over"] = kwargs["prev_chunk_left_over"].to(device=device, dtype=dtype)
            generator = torch.Generator(device=device).manual_seed(supplied_seed or 0)
            noise = torch.randn(
                (1, self.horizon, self.policy.config.max_action_dim),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            # Each RPC supplies a complete observation, independent of other clients.
            self.policy.reset()
            original = self.policy.predict_action_chunk(processed, noise=noise, **kwargs)
            base_actions = self.backend.postprocessor(original.clone())[0].cpu()
            actions = base_actions.clone()
            original = original[0].cpu()
            live_reward_detected = False
            live_reward_prob = 0.0
            live_pre_grasp_prob = 0.0
            live_edit_norm = 0.0
            if isinstance(value, str):
                self.latest_frame_b64 = value
            if getattr(self, "expo_critic", None) is not None and getattr(self, "multiframe_detector", None) is not None:
                import torch.nn.functional as F

                img_t = observation["observation.images.front"].unsqueeze(0).to(self.device)
                img_128 = F.interpolate(img_t, size=(128, 128), mode="bilinear", align_corners=False)
                fmap = self.multiframe_detector.stem(self.multiframe_detector.norm(img_128))
                gap = fmap.mean(dim=(-2, -1))
                quad = F.adaptive_avg_pool2d(fmap, (2, 2)).flatten(1)
                jaw_roi = fmap[:, :, 2:, 1:3].mean(dim=(-2, -1))
                vis_raw = torch.cat([gap, quad, jaw_roi], dim=-1)
                p12 = F.adaptive_avg_pool2d(fmap, (3, 4)).reshape(1, 1, 512, 12).permute(0, 1, 3, 2)
                ctx_pooled = p12.expand(1, 2, 12, 512).reshape(1, 24, 512)

                state_t = state.unsqueeze(0).to(self.device, dtype=torch.float32)
                if self.rtc_enabled and "prev_chunk_left_over" not in kwargs:
                    self.current_episode_steps.clear()
                ep_step_idx = len(self.current_episode_steps)
                t_norm = torch.tensor([[min(1.0, ep_step_idx / 80.0)]], device=self.device, dtype=torch.float32)
                if ep_step_idx == 0:
                    self.expo_critic.reset_episode()
                edited = self.expo_critic.forward_edit(
                    ctx_pooled, state_t, t_norm, base_actions.unsqueeze(0).to(self.device)
                )[0].cpu()
                live_edit_norm = float((edited - base_actions).abs().mean().item())
                # Smooth low-amplitude exploration perturbation on arm joints for Score-Centered RL innovation
                expl_gen = torch.Generator().manual_seed(
                    int((supplied_seed or 0) * 1009 + self.frame_index * 37 + len(self.online_rollout_buffer) * 997)
                )
                arm_noise = torch.randn((1, 5), generator=expl_gen) * 1.0
                ramp = torch.sin(torch.linspace(0.0, math.pi, self.horizon)).unsqueeze(-1)
                expl_delta = torch.zeros_like(edited)
                expl_delta[:, :5] = ramp * arm_noise
                actions = edited + expl_delta
                self.current_episode_steps.append(
                    {
                        "ctx": ctx_pooled[0].detach().cpu(),
                        "vis_raw": vis_raw[0].detach().cpu(),
                        "state": state.detach().cpu(),
                        "t_norm": t_norm[0].detach().cpu(),
                        "base_action": base_actions.detach().cpu(),
                        "exec_action": actions.detach().cpu(),
                        "det_prob": 0.0,
                        "pre_grasp_prob": 0.0,
                        "edit_norm": live_edit_norm,
                    }
                )
                seq_len = self.multiframe_detector.seq_len
                n_st = len(self.current_episode_steps)
                if n_st >= 5:
                    end_i = n_st - 1
                    cand_vis, cand_st = [], []
                    for stride in (1, 2):
                        if stride == 2 and n_st < seq_len + 2:
                            continue
                        idxs = [max(0, end_i - (seq_len - 1 - k) * stride) for k in range(seq_len)]
                        cand_vis.append(torch.stack([self.current_episode_steps[j]["vis_raw"] for j in idxs], dim=0))
                        cand_st.append(torch.stack([self.current_episode_steps[j]["state"] for j in idxs], dim=0))
                    vb = torch.stack(cand_vis, dim=0).to(self.device)
                    sb = torch.stack(cand_st, dim=0).to(self.device)
                    dummy_f = torch.empty(vb.shape[0], seq_len, 3, 1, 1, device=self.device)
                    joint_logits, pre_logits = self.multiframe_detector.forward_with_aux(
                        dummy_f, sb, precomputed_vis=vb
                    )
                    probs = torch.sigmoid(joint_logits)
                    pre_probs = torch.sigmoid(pre_logits)
                    best_k = int(probs.argmax().item())
                    live_reward_prob = float(probs[best_k].item())
                    live_pre_grasp_prob = float(pre_probs[best_k].item())
                    self.current_episode_steps[-1]["det_prob"] = live_reward_prob
                    self.current_episode_steps[-1]["pre_grasp_prob"] = live_pre_grasp_prob
                    if live_reward_prob >= 0.80:
                        live_reward_detected = True
        else:
            frames = _frame(value)
            if frames.ndim != 4 or frames.shape[0] != HISTORY:
                raise ValueError("video_vam requires five frames in images_front_u8_png_b64 or raw list")
            frames = torch.stack([_validate_frame(frame) for frame in frames])
            batch = {
                self.history_key: frames.permute(3, 0, 1, 2).unsqueeze(0).float().div(255.0),
                "observation.state": state.unsqueeze(0),
                self.history_index_key: torch.tensor([frame_index], dtype=torch.long),
            }
            if supplied_seed is not None:
                kwargs["feature_noise_seed"] = supplied_seed
            if "prev_chunk_left_over" in kwargs and self.backend.rtc_processor is None:
                raise RuntimeError("RTC leftover received but rtc_processor is not initialized")
            if getattr(self, "expo_critic", None) is not None:
                from lerobot.policies.vam.modeling_video_vam import restore_uint8_history

                images = restore_uint8_history(batch[self.history_key].to(self.device))
                context = self.backend._extract_context(images, int(supplied_seed or frame_index))
                state_t = state.unsqueeze(0).to(self.device, dtype=torch.float32)
                t_norm = torch.tensor([[min(1.0, frame_index / 360.0)]], device=self.device, dtype=torch.float32)
                base_actions = self.backend.decoder.sample_actions(
                    state_t, context, seed=int(supplied_seed or 0), use_cuda_graph=False
                )[0].cpu()
                actions = self.expo_critic.select_best_edited_chunk(
                    self.backend.decoder, state_t, context, t_norm, base_seed=int(supplied_seed or 0)
                )[0].cpu()
                if self.lower is not None and self.upper is not None:
                    lo = torch.tensor(self.lower, dtype=actions.dtype)
                    hi = torch.tensor(self.upper, dtype=actions.dtype)
                    actions = torch.max(torch.min(actions, hi), lo)
                if hasattr(self, "current_episode_steps"):
                    ctx_pooled = self.expo_critic.pool_ctx_tokens(context)[0].detach().cpu()
                    self.current_episode_steps.append(
                        {
                            "ctx": ctx_pooled,
                            "state": state.detach().cpu(),
                            "t_norm": t_norm[0].detach().cpu(),
                            "base_action": base_actions.detach().cpu(),
                            "exec_action": actions.detach().cpu(),
                        }
                    )
            else:
                actions = self.backend.predict_action_chunk(batch, **kwargs)[0].cpu()
            actions = _smooth_chunk_nyquist(actions, passes=2)
            original = actions.clone()
        actions = validate_chunk(actions, horizon=self.horizon)
        original = validate_chunk(original, horizon=self.horizon, name="original_chunk")
        if self.lower is not None and self.upper is not None:
            lo = torch.tensor(self.lower, dtype=actions.dtype)
            hi = torch.tensor(self.upper, dtype=actions.dtype)
            actions = torch.clamp(actions, min=lo, max=hi)
        actions[:, 5] = torch.clamp(actions[:, 5], min=0.0, max=100.0)
        if self.name == "video_vam":
            original = actions.clone()
        self.frame_index += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "action_chunk": actions.float().tolist(),
            "original_chunk": original.float().tolist(),
            "original_space": self.original_space,
            "identity": self.identity,
            "instance_id": self.instance_id,
            "request_id": request_id,
            "policy": self.name,
            "latency_s": time.perf_counter() - started,
            "reward_detected": live_reward_detected if self.name == "smolvla" else False,
            "reward_prob": live_reward_prob if self.name == "smolvla" else 0.0,
            "pre_grasp_prob": live_pre_grasp_prob if self.name == "smolvla" else 0.0,
            "edit_norm": live_edit_norm if self.name == "smolvla" else 0.0,
            "chunk_idx": len(getattr(self, "current_episode_steps", ())),
        }

    def complete_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("identity") != self.identity or payload.get("instance_id") != self.instance_id:
            raise ValueError("server identity/instance mismatch")
        spawn_group = int(payload.get("spawn_group", 0))
        update_critic = bool(payload.get("update_critic", False))
        update_steps = int(payload.get("update_steps", 600))
        manual_reward = payload.get("manual_reward")

        steps = list(getattr(self, "current_episode_steps", []))
        if hasattr(self, "current_episode_steps"):
            self.current_episode_steps.clear()
        if getattr(self, "expo_critic", None) is not None:
            self.expo_critic.reset_episode()

        reward_prob = 0.0
        if manual_reward is not None:
            reward_prob = float(manual_reward)
            reward = 1.0 if reward_prob >= 0.5 else 0.0
        elif getattr(self, "multiframe_detector", None) is not None and steps and "vis_raw" in steps[0]:
            seq_len = self.multiframe_detector.seq_len
            n_st = len(steps)
            win_vis_list = []
            win_st_list = []
            win_ends = []
            start_end_i = 4 if n_st > 4 else max(0, n_st - 1)
            for end_i in range(start_end_i, n_st):
                for stride in (1, 2):
                    if stride == 2 and end_i < seq_len:
                        continue
                    idxs = [max(0, end_i - (seq_len - 1 - k) * stride) for k in range(seq_len)]
                    win_vis_list.append(torch.stack([steps[j]["vis_raw"] for j in idxs], dim=0))
                    win_st_list.append(torch.stack([steps[j]["state"] for j in idxs], dim=0))
                    win_ends.append(end_i)
            vis_batch = torch.stack(win_vis_list, dim=0).to(self.device)
            st_batch = torch.stack(win_st_list, dim=0).to(self.device)
            dummy_f = torch.empty(vis_batch.shape[0], seq_len, 3, 1, 1, device=self.device)
            probs = self.multiframe_detector.predict_proba(dummy_f, st_batch, precomputed_vis=vis_batch)
            best_w = int(probs.argmax().item())
            reward_prob = float(probs[best_w].item())
            reward = 1.0 if reward_prob >= 0.80 else 0.0
            if reward >= 0.5:
                peak_end = win_ends[best_w]
                steps = steps[: min(n_st, peak_end + 2)]
        elif getattr(self, "shared_detector", None) is not None and steps:
            seq_len = self.shared_detector.seq_len
            if len(steps) >= seq_len:
                window = steps[-seq_len:]
            else:
                window = [steps[0]] * (seq_len - len(steps)) + steps
            seq_ctx = torch.stack([s["ctx"] for s in window], dim=0).unsqueeze(0).to(self.device)
            seq_states = torch.stack([s["state"] for s in window], dim=0).unsqueeze(0).to(self.device)
            reward_prob = float(self.shared_detector.predict_proba(seq_ctx, seq_states).item())
            reward = 1.0 if reward_prob >= 0.5 else 0.0
        else:
            reward = 0.0

        if not hasattr(self, "online_rollout_buffer"):
            self.online_rollout_buffer = []
        if steps:
            self.online_rollout_buffer.append(
                {
                    "steps": steps,
                    "reward": reward,
                    "reward_prob": reward_prob,
                    "spawn_group": spawn_group,
                    "human_labeled": manual_reward is not None,
                }
            )

        from lerobot.policies.vam.expo_residual import compute_dr_grpo_advantages

        all_rewards = [float(ep["reward"]) for ep in self.online_rollout_buffer]
        all_groups = [int(ep["spawn_group"]) for ep in self.online_rollout_buffer]
        adv_all = compute_dr_grpo_advantages(all_rewards, all_groups) if all_rewards else torch.zeros(1)
        latest_adv = float(adv_all[-1].item()) if len(all_rewards) > 0 else 0.0
        group_rewards = [r for r, g in zip(all_rewards, all_groups, strict=False) if g == spawn_group]
        group_mean = float(sum(group_rewards) / len(group_rewards)) if group_rewards else 0.0

        critic_updated = False
        update_loss = None
        detector_updated = False
        detector_loss = None
        if update_critic and getattr(self, "expo_critic", None) is not None and self.online_rollout_buffer:
            stats = self.expo_critic.update_from_episode_buffer(
                self.online_rollout_buffer, steps=update_steps
            )
            critic_updated = True
            update_loss = float(stats["loss"])
            self.last_update_loss = update_loss
            save_path = payload.get("save_checkpoint") or getattr(self, "expo_checkpoint_path", None)
            if save_path:
                out_p = Path(save_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": self.expo_critic.state_dict(),
                        "config": {
                            "ctx_dim": self.expo_critic.ctx_dim,
                            "state_dim": self.expo_critic.state_dim,
                            "chunk_len": self.expo_critic.chunk_len,
                            "use_init_anchor": self.expo_critic.use_init_anchor,
                            "base_alpha": self.expo_critic.base_alpha,
                            "curr_scale": self.expo_critic.curr_scale,
                            "max_delta_deg": self.expo_critic.max_delta_deg,
                        },
                    },
                    out_p,
                )
            if getattr(self, "multiframe_detector", None) is not None:
                det_stats = self.multiframe_detector.update_from_episode_buffer(
                    self.online_rollout_buffer,
                    anchor_cache=self.detector_anchor_cache,
                    steps=80,
                )
                if det_stats["num_windows"] > 0:
                    detector_updated = True
                    detector_loss = float(det_stats["loss"])
                    self.last_detector_loss = detector_loss
                    if self.detector_checkpoint_path:
                        det_p = Path(self.detector_checkpoint_path)
                        torch.save(
                            {
                                "detector_type": "multiframe",
                                "seq_len": self.multiframe_detector.seq_len,
                                "state_dict": self.multiframe_detector.state_dict(),
                            },
                            det_p,
                        )

        return {
            "episode_count": len(self.online_rollout_buffer),
            "recorded_chunks": len(steps),
            "reward": reward,
            "reward_prob": reward_prob,
            "spawn_group": spawn_group,
            "group_mean_reward": group_mean,
            "advantage": latest_adv,
            "critic_updated": critic_updated,
            "update_loss": update_loss,
            "detector_updated": detector_updated,
            "detector_loss": detector_loss,
        }

    def live_state(self) -> dict[str, Any]:
        from lerobot.policies.vam.expo_residual import compute_dr_grpo_advantages

        steps = list(getattr(self, "current_episode_steps", []))
        last_step = steps[-1] if steps else None
        state_vec = last_step["state"].tolist() if last_step is not None and "state" in last_step else [0.0] * 6
        det_trace = [round(float(s.get("det_prob", 0.0)), 4) for s in steps]
        pre_trace = [round(float(s.get("pre_grasp_prob", 0.0)), 4) for s in steps]
        edit_trace = [round(float(s.get("edit_norm", 0.0)), 3) for s in steps]
        buf = list(getattr(self, "online_rollout_buffer", []))
        all_rewards = [float(ep["reward"]) for ep in buf]
        all_groups = [int(ep["spawn_group"]) for ep in buf]
        adv_all = compute_dr_grpo_advantages(all_rewards, all_groups).tolist() if all_rewards else []
        episodes_summary = [
            {
                "ep": idx + 1,
                "reward": float(ep["reward"]),
                "reward_prob": round(float(ep.get("reward_prob", 0.0)), 3),
                "chunks": len(ep.get("steps", [])),
                "advantage": round(float(adv_all[idx]), 3) if idx < len(adv_all) else 0.0,
            }
            for idx, ep in enumerate(buf)
        ]
        return {
            "policy": self.name,
            "checkpoint": Path(self.checkpoint_path).parent.name + "/" + Path(self.checkpoint_path).name,
            "current_chunk": len(steps),
            "state": [round(float(x), 1) for x in state_vec],
            "det_prob": det_trace[-1] if det_trace else 0.0,
            "pre_grasp_prob": pre_trace[-1] if pre_trace else 0.0,
            "edit_norm": edit_trace[-1] if edit_trace else 0.0,
            "det_trace": det_trace,
            "pre_trace": pre_trace,
            "edit_trace": edit_trace,
            "episodes": episodes_summary,
            "mean_reward": round(sum(all_rewards) / len(all_rewards), 3) if all_rewards else 0.0,
            "last_update_loss": self.last_update_loss,
            "frame_b64": getattr(self, "latest_frame_b64", None),
        }

    def override_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        buf = getattr(self, "online_rollout_buffer", [])
        if not buf:
            return {"ok": False, "error": "no completed episodes in buffer"}
        ep_idx = int(payload.get("ep_index", len(buf) - 1))
        if not (0 <= ep_idx < len(buf)):
            ep_idx = len(buf) - 1
        new_rew = payload.get("reward")
        if new_rew is None:
            new_rew = 0.0 if float(buf[ep_idx]["reward"]) >= 0.5 else 1.0
        buf[ep_idx]["reward"] = float(new_rew)
        buf[ep_idx]["human_labeled"] = True
        if getattr(self, "expo_critic", None) is not None:
            stats = self.expo_critic.update_from_episode_buffer(buf, steps=300)
            self.last_update_loss = float(stats["loss"])
            if getattr(self, "expo_checkpoint_path", None):
                out_p = Path(self.expo_checkpoint_path)
                out_p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "state_dict": self.expo_critic.state_dict(),
                        "config": {
                            "ctx_dim": self.expo_critic.ctx_dim,
                            "state_dim": self.expo_critic.state_dim,
                            "chunk_len": self.expo_critic.chunk_len,
                            "use_init_anchor": self.expo_critic.use_init_anchor,
                            "base_alpha": self.expo_critic.base_alpha,
                            "curr_scale": self.expo_critic.curr_scale,
                            "max_delta_deg": self.expo_critic.max_delta_deg,
                        },
                    },
                    out_p,
                )
        if getattr(self, "multiframe_detector", None) is not None:
            det_stats = self.multiframe_detector.update_from_episode_buffer(
                buf, anchor_cache=self.detector_anchor_cache, steps=80
            )
            if det_stats["num_windows"] > 0:
                self.last_detector_loss = float(det_stats["loss"])
                if self.detector_checkpoint_path:
                    torch.save(
                        {
                            "detector_type": "multiframe",
                            "seq_len": self.multiframe_detector.seq_len,
                            "state_dict": self.multiframe_detector.state_dict(),
                        },
                        Path(self.detector_checkpoint_path),
                    )
        return {
            "ok": True,
            "ep": ep_idx + 1,
            "new_reward": buf[ep_idx]["reward"],
            "loss": self.last_update_loss,
            "detector_loss": self.last_detector_loss,
        }

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if closer is not None:
            closer()


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SO-101 Live EXPO + MultiFrame Detector Dashboard</title>
<style>
  body { background: #0f172a; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px; }
  .grid { display: grid; grid-template-columns: 420px 1fr; gap: 16px; max-width: 1360px; margin: 0 auto; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }
  h2 { margin: 0 0 10px 0; font-size: 16px; color: #38bdf8; }
  .cam { width: 100%; border-radius: 8px; border: 1px solid #475569; background: #000; min-height: 280px; object-fit: contain; }
  .bar-wrap { background: #0f172a; border-radius: 6px; height: 22px; overflow: hidden; margin: 6px 0 12px 0; border: 1px solid #334155; }
  .bar { height: 100%; transition: width 0.2s ease; }
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }
  .stat { background: #0f172a; padding: 8px 10px; border-radius: 6px; border: 1px solid #334155; }
  .stat .lbl { font-size: 11px; color: #94a3b8; }
  .stat .val { font-size: 18px; font-weight: 700; margin-top: 2px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 7px 8px; text-align: left; border-bottom: 1px solid #334155; }
  th { color: #94a3b8; }
  button { background: #2563eb; color: #fff; border: none; border-radius: 5px; padding: 5px 10px; cursor: pointer; font-size: 12px; }
  button:hover { background: #1d4ed8; }
</style>
</head>
<body>
<div class="grid">
  <div class="card">
    <h2>Wrist Camera (/dev/video0)</h2>
    <img id="cam" class="cam" alt="Waiting for frame..." />
    <div style="margin-top:12px;">
      <div style="display:flex;justify-content:space-between;font-size:13px;"><span>Neural Success Prob (det_p ≥ 0.80)</span><b id="det_val">0.000</b></div>
      <div class="bar-wrap"><div id="det_bar" class="bar" style="width:0%;background:#22c55e;"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:13px;"><span>Pre-Grasp Carry Gate (pre_p)</span><b id="pre_val">0.000</b></div>
      <div class="bar-wrap"><div id="pre_bar" class="bar" style="width:0%;background:#38bdf8;"></div></div>
      <div style="display:flex;justify-content:space-between;font-size:13px;"><span>EXPO Action Edit |Δa_φ| (deg)</span><b id="edit_val">0.00°</b></div>
      <div class="bar-wrap"><div id="edit_bar" class="bar" style="width:0%;background:#f59e0b;"></div></div>
    </div>
  </div>
  <div class="card">
    <h2>Online EXPO + Multi-Frame Transformer Telemetry</h2>
    <div class="metrics">
      <div class="stat"><div class="lbl">Current Ep Chunk</div><div class="val" id="m_chunk">0</div></div>
      <div class="stat"><div class="lbl">Gripper / Pan</div><div class="val" id="m_grip">0.0° / 0.0°</div></div>
      <div class="stat"><div class="lbl">Session Success Rate</div><div class="val" id="m_sr">0%</div></div>
      <div class="stat"><div class="lbl">Last Critic Loss</div><div class="val" id="m_loss">—</div></div>
    </div>
    <div style="font-size:12px;color:#94a3b8;margin-bottom:6px;">Current Episode Detector Trace (det_p per chunk): <span id="trace_txt" style="color:#e2e8f0;">[]</span></div>
    <h2 style="margin-top:14px;">Completed Episodes (Click Flip to Override Misdetection & Re-Train Critic)</h2>
    <table>
      <thead><tr><th>Ep</th><th>Status</th><th>Peak det_p</th><th>Chunks</th><th>Dr. GRPO Adv</th><th>Override</th></tr></thead>
      <tbody id="ep_body"></tbody>
    </table>
  </div>
</div>
<script>
async function flipEp(idx) {
  await fetch('/override_episode', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ep_index: idx})});
  refresh();
}
async function refresh() {
  try {
    const r = await fetch('/live_state');
    const d = await r.json();
    if (d.frame_b64) document.getElementById('cam').src = 'data:image/jpeg;base64,' + d.frame_b64;
    document.getElementById('det_val').textContent = d.det_prob.toFixed(3);
    document.getElementById('det_bar').style.width = Math.min(100, d.det_prob * 100) + '%';
    document.getElementById('det_bar').style.background = d.det_prob >= 0.80 ? '#22c55e' : '#64748b';
    document.getElementById('pre_val').textContent = d.pre_grasp_prob.toFixed(3);
    document.getElementById('pre_bar').style.width = Math.min(100, d.pre_grasp_prob * 100) + '%';
    document.getElementById('edit_val').textContent = d.edit_norm.toFixed(2) + '°';
    document.getElementById('edit_bar').style.width = Math.min(100, d.edit_norm * 15) + '%';
    document.getElementById('m_chunk').textContent = d.current_chunk;
    document.getElementById('m_grip').textContent = (d.state[5]||0).toFixed(1) + '° / ' + (d.state[0]||0).toFixed(1) + '°';
    document.getElementById('m_sr').textContent = (d.mean_reward * 100).toFixed(0) + '% (' + d.episodes.length + ' eps)';
    document.getElementById('m_loss').textContent = d.last_update_loss !== null ? d.last_update_loss.toFixed(4) : '—';
    document.getElementById('trace_txt').textContent = d.det_trace.slice(-16).map(x => x.toFixed(2)).join(' → ');
    const tbody = document.getElementById('ep_body');
    tbody.innerHTML = d.episodes.slice().reverse().map(e => `
      <tr>
        <td><b>#${e.ep}</b></td>
        <td style="color:${e.reward>=0.5?'#22c55e':'#ef4444'}">${e.reward>=0.5?'✅ SUCCESS (1.0)':'❌ FAIL (0.0)'}</td>
        <td>${e.reward_prob.toFixed(3)}</td>
        <td>${e.chunks}</td>
        <td>${e.advantage >= 0 ? '+' : ''}${e.advantage.toFixed(2)}</td>
        <td><button onclick="flipEp(${e.ep - 1})">Flip 0 ↔ 1 & Update</button></td>
      </tr>
    `).join('');
  } catch (e) {}
}
setInterval(refresh, 400);
refresh();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    app: RPCApplication
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(60.0)

    def handle(self) -> None:
        with contextlib.suppress(TimeoutError, ConnectionResetError, BrokenPipeError):
            super().handle()

    def do_GET(self) -> None:
        if self.path in ("/", "/dashboard"):
            encoded = DASHBOARD_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
            return
        if self.path == "/live_state":
            self._send(200, self.app.live_state())
            return
        if self.path != "/healthz":
            self._send(404, {"error": "not found"})
            return
        self._send(200, self.app.health())

    def do_POST(self) -> None:
        if self.path not in ("/predict", "/episode_complete", "/override_episode"):
            self._send(404, {"error": "use POST /predict, /episode_complete, or /override_episode"})
            return
        try:
            self.connection.settimeout(30.0 if self.path in ("/episode_complete", "/override_episode") else 10.0)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            # Health remains responsive, but never queue concurrent GPU predictions.
            is_blocking = self.path in ("/episode_complete", "/override_episode")
            acquired = (
                self.app.predict_lock.acquire(blocking=True, timeout=15.0)
                if is_blocking
                else self.app.predict_lock.acquire(blocking=False)
            )
            if not acquired:
                self._send(409, {"error": "inference already in progress"})
                return
            try:
                if self.path == "/episode_complete":
                    result = self.app.complete_episode(payload)
                elif self.path == "/override_episode":
                    result = self.app.override_episode(payload)
                else:
                    result = self.app.predict(payload)
            finally:
                self.app.predict_lock.release()
            self._send(200, result)
        except Exception as exc:
            self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            with contextlib.suppress(OSError):
                self.connection.settimeout(60.0)

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        try:
            self.end_headers()
            self.wfile.write(encoded)
        except (TimeoutError, BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="Print identity without loading a model")
    parser.add_argument("--instance-id", default="unmanaged")
    parser.add_argument("--rtc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy", choices=("smolvla", "video_vam"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--joint-limits-min", type=float, nargs=6)
    parser.add_argument("--joint-limits-max", type=float, nargs=6)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=10,
        help="RTC leftover prefix length (overridable per request)",
    )
    parser.add_argument("--max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the Cosmos DiT with max-autotune (default). Pass --no-compile for eager.",
    )
    parser.add_argument(
        "--euler-steps",
        type=int,
        default=None,
        help="Video-VAM action flow Euler steps (default: checkpoint value, 10). E.g. 3 for lower latency.",
    )
    parser.add_argument(
        "--jepa-world-model",
        type=Path,
        default=None,
        help="Optional ControlAwareJEPAWorldModel .safetensors checkpoint for MuZero latent search.",
    )
    parser.add_argument(
        "--jepa-planning-mode",
        choices=("none", "best_of_n", "mppi"),
        default="none",
        help="Latent world-model search mode when --jepa-world-model is set.",
    )
    parser.add_argument(
        "--jepa-num-candidates",
        type=int,
        default=8,
        help="Number of SmolExpert candidate chunks sampled per control step for JEPA search.",
    )
    parser.add_argument(
        "--expo-checkpoint",
        type=Path,
        default=None,
        help="Optional ExpoResidualCritic (.pt) checkpoint for score-centered residual action editing.",
    )
    parser.add_argument(
        "--shared-detector-checkpoint",
        type=Path,
        default=None,
        help="Optional SharedFeatureSequenceDetector (.pt) checkpoint for automated online reward scoring.",
    )
    args = parser.parse_args(argv)
    if args.euler_steps is not None and args.euler_steps < 1:
        parser.error("--euler-steps must be >= 1")
    if args.execution_horizon < 1:
        parser.error("--execution-horizon must be >= 1")
    if not math.isfinite(args.max_guidance_weight) or args.max_guidance_weight <= 0:
        parser.error("--max-guidance-weight must be positive")
    if (args.joint_limits_min is None) != (args.joint_limits_max is None):
        parser.error("joint limits must be supplied together")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if args.joint_limits_min is not None and (
        any(not math.isfinite(x) for x in args.joint_limits_min + args.joint_limits_max)
        or any(low >= high for low, high in zip(args.joint_limits_min, args.joint_limits_max, strict=True))
    ):
        parser.error("joint limits must be finite and ordered")
    if args.describe:
        print(json.dumps(checkpoint_identity(args), sort_keys=True))
        return 0
    app = RPCApplication(args)
    Handler.app = app
    try:
        ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
