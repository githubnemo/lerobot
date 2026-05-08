#!/usr/bin/env python3
"""
Residual RL fine-tuning of a frozen SmolVLA base policy (ResFit-style).

Pipeline:
  obs ─┬─> SmolVLA (frozen, chunked: 1 call per N env steps) ─> a_base
       └─> encoder ─> residual actor ─> delta_a (bounded ±residual_scale)
                                                                │
                                            a_final = clip(a_base + delta_a)
                                                                │
                                                         SafetyLayer ─> robot
                                                                │
                      img ─> trained reward classifier ─> r (task success ∈ {0,1})

Replay buffer stores (img, q, a_base, delta_a, r, next_img, next_q, next_a_base, done).
The critic learns Q(s, a_base, delta_a); the actor outputs delta_a.

Prereqs:
  - Trained SmolVLA at outputs/train/cube_out_of_box_il_smolvla/checkpoints/last/pretrained_model
  - Trained reward classifier at outputs/reward_classifier/cube_out_of_box/checkpoints/last/pretrained_model
  - Demos dataset at data/cube_out_of_box_dataset

Usage:
  python robot/minimal_rl_resfit.py
"""

import os
import sys
import time
import json
import random
import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots import make_robot_from_config
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Rotation
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.constants import OBS_STATE, OBS_IMAGE
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

from safety import SafetyLayer
from minimal_rl import (
    make_mlp,
    weight_normalize_,
    preprocess_obs as _preprocess_obs_base,
    reset_robot,
)


# --- Configuration ---
CONFIG = {
    "port": "/dev/ttyACM0",
    "robot_id": "shabby",
    "fps": 10,
    "lr": 3e-4,
    "batch_size": 64,
    "buffer_size": 50000,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha_init": 0.01,
    "utd_ratio": 2,
    "warmup_steps": 100,
    "residual_warmup_steps": 1000,  # ramp residual_scale 0 → 1 over first N env steps
    "max_episode_steps": 300,
    "img_size": (64, 64),
    "seed": 420,
    "use_bf16": True,
    "residual_scale": 0.2,       # ±20% of action range
    "smolvla_chunk_size": 5,     # actions consumed per SmolVLA forward call
    "policy_delay": 2,
    "use_ln": True,
    "use_wn": True,
    "pure_base": False,
    "torque_penalty_weight": 0.0,
    "classifier_reward_weight": 1.0,
    "device": "cuda",
    "safety_max_delta_deg": 5.0,
    "safety_threshold_mA": 200.0,
    "safety_limit_mA": 500.0,
    "smolvla_path": "outputs/train/cube_out_of_box_il_smolvla/checkpoints/last/pretrained_model",
    "classifier_path": "outputs/reward_classifier/cube_out_of_box/checkpoints/last/pretrained_model",
    "dataset_root": "data/cube_out_of_box_dataset",
    "dataset_repo_id": "hubnemo/cube_out_of_box_dataset",
    "preload_demos": True,
    "runs_dir": "robot/runs_resfit",
    "use_wandb": True,
    "wandb_project": "cube_out_of_box_resfit",
}


# ---------------------------------------------------------------------------
# Observation handling
# ---------------------------------------------------------------------------

def get_camera_key(obs_dict):
    """Find the first camera key in the obs dict."""
    for k in obs_dict:
        if isinstance(obs_dict[k], np.ndarray) and obs_dict[k].ndim == 3:
            return k
    return None


def extract_obs(obs_dict, motor_names_pos, img_size, device):
    """
    Returns:
      img_rgb_raw (H,W,3 uint8)       - full-res image for SmolVLA + classifier
      img_resized (B,3,H,W float32)   - small image for our residual encoder
      joints      (B,D float32)       - joint positions in degrees
    """
    cam_key = get_camera_key(obs_dict)
    if cam_key is None:
        raise RuntimeError("No camera in observation. ResFit requires a camera.")

    img_full = obs_dict[cam_key]  # HxWx3 uint8
    img_small = cv2.resize(img_full, img_size)
    img_t = torch.from_numpy(img_small).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    joints_raw = np.array([obs_dict[k] for k in motor_names_pos], dtype=np.float32)
    joints_t = torch.from_numpy(joints_raw).unsqueeze(0) / 100.0

    return img_full, img_t.to(device), joints_t.to(device), joints_raw


# ---------------------------------------------------------------------------
# SmolVLA wrapper — produces a_base with action chunking
# ---------------------------------------------------------------------------

class SmolVLABase:
    """Wraps a frozen SmolVLA policy with an internal action-chunk queue.

    Every `chunk_size` calls, it does one expensive VLA forward pass.
    """

    def __init__(self, policy_path: str, chunk_size: int, device: torch.device):
        print(f"Loading SmolVLA from {policy_path} ...")
        self.policy: SmolVLAPolicy = SmolVLAPolicy.from_pretrained(policy_path)
        self.policy.to(device)
        self.policy.eval()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.policy.config.n_action_steps = chunk_size
        self.policy.reset()
        self.device = device
        self.chunk_size = chunk_size
        self.last_inference_dt = 0.0
        # Which observation keys does this policy expect? (image + state)
        self.image_keys = [k for k in self.policy.config.input_features
                           if k.startswith("observation.image")]
        if not self.image_keys:
            raise RuntimeError("SmolVLA policy has no image input features.")
        print(f"  SmolVLA expects image keys: {self.image_keys}")

        # Load the processor pipeline — handles tokenization, normalization,
        # batch dim, device, image permutation, etc.
        print(f"  Loading SmolVLA pre/post-processors ...")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=policy_path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
            },
        )

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def __call__(self, img_full: np.ndarray, joints_raw: np.ndarray, task: str):
        """Return a_base as a 1D numpy array (action_dim,) in the raw action space."""
        # Build a raw observation dict — the preprocessor will do the heavy lifting.
        # Note: preprocessor expects HWC uint8 images → converts internally.
        obs = {OBS_STATE: torch.from_numpy(joints_raw).float()}
        img_t = torch.from_numpy(img_full).float().permute(2, 0, 1) / 255.0  # CHW
        for key in self.image_keys:
            obs[key] = img_t
        obs["task"] = task

        t0 = time.perf_counter()
        batch = self.preprocessor(obs)
        action = self.policy.select_action(batch)  # (1, action_dim) on device
        action = self.postprocessor(action)
        self.last_inference_dt = time.perf_counter() - t0
        return action.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Reward Classifier wrapper
# ---------------------------------------------------------------------------

class RewardClassifier:
    def __init__(self, classifier_path: str, device: torch.device):
        print(f"Loading reward classifier from {classifier_path} ...")
        self.model: Classifier = Classifier.from_pretrained(classifier_path)
        self.model.to(device)
        self.model.eval()
        # Detect key-name drift in the resnet10 GroupNorm wrapper:
        # new code has MyGroupNorm.group_norm, old checkpoints stored keys flat.
        # If the group_norm weights are still the default (all ones), try to
        # remap from the safetensors file directly.
        self._patch_groupnorm_if_needed(classifier_path, device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        # Which image key does the classifier expect?
        self.image_keys = [k for k in self.model.config.input_features
                           if k.startswith(OBS_IMAGE)]
        if not self.image_keys:
            raise RuntimeError("Classifier has no image input features.")
        # Read expected image size from config (e.g. [3, 128, 128])
        first_img_feat = self.model.config.input_features[self.image_keys[0]]
        shape = tuple(first_img_feat.shape)
        # shape is (C, H, W)
        self.img_h = int(shape[1])
        self.img_w = int(shape[2])
        print(f"  Classifier expects image size: {self.img_h}x{self.img_w}")

    def _patch_groupnorm_if_needed(self, classifier_path: str, device: torch.device):
        """Check if GroupNorm params are still at init (all ones/zeros),
        which would mean the checkpoint keys didn't match. If so, find the
        matching checkpoint keys (minus ".group_norm") and copy them.
        """
        from safetensors.torch import load_file as _safe_load

        # Find suspicious modules: MyGroupNorm wrappers with default params
        sd = {k: v for k, v in self.model.state_dict().items() if "group_norm" in k}
        if not sd:
            return

        # Heuristic: weight tensor is all-ones (default GroupNorm init)
        first_key = next(iter(sd))
        if not torch.allclose(sd[first_key], torch.ones_like(sd[first_key])):
            return  # classifier already has trained norm params

        # Attempt remap
        from pathlib import Path as _P
        sf = None
        base = _P(classifier_path)
        if base.is_dir():
            for f in base.glob("*.safetensors"):
                sf = f
                break
        if sf is None:
            return

        print(f"  Patching GroupNorm keys: remapping from {sf.name} ...")
        ckpt = _safe_load(str(sf))
        # Build map: for each "X.group_norm.Y" target, look for "X.Y" in checkpoint
        fixes = {}
        for target_key in self.model.state_dict().keys():
            if ".group_norm." in target_key:
                flat_key = target_key.replace(".group_norm.", ".")
                if flat_key in ckpt:
                    fixes[target_key] = ckpt[flat_key]

        if fixes:
            sd = self.model.state_dict()
            for k, v in fixes.items():
                sd[k] = v.to(device)
            self.model.load_state_dict(sd, strict=False)
            print(f"  Remapped {len(fixes)} GroupNorm tensors.")
        else:
            print("  No matching keys found — classifier norms may be random.")

    @torch.no_grad()
    def __call__(self, img_full: np.ndarray) -> float:
        # Resize to what classifier was trained on (e.g. 128x128)
        img_resized = cv2.resize(img_full, (self.img_w, self.img_h))
        img_chw = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        batch = {key: img_chw.to(self.device) for key in self.image_keys}
        r = self.model.predict_reward(batch)  # (1,) tensor of 0/1
        return float(r.item())


# ---------------------------------------------------------------------------
# Residual actor / critic
# ---------------------------------------------------------------------------

class ResidualEncoder(nn.Module):
    """Small CNN + joint MLP — same style as minimal_rl's TinyEncoder but always uses camera."""

    def __init__(self, state_dim, img_shape=(3, 64, 64), use_ln=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, *img_shape)).shape[1]
        in_dim = cnn_out_dim + state_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256) if use_ln else nn.Identity(),
            nn.ReLU(),
        )
        self.out_dim = 256

    def forward(self, img, joints):
        return self.fc(torch.cat([self.cnn(img), joints], dim=-1))


class ResidualActor(nn.Module):
    """Outputs delta_a conditioned on (state_feat, a_base). Bounded to ±1 via tanh.

    Initialized to output (mu=0, log_std=-3) so that at the start of training
    the residual is ~ tanh(N(0, 0.05)) ≈ 0. This keeps the VLA base intact
    until the critic has learned something meaningful.
    """

    def __init__(self, state_dim, action_dim, use_ln=True):
        super().__init__()
        self.net = make_mlp(state_dim + action_dim, 256, use_ln=use_ln)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        # Zero-init so the initial policy defaults to "no residual, small noise"
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -3.0)  # std = e^-3 ≈ 0.05

    def forward(self, feat, a_base):
        x = self.net(torch.cat([feat, a_base], dim=-1))
        return self.mu(x), torch.clamp(self.log_std(x), -20, 2)

    def sample(self, feat, a_base):
        mu, log_std = self(feat, a_base)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        delta = torch.tanh(u)  # in [-1, 1], later scaled by residual_scale
        log_prob = dist.log_prob(u) - torch.log(1 - delta.pow(2) + 1e-6)
        return delta, log_prob.sum(dim=-1, keepdim=True)


class ResidualCritic(nn.Module):
    """Q(state_feat, a_base, delta_a). Double-Q."""

    def __init__(self, state_dim, action_dim, use_ln=True):
        super().__init__()
        inp = state_dim + 2 * action_dim
        self.q1_net = make_mlp(inp, 256, use_ln=use_ln)
        self.q1_head = nn.Linear(256, 1)
        self.q2_net = make_mlp(inp, 256, use_ln=use_ln)
        self.q2_head = nn.Linear(256, 1)

    def forward(self, feat, a_base, delta):
        xa = torch.cat([feat, a_base, delta], dim=-1)
        return self.q1_head(self.q1_net(xa)), self.q2_head(self.q2_net(xa))


# ---------------------------------------------------------------------------
# Demo preloading
# ---------------------------------------------------------------------------

def preload_demos(dataset_root: str, repo_id: str, img_size, device, max_samples=10000):
    """Load demos from the local LeRobot dataset into transition tuples.

    Each tuple: (img, q, a_base, delta_a, r, next_img, next_q, next_a_base, done)
    a_base = the demo action itself (best a priori guess), delta_a = 0.
    r = 0 for all but last step of episode = 1 (success).
    """
    print(f"Loading demos from {dataset_root} ...")
    try:
        ds = LeRobotDataset(repo_id, root=dataset_root)
    except Exception as e:
        print(f"  Failed to load dataset: {e}")
        return []

    transitions = []
    n = min(len(ds) - 1, max_samples)
    print(f"  Dataset length: {len(ds)}, using first {n} transitions")

    def to_img_tensor(item):
        # LeRobot datasets return images as float tensors in [0,1] with shape (C,H,W)
        img = None
        for k, v in item.items():
            if k.startswith("observation.image") and isinstance(v, torch.Tensor) and v.ndim == 3:
                img = v
                break
        if img is None:
            return None
        if img.dtype != torch.float32:
            img = img.float() / 255.0
        img = img.unsqueeze(0)  # add batch dim
        img = F.interpolate(img, size=img_size, mode="bilinear", align_corners=False)
        return img.cpu()

    for i in range(n):
        item = ds[i]
        nxt = ds[i + 1]
        # Skip transitions that cross episode boundaries
        if item.get("episode_index") != nxt.get("episode_index"):
            continue

        img = to_img_tensor(item)
        next_img = to_img_tensor(nxt)
        if img is None or next_img is None:
            continue

        q = item["observation.state"].float().unsqueeze(0) / 100.0
        next_q = nxt["observation.state"].float().unsqueeze(0) / 100.0
        a = item["action"].float().numpy()   # raw action in degrees
        next_a = nxt["action"].float().numpy()

        # is_last: end of episode?
        done = False
        if "next.done" in item:
            done = bool(item["next.done"])
        elif i + 2 >= len(ds) or ds[i + 1].get("episode_index") != ds[i + 2].get("episode_index"):
            done = True

        r = 1.0 if done else 0.0

        transitions.append((
            img.cpu(), q.cpu(),
            torch.from_numpy(a).float(),           # a_base
            torch.zeros_like(torch.from_numpy(a)).float(),  # delta_a = 0
            float(r),
            next_img.cpu(), next_q.cpu(),
            torch.from_numpy(next_a).float(),      # next_a_base
            float(done),
        ))

    print(f"  Loaded {len(transitions)} demo transitions")
    return transitions


# ---------------------------------------------------------------------------
# RunLogger (minimal)
# ---------------------------------------------------------------------------

class SimpleLogger:
    def __init__(self, cfg, run_name=None):
        run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(cfg["runs_dir"]) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self.jsonl = open(self.run_dir / "metrics.jsonl", "w")
        self.data = []
        self.use_wandb = HAS_WANDB and cfg.get("use_wandb", False)
        if self.use_wandb:
            wandb.init(project=cfg.get("wandb_project", "resfit"),
                       name=run_name, config=cfg)

    def log(self, d):
        self.jsonl.write(json.dumps(d) + "\n")
        self.jsonl.flush()
        self.data.append(d)
        if self.use_wandb:
            wandb.log(d)

    def finish(self):
        self.jsonl.close()
        # Plot simple learning curves
        if self.data:
            keys = [k for k in ["reward", "q_val", "alpha", "hz",
                                 "loss_q", "loss_a", "safety_atten"]
                    if k in self.data[0]]
            n = len(keys)
            if n > 0:
                fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n))
                if n == 1:
                    axes = [axes]
                steps = [d["total_steps"] for d in self.data]
                for ax, k in zip(axes, keys):
                    ax.plot(steps, [d.get(k, 0) for d in self.data])
                    ax.set_title(k)
                    ax.grid(alpha=0.3)
                plt.tight_layout()
                plot_path = self.run_dir / "learning_curves.jpg"
                plt.savefig(plot_path, format="jpeg", dpi=80)
                if self.use_wandb:
                    wandb.log({"learning_curves": wandb.Image(str(plot_path))})
                plt.close()
        if self.use_wandb:
            wandb.finish()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=CONFIG["device"])
    parser.add_argument("--port", type=str, default=CONFIG["port"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--residual-scale", type=float, default=CONFIG["residual_scale"],
                        help="Max fraction of action range the residual can add (0.2 = ±20%)")
    parser.add_argument("--residual-warmup-steps", type=int, default=CONFIG["residual_warmup_steps"],
                        help="Env steps over which residual_scale ramps 0 → 1")
    parser.add_argument("--chunk-size", type=int, default=CONFIG["smolvla_chunk_size"],
                        help="How many SmolVLA actions to consume per VLA forward pass")
    parser.add_argument("--smolvla-path", type=str, default=CONFIG["smolvla_path"])
    parser.add_argument("--classifier-path", type=str, default=CONFIG["classifier_path"])
    parser.add_argument("--dataset-root", type=str, default=CONFIG["dataset_root"])
    parser.add_argument("--no-preload", action="store_true", help="Skip preloading demos")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--torque-weight", type=float, default=CONFIG["torque_penalty_weight"])
    parser.add_argument("--task", type=str, default="take cube out of box")
    parser.add_argument("--pure-base", action="store_true",
                        help="Run ONLY the frozen SmolVLA base policy (no residual, no training). "
                             "Useful for sanity-checking the IL policy through this pipeline.")
    args = parser.parse_args()

    CONFIG["device"] = args.device
    CONFIG["port"] = args.port
    CONFIG["seed"] = args.seed
    CONFIG["residual_scale"] = args.residual_scale
    CONFIG["residual_warmup_steps"] = args.residual_warmup_steps
    CONFIG["smolvla_chunk_size"] = args.chunk_size
    CONFIG["smolvla_path"] = args.smolvla_path
    CONFIG["classifier_path"] = args.classifier_path
    CONFIG["dataset_root"] = args.dataset_root
    CONFIG["torque_penalty_weight"] = args.torque_weight
    if args.no_preload:
        CONFIG["preload_demos"] = False
    if args.no_wandb:
        CONFIG["use_wandb"] = False
    if args.no_bf16:
        CONFIG["use_bf16"] = False
    CONFIG["pure_base"] = args.pure_base
    if args.pure_base:
        print("*** PURE-BASE MODE: residual disabled, no training. ***")
        CONFIG["preload_demos"] = False

    # Seeds
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])

    # Device
    dev = CONFIG["device"]
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        dev = "cpu"
    device = torch.device(dev)
    print(f"Using device: {device}")

    # Robot
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=0, width=640, height=480, fps=30,
            rotation=Cv2Rotation.ROTATE_180,
        ),
    }
    robot_cfg = SOFollowerRobotConfig(port=CONFIG["port"], id=CONFIG["robot_id"], cameras=cameras)
    robot = make_robot_from_config(robot_cfg)
    robot.connect()

    motor_names = list(robot.bus.motors.keys())
    motor_names_pos = [f"{n}.pos" for n in motor_names]
    action_dim = len(motor_names)
    q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])

    safety = SafetyLayer(
        robot.bus, motor_names,
        max_delta_deg=CONFIG["safety_max_delta_deg"],
        current_threshold_mA=CONFIG["safety_threshold_mA"],
        current_limit_mA=CONFIG["safety_limit_mA"],
    )

    # Frozen base + classifier
    base = SmolVLABase(CONFIG["smolvla_path"], CONFIG["smolvla_chunk_size"], device)
    reward_fn = RewardClassifier(CONFIG["classifier_path"], device)

    # Residual networks
    img_shape = (3, *CONFIG["img_size"])
    encoder = ResidualEncoder(action_dim, img_shape=img_shape, use_ln=CONFIG["use_ln"]).to(device)
    actor = ResidualActor(encoder.out_dim, action_dim, use_ln=CONFIG["use_ln"]).to(device)
    critic = ResidualCritic(encoder.out_dim, action_dim, use_ln=CONFIG["use_ln"]).to(device)
    critic_target = ResidualCritic(encoder.out_dim, action_dim, use_ln=CONFIG["use_ln"]).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()
    for p in critic_target.parameters():
        p.requires_grad_(False)

    # Automated entropy tuning — note: target entropy for bounded delta_a
    target_entropy = -float(action_dim)
    log_alpha = torch.tensor(np.log(CONFIG["alpha_init"]), requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=CONFIG["lr"])
    opt_enc = optim.Adam(encoder.parameters(), lr=CONFIG["lr"])
    opt_actor = optim.Adam(actor.parameters(), lr=CONFIG["lr"])
    opt_critic = optim.Adam(critic.parameters(), lr=CONFIG["lr"])

    # Replay buffer
    replay = deque(maxlen=CONFIG["buffer_size"])

    # Preload demos
    if CONFIG["preload_demos"]:
        demos = preload_demos(
            CONFIG["dataset_root"], CONFIG["dataset_repo_id"],
            CONFIG["img_size"], device,
        )
        for t in demos:
            replay.append(t)
        print(f"Replay buffer preloaded with {len(replay)} transitions")

    logger = SimpleLogger(CONFIG)

    # Training loop state
    total_steps = 0
    episode_num = 0
    BATCH = CONFIG["batch_size"]
    GAMMA = CONFIG["gamma"]
    TAU = CONFIG["tau"]
    UTD = CONFIG["utd_ratio"]
    POLICY_DELAY = CONFIG["policy_delay"]
    RESIDUAL_SCALE = CONFIG["residual_scale"]
    TASK = args.task

    # Action scaling: SmolVLA outputs in "raw action" degrees (same as dataset)
    # Our residual is bounded to ±residual_scale * action_range.
    # For cube-out-of-box, joints span ~180deg so residual_scale=0.2 ≈ ±36 deg worst case,
    # but the safety layer clamps per-step deltas to 5° anyway.
    # We'll express the residual in the same units as a_base (degrees), scaled by a fixed range.
    ACTION_RANGE_DEG = 90.0  # ± per-joint range the residual can span
    residual_deg_scale = RESIDUAL_SCALE * ACTION_RANGE_DEG

    print("\nStarting ResFit RL loop. Press Ctrl+C to stop.\n")
    try:
        while True:
            episode_num += 1
            episode_reward = 0.0
            episode_steps = 0
            reset_robot(robot, q_home, motor_names_pos)
            base.reset()

            while episode_steps < CONFIG["max_episode_steps"]:
                t_start = time.perf_counter()
                step_period = 1.0 / CONFIG["fps"]
                # Minimum time between sending an action and observing its
                # effect. Keeps action→obs lag reasonable even if training below
                # takes longer than the remaining step budget.
                action_to_obs_dt = 0.05  # 50ms

                # 1. Observe
                obs_dict = robot.get_observation()
                img_full, img_small, joints_t, joints_raw = extract_obs(
                    obs_dict, motor_names_pos, CONFIG["img_size"], device,
                )

                # 2. Base policy (chunked)
                a_base = base(img_full, joints_raw, TASK)  # (action_dim,) in degrees
                a_base_t = torch.from_numpy(a_base).float().unsqueeze(0).to(device) / 100.0

                # 3. Residual policy
                if CONFIG["pure_base"]:
                    # Skip residual entirely — robot executes SmolVLA's a_base.
                    delta_unit = torch.zeros(1, action_dim, device=device)
                    delta_np = np.zeros(action_dim, dtype=np.float32)
                    ramp = 0.0
                else:
                    encoder.eval(); actor.eval(); critic.eval()
                    with torch.no_grad():
                        feat = encoder(img_small, joints_t)
                        delta_unit, _ = actor.sample(feat, a_base_t)
                    # Linear residual ramp 0 → 1 over `residual_warmup_steps`. Lets
                    # the VLA run effectively untouched at first, then blends in.
                    ramp = min(1.0, total_steps / max(1, CONFIG["residual_warmup_steps"]))
                    delta_np = delta_unit.squeeze(0).cpu().numpy() * (residual_deg_scale * ramp)

                # 4. Combine and act
                target_q_vals = a_base + delta_np
                target_q_vals = np.clip(target_q_vals, q_home - 90, q_home + 90)
                q_safe, safety_info = safety(target_q_vals, q_home=q_home)
                robot.send_action({n: float(v) for n, v in zip(motor_names_pos, q_safe)})

                # 5. Wait a fixed short time for the robot to respond before observing
                precise_sleep(action_to_obs_dt)
                next_obs = robot.get_observation()
                next_img_full, next_img_small, next_joints_t, next_joints_raw = extract_obs(
                    next_obs, motor_names_pos, CONFIG["img_size"], device,
                )
                next_a_base = a_base  # within-chunk approximation

                # 6. Reward
                r_task = reward_fn(next_img_full)
                currents = safety_info["currents"]
                torque_sq = float(np.sum(currents ** 2)) / 1000.0
                r_torque = -CONFIG["torque_penalty_weight"] * torque_sq
                reward = CONFIG["classifier_reward_weight"] * r_task + r_torque
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                done = (r_task > 0.5) or (episode_steps >= CONFIG["max_episode_steps"])

                # 7. Store (skip in pure-base mode — no RL learning needed)
                if not CONFIG["pure_base"]:
                    replay.append((
                        img_small.cpu(), joints_t.cpu(),
                        a_base_t.cpu().squeeze(0),
                        delta_unit.cpu().squeeze(0),   # store UNSCALED delta in [-1,1]
                        float(reward),
                        next_img_small.cpu(), next_joints_t.cpu(),
                        torch.from_numpy(next_a_base).float() / 100.0,
                        float(done),
                    ))

                # 8. Train
                loss_q_val = loss_a_val = mean_q_val = 0.0
                alpha_val = log_alpha.exp().item()

                if (not CONFIG["pure_base"]
                        and len(replay) > BATCH
                        and total_steps > CONFIG["warmup_steps"]):
                    encoder.train(); actor.train(); critic.train()
                    loss_a_t = torch.tensor(0.0, device=device)
                    for upd in range(UTD):
                        batch = random.sample(replay, BATCH)
                        b_img, b_q, b_ab, b_d, b_r, b_nimg, b_nq, b_nab, b_done = zip(*batch)

                        b_img = torch.cat([x.unsqueeze(0) if x.ndim == 3 else x for x in b_img]).to(device)
                        b_q = torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in b_q]).to(device)
                        b_ab = torch.stack(b_ab).to(device)
                        b_d = torch.stack(b_d).to(device)
                        b_r = torch.tensor(b_r, dtype=torch.float32, device=device).unsqueeze(1)
                        b_nimg = torch.cat([x.unsqueeze(0) if x.ndim == 3 else x for x in b_nimg]).to(device)
                        b_nq = torch.cat([x.unsqueeze(0) if x.ndim == 1 else x for x in b_nq]).to(device)
                        b_nab = torch.stack(b_nab).to(device)
                        b_done = torch.tensor(b_done, dtype=torch.float32, device=device).unsqueeze(1)

                        curr_alpha = log_alpha.exp()

                        with torch.amp.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=CONFIG["use_bf16"] and device.type == "cuda",
                        ):
                            with torch.no_grad():
                                next_feat = encoder(b_nimg, b_nq)
                                next_d, next_lp = actor.sample(next_feat, b_nab)
                                q1_t, q2_t = critic_target(next_feat, b_nab, next_d)
                                target_q = b_r + (1 - b_done) * GAMMA * (
                                    torch.min(q1_t, q2_t) - curr_alpha * next_lp
                                )

                            curr_feat = encoder(b_img, b_q)
                            q1, q2 = critic(curr_feat, b_ab, b_d)
                            loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                        opt_enc.zero_grad()
                        opt_critic.zero_grad()
                        loss_q.backward()
                        opt_enc.step()
                        opt_critic.step()
                        if CONFIG["use_wn"]:
                            weight_normalize_(encoder)
                            weight_normalize_(critic)

                        # Delayed actor & alpha updates
                        if upd % POLICY_DELAY == 0:
                            critic.eval()
                            with torch.amp.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=CONFIG["use_bf16"] and device.type == "cuda",
                            ):
                                curr_feat_detached = curr_feat.detach()
                                new_d, new_lp = actor.sample(curr_feat_detached, b_ab)
                                q1_new, q2_new = critic(curr_feat_detached, b_ab, new_d)
                                min_q = torch.min(q1_new, q2_new)
                                loss_a_t = (curr_alpha.detach() * new_lp - min_q).mean()
                            opt_actor.zero_grad()
                            loss_a_t.backward()
                            opt_actor.step()
                            critic.train()
                            if CONFIG["use_wn"]:
                                weight_normalize_(actor)

                            # alpha
                            loss_alpha = -(log_alpha * (new_lp.detach() + target_entropy)).mean()
                            opt_alpha.zero_grad()
                            loss_alpha.backward()
                            opt_alpha.step()

                        # Soft target update
                        with torch.no_grad():
                            for p, pt in zip(critic.parameters(), critic_target.parameters()):
                                pt.data.mul_(1 - TAU).add_(p.data, alpha=TAU)

                        loss_q_val = float(loss_q.item())
                        loss_a_val = float(loss_a_t.item())
                        mean_q_val = float(torch.min(q1, q2).mean().item())

                # 9. Log
                dt = time.perf_counter() - t_start
                hz = 1.0 / dt if dt > 0 else 0.0
                # Time spent working vs sleeping is useful for diagnosing
                # whether training is eating the step budget.
                work_dt = time.perf_counter() - t_start
                sleep_budget = step_period - work_dt
                metrics = {
                    "total_steps": total_steps,
                    "episode": episode_num,
                    "reward": float(reward),
                    "r_task": float(r_task),
                    "q_val": mean_q_val,
                    "loss_q": loss_q_val,
                    "loss_a": loss_a_val,
                    "alpha": alpha_val,
                    "hz": hz,
                    "work_dt_ms": float(work_dt * 1000),
                    "sleep_budget_ms": float(sleep_budget * 1000),
                    "safety_atten": float(safety_info["attenuation"].mean()),
                    "vla_inf_ms": float(base.last_inference_dt * 1000),
                    "delta_abs_mean": float(np.abs(delta_np).mean()),
                    "residual_ramp": float(ramp),
                }
                if CONFIG["torque_penalty_weight"] > 0:
                    metrics["torque"] = torque_sq
                    metrics["r_torque"] = float(r_torque)
                logger.log(metrics)

                if total_steps % 10 == 0:
                    over = "!" if sleep_budget < 0 else " "
                    print(f"Ep {episode_num:3d} | S {episode_steps:3d} | "
                          f"R {reward:+5.2f} (task {r_task:.0f}) | "
                          f"Q {mean_q_val:+5.2f} | "
                          f"|Δ| {np.abs(delta_np).mean():4.1f}° | "
                          f"Hz {hz:4.1f}{over}| "
                          f"VLA {base.last_inference_dt*1000:4.0f}ms | "
                          f"S {float(safety_info['attenuation'].mean()):.0%} | "
                          f"α {alpha_val:.3f}")

                if done:
                    print(f"--- Episode {episode_num} done. reward={episode_reward:.2f} "
                          f"steps={episode_steps} success={r_task>0.5}")
                    break

                # 10. Pace: sleep until the end of this step period so commands
                # go out at a steady `fps` rate. If training was slow, we won't
                # sleep and Hz will drop — watch `sleep_budget_ms` / "!" marker.
                precise_sleep(max(0.0, step_period - (time.perf_counter() - t_start)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        logger.finish()
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
