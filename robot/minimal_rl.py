#!/usr/bin/env python3
"""
Minimal Single-Process RL script for SO-101.
Bypasses the distributed architecture for direct debugging.

Usage:
    python robot/minimal_rl.py
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import deque
import random
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
try:
    import umap
    from scipy.sparse import lil_matrix, csr_matrix
    from sklearn.neighbors import NearestNeighbors
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots import make_robot_from_config
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.configs import Cv2Rotation
from lerobot.utils.robot_utils import precise_sleep

from safety import SafetyLayer

# --- Configuration ---
CONFIG = {
    "port": "/dev/ttyACM0",
    "robot_id": "shabby",
    "fps": 10,
    "lr": 3e-4,
    "batch_size": 64,
    "buffer_size": 10000,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha_init": 0.2,
    "utd_ratio": 4,
    "warmup_steps": 50,
    "max_episode_steps": 200,
    "img_size": (64, 64),
    "use_camera": False,
    "use_joints": True,
    "seed": 420,
    "use_bf16": True,
    "pos_reward_weight": 0.0,
    "torque_penalty_weight": 100.0, # 0.0 means off
    "use_simple_torque": True, # True: sum(I^2), False: sigmoid(sum(I^2))
    "policy_delay": 2, # Update policy every N steps
    "use_bn": False, # Use Batch Norm (XQC style) - broken right now.
    "use_ln": True, # Use Layer Norm (SimBa/BRO style) - stable combo with WeightNorm
    "use_wn": True, # Use Weight Norm projection (XQC style)
    "use_reward_norm": False, # Use reward normalization - kind of broken right now.
    "use_wandb": True, # Use Weights & Biases
    "use_analysis": True, # Run post-training analysis (heatmaps, UMAP)
    "device": "cuda", # "cuda" or "cpu"
    "safety_max_delta_deg": 5.0, # Max position change per step (degrees) — proactive
    "safety_threshold_mA": 150.0, # Current below which RL has full authority
    "safety_limit_mA": 400.0, # Current at which RL authority drops to zero
    "runs_dir": "robot/runs",
}

# Mapping for easier access in code
PORT = CONFIG["port"]
ROBOT_ID = CONFIG["robot_id"]
FPS = CONFIG["fps"]
LR = CONFIG["lr"]
BATCH_SIZE = CONFIG["batch_size"]
BUFFER_SIZE = CONFIG["buffer_size"]
GAMMA = CONFIG["gamma"]
TAU = CONFIG["tau"]
ALPHA_INIT = CONFIG["alpha_init"]
UTD_RATIO = CONFIG["utd_ratio"]
WARMUP_STEPS = CONFIG["warmup_steps"]
MAX_EPISODE_STEPS = CONFIG["max_episode_steps"]
IMG_SIZE = CONFIG["img_size"]
USE_CAMERA = CONFIG["use_camera"]
USE_JOINTS = CONFIG["use_joints"]
POS_REWARD_WEIGHT = CONFIG["pos_reward_weight"]
TORQUE_PENALTY_WEIGHT = CONFIG["torque_penalty_weight"]
USE_SIMPLE_TORQUE = CONFIG["use_simple_torque"]
POLICY_DELAY = CONFIG["policy_delay"]
USE_BN = CONFIG["use_bn"]
USE_LN = CONFIG["use_ln"]
USE_WN = CONFIG["use_wn"]
SEED = CONFIG["seed"]
USE_BF16 = CONFIG["use_bf16"]
RUNS_DIR = Path(CONFIG["runs_dir"])

# --- Logging ---

class RunLogger:
    def __init__(self, config, run_name=None):
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RUNS_DIR / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / "metrics.jsonl"
        self.run_name = run_name
        
        # Save config
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)

        # Init wandb
        self.use_wandb = HAS_WANDB and USE_WANDB
        if self.use_wandb:
            try:
                wandb.init(project="minimal-rl", name=run_name, config=config,
                           dir=str(self.run_dir))
                print(f"W&B run: {wandb.run.url}")
            except Exception as e:
                print(f"W&B init failed: {e}")
                self.use_wandb = False
            
        self.data = []
        print(f"Logging to {self.run_dir}")

    def log(self, metrics):
        self.data.append(metrics)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.use_wandb:
            wandb.log(metrics, step=metrics.get("total_steps", None))

    def finish(self):
        """Upload final artifacts and close wandb."""
        if self.use_wandb:
            for img_path in self.run_dir.glob("*.jpg"):
                wandb.log({img_path.stem: wandb.Image(str(img_path))})
            wandb.finish()

    def plot(self):
        if not self.data: return
        all_keys = ["reward", "reward_pos", "torque_penalty", "q_val", "torque", "hz", "alpha"]
        keys = [k for k in all_keys if k in self.data[0]]
        if not keys: return

        fig, axes = plt.subplots(len(keys), 1, figsize=(10, 3 * len(keys)), sharex=True)
        if len(keys) == 1:
            axes = [axes]
        steps = [d["total_steps"] for d in self.data]
        
        for i, key in enumerate(keys):
            vals = [d.get(key, 0) for d in self.data]
            axes[i].plot(steps, vals)
            axes[i].set_ylabel(key.capitalize())
            axes[i].grid(True)
        
        axes[-1].set_xlabel("Total Steps")
        plt.tight_layout()
        plot_path = self.run_dir / "learning_curves.jpg"
        plt.savefig(plot_path, dpi=150, format="jpeg")
        plt.close()
        print(f"Plot saved to {plot_path}")

    def plot_analysis(self, q_matrix, policy_matrix, motor_names, range_deg):
        fig, (ax1, axes2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Q-Value Heatmap
        sns.heatmap(q_matrix, xticklabels=np.round(range_deg, 1), yticklabels=motor_names, ax=ax1, cmap="viridis")
        ax1.set_title("Critic Q-Values (Varying each joint +/- 45°)")
        ax1.set_xlabel("Relative Joint Position (Degrees)")
        
        # Policy Heatmap (Delta action for the joint being varied)
        sns.heatmap(policy_matrix, xticklabels=np.round(range_deg, 1), yticklabels=motor_names, ax=axes2, cmap="RdBu_r", center=0)
        axes2.set_title("Actor Delta Actions (Response of varied joint)")
        axes2.set_xlabel("Relative Joint Position (Degrees)")
        
        plt.tight_layout()
        plot_path = self.run_dir / "post_analysis_heatmaps.jpg"
        plt.savefig(plot_path, dpi=150, format="jpeg")
        plt.close()
        print(f"Analysis heatmaps saved to {plot_path}")

    @staticmethod
    def _sym_kl_pair(mu_i, std_i, mu_j, std_j):
        """Symmetrized KL between two diagonal Gaussians (numpy vectors)."""
        var_i = np.maximum(std_i, 1e-6) ** 2
        var_j = np.maximum(std_j, 1e-6) ** 2
        d = mu_i.shape[0]
        diff = mu_i - mu_j
        kl_ij = 0.5 * ((var_i / var_j).sum() + (diff**2 / var_j).sum() - d + np.log(var_j / var_i).sum())
        kl_ji = 0.5 * ((var_j / var_i).sum() + (diff**2 / var_i).sum() - d + np.log(var_i / var_j).sum())
        return max(0.5 * (kl_ij + kl_ji), 0.0)

    def _plot_umap_panels(self, embedding, states, std, timestamps, ep_ids, ep_steps, title_prefix, filename, sizes=None):
        """Shared 6-panel UMAP scatter plot."""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        fig.suptitle(title_prefix, fontsize=14, y=1.02)
        s = sizes if sizes is not None else 8

        # 1. Timestep (Total)
        sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=timestamps, cmap="viridis", s=s, alpha=0.7)
        axes[0].set_title("Total Timestep")
        plt.colorbar(sc0, ax=axes[0], label="Step")

        # 2. Episode ID
        sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=ep_ids, cmap="tab20", s=s, alpha=0.7)
        axes[1].set_title("Episode ID")
        plt.colorbar(sc1, ax=axes[1], label="Ep")

        # 3. Timestep in Episode
        sc2 = axes[2].scatter(embedding[:, 0], embedding[:, 1], c=ep_steps, cmap="plasma", s=s, alpha=0.7)
        axes[2].set_title("Timestep in Episode")
        plt.colorbar(sc2, ax=axes[2], label="Ep Step")

        # 4. Policy Entropy
        entropy = np.sum(np.log(std + 1e-8), axis=1)
        sc3 = axes[3].scatter(embedding[:, 0], embedding[:, 1], c=entropy, cmap="coolwarm", s=s, alpha=0.7)
        axes[3].set_title("Policy Entropy")
        plt.colorbar(sc3, ax=axes[3], label="Σ log(σ)")

        # 5. Distance to Home
        dist_home = np.linalg.norm(states, axis=1)
        sc4 = axes[4].scatter(embedding[:, 0], embedding[:, 1], c=dist_home, cmap="magma", s=s, alpha=0.7)
        axes[4].set_title("Dist-to-home")
        plt.colorbar(sc4, ax=axes[4], label="||q - q_home||")

        # 6. Empty or just repeat one for now
        axes[5].axis("off")

        plt.tight_layout()
        path = self.run_dir / filename
        plt.savefig(path.with_suffix(".jpg"), dpi=150, format="jpeg")
        plt.close()
        print(f"  Saved to {path.with_suffix('.jpg')}")

    @staticmethod
    def _safe_umap(dist_input, n_neighbors, is_sparse=False):
        """Run UMAP with automatic n_neighbors clamping and error handling."""
        if is_sparse:
            # Compute minimum row degree from sparse matrix
            row_nnz = np.diff(dist_input.indptr)
            min_degree = int(row_nnz[row_nnz > 0].min()) if np.any(row_nnz > 0) else 0
            if min_degree < 2:
                print(f"  Warning: graph too sparse (min degree={min_degree}), skipping.")
                return None
            n_neighbors = min(n_neighbors, min_degree)
        try:
            reducer = umap.UMAP(metric="precomputed", n_neighbors=n_neighbors, random_state=42)
            return reducer.fit_transform(dist_input)
        except Exception as e:
            print(f"  UMAP failed: {e}")
            return None

    def plot_umap_knn_density(self, states, mu, std, timestamps, ep_ids, ep_steps, episode_ends,
                              k=30, filename="umap_knn_density.jpg"):
        """k-NN + analytical transition density UMAP.
        Uses Euclidean k-NN for candidate neighbors, then evaluates the
        analytical Gaussian log-density p(s'|s) for each neighbor pair.
        Also force-includes actual replay successors.
        """
        if not HAS_UMAP:
            print("umap-learn not installed, skipping UMAP. pip install umap-learn")
            return
        n = len(states)
        if n < 20:
            print("Not enough data for k-NN density UMAP.")
            return

        max_pts = 2000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
            idx.sort()
            states = states[idx]
            mu = mu[idx]
            std = std[idx]
            timestamps = timestamps[idx]
            ep_ids = ep_ids[idx]
            ep_steps = ep_steps[idx]
            n = max_pts

        k_actual = min(k, n - 1)
        print(f"  Building k-NN graph (n={n}, k={k_actual})...")
        nn = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nn.fit(states)
        _, neighbor_indices = nn.kneighbors(states)

        # Force-include actual replay successors
        episode_ends_set = set(episode_ends.tolist()) if len(episode_ends) else set()
        successor_pairs = set()
        for t in range(n - 1):
            if t not in episode_ends_set:
                successor_pairs.add((t, t + 1))

        # Build sparse log-density matrix
        print(f"  Computing analytical log-densities...")
        log_P = lil_matrix((n, n), dtype=np.float64)
        stds_safe = np.maximum(std, 1e-6)

        for i in range(n):
            # k-NN neighbors
            js = neighbor_indices[i]
            x_neighbors = states[js]
            diff = x_neighbors - mu[i]
            log_probs = -0.5 * np.sum(
                (diff / stds_safe[i]) ** 2 + 2 * np.log(stds_safe[i]) + np.log(2 * np.pi),
                axis=1
            )
            for idx_k, j in enumerate(js):
                if i != j:
                    log_P[i, j] = log_probs[idx_k]

        # Force-add successor transitions
        for i, j in successor_pairs:
            if log_P[i, j] == 0:
                diff = states[j] - mu[i]
                lp = -0.5 * np.sum(
                    (diff / stds_safe[i]) ** 2 + 2 * np.log(stds_safe[i]) + np.log(2 * np.pi)
                )
                log_P[i, j] = lp

        log_P_csr = csr_matrix(log_P)

        # Symmetrize in log-density space
        log_P_sym = (log_P_csr + log_P_csr.T) / 2.0

        # Convert to distance: negate (high density = short distance)
        D = log_P_sym.copy()
        D.data = -D.data
        D.data -= D.data.min()

        D_csr = csr_matrix(D)

        embedding = self._safe_umap(D_csr, n_neighbors=min(15, k_actual), is_sparse=True)
        if embedding is None:
            print("  Sparse k-NN density failed, falling back to dense...")
            D_dense = D_csr.toarray()
            embedding = self._safe_umap(D_dense, n_neighbors=min(15, n - 1))
            if embedding is None:
                return

        self._plot_umap_panels(embedding, states, std, timestamps, ep_ids, ep_steps,
                               f"UMAP k-NN + analytical density (k={k_actual})", filename)

    def plot_umap_full_density(self, states, mu, std, timestamps, ep_ids, ep_steps,
                               filename="umap_full_density.jpg"):
        """Full pairwise UMAP using analytical transition density (not KL)."""
        if not HAS_UMAP:
            print("umap-learn not installed, skipping UMAP. pip install umap-learn")
            return
        n = len(states)
        if n < 20:
            print("Not enough data for full density UMAP.")
            return

        max_pts = 2000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
            idx.sort()
            states = states[idx]
            mu = mu[idx]
            std = std[idx]
            timestamps = timestamps[idx]
            ep_ids = ep_ids[idx]
            ep_steps = ep_steps[idx]
            n = max_pts

        stds_safe = np.maximum(std, 1e-6)

        print(f"  Computing {n}x{n} pairwise analytical densities...")
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                # log p(s_j | s_i) under policy at s_i
                diff_ij = states[j] - mu[i]
                lp_ij = -0.5 * np.sum(
                    (diff_ij / stds_safe[i]) ** 2 + 2 * np.log(stds_safe[i]) + np.log(2 * np.pi)
                )
                # log p(s_i | s_j) under policy at s_j
                diff_ji = states[i] - mu[j]
                lp_ji = -0.5 * np.sum(
                    (diff_ji / stds_safe[j]) ** 2 + 2 * np.log(stds_safe[j]) + np.log(2 * np.pi)
                )
                # Symmetrize and convert to distance
                avg_log_p = 0.5 * (lp_ij + lp_ji)
                dist_matrix[i, j] = -avg_log_p
                dist_matrix[j, i] = -avg_log_p

        # Shift so minimum distance is 0
        dist_matrix -= dist_matrix[dist_matrix > 0].min() if np.any(dist_matrix > 0) else 0

        embedding = self._safe_umap(dist_matrix, n_neighbors=min(15, n - 1))
        if embedding is None:
            return

        self._plot_umap_panels(embedding, states, std, timestamps, ep_ids, ep_steps,
                               "UMAP full analytical density", filename)

    def plot_umap_full_kl(self, states, mu, std, timestamps, ep_ids, ep_steps,
                          filename="umap_full_kl.jpg"):
        """Full pairwise UMAP using symmetrized KL divergence between policy distributions."""
        if not HAS_UMAP:
            print("umap-learn not installed, skipping UMAP. pip install umap-learn")
            return

        n = len(states)
        if n < 20:
            print("Not enough data for full KL UMAP.")
            return

        max_pts = 2000
        if n > max_pts:
            idx = np.random.choice(n, max_pts, replace=False)
            idx.sort()
            states = states[idx]
            mu = mu[idx]
            std = std[idx]
            timestamps = timestamps[idx]
            ep_ids = ep_ids[idx]
            ep_steps = ep_steps[idx]
            n = max_pts

        print(f"  Computing {n}x{n} pairwise KL distances...")
        dist_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._sym_kl_pair(mu[i], std[i], mu[j], std[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        embedding = self._safe_umap(dist_matrix, n_neighbors=min(15, n - 1))
        if embedding is None:
            return

        self._plot_umap_panels(embedding, states, std, timestamps, ep_ids, ep_steps,
                               "UMAP full pairwise KL", filename)

class RunningMeanStd(nn.Module):
    """Welford online algorithm for tracking running mean and variance."""
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
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
    def std(self) -> torch.Tensor:
        return (self.var + self.epsilon).sqrt()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.std

# --- Models ---

def weight_normalize_(module):
    """Project Linear layer weights to unit sphere (XQC style)."""
    with torch.no_grad():
        for name, m in module.named_modules():
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                # Do not normalize the final output layers of Actor and Critic
                if "mu" in name or "log_std" in name or "head" in name:
                    continue
                m.weight.data = F.normalize(m.weight.data, dim=1)

def make_mlp(in_dim, out_dim, hidden_dim=256, use_bn=False, use_ln=False):
    """Utility to build MLP with optional BatchNorm (XQC) or LayerNorm (SimBa/BRO).
    Order: Linear -> Norm -> ReLU.  LayerNorm is safer than BatchNorm for RL
    (no train/eval mismatch, no target-net desync).
    """
    layers = []
    # If using BN, XQC starts with a BN on raw input (per-feature normalization).
    # LN on raw input is less standard; we skip it.
    if use_bn:
        layers.append(nn.BatchNorm1d(in_dim))

    # Layer 1
    layers.append(nn.Linear(in_dim, hidden_dim))
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    elif use_ln:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())

    # Layer 2
    layers.append(nn.Linear(hidden_dim, hidden_dim))
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    elif use_ln:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)

class TinyEncoder(nn.Module):
    def __init__(self, state_dim, img_shape=(3, 64, 64), use_camera=False, use_joints=True, use_bn=False, use_ln=False):
        super().__init__()
        self.use_camera = use_camera
        self.use_joints = use_joints
        self.use_bn = use_bn
        self.use_ln = use_ln
        if self.use_camera:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.Flatten()
            )
            with torch.no_grad():
                dummy = torch.zeros(1, *img_shape)
                cnn_out_dim = self.cnn(dummy).shape[1]
        else:
            cnn_out_dim = 0
        
        in_dim = cnn_out_dim + (state_dim if self.use_joints else 0)
        if in_dim == 0:
            raise ValueError("Both camera and joints are disabled. Nothing to encode!")

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256) if use_bn else (nn.LayerNorm(256) if use_ln else nn.Identity()),
            nn.ReLU()
        )
        self.out_dim = 256

    def forward(self, img, state):
        parts = []
        if self.use_camera:
            parts.append(self.cnn(img))
        if self.use_joints:
            parts.append(state)
        
        x = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]
        # BN expects at least 2 samples if training. In inference (1 sample), we must be in eval mode.
        if x.shape[0] == 1 and self.training and self.use_bn:
            # Fallback or handled by caller setting eval()
            pass
        return self.fc(x)

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, use_bn=False, use_ln=False):
        super().__init__()
        self.net = make_mlp(input_dim, 256, use_bn=use_bn, use_ln=use_ln)
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

    def sample(self, x):
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, input_dim, action_dim, use_bn=False, use_ln=False):
        super().__init__()
        self.q1_net = make_mlp(input_dim + action_dim, 256, use_bn=use_bn, use_ln=use_ln)
        self.q1_head = nn.Linear(256, 1)
        
        self.q2_net = make_mlp(input_dim + action_dim, 256, use_bn=use_bn, use_ln=use_ln)
        self.q2_head = nn.Linear(256, 1)

    def forward(self, x, a):
        xa = torch.cat([x, a], dim=-1)
        return self.q1_head(self.q1_net(xa)), self.q2_head(self.q2_net(xa))

# --- Utilities ---

def preprocess_obs(obs_dict, device, motor_names_pos, use_camera=False):
    # Joints
    joints = [obs_dict[k] for k in motor_names_pos]
    joints = torch.tensor(joints, dtype=torch.float32, device=device).unsqueeze(0) / 100.0
    
    img = None
    if use_camera:
        # Image (Take first available camera)
        cam_key = next((k for k in obs_dict.keys() if "observation.images." in k), None)
        if cam_key is None:
            # Fallback to zeros if no camera found
            img = torch.zeros(1, 3, *IMG_SIZE, device=device)
        else:
            img_raw = obs_dict[cam_key]
            img_resized = cv2.resize(img_raw, IMG_SIZE)
            # Show what the robot sees
            try:
                cv2.imshow("Robot View", cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            except:
                pass
            img = torch.tensor(img_resized, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    return img, joints

def reset_robot(robot, q_home, motor_names_pos):
    print("Resetting to home position...")
    # Move slowly
    steps = 50
    curr_obs = robot.get_observation()
    q_curr = np.array([curr_obs[k] for k in motor_names_pos])
    
    for i in range(steps):
        interp_q = q_curr + (q_home - q_curr) * (i + 1) / steps
        robot_action = {name: float(val) for name, val in zip(motor_names_pos, interp_q)}
        robot.send_action(robot_action)
        time.sleep(0.05)
    print("Reset complete.")

@torch.no_grad()
def run_post_analysis(encoder, actor, critic, q_home, motor_names_pos, device, last_img, logger):
    print("\nStarting post-training analysis...")
    num_joints = len(motor_names_pos)
    num_points = 20
    range_vals = np.linspace(-45, 45, num_points)
    
    q_matrix = np.zeros((num_joints, num_points))
    policy_matrix = np.zeros((num_joints, num_points))
    
    motor_names_short = [n.removesuffix(".pos") for n in motor_names_pos]
    
    encoder.eval(); actor.eval(); critic.eval()
    
    # Placeholder image if none exists
    if last_img is None:
        last_img = torch.zeros(1, 3, *IMG_SIZE).to(device)
    else:
        last_img = last_img.to(device)

    for i in range(num_joints):
        # Create a batch of states where joint i varies
        states = []
        for v in range_vals:
            q_state = q_home.copy()
            q_state[i] += v
            states.append(q_state)
        
        q_torch = torch.tensor(np.array(states), dtype=torch.float32, device=device) / 100.0
        
        # Expand image to match batch size
        img_batch = last_img.repeat(num_points, 1, 1, 1)
        
        # Forward pass
        feats = encoder(img_batch, q_torch if USE_JOINTS else None)
        actions, _ = actor.sample(feats)
        q1, q2 = critic(feats, actions)
        
        q_vals = torch.min(q1, q2).cpu().numpy().flatten()
        delta_actions = actions.cpu().numpy()[:, i] # Take delta for the current joint
        
        q_matrix[i, :] = q_vals
        policy_matrix[i, :] = delta_actions
        
    logger.plot_analysis(q_matrix, policy_matrix, motor_names_short, range_vals)

@torch.no_grad()
def run_umap_analysis(encoder, actor, replay_buffer, q_home, motor_names_pos, device, logger):
    """Build UMAP of visited states using symmetrized KL of policy distributions."""
    if not HAS_UMAP:
        print("umap-learn not installed, skipping UMAP.")
        return
    if len(replay_buffer) < 50:
        print("Replay buffer too small for UMAP.")
        return

    print("\nBuilding UMAP from replay buffer...")
    encoder.eval(); actor.eval()

    # Buffer format: (img, q, action, reward, next_img, next_q, done)
    all_q = []
    all_q_raw = []
    all_img = []
    episode_ends = []
    ep_ids = []
    ep_steps = []
    
    curr_ep_id = 0
    curr_ep_step = 0
    
    for idx, (img, q, a, r, ni, nq, done) in enumerate(replay_buffer):
        all_q.append(q)
        all_q_raw.append(q.numpy().flatten() * 100.0)
        if img is not None:
            all_img.append(img)
        
        ep_ids.append(curr_ep_id)
        ep_steps.append(curr_ep_step)
        
        if done:
            episode_ends.append(idx)
            curr_ep_id += 1
            curr_ep_step = 0
        else:
            curr_ep_step += 1

    timestamps = np.arange(len(all_q))
    all_q_raw = np.array(all_q_raw) - q_home
    episode_ends = np.array(episode_ends)
    ep_ids = np.array(ep_ids)
    ep_steps = np.array(ep_steps)

    # Batch forward pass to get policy (mu, std) for every state
    batch_sz = 256
    all_mu = []
    all_std = []
    for start in range(0, len(all_q), batch_sz):
        end = min(start + batch_sz, len(all_q))
        q_batch = torch.cat(all_q[start:end]).to(device)
        img_batch = None
        if USE_CAMERA and all_img:
            img_batch = torch.cat(all_img[start:end]).to(device)
        feat = encoder(img_batch, q_batch)
        mu, log_std = actor(feat)
        all_mu.append(mu.cpu().numpy())
        all_std.append(log_std.exp().cpu().numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_std = np.concatenate(all_std, axis=0)

    # 1. k-NN + analytical transition density (fast, best quality)
    print("UMAP 1/4: k-NN + analytical density (fast)...")
    logger.plot_umap_knn_density(all_q_raw, all_mu, all_std, timestamps, ep_ids, ep_steps, episode_ends)

    # 2. Full pairwise analytical density (slow, for comparison)
    print("UMAP 2/4: Full pairwise analytical density (slow)...")
    logger.plot_umap_full_density(all_q_raw, all_mu, all_std, timestamps, ep_ids, ep_steps)

    # 3. Full pairwise symmetrized KL (slow, behavioral distance)
    print("UMAP 3/4: Full pairwise KL divergence (slow)...")
    logger.plot_umap_full_kl(all_q_raw, all_mu, all_std, timestamps, ep_ids, ep_steps)

    # 4. k-NN + analytical density with larger k (comparison)
    print("UMAP 4/4: k-NN + analytical density k=50 (fast)...")
    logger.plot_umap_knn_density(all_q_raw, all_mu, all_std, timestamps, ep_ids, ep_steps, episode_ends,
                                 k=50, filename="umap_knn_density_k50.jpg")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=CONFIG["device"], help="cuda or cpu")
    parser.add_argument("--port", type=str, default=CONFIG["port"])
    parser.add_argument("--utd", type=int, default=CONFIG["utd_ratio"])
    parser.add_argument("--camera", action="store_true", default=CONFIG["use_camera"])
    parser.add_argument("--no-joints", action="store_true", help="Disable joint positions in observation")
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--no-bf16", action="store_true", help="Disable bfloat16 mixed precision")
    parser.add_argument("--pos-weight", type=float, default=CONFIG["pos_reward_weight"])
    parser.add_argument("--torque-weight", type=float, default=CONFIG["torque_penalty_weight"])
    parser.add_argument("--use-sigmoid-torque", action="store_true", help="Use sigmoid instead of simple squared torque")
    parser.add_argument("--policy-delay", type=int, default=CONFIG["policy_delay"], help="Update policy every N steps")
    parser.add_argument("--bn", action="store_true", default=CONFIG["use_bn"], help="Use Batch Norm (XQC style)")
    parser.add_argument("--no-ln", action="store_true", help="Disable Layer Norm (SimBa/BRO style)")
    parser.add_argument("--wn", action="store_true", default=CONFIG["use_wn"], help="Use Weight Norm projection (XQC style)")
    parser.add_argument("--reward-norm", action="store_true", default=CONFIG["use_reward_norm"], help="Enable reward normalization")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases")
    parser.add_argument("--no-analysis", action="store_true", help="Disable post-training analysis")
    parser.add_argument("--safety-max-delta", type=float, default=CONFIG["safety_max_delta_deg"], help="Max degrees per step (safety)")
    parser.add_argument("--safety-threshold", type=float, default=CONFIG["safety_threshold_mA"], help="Current (mA) below which full authority")
    parser.add_argument("--safety-limit", type=float, default=CONFIG["safety_limit_mA"], help="Current (mA) at which authority=0")
    args = parser.parse_args()

    # Override config with args
    CONFIG["device"] = args.device
    CONFIG["port"] = args.port
    CONFIG["utd_ratio"] = args.utd
    CONFIG["use_camera"] = args.camera
    CONFIG["seed"] = args.seed
    CONFIG["pos_reward_weight"] = args.pos_weight
    CONFIG["torque_penalty_weight"] = args.torque_weight
    CONFIG["policy_delay"] = args.policy_delay
    CONFIG["use_bn"] = args.bn
    if args.no_ln:
        CONFIG["use_ln"] = False
    CONFIG["use_wn"] = args.wn
    CONFIG["use_reward_norm"] = args.reward_norm
    if args.no_joints:
        CONFIG["use_joints"] = False
    if args.no_bf16:
        CONFIG["use_bf16"] = False
    if args.use_sigmoid_torque:
        CONFIG["use_simple_torque"] = False
    if args.no_wandb:
        CONFIG["use_wandb"] = False
    if args.no_analysis:
        CONFIG["use_analysis"] = False
    CONFIG["safety_max_delta_deg"] = args.safety_max_delta
    CONFIG["safety_threshold_mA"] = args.safety_threshold
    CONFIG["safety_limit_mA"] = args.safety_limit

    # 0. Set Seed
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Mapping for easier access in code
    global PORT, ROBOT_ID, FPS, LR, BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU, ALPHA_INIT, UTD_RATIO, WARMUP_STEPS, MAX_EPISODE_STEPS, IMG_SIZE, USE_CAMERA, USE_JOINTS, POS_REWARD_WEIGHT, TORQUE_PENALTY_WEIGHT, USE_SIMPLE_TORQUE, POLICY_DELAY, USE_BN, USE_LN, USE_WN, USE_REWARD_NORM, USE_WANDB, USE_ANALYSIS, SEED, USE_BF16, RUNS_DIR
    PORT = CONFIG["port"]
    ROBOT_ID = CONFIG["robot_id"]
    FPS = CONFIG["fps"]
    LR = CONFIG["lr"]
    BATCH_SIZE = CONFIG["batch_size"]
    BUFFER_SIZE = CONFIG["buffer_size"]
    GAMMA = CONFIG["gamma"]
    TAU = CONFIG["tau"]
    ALPHA_INIT = CONFIG["alpha_init"]
    UTD_RATIO = CONFIG["utd_ratio"]
    WARMUP_STEPS = CONFIG["warmup_steps"]
    MAX_EPISODE_STEPS = CONFIG["max_episode_steps"]
    IMG_SIZE = CONFIG["img_size"]
    USE_CAMERA = CONFIG["use_camera"]
    USE_JOINTS = CONFIG["use_joints"]
    POS_REWARD_WEIGHT = CONFIG["pos_reward_weight"]
    TORQUE_PENALTY_WEIGHT = CONFIG["torque_penalty_weight"]
    USE_SIMPLE_TORQUE = CONFIG["use_simple_torque"]
    POLICY_DELAY = CONFIG["policy_delay"]
    USE_BN = CONFIG["use_bn"]
    USE_LN = CONFIG["use_ln"]
    USE_WN = CONFIG["use_wn"]
    USE_REWARD_NORM = CONFIG["use_reward_norm"]
    USE_WANDB = CONFIG["use_wandb"]
    USE_ANALYSIS = CONFIG["use_analysis"]
    SEED = CONFIG["seed"]
    USE_BF16 = CONFIG["use_bf16"]
    RUNS_DIR = Path(CONFIG["runs_dir"])

    requested_device = CONFIG["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"Using device: {device}")

    # 1. Setup Robot
    cameras = {}
    if USE_CAMERA:
        cameras = {
            "front": OpenCVCameraConfig(
                index_or_path=0,
                width=640,
                height=480,
                fps=30,
                rotation=Cv2Rotation.ROTATE_180
            )
        }
    
    robot_config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, cameras=cameras)
    robot = make_robot_from_config(robot_config)
    robot.connect()
    
    # Use canonical motor order from the bus to match config [pan, lift, flex, flex, roll, gripper]
    motor_names = list(robot.bus.motors.keys())
    motor_names_pos = [f"{n}.pos" for n in motor_names]
    action_dim = len(motor_names)
    
    # Starting position from rl_config.json: [0.0, 0.0, 0.0, 0.0, 0.0, 25.0]
    q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])
    print(f"Target home position: {dict(zip(motor_names, q_home))}")

    # Safety Layer
    safety = SafetyLayer(
        robot.bus, motor_names,
        max_delta_deg=CONFIG["safety_max_delta_deg"],
        current_threshold_mA=CONFIG["safety_threshold_mA"],
        current_limit_mA=CONFIG["safety_limit_mA"],
    )
    print(f"Safety: max_delta={CONFIG['safety_max_delta_deg']}° "
          f"threshold={CONFIG['safety_threshold_mA']}mA "
          f"limit={CONFIG['safety_limit_mA']}mA")

    # 2. Init Models
    encoder = TinyEncoder(state_dim=action_dim, use_camera=USE_CAMERA, use_joints=USE_JOINTS, use_bn=USE_BN, use_ln=USE_LN).to(device)
    actor = Actor(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)
    critic = Critic(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)
    critic_target = Critic(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    reward_rms = RunningMeanStd().to(device)

    # Automated entropy tuning
    target_entropy = -float(action_dim)
    log_alpha = torch.tensor(np.log(ALPHA_INIT), requires_grad=True, device=device)
    opt_alpha = optim.Adam([log_alpha], lr=LR)

    opt_encoder = optim.Adam(encoder.parameters(), lr=LR)
    opt_actor = optim.Adam(actor.parameters(), lr=LR)
    opt_critic = optim.Adam(critic.parameters(), lr=LR)

    replay_buffer = deque(maxlen=BUFFER_SIZE)
    logger = RunLogger(CONFIG)
    last_img_cache = None

    # 3. RL Loop
    print("Starting RL loop. Press Ctrl+C to stop.")
    total_steps = 0
    episode_num = 0
    try:
        while True:
            # Start of Episode
            episode_num += 1
            episode_steps = 0
            episode_reward = 0
            
            # Reset to home at start of each episode
            reset_robot(robot, q_home, motor_names_pos)
            
            while episode_steps < MAX_EPISODE_STEPS:
                t_start = time.perf_counter()
                
                # Observe
                obs_dict = robot.get_observation()
                img, q_torch = preprocess_obs(obs_dict, device, motor_names_pos, use_camera=USE_CAMERA)
                if img is not None:
                    last_img_cache = img
                
                encoder.eval(); actor.eval(); critic.eval()
                with torch.no_grad():
                    feat = encoder(img, q_torch)
                    #if total_steps < WARMUP_STEPS:
                    #    action_torch = torch.empty(1, action_dim).uniform_(-1, 1).to(device)
                    #else:
                    action_torch, _ = actor.sample(feat)
                
                # Act
                action_np = action_torch.cpu().numpy()[0]
                curr_q_vals = np.array([obs_dict[k] for k in motor_names_pos])
                target_q_vals = curr_q_vals + action_np * 5.0
                target_q_vals = np.clip(target_q_vals, q_home - 45, q_home + 45)
                
                # Safety filter: clamp step + attenuate by current
                q_safe, safety_info = safety(target_q_vals, q_home=q_home)
                robot.send_action({name: float(val) for name, val in zip(motor_names_pos, q_safe)})
                
                # Wait for next step and get next obs
                precise_sleep(max(1.0/FPS - (time.perf_counter() - t_start), 0))
                next_obs_dict = robot.get_observation()
                next_img, next_q_torch = preprocess_obs(next_obs_dict, device, motor_names_pos, use_camera=USE_CAMERA)
                
                # Train (same as before)
                loss_q_val, loss_a_val, mean_q_val = 0, 0, 0
                alpha_val = log_alpha.exp().item()
                
                # Torque from safety layer (already read, no extra bus call)
                currents = safety_info["currents"]
                torque_sq_sum = float(np.sum(currents ** 2)) / 1000.0
                safety_attenuation = float(safety_info["attenuation"].mean())

                # Reward: distance to home (punish moving away)
                current_q = np.array([next_obs_dict[k] for k in motor_names_pos])
                dist_sq = np.sum((current_q - q_home)**2)
                reward_pos = -dist_sq / 100.0 
                
                # Torque Penalty
                torque_penalty = 0.0
                if TORQUE_PENALTY_WEIGHT > 0:
                    if USE_SIMPLE_TORQUE:
                        # Simple squared sum (normalized by 1000, so 10k -> 10.0)
                        torque_penalty = torque_sq_sum
                    else:
                        # Sigmoid centered at 14k, scale from 0 to 100
                        torque_penalty = 100.0 / (1.0 + np.exp(-0.9 * (torque_sq_sum - 14.0)))
                
                reward = (POS_REWARD_WEIGHT * reward_pos) - (TORQUE_PENALTY_WEIGHT * torque_penalty)
                
                episode_steps += 1
                total_steps += 1
                episode_reward += reward
                
                done = (episode_steps >= MAX_EPISODE_STEPS)
                
                # Store
                img_to_store = img.cpu() if img is not None else None
                next_img_to_store = next_img.cpu() if next_img is not None else None
                replay_buffer.append((img_to_store, q_torch.cpu(), action_torch.cpu(), reward, next_img_to_store, next_q_torch.cpu(), done))
                
                if len(replay_buffer) > BATCH_SIZE and total_steps > WARMUP_STEPS:
                    encoder.train(); actor.train(); critic.train()
                    loss_a = torch.tensor(0.0, device=device) # Initialize for logging
                    for i in range(UTD_RATIO):
                        batch = random.sample(replay_buffer, BATCH_SIZE)
                        b_img, b_q, b_a, b_r, b_next_img, b_next_q, b_d = zip(*batch)
                        
                        b_img = torch.cat(b_img).to(device) if USE_CAMERA else None
                        b_q = torch.cat(b_q).to(device)
                        b_a = torch.cat(b_a).to(device)
                        b_r = torch.tensor(b_r, dtype=torch.float32, device=device).unsqueeze(1)
                        b_next_img = torch.cat(b_next_img).to(device) if USE_CAMERA else None
                        b_next_q = torch.cat(b_next_q).to(device)
                        b_d = torch.tensor(b_d, dtype=torch.float32, device=device).unsqueeze(1)
                        
                        curr_alpha = log_alpha.exp()

                        # Normalize rewards with running statistics
                        if USE_REWARD_NORM:
                            reward_rms.update(b_r)
                            b_r_norm = reward_rms.normalize(b_r)
                        else:
                            b_r_norm = b_r

                        # Use mixed precision for training speedup
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BF16 and device.type == 'cuda'):
                            if USE_BN:
                                # Joined encoder pass so BN sees both s and s'
                                j_img = torch.cat([b_img, b_next_img]) if USE_CAMERA else None
                                j_q = torch.cat([b_q, b_next_q])
                                j_feat = encoder(j_img, j_q)
                                curr_feat, next_feat = j_feat.chunk(2)

                                with torch.no_grad():
                                    next_a, next_lp = actor.sample(next_feat)
                                    q1_t, q2_t = critic_target(next_feat, next_a)
                                    target_q = b_r_norm + (1 - b_d) * GAMMA * (torch.min(q1_t, q2_t) - curr_alpha * next_lp)

                                # Joined critic pass for consistent BN stats
                                j_a = torch.cat([b_a, next_a])
                                j_q1, j_q2 = critic(torch.cat([curr_feat, next_feat.detach()]), j_a)
                                q1, _ = j_q1.chunk(2)
                                q2, _ = j_q2.chunk(2)
                            else:
                                with torch.no_grad():
                                    next_feat_for_a = encoder(b_next_img, b_next_q)
                                    next_a, next_lp = actor.sample(next_feat_for_a)
                                    q1_t, q2_t = critic_target(next_feat_for_a, next_a)
                                    target_q = b_r_norm + (1 - b_d) * GAMMA * (torch.min(q1_t, q2_t) - curr_alpha * next_lp)

                                curr_feat = encoder(b_img, b_q)
                                q1, q2 = critic(curr_feat, b_a)

                            loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                        
                        opt_critic.zero_grad()
                        opt_encoder.zero_grad()
                        loss_q.backward()
                        opt_critic.step()
                        opt_encoder.step()
                        
                        if USE_WN:
                            weight_normalize_(encoder)
                            weight_normalize_(critic)

                        # Delayed Policy Updates (TD3-style)
                        if (total_steps * UTD_RATIO + i) % POLICY_DELAY == 0:
                            # Set critic to eval mode for actor update to use consistent BN stats 
                            # from the joined pass without updating them again with a different batch size.
                            critic.eval()
                            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BF16 and device.type == 'cuda'):
                                # Update Actor - Use curr_feat from joined pass to maintain BN consistency
                                new_a, log_p = actor.sample(curr_feat.detach())
                                q1_new, q2_new = critic(curr_feat.detach(), new_a)
                                loss_a = (curr_alpha * log_p - torch.min(q1_new, q2_new)).mean()
                            
                            opt_actor.zero_grad()
                            loss_a.backward()
                            opt_actor.step()
                            critic.train() # Set back to train for next UTD iteration
                            
                            if USE_WN:
                                weight_normalize_(actor)

                            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_BF16 and device.type == 'cuda'):
                                # Update Alpha
                                loss_alpha = -(log_alpha * (log_p + target_entropy).detach()).mean()
                            
                            opt_alpha.zero_grad()
                            loss_alpha.backward()
                            opt_alpha.step()
                            
                            # Soft Update Target (parameters + BN buffers)
                            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                            if USE_BN:
                                for buf, target_buf in zip(critic.buffers(), critic_target.buffers()):
                                    target_buf.data.copy_(TAU * buf.data + (1 - TAU) * target_buf.data)

                    loss_q_val = loss_q.item()
                    loss_a_val = loss_a.item()
                    mean_q_val = torch.min(q1, q2).mean().item()
                    alpha_val = log_alpha.exp().item()

                dt = time.perf_counter() - t_start
                actual_hz = 1.0 / dt if dt > 0 else 0
                
                # Log metrics
                step_metrics = {
                    "total_steps": total_steps,
                    "episode_num": episode_num,
                    "reward": float(reward),
                    "reward_pos": float(reward_pos),
                    "q_val": float(mean_q_val),
                    "hz": float(actual_hz),
                    "alpha": float(alpha_val),
                    "safety_atten": float(safety_attenuation),
                }
                if TORQUE_PENALTY_WEIGHT > 0:
                    step_metrics["torque_penalty"] = float(torque_penalty)
                    step_metrics["torque"] = float(torque_sq_sum)
                logger.log(step_metrics)

                if total_steps % 10 == 0:
                    print(f"Ep: {episode_num:3d} | Step: {episode_steps:3d}/{MAX_EPISODE_STEPS} | "
                          f"Rew: {reward:6.2f} (P:{POS_REWARD_WEIGHT*reward_pos:5.1f} T:-{TORQUE_PENALTY_WEIGHT*torque_penalty:4.1f}) | "
                          f"Q: {mean_q_val:6.2f} | T: {torque_sq_sum:4.1f}k | Hz: {actual_hz:4.1f} | A: {alpha_val:.3f} | S: {safety_attenuation:.0%}")
                
                if done:
                    avg_rew = episode_reward / episode_steps
                    print(f"--- Episode {episode_num} finished! Steps: {total_steps} | Avg Rew: {avg_rew:.3f} ---")
                    break

    except KeyboardInterrupt:
        print("\nStopping RL loop.")
    finally:
        logger.plot()
        if USE_ANALYSIS:
            run_post_analysis(encoder, actor, critic, q_home, motor_names_pos, device, last_img_cache, logger)
            run_umap_analysis(encoder, actor, replay_buffer, q_home, motor_names_pos, device, logger)
        logger.finish()
        if USE_CAMERA:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        robot.disconnect()

if __name__ == "__main__":
    main()
