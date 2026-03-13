#!/usr/bin/env python3
"""
Minimal Single-Process RL script for SO-101 or MuJoCo environments.

Usage:
    python robot/minimal_rl.py                          # Real robot
    python robot/minimal_rl.py --env HalfCheetah-v4     # MuJoCo sim
"""

import time
import os
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
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
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

# --- Configuration ---
CONFIG = {
    "port": "/dev/ttyACM0",
    "robot_id": "shabby",
    "fps": 10,
    "lr": 3e-4,
    "batch_size": 256,
    "buffer_size": 10000,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha_init": None, # Set conditionally based on reward_norm
    "entropy_scale": None, # Set conditionally based on reward_norm
    "utd_ratio": 2,
    "warmup_steps": 50,
    "max_episode_steps": 200,
    "img_size": (64, 64),
    "use_camera": False,
    "use_joints": True,
    "seed": 42,
    "use_bf16": True,
    "pos_reward_weight": 1.0,
    "torque_penalty_weight": 1.0, # 0.0 means off
    "use_simple_torque": True, # True: sum(I^2), False: sigmoid(sum(I^2))
    "policy_delay": 3, # Update policy every N steps
    "use_bn": True, # Use Batch Norm (XQC style)
    "use_wn": True, # Use Weight Norm projection (XQC style)
    "use_reward_norm": True, # Use reward normalization
    "use_obs_norm": False, # Use observation normalization
    "use_c51": True, # Use C51 categorical critic
    "device": "cuda", # "cuda" or "cpu"
    "runs_dir": "robot/runs",
    "env_name": None, # Gymnasium env name
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
        self.use_wandb = HAS_WANDB
        if self.use_wandb:
            try:
                env_name = config.get("env_name")
                project_name = f"minimal-rl-{env_name}" if env_name else "minimal-rl-robot"
                wandb.init(project=project_name, name=run_name, config=config,
                           dir=str(self.run_dir))
                print(f"W&B run: {wandb.run.url}")
            except Exception as e:
                print(f"W&B init failed: {e}")
                self.use_wandb = False
            
        self.data = []
        print(f"Logging to {self.run_dir}")

    def log(self, metrics):
        # Create a copy for json logging that excludes wandb objects
        json_metrics = {k: v for k, v in metrics.items() if not (HAS_WANDB and isinstance(v, wandb.Video))}
        
        self.data.append(json_metrics)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(json_metrics) + "\n")
            
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

    def _plot_umap_panels(self, embedding, states, std, timestamps, title_prefix, filename, sizes=None):
        """Shared 3-panel UMAP scatter plot."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(title_prefix, fontsize=14, y=1.02)
        s = sizes if sizes is not None else 8

        sc0 = axes[0].scatter(embedding[:, 0], embedding[:, 1], c=timestamps, cmap="viridis", s=s, alpha=0.7)
        axes[0].set_title("Colored by timestep")
        plt.colorbar(sc0, ax=axes[0], label="Avg step" if sizes is not None else "Step")

        entropy = np.sum(np.log(std + 1e-8), axis=1)
        sc1 = axes[1].scatter(embedding[:, 0], embedding[:, 1], c=entropy, cmap="coolwarm", s=s, alpha=0.7)
        axes[1].set_title("Colored by policy entropy")
        plt.colorbar(sc1, ax=axes[1], label="Σ log(σ)")

        dist_home = np.linalg.norm(states, axis=1)
        sc2 = axes[2].scatter(embedding[:, 0], embedding[:, 1], c=dist_home, cmap="magma", s=s, alpha=0.7)
        axes[2].set_title("Colored by dist-to-home")
        plt.colorbar(sc2, ax=axes[2], label="||q - q_home||")

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

    def plot_umap_knn_density(self, states, mu, std, timestamps, episode_ends,
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

        self._plot_umap_panels(embedding, states, std, timestamps,
                               f"UMAP k-NN + analytical density (k={k_actual})", filename)

    def plot_umap_full_density(self, states, mu, std, timestamps,
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

        self._plot_umap_panels(embedding, states, std, timestamps,
                               "UMAP full analytical density", filename)

    def plot_umap_full_kl(self, states, mu, std, timestamps,
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

        self._plot_umap_panels(embedding, states, std, timestamps,
                               "UMAP full pairwise KL", filename)

class RunningMeanStd(nn.Module):
    """Welford online algorithm for tracking running mean and variance."""
    def __init__(self, epsilon: float = 1e-4, shape=()):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(dim=0) if x.dim() > 1 else x.mean()
        batch_var = x.var(dim=0, unbiased=False) if x.dim() > 1 and x.shape[0] > 1 else torch.zeros_like(batch_mean)
        batch_count = x.shape[0] if x.dim() > 1 else x.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count

        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total_count)
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

def apply_ortho_init(m):
    """Apply orthogonal initialization to linear layers."""
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_mlp(in_dim, out_dim, hidden_dim=256, use_bn=False, use_ln=False):
    """Utility to build MLP with optional BatchNorm or LayerNorm."""
    layers = []
    if use_bn:
        layers.append(nn.BatchNorm1d(in_dim))
    elif use_ln:
        layers.append(nn.LayerNorm(in_dim))
    
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

        norm_layer = nn.BatchNorm1d(256) if use_bn else (nn.LayerNorm(256) if use_ln else nn.Identity())
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            norm_layer,
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


class CategoricalCritic(nn.Module):
    """C51-style distributional critic with two Q-networks."""
    def __init__(self, input_dim, action_dim, n_atoms=101, v_min=-5.0, v_max=5.0,
                 use_bn=False, use_ln=False):
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.q1_net = make_mlp(input_dim + action_dim, 256, use_bn=use_bn, use_ln=use_ln)
        self.q1_head = nn.Linear(256, n_atoms)

        self.q2_net = make_mlp(input_dim + action_dim, 256, use_bn=use_bn, use_ln=use_ln)
        self.q2_head = nn.Linear(256, n_atoms)

    def forward_logits(self, x, a):
        """Return raw logits for both critics."""
        xa = torch.cat([x, a], dim=-1)
        return self.q1_head(self.q1_net(xa)), self.q2_head(self.q2_net(xa))

    def forward(self, x, a):
        """Return scalar Q-values (expected value under the learned distribution)."""
        logits1, logits2 = self.forward_logits(x, a)
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        q1 = (probs1 * self.support).sum(dim=-1, keepdim=True)
        q2 = (probs2 * self.support).sum(dim=-1, keepdim=True)
        return q1, q2

    def compute_target_distribution(self, rewards, dones, gamma, target_q_probs, support):
        """Project the Bellman update onto the categorical support."""
        batch_size = rewards.shape[0]
        # Tz = r + gamma * (1 - done) * z_j, clipped to [v_min, v_max]
        tz = rewards + (1.0 - dones) * gamma * support.unsqueeze(0)  # (B, n_atoms)
        tz = tz.clamp(self.v_min, self.v_max)

        # Compute the projection indices
        b = (tz - self.v_min) / self.delta_z  # (B, n_atoms), float indices into support
        l = b.floor().long().clamp(0, self.n_atoms - 1)
        u = b.ceil().long().clamp(0, self.n_atoms - 1)

        # Distribute probability mass
        m = torch.zeros(batch_size, self.n_atoms, device=rewards.device)
        offset = torch.arange(batch_size, device=rewards.device).unsqueeze(1) * self.n_atoms

        m.view(-1).index_add_(0, (l + offset).view(-1), (target_q_probs * (u.float() - b)).view(-1))
        m.view(-1).index_add_(0, (u + offset).view(-1), (target_q_probs * (b - l.float())).view(-1))

        return m

# --- Utilities ---

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
    for idx, (img, q, a, r, ni, nq, done) in enumerate(replay_buffer):
        all_q.append(q)
        all_q_raw.append(q.numpy().flatten() * 100.0)
        if img is not None:
            all_img.append(img)
        if done:
            episode_ends.append(idx)

    timestamps = np.arange(len(all_q))
    all_q_raw = np.array(all_q_raw) - q_home
    episode_ends = np.array(episode_ends)

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
    logger.plot_umap_knn_density(all_q_raw, all_mu, all_std, timestamps, episode_ends)

    # 2. Full pairwise analytical density (slow, for comparison)
    print("UMAP 2/4: Full pairwise analytical density (slow)...")
    logger.plot_umap_full_density(all_q_raw, all_mu, all_std, timestamps)

    # 3. Full pairwise symmetrized KL (slow, behavioral distance)
    print("UMAP 3/4: Full pairwise KL divergence (slow)...")
    logger.plot_umap_full_kl(all_q_raw, all_mu, all_std, timestamps)

    # 4. k-NN + analytical density with larger k (comparison)
    print("UMAP 4/4: k-NN + analytical density k=50 (fast)...")
    logger.plot_umap_knn_density(all_q_raw, all_mu, all_std, timestamps, episode_ends,
                                 k=50, filename="umap_knn_density_k50.png")

class UnifiedEnv:
    """Wraps either a real robot or a gymnasium env behind a common interface."""

    def __init__(self, env_name=None, robot=None, motor_names_pos=None,
                 q_home=None, device="cuda", img_size=(64, 64),
                 use_camera=False, fps=10, seed=42, render_mode=None):
        self.is_sim = env_name is not None
        self.device = device
        self.img_size = img_size
        self.use_camera = use_camera
        self.fps = fps
        self.seed = seed

        if self.is_sim:
            if not HAS_GYM:
                raise ImportError("gymnasium is required for sim mode: pip install gymnasium[mujoco]")
            self.env = gym.make(env_name, render_mode=render_mode)
            self._last_obs = None
        else:
            self.robot = robot
            self.motor_names_pos = motor_names_pos
            self.q_home = q_home

        # Set dimensions
        if self.is_sim:
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
        else:
            self.state_dim = len(motor_names_pos)
            self.action_dim = len(motor_names_pos)

    def reset(self):
        if self.is_sim:
            obs, _ = self.env.reset(seed=self.seed)
            self._last_obs = obs
            return None, torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        steps = 50
        curr_obs = self.robot.get_observation()
        q_curr = np.array([curr_obs[k] for k in self.motor_names_pos])
        for i in range(steps):
            interp = q_curr + (self.q_home - q_curr) * (i + 1) / steps
            self.robot.send_action({n: float(v) for n, v in zip(self.motor_names_pos, interp)})
            time.sleep(0.05)
        obs_dict = self.robot.get_observation()
        return self._robot_obs(obs_dict)

    def step(self, action_np):
        if self.is_sim:
            obs, reward, terminated, truncated, info = self.env.step(action_np)
            self._last_obs = obs
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return (None, state), float(reward), terminated or truncated, info

        t0 = time.perf_counter()
        obs_dict = self.robot.get_observation()
        curr_q = np.array([obs_dict[k] for k in self.motor_names_pos])
        target_q = np.clip(curr_q + action_np * 5.0, self.q_home - 45, self.q_home + 45)
        self.robot.send_action({n: float(v) for n, v in zip(self.motor_names_pos, target_q)})
        from lerobot.utils.robot_utils import precise_sleep
        precise_sleep(max(1.0 / self.fps - (time.perf_counter() - t0), 0))

        next_dict = self.robot.get_observation()
        next_q = np.array([next_dict[k] for k in self.motor_names_pos])

        try:
            cur = self.robot.bus.sync_read("Present_Current")
            currents = [cur.get(n.removesuffix(".pos"), 0) for n in self.motor_names_pos]
            torque = sum(c ** 2 for c in currents) / 1000.0
        except Exception:
            torque = 0.0

        dist_sq = float(np.sum((next_q - self.q_home) ** 2))
        reward_pos = -dist_sq / 100.0
        tp = torque if CONFIG["use_simple_torque"] else 100.0 / (1.0 + np.exp(-0.9 * (torque - 14.0)))
        if CONFIG["torque_penalty_weight"] <= 0:
            tp = 0.0
        reward = CONFIG["pos_reward_weight"] * reward_pos - CONFIG["torque_penalty_weight"] * tp
        info = {"reward_pos": reward_pos, "torque_penalty": tp, "torque": torque}
        return self._robot_obs(next_dict), float(reward), False, info

    def close(self):
        if self.is_sim:
            self.env.close()
        else:
            self.robot.disconnect()

    def render(self):
        if self.is_sim:
            return self.env.render()
        return None

    def _robot_obs(self, obs_dict):
        joints = torch.tensor(
            [obs_dict[k] for k in self.motor_names_pos],
            dtype=torch.float32, device=self.device,
        ).unsqueeze(0) / 100.0
        img = None
        if self.use_camera:
            cam_key = next((k for k in obs_dict if "observation.images." in k), None)
            if cam_key is None:
                img = torch.zeros(1, 3, *self.img_size, device=self.device)
            else:
                raw = obs_dict[cam_key]
                resized = cv2.resize(raw, self.img_size)
                img = torch.tensor(resized, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255.0
        return img, joints


@torch.no_grad()
def evaluate_policy(env, encoder, actor, device, n_episodes=10, max_steps=1000,
                    use_camera=False, use_joints=True, video_path=None, obs_rms=None):
    """Run n deterministic episodes (mean action) and return avg episodic return."""
    encoder.eval(); actor.eval()
    returns = []
    for ep in range(n_episodes):
        img, state = env.reset()
        if obs_rms is not None and state is not None:
            state = torch.clamp(obs_rms.normalize(state), -10.0, 10.0)
            
        ep_ret = 0.0
        frames = []
        for _ in range(max_steps):
            if ep == 0 and video_path is not None and env.is_sim:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
                    
            feat = encoder(img, state if use_joints else None)
            mu, _ = actor(feat)
            action = torch.tanh(mu)
            action_np = action.cpu().numpy()[0]
            (img, state), reward, done, _ = env.step(action_np)
            if obs_rms is not None and state is not None:
                state = torch.clamp(obs_rms.normalize(state), -10.0, 10.0)
                
            ep_ret += reward
            if done:
                break
        returns.append(ep_ret)
        
        if ep == 0 and video_path is not None and frames:
            try:
                import imageio
                # Force macro_block_size=None to avoid dimension issues with some codecs
                imageio.mimsave(video_path, frames, fps=30, macro_block_size=None)
            except Exception as e:
                print(f"Warning: imageio failed to save video ({e}). Falling back to cv2 mp4.")
                # Fallback to mp4 if imageio is missing or fails
                if video_path.endswith('.webm'):
                    video_path = video_path[:-5] + '.mp4'
                h, w, c = frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                for f in frames:
                    out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                out.release()
                
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, help="Gymnasium env name (e.g. HalfCheetah-v4). Omit for real robot.")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Evaluate every N steps (0 to disable)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes per evaluation")
    parser.add_argument("--batch-size", type=int, default=CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--lr", type=float, default=CONFIG["lr"], help="Learning rate")
    parser.add_argument("--c51", action=argparse.BooleanOptionalAction, default=CONFIG["use_c51"], help="Use C51 categorical critic (XQC style)")
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
    parser.add_argument("--bn", action=argparse.BooleanOptionalAction, default=CONFIG["use_bn"], help="Use Batch Norm (XQC style)")
    parser.add_argument("--ln", action="store_true", default=False, help="Use Layer Norm")
    parser.add_argument("--ortho-init", action="store_true", default=False, help="Use Orthogonal Initialization")
    parser.add_argument("--wn", action=argparse.BooleanOptionalAction, default=CONFIG["use_wn"], help="Use Weight Norm projection (XQC style)")
    parser.add_argument("--reward-norm", action=argparse.BooleanOptionalAction, default=CONFIG["use_reward_norm"], help="Enable reward normalization")
    parser.add_argument("--obs-norm", action=argparse.BooleanOptionalAction, default=CONFIG["use_obs_norm"], help="Enable observation normalization")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Global gradient clipping norm (0.0 to disable)")
    parser.add_argument("--alpha-init", type=float, default=None, help="Initial temperature for entropy (default depends on reward-norm)")
    parser.add_argument("--entropy-scale", type=float, default=None, help="Scale for target entropy (default depends on reward-norm)")
    args = parser.parse_args()

    # Override config with args
    CONFIG["lr"] = args.lr
    CONFIG["batch_size"] = args.batch_size
    CONFIG["use_c51"] = args.c51
    CONFIG["device"] = args.device
    CONFIG["port"] = args.port
    CONFIG["utd_ratio"] = args.utd
    CONFIG["use_camera"] = args.camera
    CONFIG["seed"] = args.seed
    CONFIG["pos_reward_weight"] = args.pos_weight
    CONFIG["torque_penalty_weight"] = args.torque_weight
    CONFIG["policy_delay"] = args.policy_delay
    CONFIG["use_bn"] = args.bn
    CONFIG["use_ln"] = args.ln
    CONFIG["ortho_init"] = args.ortho_init
    CONFIG["use_wn"] = args.wn
    CONFIG["use_reward_norm"] = args.reward_norm
    CONFIG["use_obs_norm"] = args.obs_norm
    CONFIG["grad_clip"] = args.grad_clip
    CONFIG["env_name"] = args.env

    # Set conditional defaults for entropy based on whether reward_norm is active
    if CONFIG["use_reward_norm"]:
        default_alpha_init = 0.05
        default_entropy_scale = 0.5
    else:
        default_alpha_init = 0.2
        default_entropy_scale = 1.0

    CONFIG["alpha_init"] = args.alpha_init if args.alpha_init is not None else default_alpha_init
    CONFIG["entropy_scale"] = args.entropy_scale if args.entropy_scale is not None else default_entropy_scale
    if args.no_joints:
        CONFIG["use_joints"] = False
    if args.no_bf16:
        CONFIG["use_bf16"] = False
    if args.use_sigmoid_torque:
        CONFIG["use_simple_torque"] = False

    # 0. Set Seed
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Mapping for easier access in code
    global PORT, ROBOT_ID, FPS, LR, BATCH_SIZE, BUFFER_SIZE, GAMMA, TAU, ALPHA_INIT, ENTROPY_SCALE, UTD_RATIO, WARMUP_STEPS, MAX_EPISODE_STEPS, IMG_SIZE, USE_CAMERA, USE_JOINTS, POS_REWARD_WEIGHT, TORQUE_PENALTY_WEIGHT, USE_SIMPLE_TORQUE, POLICY_DELAY, USE_BN, USE_LN, ORTHO_INIT, USE_WN, USE_REWARD_NORM, USE_OBS_NORM, GRAD_CLIP, SEED, USE_BF16, RUNS_DIR, USE_C51
    PORT = CONFIG["port"]
    ROBOT_ID = CONFIG["robot_id"]
    FPS = CONFIG["fps"]
    LR = CONFIG["lr"]
    BATCH_SIZE = CONFIG["batch_size"]
    BUFFER_SIZE = CONFIG["buffer_size"]
    GAMMA = CONFIG["gamma"]
    TAU = CONFIG["tau"]
    ALPHA_INIT = CONFIG["alpha_init"]
    ENTROPY_SCALE = CONFIG.get("entropy_scale", 1.0)
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
    USE_LN = CONFIG.get("use_ln", False)
    ORTHO_INIT = CONFIG.get("ortho_init", False)
    USE_WN = CONFIG["use_wn"]
    USE_REWARD_NORM = CONFIG["use_reward_norm"]
    USE_OBS_NORM = CONFIG["use_obs_norm"]
    GRAD_CLIP = CONFIG["grad_clip"]
    SEED = CONFIG["seed"]
    USE_BF16 = CONFIG["use_bf16"]
    USE_C51 = CONFIG.get("use_c51", False)
    RUNS_DIR = Path(CONFIG["runs_dir"])

    requested_device = CONFIG["device"]
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    print(f"Using device: {device}")

    # 1. Setup environment
    is_sim = args.env is not None
    eval_freq = args.eval_freq
    eval_episodes = args.eval_episodes

    if is_sim:
        env = UnifiedEnv(env_name=args.env, device=device, seed=args.seed)
        eval_env = UnifiedEnv(env_name=args.env, device=device, seed=args.seed + 100, render_mode="rgb_array")
        env.env.action_space.seed(args.seed)
        env.env.observation_space.seed(args.seed)
        eval_env.env.action_space.seed(args.seed + 100)
        eval_env.env.observation_space.seed(args.seed + 100)
        action_dim = env.action_dim
        state_dim = env.state_dim
        print(f"Sim env: {args.env}  |  state_dim={state_dim}  action_dim={action_dim}")
    else:
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        from lerobot.robots import make_robot_from_config
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.cameras.configs import Cv2Rotation

        cameras = {}
        if USE_CAMERA:
            cameras = {
                "front": OpenCVCameraConfig(
                    index_or_path=0, width=640, height=480, fps=30,
                    rotation=Cv2Rotation.ROTATE_180,
                )
            }
        robot_config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, cameras=cameras)
        robot = make_robot_from_config(robot_config)
        robot.connect()
        motor_names = list(robot.bus.motors.keys())
        motor_names_pos = [f"{n}.pos" for n in motor_names]
        q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])
        action_dim = len(motor_names)
        state_dim = action_dim
        env = UnifiedEnv(robot=robot, motor_names_pos=motor_names_pos,
                         q_home=q_home, device=device, img_size=IMG_SIZE,
                         use_camera=USE_CAMERA, fps=FPS)
        eval_env = env
        print(f"Robot: {ROBOT_ID}  |  motors={motor_names}  home={q_home}")

    # 2. Init Models
    encoder = TinyEncoder(state_dim=state_dim, use_camera=USE_CAMERA, use_joints=USE_JOINTS, use_bn=USE_BN, use_ln=USE_LN).to(device)
    actor = Actor(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)

    if USE_C51:
        c51_v_min, c51_v_max, c51_n_atoms = -5.0, 5.0, 101
        critic = CategoricalCritic(encoder.out_dim, action_dim,
                                   n_atoms=c51_n_atoms, v_min=c51_v_min, v_max=c51_v_max,
                                   use_bn=USE_BN, use_ln=USE_LN).to(device)
        print(f"C51 critic: {c51_n_atoms} atoms, support [{c51_v_min}, {c51_v_max}]")
    else:
        critic = Critic(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)

    if ORTHO_INIT:
        encoder.apply(apply_ortho_init)
        actor.apply(apply_ortho_init)
        critic.apply(apply_ortho_init)
        for m in [actor.mu, actor.log_std, critic.q1_head, critic.q2_head]:
            nn.init.orthogonal_(m.weight, gain=1.0)

    if USE_C51:
        critic_target = CategoricalCritic(encoder.out_dim, action_dim,
                                          n_atoms=c51_n_atoms, v_min=c51_v_min, v_max=c51_v_max,
                                          use_bn=USE_BN, use_ln=USE_LN).to(device)
    else:
        critic_target = Critic(encoder.out_dim, action_dim, use_bn=USE_BN, use_ln=USE_LN).to(device)
    critic_target.load_state_dict(critic.state_dict())
    critic_target.eval()

    reward_rms = RunningMeanStd().to(device)
    obs_rms = RunningMeanStd(shape=(state_dim,)).to(device) if USE_OBS_NORM else None

    target_entropy = -float(action_dim) * ENTROPY_SCALE
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
            episode_num += 1
            episode_steps = 0
            episode_reward = 0.0
            running_ret = 0.0

            img, q_torch = env.reset()
            if USE_OBS_NORM and q_torch is not None:
                obs_rms.update(q_torch)
                q_torch = torch.clamp(obs_rms.normalize(q_torch), -10.0, 10.0)
            if img is not None:
                last_img_cache = img

            while episode_steps < MAX_EPISODE_STEPS:
                t_start = time.perf_counter()

                encoder.eval(); actor.eval(); critic.eval()
                with torch.no_grad():
                    feat = encoder(img, q_torch if USE_JOINTS else None)
                    if total_steps < WARMUP_STEPS:
                        action_torch = torch.empty(1, action_dim, device=device).uniform_(-1, 1)
                    else:
                        action_torch, _ = actor.sample(feat)

                action_np = action_torch.cpu().numpy()[0]
                (next_img, next_q_torch), reward, done, info = env.step(action_np)
                if USE_OBS_NORM and next_q_torch is not None:
                    obs_rms.update(next_q_torch)
                    next_q_torch = torch.clamp(obs_rms.normalize(next_q_torch), -10.0, 10.0)
                if next_img is not None:
                    last_img_cache = next_img

                episode_steps += 1
                total_steps += 1
                episode_reward += reward

                # Scale reward *before* inserting into replay buffer
                scaled_reward = reward
                if USE_REWARD_NORM:
                    running_ret = reward + GAMMA * running_ret
                    reward_rms.update(torch.tensor([running_ret], dtype=torch.float32, device=device))
                    scaled_reward = reward / (reward_rms.std.item() + 1e-8)
                    scaled_reward = float(np.clip(scaled_reward, -10.0, 10.0))

                if episode_steps >= MAX_EPISODE_STEPS:
                    done = True

                img_s = img.cpu() if img is not None else None
                nimg_s = next_img.cpu() if next_img is not None else None
                replay_buffer.append((img_s, q_torch.cpu(), action_torch.cpu(),
                                      scaled_reward, nimg_s, next_q_torch.cpu(), done))

                # --- Training ---
                loss_q_val, loss_a_val, mean_q_val = 0.0, 0.0, 0.0
                alpha_val = log_alpha.exp().item()

                if len(replay_buffer) > BATCH_SIZE and total_steps > WARMUP_STEPS:
                    encoder.train(); actor.train(); critic.train()
                    loss_a = torch.tensor(0.0, device=device)
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
                        
                        b_r_norm = b_r # Reward is already normalized before adding to buffer if USE_REWARD_NORM is True

                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=USE_BF16 and device.type == 'cuda'):
                            if USE_BN:
                                j_img = torch.cat([b_img, b_next_img]) if USE_CAMERA else None
                                j_q = torch.cat([b_q, b_next_q])
                                j_feat = encoder(j_img, j_q)
                                curr_feat, next_feat = j_feat.chunk(2)

                                with torch.no_grad():
                                    next_a, next_lp = actor.sample(next_feat)

                                if USE_C51:
                                    with torch.no_grad():
                                        tgt_logits1, tgt_logits2 = critic_target.forward_logits(next_feat, next_a)
                                        tgt_probs1 = F.softmax(tgt_logits1, dim=-1)
                                        tgt_probs2 = F.softmax(tgt_logits2, dim=-1)
                                        # Use the distribution from the min-Q network
                                        q1_t = (tgt_probs1 * critic_target.support).sum(dim=-1, keepdim=True)
                                        q2_t = (tgt_probs2 * critic_target.support).sum(dim=-1, keepdim=True)
                                        min_mask = (q1_t - curr_alpha * next_lp <= q2_t - curr_alpha * next_lp).float()
                                        tgt_probs = min_mask * tgt_probs1 + (1 - min_mask) * tgt_probs2
                                        target_dist = critic.compute_target_distribution(
                                            b_r_norm, b_d, GAMMA, tgt_probs, critic.support)

                                    j_a = torch.cat([b_a, next_a])
                                    j_logits1, j_logits2 = critic.forward_logits(
                                        torch.cat([curr_feat, next_feat.detach()]), j_a)
                                    logits1, _ = j_logits1.chunk(2)
                                    logits2, _ = j_logits2.chunk(2)
                                    loss_q = -(target_dist * F.log_softmax(logits1, dim=-1)).sum(-1).mean() \
                                           + -(target_dist * F.log_softmax(logits2, dim=-1)).sum(-1).mean()
                                    # For logging: compute scalar Q from current critic
                                    with torch.no_grad():
                                        q1 = (F.softmax(logits1, dim=-1) * critic.support).sum(-1, keepdim=True)
                                        q2 = (F.softmax(logits2, dim=-1) * critic.support).sum(-1, keepdim=True)
                                else:
                                    with torch.no_grad():
                                        q1_t, q2_t = critic_target(next_feat, next_a)
                                        target_q = b_r_norm + (1 - b_d) * GAMMA * (torch.min(q1_t, q2_t) - curr_alpha * next_lp)

                                    j_a = torch.cat([b_a, next_a])
                                    j_q1, j_q2 = critic(torch.cat([curr_feat, next_feat.detach()]), j_a)
                                    q1, _ = j_q1.chunk(2)
                                    q2, _ = j_q2.chunk(2)
                                    loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
                            else:
                                with torch.no_grad():
                                    next_feat_for_a = encoder(b_next_img, b_next_q)
                                    next_a, next_lp = actor.sample(next_feat_for_a)

                                if USE_C51:
                                    with torch.no_grad():
                                        tgt_logits1, tgt_logits2 = critic_target.forward_logits(next_feat_for_a, next_a)
                                        tgt_probs1 = F.softmax(tgt_logits1, dim=-1)
                                        tgt_probs2 = F.softmax(tgt_logits2, dim=-1)
                                        q1_t = (tgt_probs1 * critic_target.support).sum(dim=-1, keepdim=True)
                                        q2_t = (tgt_probs2 * critic_target.support).sum(dim=-1, keepdim=True)
                                        min_mask = (q1_t - curr_alpha * next_lp <= q2_t - curr_alpha * next_lp).float()
                                        tgt_probs = min_mask * tgt_probs1 + (1 - min_mask) * tgt_probs2
                                        target_dist = critic.compute_target_distribution(
                                            b_r_norm, b_d, GAMMA, tgt_probs, critic.support)

                                    curr_feat = encoder(b_img, b_q)
                                    logits1, logits2 = critic.forward_logits(curr_feat, b_a)
                                    loss_q = -(target_dist * F.log_softmax(logits1, dim=-1)).sum(-1).mean() \
                                           + -(target_dist * F.log_softmax(logits2, dim=-1)).sum(-1).mean()
                                    with torch.no_grad():
                                        q1 = (F.softmax(logits1, dim=-1) * critic.support).sum(-1, keepdim=True)
                                        q2 = (F.softmax(logits2, dim=-1) * critic.support).sum(-1, keepdim=True)
                                else:
                                    with torch.no_grad():
                                        q1_t, q2_t = critic_target(next_feat_for_a, next_a)
                                        target_q = b_r_norm + (1 - b_d) * GAMMA * (torch.min(q1_t, q2_t) - curr_alpha * next_lp)

                                    curr_feat = encoder(b_img, b_q)
                                    q1, q2 = critic(curr_feat, b_a)
                                    loss_q = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

                        opt_critic.zero_grad()
                        opt_encoder.zero_grad()
                        loss_q.backward()
                        if GRAD_CLIP > 0.0:
                            nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                            nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
                        opt_critic.step()
                        opt_encoder.step()

                        if USE_WN:
                            weight_normalize_(encoder)
                            weight_normalize_(critic)

                        if (total_steps * UTD_RATIO + i) % POLICY_DELAY == 0:
                            critic.eval()
                            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                    enabled=USE_BF16 and device.type == 'cuda'):
                                new_a, log_p = actor.sample(curr_feat.detach())
                                q1_new, q2_new = critic(curr_feat.detach(), new_a)
                                loss_a = (curr_alpha * log_p - torch.min(q1_new, q2_new)).mean()

                            opt_actor.zero_grad()
                            loss_a.backward()
                            if GRAD_CLIP > 0.0:
                                nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                            opt_actor.step()
                            critic.train()

                            if USE_WN:
                                weight_normalize_(actor)

                            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                    enabled=USE_BF16 and device.type == 'cuda'):
                                loss_alpha = -(log_alpha * (log_p + target_entropy).detach()).mean()

                            opt_alpha.zero_grad()
                            loss_alpha.backward()
                            opt_alpha.step()

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

                # --- Log step metrics ---
                step_metrics = {
                    "total_steps": total_steps,
                    "episode_num": episode_num,
                    "reward": float(reward),
                    "q_val": float(mean_q_val),
                    "loss_q": float(loss_q_val),
                    "loss_a": float(loss_a_val),
                    "hz": float(actual_hz),
                    "alpha": float(alpha_val),
                }
                if not is_sim:
                    step_metrics["reward_pos"] = float(info.get("reward_pos", 0))
                    if TORQUE_PENALTY_WEIGHT > 0:
                        step_metrics["torque_penalty"] = float(info.get("torque_penalty", 0))
                        step_metrics["torque"] = float(info.get("torque", 0))
                logger.log(step_metrics)

                if total_steps % 10 == 0:
                    log_str = (f"Ep {episode_num:3d} | S {episode_steps:3d}/{MAX_EPISODE_STEPS} | "
                               f"R {reward:7.2f} | Q {mean_q_val:7.2f} | "
                               f"Lq {loss_q_val:.3f} | La {loss_a_val:.3f} | "
                               f"a {alpha_val:.3f} | {actual_hz:5.1f}Hz")
                    if not is_sim:
                        log_str += f" | T {info.get('torque', 0):.1f}k"
                    print(log_str)

                # --- Periodic evaluation ---
                if eval_freq > 0 and total_steps % eval_freq == 0 and total_steps > WARMUP_STEPS:
                    video_file = None
                    if is_sim:
                        video_file = str(logger.run_dir / f"eval_step_{total_steps}.mp4")
                        
                    eval_mean, eval_std = evaluate_policy(
                        eval_env, encoder, actor, device,
                        n_episodes=eval_episodes,
                        max_steps=MAX_EPISODE_STEPS,
                        use_camera=USE_CAMERA, use_joints=USE_JOINTS,
                        video_path=video_file,
                        obs_rms=obs_rms
                    )
                    eval_msg = (f"[EVAL @ {total_steps}]  {eval_episodes} episodes  |  "
                                f"mean_return = {eval_mean:.2f} +/- {eval_std:.2f}")
                    print(eval_msg)
                    eval_metrics = {
                        "total_steps": total_steps,
                        "eval/mean_return": eval_mean,
                        "eval/std_return": eval_std,
                    }
                    if HAS_WANDB and logger.use_wandb and video_file:
                        if os.path.exists(video_file):
                            eval_metrics["eval/video"] = wandb.Video(video_file, format="mp4")
                        
                    logger.log(eval_metrics)

                # Advance observation
                img, q_torch = next_img, next_q_torch

                if done:
                    avg_rew = episode_reward / max(episode_steps, 1)
                    print(f"--- Ep {episode_num} done | {episode_steps} steps | "
                          f"total_steps {total_steps} | return {episode_reward:.2f} | "
                          f"avg {avg_rew:.3f} ---")
                    if HAS_WANDB and logger.use_wandb:
                        wandb.log({"episode_return": episode_reward,
                                   "avg_episode_reward": avg_rew}, step=total_steps)
                    break

    except KeyboardInterrupt:
        print("\nStopping RL loop.")
    finally:
        logger.plot()
        if not is_sim:
            q_home_local = q_home
            motor_names_pos_local = motor_names_pos
            run_post_analysis(encoder, actor, critic, q_home_local, motor_names_pos_local,
                              device, last_img_cache, logger)
            run_umap_analysis(encoder, actor, replay_buffer, q_home_local, motor_names_pos_local,
                              device, logger)
        logger.finish()
        if USE_CAMERA:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()
        if is_sim and eval_env is not env:
            eval_env.close()

if __name__ == "__main__":
    main()
