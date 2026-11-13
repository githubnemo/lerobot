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

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.envs.registration import registry as gym_registry

SCENE_XML_PATH = Path(__file__).resolve().parent / "so101_assets" / "so101_cube_box_scene.xml"

DEFAULT_TASK_INSTRUCTION = "Grab the purple cube and put it into the cardboard box, ignore the other objects"

# Initial joint states (in LeRobot SO-101 degrees + [0..100] gripper units) representative of Orellius/so101_sort_cubes_no_top
START_POSES_DEG = np.array(
    [
        [-0.1, -22.2, 30.0, 52.1, -99.2, 3.1],
        [-1.5, -83.6, 93.6, 74.1, -55.0, 4.0],
        [1.1, -75.2, 80.5, 65.5, -53.8, 8.0],
        [-3.1, -86.8, 90.7, 76.6, -53.9, 10.3],
    ],
    dtype=np.float32,
)

# Representative cube spawn positions on the desk (in meters, robot base frame)
CUBE_SPAWN_POSITIONS = np.array(
    [
        [0.332, -0.056, 0.026],
        [0.275, -0.020, 0.026],
        [0.255, 0.035, 0.026],
        [0.300, 0.015, 0.026],
    ],
    dtype=np.float64,
)

BOX_CENTER_XY = np.array([0.165, -0.300], dtype=np.float64)
BOX_HALF_SIZE_XY = 0.090


def lerobot_to_mujoco_qpos(q_deg: np.ndarray, jnt_range: np.ndarray | None = None) -> np.ndarray:
    """Convert 6-D LeRobot SO-101 state/action (5 arm angles in deg + gripper in [0..100]) to MuJoCo radians."""
    q_rad = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64).copy())
    q_rad[5] = np.deg2rad((float(q_deg[5]) - 12.0) * 2.2)
    if jnt_range is not None:
        q_rad = np.clip(q_rad, jnt_range[:6, 0], jnt_range[:6, 1])
    return q_rad


def mujoco_to_lerobot_qpos(q_rad: np.ndarray) -> np.ndarray:
    """Convert 6-D MuJoCo joint positions in radians to LeRobot SO-101 state (deg + [0..100] gripper)."""
    q_deg = np.rad2deg(np.asarray(q_rad[:6], dtype=np.float64).copy())
    q_deg[5] = q_deg[5] / 2.2 + 12.0
    return q_deg.astype(np.float32)


class SO101CubeBoxEnv(gym.Env):
    """MuJoCo digital-twin environment for the SO-101 wrist-camera 'Cube to Cardboard Box' task."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        observation_width: int = 640,
        observation_height: int = 480,
        visualization_width: int = 640,
        visualization_height: int = 480,
        max_episode_steps: int = 600,
        n_substeps: int = 16,
        randomize: bool = False,
        reward_mode: str = "dense",
        task_instruction: str = DEFAULT_TASK_INSTRUCTION,
    ):
        super().__init__()
        if reward_mode not in ("dense", "sparse"):
            raise ValueError(f"Unsupported reward_mode={reward_mode!r}; expected 'dense' or 'sparse'.")
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.max_episode_steps = max_episode_steps
        self.n_substeps = n_substeps
        self.randomize = randomize
        self.reward_mode = reward_mode
        self.task = task_instruction
        self.task_description = task_instruction

        if not SCENE_XML_PATH.exists():
            raise FileNotFoundError(f"SO-101 MuJoCo scene XML not found at {SCENE_XML_PATH}")

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None

        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        self._fixed_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad")
        self._moving_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad")
        self._tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")

        self.action_space = gym.spaces.Box(
            low=np.array([-110.0, -100.0, -100.0, -95.0, -160.0, 0.0], dtype=np.float32),
            high=np.array([110.0, 100.0, 100.0, 95.0, 160.0, 100.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        obs_spaces: dict[str, gym.Space] = {
            "agent_pos": gym.spaces.Box(
                low=np.array([-180.0, -180.0, -180.0, -180.0, -180.0, -20.0], dtype=np.float32),
                high=np.array([180.0, 180.0, 180.0, 180.0, 180.0, 120.0], dtype=np.float32),
                shape=(6,),
                dtype=np.float32,
            ),
        }
        if "pixels" in self.obs_type:
            obs_spaces["pixels"] = gym.spaces.Dict(
                {
                    "front": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self._step_count = 0
        self._episode_idx = 0

    def _ensure_renderer(self) -> mujoco.Renderer:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.observation_height, width=self.observation_width
            )
        return self._renderer

    def _render_camera(self, camera_name: str = "wrist_cam") -> np.ndarray:
        renderer = self._ensure_renderer()
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render().copy()

    def _get_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {
            "agent_pos": mujoco_to_lerobot_qpos(self.data.qpos[:6]),
        }
        if "pixels" in self.obs_type:
            obs["pixels"] = {"front": self._render_camera("wrist_cam")}
        return obs

    def _compute_reward_and_success(self) -> tuple[float, bool, dict[str, Any]]:
        cube_pos = self.data.xpos[self._cube_body_id].copy()
        p1 = self.data.geom_xpos[self._fixed_pad_id]
        p2 = self.data.geom_xpos[self._moving_pad_id]
        jaw_mid = 0.5 * (p1 + p2)

        dist_to_cube = float(np.linalg.norm(jaw_mid - cube_pos))
        reach_reward = 0.25 * float(np.clip(1.0 - dist_to_cube / 0.35, 0.0, 1.0))

        is_lifted = bool(cube_pos[2] > 0.042 and dist_to_cube < 0.045)
        lift_reward = 0.25 if is_lifted else 0.0

        dist_to_box_xy = float(np.linalg.norm(cube_pos[:2] - BOX_CENTER_XY))
        transport_reward = (
            0.25 * float(np.clip(1.0 - dist_to_box_xy / 0.40, 0.0, 1.0)) if cube_pos[2] > 0.038 else 0.0
        )

        in_box_x = abs(cube_pos[0] - BOX_CENTER_XY[0]) <= BOX_HALF_SIZE_XY
        in_box_y = abs(cube_pos[1] - BOX_CENTER_XY[1]) <= BOX_HALF_SIZE_XY
        in_box_z = 0.015 <= cube_pos[2] <= 0.095
        is_success = bool(in_box_x and in_box_y and in_box_z)

        dense_reward = 1.0 if is_success else (reach_reward + lift_reward + transport_reward)
        reward = (1.0 if is_success else 0.0) if self.reward_mode == "sparse" else dense_reward
        info = {
            "is_success": is_success,
            "cube_pos": cube_pos,
            "dist_to_cube": dist_to_cube,
            "is_lifted": is_lifted,
            "dense_reward": dense_reward,
            "sparse_reward": 1.0 if is_success else 0.0,
            "metrics": {
                "dense_reward": float(dense_reward),
                "sparse_reward": 1.0 if is_success else 0.0,
            },
        }
        return reward, is_success, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        if self._step_count > 0:
            self._episode_idx += 1
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        if self.randomize:
            pose_idx = int(self.np_random.integers(0, len(START_POSES_DEG)))
            start_deg = START_POSES_DEG[pose_idx].copy()
            start_deg[:5] += self.np_random.uniform(-2.0, 2.0, size=5).astype(np.float32)
            cube_xy = self.np_random.uniform(low=[0.24, -0.06], high=[0.335, 0.04])
            cube_pos = np.array([cube_xy[0], cube_xy[1], 0.026], dtype=np.float64)
        else:
            idx = self._episode_idx % len(CUBE_SPAWN_POSITIONS)
            start_deg = START_POSES_DEG[0 if idx == 0 else (idx % len(START_POSES_DEG))].copy()
            cube_pos = CUBE_SPAWN_POSITIONS[idx].copy()

        q_init = lerobot_to_mujoco_qpos(start_deg, self.model.jnt_range)
        self.data.qpos[:6] = q_init
        self.data.ctrl[:6] = q_init

        # Freejoint for target_cube is at qpos[6:13] = [x, y, z, qw, qx, qy, qz]
        self.data.qpos[6:9] = cube_pos
        self.data.qpos[9:13] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)
        _, is_success, info = self._compute_reward_and_success()
        info["is_success"] = is_success
        return self._get_obs(), info

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.shape[0] != 6:
            raise ValueError(f"Expected 6-D action for SO-101, got shape {action_arr.shape}")

        self.data.ctrl[:6] = lerobot_to_mujoco_qpos(action_arr, self.model.jnt_range)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        reward, is_success, info = self._compute_reward_and_success()
        terminated = bool(is_success)
        truncated = bool(self._step_count >= self.max_episode_steps)
        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._render_camera("wrist_cam")

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if "lerobot/SO101CubeBox-v0" not in gym_registry:
    gym.register(
        id="lerobot/SO101CubeBox-v0",
        entry_point="lerobot.envs.so101_cube_box:SO101CubeBoxEnv",
    )

if "lerobot/SO101CubeBoxSparse-v0" not in gym_registry:
    gym.register(
        id="lerobot/SO101CubeBoxSparse-v0",
        entry_point="lerobot.envs.so101_cube_box:SO101CubeBoxEnv",
        kwargs={"reward_mode": "sparse"},
    )
