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

import numpy as np

from lerobot.envs.factory import make_env, make_env_config
from lerobot.envs.so101_cube_box import lerobot_to_mujoco_qpos, mujoco_to_lerobot_qpos
from lerobot.envs.utils import close_envs, prepare_canonical_observation, preprocess_observation


def test_so101_joint_conversion_roundtrip():
    q_deg = np.array([-0.1, -22.2, 30.0, 52.1, -99.2, 23.3], dtype=np.float32)
    q_rad = lerobot_to_mujoco_qpos(q_deg)
    q_roundtrip = mujoco_to_lerobot_qpos(q_rad)
    np.testing.assert_allclose(q_deg, q_roundtrip, atol=1e-4)


def test_so101_cube_box_env_creation_and_step():
    cfg = make_env_config("so101_cube_box", episode_length=50)
    envs = make_env(cfg, n_envs=1)
    try:
        vec_env = next(iter(envs.values()))[0]
        obs, info = vec_env.reset(seed=0)
        proc = (
            prepare_canonical_observation(obs)
            if vec_env.metadata.get("canonical_observations", False)
            else preprocess_observation(obs)
        )
        assert "observation.images.front" in proc
        assert "observation.state" in proc
        assert proc["observation.images.front"].shape == (1, 3, 480, 640)
        assert proc["observation.state"].shape == (1, 6)

        action = np.array([[0.0, -20.0, 30.0, 50.0, -90.0, 20.0]], dtype=np.float32)
        obs2, rew, term, trunc, info2 = vec_env.step(action)
        assert rew.shape == (1,)
        assert not term[0]
        assert not trunc[0]
        assert "is_success" in info2
    finally:
        close_envs(envs)


def test_so101_cube_box_sparse_reward_mode():
    cfg = make_env_config("so101_cube_box", episode_length=50, reward_mode="sparse")
    envs = make_env(cfg, n_envs=1)
    try:
        vec_env = next(iter(envs.values()))[0]
        vec_env.reset(seed=0)
        action = np.array([[0.0, -20.0, 30.0, 50.0, -90.0, 20.0]], dtype=np.float32)
        _, rew, term, _, info = vec_env.step(action)
        assert not term[0]
        assert rew[0] == 0.0
        dense_reward = info["metrics"]["dense_reward"] if "metrics" in info else info["dense_reward"]
        assert dense_reward[0] > 0.0
    finally:
        close_envs(envs)
