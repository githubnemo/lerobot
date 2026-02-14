# Robot RL Cheatsheet

## Action Formats

| Stage | Format | Range | Description |
|---|---|---|---|
| **Data collection** (teleop) | Absolute joint positions | `[-100, 100]` (body), `[0, 100]` (gripper) | Leader arm encoder readings, normalized by `motors_bus._normalize()` |
| **LeRobotDataset** | Absolute joint positions | Same as above | Stored directly from teleop |
| **Replay buffer** (after conversion) | **Delta** (normalized) | `[-1, 1]` | `delta = (action - obs_state) / action_scale` |
| **SAC policy output** | **Delta** (normalized) | `[-1, 1]` | `tanh`-squashed Gaussian |
| **Environment step** | Delta → absolute | `[-action_scale, +action_scale]` per step | `target = current_pos + action * action_scale` |

## action_scale_per_s (FPS-independent scaling)

```
action_scale = action_scale_per_s / fps
```

- Defined in `rl_config.json` → `env.action_scale_per_s`
- The policy outputs `a ∈ [-1, 1]`; the env applies `delta = a * action_scale` per step
- Making it per-second means changing FPS doesn't change the max velocity of the robot

| action_scale_per_s | fps | action_scale/step | Max delta/step |
|---|---|---|---|
| 50 | 10 | 5.0 | ±5.0 |
| 50 | 30 | 1.67 | ±1.67 |
| 100 | 10 | 10.0 | ±10.0 |
| 100 | 30 | 3.33 | ±3.33 |

## Delta Interpolation (buffer conversion)

When converting the expert dataset (absolute positions) to deltas for the replay buffer,
some transitions have deltas larger than `action_scale` (the policy can't express them in
one step). These are automatically split:

1. Compute `delta = action - observation.state` for each frame
2. If `max(|delta_j|) > action_scale` for any joint j:
   - `N = ceil(max(|delta_j|) / action_scale)`
   - Split into N sub-transitions, each with `sub_action = delta / N / action_scale`
   - **Images**: zero-order hold (repeat current frame)
   - **Proprioception**: linear interpolation
   - **Reward/done**: assigned only to last sub-step
3. If delta fits → single transition, no interpolation

This preserves original timing for small movements and only slows down transitions
that would violate the agent's action-size capabilities.

## Dataset Delta Distribution (cube_out_of_box, 10 FPS)

```
Joint           min      max      mean     std      median
shoulder_pan   -33.8    75.5      1.0      4.9       0.2
shoulder_lift  -39.6    36.4     -2.2      5.5      -1.7
elbow_flex     -36.9    47.4     -3.4      4.9      -3.3
wrist_flex     -36.9    51.0     -0.4      4.8      -0.4
wrist_roll     -15.3    53.7     -0.1      2.4       0.0
gripper        -32.4    24.4     -2.8      6.7       0.0
```

**|delta| percentiles:**
```
       shoulder_pan  shoulder_lift  elbow_flex  wrist_flex  wrist_roll  gripper
p50       0.80          2.37          3.84        0.83        0.20       0.42
p75       2.55          5.06          5.43        2.75        0.66       3.69
p90       6.79          9.94          9.01        7.32        2.46      15.90
p95      10.21         13.67         12.51       10.90        3.73      18.91
p99      20.36         19.89         20.47       19.26        8.73      20.61
p100     75.49         39.64         47.38       50.99       53.69      32.43
```

**Frames needing interpolation by action_scale:**
```
action_scale=5  (per_s=50  @ 10fps): 65.2% interpolated
action_scale=10 (per_s=100 @ 10fps): 32.6% interpolated
action_scale=15 (per_s=150 @ 10fps): 17.9% interpolated
action_scale=20 (per_s=200 @ 10fps):  5.6% interpolated
```

## Key Files

| File | What it does |
|---|---|
| `src/lerobot/rl/gym_manipulator.py` | `RobotEnv.step()` — applies delta actions to robot |
| `src/lerobot/rl/buffer.py` | `ReplayBuffer.from_lerobot_dataset()` — converts dataset to deltas with interpolation |
| `src/lerobot/rl/learner.py` | `initialize_offline_replay_buffer()` — orchestrates buffer creation |
| `src/lerobot/policies/sac/modeling_sac.py` | SAC policy — `tanh`-squashed actor output |
| `src/lerobot/envs/configs.py` | `HILSerlRobotEnvConfig` — env config with `action_scale_per_s` |
| `src/lerobot/motors/motors_bus.py` | `_normalize()` — raw encoder → `[-100,100]` or `[0,100]` |

## Normalization Pipeline

```
Raw encoder (0-4095) → _normalize() → [-100, 100] body / [0, 100] gripper
                                           ↓
                              LeRobotDataset (stored as-is)
                                           ↓
                              Buffer conversion: delta / action_scale → [-1, 1]
                                           ↓
                              SAC policy trains on [-1, 1] deltas
                                           ↓
                              Env step: a * action_scale → delta in position-units
                                           ↓
                              send_action(current_pos + delta) → robot
```

## Discrete Time

- Both data collection and RL operate at fixed FPS (configured in `env.fps`)
- Data collection: `dataset.fps` in the recording config
- RL training env: `env.fps` in `rl_config.json`
- The `action_scale_per_s` makes the action budget FPS-independent
