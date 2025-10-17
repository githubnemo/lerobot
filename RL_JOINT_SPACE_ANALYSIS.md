# RL Joint Space Support Analysis

## Summary

**Good News**: The RL learning scripts (`actor.py`, `learner.py`) are **mostly action-space agnostic** and will work with joint space control! However, there's **one critical issue** that needs to be fixed.

## Current Architecture

### How Action Dimensions are Determined

1. **Policy Creation** (`actor.py` line 254-257):

   ```python
   policy: SACPolicy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
   ```

2. **Feature Extraction** (`policies/factory.py` line 361):

   ```python
   features = env_to_policy_features(env_cfg)
   ```

3. **Action Dimension** (`policies/sac/modeling_sac.py` line 54):
   ```python
   continuous_action_dim = config.output_features[ACTION].shape[0]
   ```

### The Problem

The `HILSerlRobotEnvConfig` (gym_manipulator) has **empty features by default**:

```python
@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    robot: RobotConfig | None = None
    teleop: TeleoperatorConfig | None = None
    processor: HILSerlProcessorConfig = field(default_factory=HILSerlProcessorConfig)

    # Features are empty dict by default (inherited from EnvConfig)
    # features: dict[str, PolicyFeature] = field(default_factory=dict)
```

This means:

- **Recording mode works** ✅ - Gets features from `teleop_device.action_features`
- **RL training mode FAILS** ❌ - No features defined, so policy can't determine action dimension

## Required Fix

You need to **set the ACTION feature in the environment config dynamically** based on the action space type and the actual environment.

### ✅ Solution: Set Features After Environment Creation

We use the actual environment's action space to set features dynamically:

**In `src/lerobot/envs/configs.py`:**

```python
@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    ...
    def set_action_features_from_env(self, action_dim: int):
        """Set action features based on actual environment action space."""
        if not self.features:
            self.features = {
                ACTION: PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(action_dim,)
                )
            }
            self.features_map = {ACTION: ACTION}
```

**In `src/lerobot/rl/gym_manipulator.py`:**

```python
def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment with dynamic feature configuration."""

    # ... create env with robot ...

    # Set features based on actual environment action space
    # This works for ANY robot (SO100, SO101, Koch, ViperX, custom, etc.)
    action_dim = env.action_space.shape[0]
    cfg.set_action_features_from_env(action_dim)

    return env, teleop_device
```

## Why This Solution is Best

This approach:

- ✅ Works for **any robot** without hardcoding dimensions
- ✅ Automatically adapts to the robot's actual motor configuration
- ✅ Maintains backward compatibility
- ✅ Requires no manual feature configuration from users
- ✅ Follows the principle: "inspect at runtime, not at config time"

## Action Items

1. ✅ Environment action space correctly changes with `action_space_type`
2. ✅ IK pipeline conditionally applied
3. ✅ Actor/Learner scripts are action-dimension agnostic
4. ❌ **Need to add**: Feature configuration in `HILSerlRobotEnvConfig`

## Files to Modify

1. `src/lerobot/envs/configs.py` - Add `__post_init__` to `HILSerlRobotEnvConfig`

Once this is fixed, the RL training will automatically work with both joint and end-effector action spaces!
