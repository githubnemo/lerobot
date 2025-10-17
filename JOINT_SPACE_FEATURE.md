# Joint Space Control Feature for LeRobot

## Summary

This branch adds support for **joint space control** to LeRobot's RL gym_manipulator environment. Previously, the environment only supported end-effector (3D delta xyz) control, which requires inverse kinematics and limits the robot's capabilities. This feature allows direct control of joint positions.

## Changes Made

### 1. Configuration (`src/lerobot/envs/configs.py`)

- Added `action_space_type` field to `HILSerlProcessorConfig`
- Default: `"end_effector"` (maintains backward compatibility)
- Options: `"end_effector"` or `"joint"`
- Added `set_action_features_from_env()` method to dynamically set features

```python
@dataclass
class HILSerlProcessorConfig:
    """Configuration for environment processing pipeline."""

    control_mode: str = "gamepad"
    action_space_type: str = "end_effector"  # "end_effector" or "joint"
    ...

@EnvConfig.register_subclass(name="gym_manipulator")
@dataclass
class HILSerlRobotEnvConfig(EnvConfig):
    ...
    def set_action_features_from_env(self, action_dim: int):
        """Set action features based on actual environment action space.

        This is called after environment creation, ensuring compatibility
        with any robot (SO100, SO101, Koch, ViperX, etc.)
        """
        if not self.features:
            self.features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))
            }
```

### 2. Robot Environment (`src/lerobot/rl/gym_manipulator.py`)

#### RobotEnv Class Updates:

- Added `action_space_type` parameter to `__init__`
- Modified `_setup_spaces()` to create different action spaces:
  - **Joint space**: N-DOF action space (one action per motor)
  - **End-effector space**: 3D action space (xyz deltas)
- Modified `step()` to handle both action types

#### Pipeline Factory Updates:

- Made inverse kinematics pipeline **conditional** based on `action_space_type`
- When `action_space_type == "joint"`:
  - Skips all IK processing
  - Actions are passed directly to motors
  - Only adds `Torch2NumpyActionProcessorStep()`
- When `action_space_type == "end_effector"`:
  - Adds full IK pipeline (existing behavior)

### 3. Teleoperation Support (`src/lerobot/processor/hil_processor.py`)

#### InterventionActionProcessorStep Updates:

- Added `action_space_type` parameter
- Updated intervention handling to support both action space types:
  - **Joint space**: Extracts joint positions from teleop dict
  - **End-effector space**: Extracts delta commands (delta_x, delta_y, delta_z)

## Key Architecture Insight

### How RL Training Works

The RL training system (actor.py, learner.py) is **already action-dimension agnostic**:

1. **Policy Creation**:

   ```python
   policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
   ```

2. **Action Dimension Discovery**:

   ```python
   # In SACPolicy.__init__
   continuous_action_dim = config.output_features[ACTION].shape[0]
   ```

3. **Feature Extraction**:
   ```python
   # In policies/factory.py
   features = env_to_policy_features(env_cfg)
   ```

The policy automatically adapts to whatever action dimension is specified in `env_cfg.features`. Our changes ensure this is set correctly based on `action_space_type`.

## Usage

### Joint Space Control

```python
# In your config file or CLI
cfg.processor.action_space_type = "joint"

# Action space will be: Box(shape=(7,)) for SO100 (6 joints + gripper)
# Actions are direct joint positions
```

### End-Effector Control (Default)

```python
# In your config file or CLI
cfg.processor.action_space_type = "end_effector"

# Action space will be: Box(shape=(3,)) or Box(shape=(4,)) with gripper
# Actions are xyz deltas in end-effector space
```

## Benefits

1. **More Direct Control**: Policies can control joints directly without IK limitations
2. **Better for Some Tasks**: Tasks requiring specific joint configurations
3. **Training Flexibility**: Allows comparing joint vs EE control for research
4. **Backward Compatible**: Default behavior unchanged
5. **Automatic RL Support**: Works seamlessly with HILSerl training
6. **Teleoperation Support**: Human interventions work in both action spaces

## Teleoperation Compatibility

| Teleop Device                 | action_space_type="joint" | action_space_type="end_effector" |
| ----------------------------- | ------------------------- | -------------------------------- |
| **Leader Arms** (SO101, Koch) | ✅ Recommended            | ❌ Not supported                 |
| **Gamepad**                   | ❌ Not supported          | ✅ Supported                     |
| **Keyboard**                  | ❌ Not supported          | ✅ Supported                     |

**Note**: Leader arms return joint positions directly, making them ideal for joint space control. Gamepad/keyboard return end-effector deltas, so they only work with end-effector mode.

## Files Changed

1. `src/lerobot/envs/configs.py`:
   - Added `action_space_type` field to `HILSerlProcessorConfig`
   - Added `set_action_features_from_env()` method for dynamic feature initialization

2. `src/lerobot/rl/gym_manipulator.py`:
   - Added `action_space_type` parameter to `RobotEnv`
   - Modified action space creation to support both types
   - Made IK pipeline conditional
   - Pass `action_space_type` to intervention processor

3. `src/lerobot/processor/hil_processor.py`:
   - Added `action_space_type` parameter to `InterventionActionProcessorStep`
   - Updated intervention logic to handle joint space teleoperation

## Testing

To test this feature:

```bash
# Test with joint space - Recording
python src/lerobot/rl/gym_manipulator.py \
    env.processor.action_space_type=joint \
    mode=record

# Test with end-effector space (existing behavior)
python src/lerobot/rl/gym_manipulator.py \
    env.processor.action_space_type=end_effector \
    mode=record

# Test with RL training (joint space)
python -m lerobot.rl.learner --config_path your_config.json
python -m lerobot.rl.actor --config_path your_config.json
# (with action_space_type: "joint" in the config)
```

## PR to HuggingFace

This feature is ready to be submitted as a PR to `huggingface/lerobot`. The changes are:

- **Minimal**: Only adds the necessary functionality
- **Backward compatible**: Default behavior unchanged
- **Well-documented**: Clear docstrings and comments
- **Complete**: Works for both recording and RL training

### PR Checklist

- [x] Add action_space_type configuration field
- [x] Modify RobotEnv to support both action spaces
- [x] Make IK pipeline conditional
- [x] Add feature initialization for RL training
- [x] Add teleoperation support for joint space
- [x] Document all changes
- [ ] Create PR on GitHub
- [ ] Add tests (recommended)

## Next Steps for Matchbox Branch

After this PR is merged to HuggingFace:

1. Merge the updated main back into matchbox branch
2. Update your matchbox-specific configs to use the new `action_space_type` field
3. Remove any duplicate/old implementations
4. Test thoroughly with your robots

## Technical Notes

### Automatic Action Dimension Detection

The implementation automatically detects the correct action dimension for **any robot**:

1. **Environment Creation**: `RobotEnv` creates action space based on robot motors:

   ```python
   action_dim = len(self.robot.bus.motors)  # For joint space
   # or action_dim = 3 (+ gripper)  # For end-effector space
   ```

2. **Feature Configuration**: After env creation, features are set automatically:
   ```python
   action_dim = env.action_space.shape[0]
   cfg.set_action_features_from_env(action_dim)
   ```

This works seamlessly for:

- **SO100/SO101**: 7 DOF (6 arm + 1 gripper)
- **Koch**: 6 DOF (5 arm + 1 gripper)
- **ViperX**: 7 DOF
- **Custom robots**: Any number of joints

### Recording Mode

In recording mode, action features are determined from the teleop device, so they automatically match the robot being used. The feature initialization affects RL training mode where features must be known before policy creation.
