# Teleoperation Joint Space Support Update

## Issue Found

The initial implementation was missing proper teleoperation support for joint space control. The `InterventionActionProcessorStep` was hardcoded to handle only end-effector delta commands (delta_x, delta_y, delta_z).

## Fix Applied

### Modified Files

#### 1. `src/lerobot/processor/hil_processor.py`

Added `action_space_type` parameter to `InterventionActionProcessorStep`:

```python
@dataclass
class InterventionActionProcessorStep(ProcessorStep):
    use_gripper: bool = False
    terminate_on_success: bool = True
    action_space_type: str = "end_effector"  # NEW: supports "joint" or "end_effector"
```

Updated intervention handling to support both action space types:

```python
if is_intervention and teleop_action is not None:
    if isinstance(teleop_action, dict):
        if self.action_space_type == "joint":
            # For joint space: teleop returns joint positions
            # e.g., {"shoulder.pos": 0.5, "elbow.pos": 1.2, ...}
            action_list = list(teleop_action.values())
        else:
            # For end-effector space: teleop returns delta commands
            # e.g., {"delta_x": 0.1, "delta_y": 0.0, "delta_z": 0.05}
            action_list = [
                teleop_action.get("delta_x", 0.0),
                teleop_action.get("delta_y", 0.0),
                teleop_action.get("delta_z", 0.0),
            ]
            if self.use_gripper:
                action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
```

#### 2. `src/lerobot/rl/gym_manipulator.py`

Pass `action_space_type` to intervention processor:

```python
InterventionActionProcessorStep(
    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
    terminate_on_success=terminate_on_success,
    action_space_type=cfg.processor.action_space_type,  # NEW
)
```

## How It Works

### Joint Space Teleoperation

When using a **leader arm** (e.g., SO101 Leader, Koch Leader) with `action_space_type="joint"`:

1. **Teleop device** returns joint positions:

   ```python
   {
       "shoulder.pos": 0.5,
       "elbow.pos": 1.2,
       "wrist.pos": -0.3,
       "gripper.pos": 50.0
   }
   ```

2. **InterventionActionProcessorStep** extracts values:

   ```python
   action_list = [0.5, 1.2, -0.3, 50.0]  # Preserves dict order (Python 3.7+)
   ```

3. **Action sent to robot**: Direct joint positions (no IK needed)

### End-Effector Teleoperation

When using **gamepad/keyboard** with `action_space_type="end_effector"`:

1. **Teleop device** returns delta commands:

   ```python
   {
       "delta_x": 0.1,
       "delta_y": 0.0,
       "delta_z": 0.05,
       "gripper": 1.0
   }
   ```

2. **InterventionActionProcessorStep** extracts deltas:

   ```python
   action_list = [0.1, 0.0, 0.05, 1.0]
   ```

3. **IK pipeline** converts to joint positions

## Supported Teleop Combinations

| Teleop Device | action_space_type | Works? | Notes                                                 |
| ------------- | ----------------- | ------ | ----------------------------------------------------- |
| SO101 Leader  | `"joint"`         | ✅     | Direct joint control - optimal                        |
| SO101 Leader  | `"end_effector"`  | ❌     | Leader returns joints, IK expects deltas              |
| Koch Leader   | `"joint"`         | ✅     | Direct joint control                                  |
| Gamepad       | `"end_effector"`  | ✅     | Returns deltas, IK converts                           |
| Gamepad       | `"joint"`         | ❌     | Gamepad returns deltas, can't directly control joints |
| Keyboard      | `"end_effector"`  | ✅     | Returns deltas, IK converts                           |

## Best Practices

1. **Leader Arms + Joint Space**: Use `action_space_type="joint"` for most direct control
2. **Gamepad/Keyboard**: Use `action_space_type="end_effector"` (only supported mode)
3. **Recording demonstrations**: Match action_space_type to your teleoperation device
4. **RL Training**: Can use either action_space_type regardless of how demos were collected

## Summary

Now the complete joint space control feature is working for:

- ✅ Environment action spaces (joint or end-effector)
- ✅ IK pipeline conditionally applied
- ✅ Teleoperation with interventions (both action space types)
- ✅ RL training (actor/learner scripts)
- ✅ Recording demonstrations
- ✅ Works with any robot (SO100, SO101, Koch, ViperX, etc.)
