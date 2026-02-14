# Human Reward RL: Implementation Summary

This document describes the changes made to enable human-in-the-loop reinforcement learning with graded keyboard rewards.

## Overview

Added the ability for humans to provide **graded rewards (0.0-0.9)** during RL training by pressing number keys, enabling RLHF-style training without requiring a pre-trained reward classifier.

## Key Features

- **Graded rewards**: Press `0-9` keys for rewards from 0.0 to 0.9
- **Punishment**: Press `-` key for -0.5 penalty
- **Episode control**: `q` to end episode, `s` for success (+1.0), `r` to discard
- **Works alongside or instead of reward classifier**
- **No dataset required** - can train from scratch with human feedback only

---

## File Changes

### 1. `src/lerobot/teleoperators/utils.py`

Added new `TeleopEvents` enum value for human rewards:

```python
class TeleopEvents(Enum):
    SUCCESS = "success"
    RERECORD_EPISODE = "rerecord_episode"
    IS_INTERVENTION = "is_intervention"
    TERMINATE_EPISODE = "terminate_episode"
    HUMAN_REWARD = "human_reward"  # NEW: graded reward from keyboard (0.0-0.9)
```

### 2. `src/lerobot/teleoperators/keyboard/teleop_keyboard.py`

Extended `get_teleop_events()` to detect number key presses:

```python
# Handle number keys 0-9 for graded rewards, '-' for punishment
key_char = getattr(key, 'char', key) if hasattr(key, 'char') else str(key)
if key_char in "0123456789":
    human_reward = int(key_char) / 10.0  # '0'->0.0, '1'->0.1, ..., '9'->0.9
    logging.info(f"[HUMAN REWARD] +{human_reward:.1f} from key '{key_char}'")
elif key_char == "-":
    human_reward = -0.5  # Punishment
    logging.info(f"[HUMAN REWARD] {human_reward:.1f} (PUNISHMENT) from key '-'")
```

### 3. `src/lerobot/teleoperators/gamepad/teleop_gamepad.py`

Added `HUMAN_REWARD: None` to gamepad's `get_teleop_events()` for compatibility.

### 4. `src/lerobot/processor/hil_processor.py`

#### New: `HumanRewardProcessorStep`

```python
@dataclass
@ProcessorStepRegistry.register("human_reward_processor")
class HumanRewardProcessorStep(ProcessorStep):
    """
    Adds reward when human presses number keys (0-9) for graded feedback.
    Can be used alongside or instead of reward classifier.
    """
    cumulative: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()
        info = new_transition.get(TransitionKey.INFO, {})
        human_reward = info.get(TeleopEvents.HUMAN_REWARD, None)

        if human_reward is not None:
            current_reward = new_transition.get(TransitionKey.REWARD, 0.0)
            if self.cumulative:
                new_reward = current_reward + human_reward
            else:
                new_reward = human_reward
            new_transition[TransitionKey.REWARD] = new_reward

        return new_transition
```

#### Fix: `InterventionActionProcessorStep`

**Bug**: Was overwriting human rewards with `float(success)`.

```python
# Before (BUG - overwrote human reward):
new_transition[TransitionKey.REWARD] = float(success)

# After (ADD to existing reward):
current_reward = new_transition.get(TransitionKey.REWARD, 0.0)
new_transition[TransitionKey.REWARD] = current_reward + float(success)
```

### 5. `src/lerobot/rl/gym_manipulator.py`

#### Added `HumanRewardProcessorStep` to action pipeline:

```python
action_pipeline_steps = [
    AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
    AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
    HumanRewardProcessorStep(cumulative=True),  # NEW
    InterventionActionProcessorStep(...),
]
```

#### Reset reward at start of each step:

```python
# Prevent reward from carrying over between steps
transition[TransitionKey.REWARD] = 0.0
```

#### Fixed reward extraction with proper type handling:

```python
action_reward = processed_action_transition.get(TransitionKey.REWARD, 0.0)
if hasattr(action_reward, 'item'):
    action_reward = action_reward.item()
action_reward = float(action_reward) if action_reward is not None else 0.0
```

### 6. `src/lerobot/rl/buffer.py`

**Fix**: Disabled `torch.compile` for image augmentation (causes permission errors on some systems):

```python
# Before (crashed with PermissionError: '/sbin/ldconfig'):
self.image_augmentation_function = torch.compile(base_function)

# After:
self.image_augmentation_function = base_function
```

### 7. `src/lerobot/rl/learner.py`

Added console logging for training progress:

```python
# Log losses to console for easy monitoring
loss_str = f"[LEARNER] Step {optimization_step}: "
loss_str += f"critic={training_infos.get('loss_critic', 0):.4f}, "
loss_str += f"actor={training_infos.get('loss_actor', 0):.4f}, "
loss_str += f"temp={training_infos.get('loss_temperature', 0):.4f}, "
loss_str += f"α={training_infos.get('temperature', 0):.4f}, "
loss_str += f"buffer={training_infos.get('replay_buffer_size', 0)}"
logging.info(loss_str)
```

Added buffer waiting status:

```python
logging.info(f"[LEARNER] Waiting for data: {buffer_size}/{online_step_before_learning} samples")
```

Added transition received logging:

```python
logging.info(f"[LEARNER] Received {len(transition_list)} transitions from actor")
```

### 8. `src/lerobot/rl/actor.py`

Added keyboard teleoperator import and reward logging:

```python
from lerobot.teleoperators import gamepad, keyboard, so_leader  # Added keyboard

# Log non-zero rewards
if reward != 0.0:
    logging.info(f"[ACTOR] Step reward: {reward}")
```

### 9. `src/lerobot/configs/train.py`

Allow running without a dataset:

```python
# Before:
if isinstance(self.dataset.repo_id, list):

# After:
if self.dataset is not None and isinstance(self.dataset.repo_id, list):
```

---

## Keyboard Controls Reference

| Key | Action |
|-----|--------|
| `0-9` | Graded reward: 0.0 to 0.9 |
| `-` | Punishment: -0.5 |
| `q` | End episode (neutral) |
| `s` | End episode (success, +1.0) |
| `r` | Discard episode (rerecord) |
| `WASD/Arrows` | Intervention (take control) |

---

## Example Config

See `robot/train_rl_human_reward_config.json` for a complete example configuration that:
- Uses keyboard teleop for human rewards
- Disables reward classifier
- Runs without a pre-existing dataset
- Uses joint-space delta actions

---

## Usage

```bash
# Terminal 1: Start learner
bash robot/train_rl_human_reward_learner.sh

# Terminal 2: Start actor
bash robot/train_rl_human_reward_actor.sh
```

Press number keys during robot execution to provide rewards. The learner will start training after collecting enough transitions (default: 200).

---

## Notes

- Rewards are applied to the **current step** when the key is pressed
- Episode reward is the sum of all step rewards
- Works with or without a reward classifier
- Can be combined with demonstrations in the replay buffer

