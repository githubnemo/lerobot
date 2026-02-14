# Cube Touch Task - RL Training Pipeline

This folder contains everything needed to train a robot to touch a cube using:
1. **Demonstration collection** with leader-follower teleoperation
2. **Reward labeling** (post-process demos to add success labels)
3. **Reward classifier training** to automate reward detection
4. **Reinforcement learning** with offline data + automated rewards

## Prerequisites

- SO101 follower robot connected (`/dev/ttyACM1`)
- SO101 leader robot connected (`/dev/ttyACM0`) - for data collection
- Camera connected (index 0)
- A cube placed in the robot's workspace

## Workflow

### Step 1: Collect Demonstration Data
```bash
./1_collect_data.sh
```
- Teleoperate with the leader arm
- **Right Arrow**: Save successful episode & move to next
- **Left Arrow**: Discard current episode & re-record
- **Escape**: Stop recording completely
- **Only save successful episodes** - where the gripper touches the cube!
- Collect ~50 successful episodes

### Step 2: Add Reward Labels
```bash
./2_add_reward_labels.sh
```
- Post-processes the collected dataset
- Marks the **last N frames** of each episode as success (`reward=1.0`)
- Marks all other frames as failure (`reward=0.0`)
- Adds `next.done` column for episode boundaries
- Required for reward classifier training

### Step 3: Train Reward Classifier
```bash
./3_train_reward_classifier.sh
```
- Trains a ResNet model to recognize "cube touched" from images
- Takes ~5-10 minutes on GPU
- Model saved to `outputs/reward_classifier/cube_touch`

### Step 4: Evaluate Reward Classifier (Optional)
```bash
./4_eval_reward_classifier.sh
```
- Test the classifier by moving the robot manually
- Verify it correctly detects when cube is touched
- Episode auto-terminates on predicted success

### Step 5: RL Training
**Terminal 1 - Start Learner:**
```bash
./5_rl_learner.sh
```

**Terminal 2 - Start Actor:**
```bash
./6_rl_actor.sh
```

The RL training:
- Loads demonstrations into offline buffer
- Uses reward classifier for automated success detection
- Mixes online exploration with offline data (50/50 per batch)
- Human can still provide additional feedback with number keys

## Files

| File | Description |
|------|-------------|
| `collect_data_config.json` | Config for demonstration collection (deprecated, using CLI) |
| `train_reward_classifier_config.json` | Config for classifier training |
| `rl_config.json` | Config for RL training (uses classifier + demos) |
| `add_reward_labels.py` | Script to post-process dataset with reward labels |
| `1_collect_data.sh` | Script to collect demonstrations |
| `2_add_reward_labels.sh` | Script to add reward labels to dataset |
| `3_train_reward_classifier.sh` | Script to train reward classifier |
| `4_eval_reward_classifier.sh` | Script to test reward classifier |
| `5_rl_learner.sh` | Script to start RL learner |
| `6_rl_actor.sh` | Script to start RL actor |

## Customization

### Change number of success frames per episode
In `2_add_reward_labels.sh`, modify:
```bash
N_SUCCESS_FRAMES=1  # Increase to mark more frames as success
```
Use more frames (e.g., 3-5) if the robot needs to learn to maintain contact, not just touch once.

### Change the reward classifier threshold
In `rl_config.json`:
```json
"reward_classifier": {
  "success_threshold": 0.7,  // Lower = more sensitive
  "success_reward": 1.0
}
```

### Disable offline data (pure online RL)
In `rl_config.json`:
```json
"dataset": null
```

### Disable reward classifier (human-only rewards)
In `rl_config.json`:
```json
"reward_classifier": null
```
Then use keyboard number keys (0-9) to provide rewards during training.
