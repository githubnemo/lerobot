# Robot Training Scripts

This directory contains scripts for training and interacting with the robot setups.

## RL Training with Actor-Learner

The `train_matchbox_rl.sh` script launches a distributed reinforcement learning training session using an Actor-Learner architecture.

### How it Works
-   **`learner.py`**: The "brain" of the operation. It runs on a powerful machine, receives data, and trains the policy.
-   **`actor.py`**: The "body". It runs on the machine connected to the robot, executes the policy, and collects data.

They communicate over the network using the settings defined in `train_rl_config.json`.

### How to Run Remotely

This setup is ideal for running the computationally-heavy learner process on a powerful remote machine (e.g., a server with a strong GPU) while the actor runs on the local machine connected to the robot.

**1. Set up the Learner Machine:**

-   Ensure the `lerobot` repository is cloned and the environment is set up on your powerful remote machine.
-   Find the local IP address of this machine (e.g., `192.168.1.100`).
-   Make sure any firewalls on this machine allow incoming connections on the port specified in the config (default is `50051`).

**2. Configure for Remote Connection:**

-   Open `robot/train_rl_config.json` on **both** the learner and actor machines.
-   Locate the `actor_learner_config` section.
-   Change `learner_host` from `"127.0.0.1"` to the IP address of your learner machine.

    ```json
    "actor_learner_config": {
      "learner_host": "192.168.1.100", // <-- CHANGE THIS
      "learner_port": 50051,
      // ...
    }
    ```

**3. Launch the Training:**

-   **On the remote learner machine (e.g., with the 4090 GPU):**
    ```bash
    python lerobot/scripts/rl/learner.py --config_path robot/train_rl_config.json
    ```

-   **On the local actor machine (connected to the robot):**
    ```bash
    python lerobot/scripts/rl/actor.py --config_path robot/train_rl_config.json
    ```

The actor will connect to the learner across the network, and your distributed training will begin!
