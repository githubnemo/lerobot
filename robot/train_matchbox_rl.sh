#!/bin/bash
set -eu

# Path to your RL configuration file
# Make sure to create this file and customize it for your setup.
CONFIG_PATH="robot/train_rl_config.json"

# Check if the config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Configuration file not found at $CONFIG_PATH"
    echo "Please create it, perhaps based on https://huggingface.co/datasets/aractingi/lerobot-example-config-files/blob/main/train_config_hilserl_so100.json"
    exit 1
fi

# Function to clean up background processes on exit
cleanup() {
    echo "Shutting down learner..."
    if ps -p $LEARNER_PID > /dev/null; then
        kill $LEARNER_PID
    fi
    wait $LEARNER_PID 2>/dev/null
    echo "Cleanup complete."
}

# Trap script exit signals to run the cleanup function
trap cleanup EXIT INT TERM

echo "Starting learner in the background..."
python lerobot/scripts/rl/learner.py --config_path "$CONFIG_PATH" &
LEARNER_PID=$!

# Give the learner a moment to initialize the gRPC server
echo "Waiting for learner to start... (PID: $LEARNER_PID)"
sleep 10

echo "Starting actor..."
# The script will stay on this line until the actor is stopped (e.g., with Ctrl+C)
python lerobot/scripts/rl/actor.py --config_path "$CONFIG_PATH"

echo "Actor finished."
