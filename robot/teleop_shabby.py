#!/usr/bin/env python3
"""
Teleoperate the SO101 follower while printing motor current (torque proxy) each step.

Usage:
    python robot/teleop_shabby.py
"""

import os
import time

from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging

os.environ.setdefault("DISPLAY", ":0.0")

FPS = 30


def main():
    init_logging()

    robot_config = SOFollowerRobotConfig(port="/dev/ttyACM1", id="shabby")
    teleop_config = SOLeaderTeleopConfig(port="/dev/ttyACM0", id="shabby_leader")

    robot = make_robot_from_config(robot_config)
    teleop = make_teleoperator_from_config(teleop_config)

    teleop_action_processor, robot_action_processor, _ = make_default_processors()

    teleop.connect()
    robot.connect()

    motor_names = list(robot.bus.motors.keys())

    try:
        while True:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            raw_action = teleop.get_action()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action = robot_action_processor((teleop_action, obs))
            robot.send_action(robot_action)

            # Read motor currents (torque proxy)
            try:
                current_dict = robot.bus.sync_read("Present_Current")
                currents = [current_dict.get(name, 0) for name in motor_names]
                total_sq = sum(c ** 2 for c in currents)
                total_abs = sum(abs(c) for c in currents)
                print(
                    f"Currents: {[f'{c:6.1f}' for c in currents]}  "
                    f"Σ|I|={total_abs:8.1f}  Σ(I²)={total_sq:10.1f}"
                )
            except Exception as e:
                print(f"Current read error: {e}")

            dt = time.perf_counter() - t0
            precise_sleep(max(1.0 / FPS - dt, 0.0))

    except KeyboardInterrupt:
        print("\nStopping teleop.")
    finally:
        teleop.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
