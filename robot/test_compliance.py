#!/usr/bin/env python3
"""
Test the SafetyLayer + compliance spring on real hardware.

The compliance spring provides a soft "return to home" behavior.
The SafetyLayer ensures no step is too large and attenuates when
current is already high.  Together they let you push the robot
around safely and feel the spring pull it back.

This is the same SafetyLayer that gets used in the RL loop.
"""

import time
import numpy as np
import argparse
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots import make_robot_from_config
from lerobot.utils.robot_utils import precise_sleep

from safety import SafetyLayer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--stiffness", type=float, default=0.8,
                        help="P gain: how firmly it holds home (0=floppy, 1=rigid). "
                             "Yielding comes from the safety layer, not from low stiffness.")
    parser.add_argument("--ki", type=float, default=0.002,
                        help="I gain: accumulated error correction (overcomes gravity/friction)")
    parser.add_argument("--max-delta", type=float, default=5.0,
                        help="Max position change per step (degrees)")
    parser.add_argument("--threshold", type=float, default=100.0,
                        help="Current (mA) below which servo holds firm")
    parser.add_argument("--limit", type=float, default=300.0,
                        help="Current (mA) at which servo fully yields")
    parser.add_argument("--fps", type=int, default=50, help="Loop frequency")
    parser.add_argument("--range", type=float, default=60.0,
                        help="Max deviation from home (degrees)")
    args = parser.parse_args()

    # --- Setup Robot ---
    robot_config = SOFollowerRobotConfig(port=args.port, id="shabby")
    robot = make_robot_from_config(robot_config)
    robot.connect()

    motor_names = list(robot.bus.motors.keys())
    motor_names_pos = [f"{n}.pos" for n in motor_names]

    # --- Safety Layer (same one we use in RL) ---
    safety = SafetyLayer(
        robot.bus, motor_names,
        max_delta_deg=args.max_delta,
        current_threshold_mA=args.threshold,
        current_limit_mA=args.limit,
    )

    # --- Move to home ---
    print("Moving to home position...")
    q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 25.0])

    curr_obs = robot.get_observation()
    q_curr = np.array([curr_obs[k] for k in motor_names_pos])
    for i in range(50):
        interp = q_curr + (q_home - q_curr) * (i + 1) / 50
        robot.send_action({name: float(val) for name, val in zip(motor_names_pos, interp)})
        time.sleep(0.02)

    # --- Compliance Loop ---
    print(f"\nCompliance Active!")
    print(f"  Kp={args.stiffness}  Ki={args.ki}  max_delta={args.max_delta}°")
    print(f"  threshold={args.threshold}mA  limit={args.limit}mA  range=±{args.range}°")
    print("Push the robot joints. Press Ctrl+C to stop.\n")

    last_print = time.time()
    dt = 1.0 / args.fps
    num_joints = len(motor_names)
    integral = np.zeros(num_joints)
    integral_limit = 10.0  # anti-windup clamp (degrees)

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Read hardware state
            q_actual, currents = safety.read_state()

            # 2. PI compliance spring
            error = q_home - q_actual

            # P term
            p_term = args.stiffness * error

            # I term with anti-windup: only integrate when not being pushed
            # (if current is low, the deviation is from gravity/friction, so integrate)
            abs_I = np.abs(currents)
            not_pushed = abs_I < args.threshold
            integral[not_pushed] += error[not_pushed] * dt
            # Reset integral for joints being actively pushed
            integral[~not_pushed] = 0.0
            integral = np.clip(integral, -integral_limit, integral_limit)
            i_term = args.ki * integral

            q_spring = q_actual + p_term + i_term

            # Clamp to allowed range
            q_spring = np.clip(q_spring, q_home - args.range, q_home + args.range)

            # 3. Safety filter (clamps step size + attenuates by current)
            q_safe, info = safety(q_spring, q_actual=q_actual, currents=currents, q_home=q_home)

            # 4. Send
            action_dict = {f"{n}.pos": float(v) for n, v in zip(motor_names, q_safe)}
            robot.send_action(action_dict)

            # 5. Display
            now = time.time()
            if now - last_print > 0.12:
                abs_I = info["abs_currents"]
                max_i = np.argmax(abs_I)
                disp = q_actual - q_home
                max_d = np.argmax(np.abs(disp))
                avg_atten = info["attenuation"].mean()
                max_int = np.argmax(np.abs(integral))
                print(
                    f"I:{abs_I[max_i]:4.0f}mA ({motor_names[max_i]:10s}) | "
                    f"Disp:{disp[max_d]:+5.1f}° ({motor_names[max_d]:10s}) | "
                    f"Int:{integral[max_int]:+5.2f} | "
                    f"Att:{avg_atten:.0%}",
                    end="\r",
                )
                last_print = now

            precise_sleep(max(dt - (time.perf_counter() - t0), 0))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
