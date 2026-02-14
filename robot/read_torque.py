#!/usr/bin/env python3
"""Quick script to read and print motor currents from the SO101 robot.

Run this while teleoperating to get a feel for torque values.
Usage:
    python robot/read_torque.py --port /dev/ttyACM1 --fps 10
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(description="Read motor currents from SO101")
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    # SO101 motor configuration (no calibration needed for current reading)
    motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    bus = FeetechMotorsBus(port=args.port, motors=motors)
    bus.connect()
    print(f"Connected to {args.port}. Reading motor currents at {args.fps} Hz...")
    print(f"Press Ctrl+C to stop.\n")

    dt = 1.0 / args.fps

    # Track running stats
    max_total_sq = 0.0
    min_total_sq = float('inf')

    try:
        while True:
            start = time.perf_counter()

            try:
                currents = bus.sync_read("Present_Current")
            except Exception as e:
                print(f"Read error: {e}")
                time.sleep(dt)
                continue

            # Compute per-motor and total
            total_sq = 0.0
            total_abs = 0.0
            parts = []
            for name, val in currents.items():
                sq = val * val
                total_sq += sq
                total_abs += abs(val)
                parts.append(f"{name}:{val:+6.0f}")

            # Track min/max for thresholding later
            max_total_sq = max(max_total_sq, total_sq)
            if total_sq > 0:
                min_total_sq = min(min_total_sq, total_sq)

            # Print one compact line (overwrite)
            line = " ".join(parts)
            print(f"\r{line}  Σ|I|={total_abs:6.0f}  Σ(I²)={total_sq:8.0f}  [min={min_total_sq:8.0f} max={max_total_sq:8.0f}]", end="", flush=True)

            # Maintain FPS
            elapsed = time.perf_counter() - start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n=== Summary ===")
        print(f"Min Σ(I²): {min_total_sq:.0f}")
        print(f"Max Σ(I²): {max_total_sq:.0f}")
        print(f"\nSuggested torque penalty threshold: ~{(min_total_sq + max_total_sq) / 2:.0f}")
        print(f"(penalize when Σ(I²) > threshold)")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    main()
