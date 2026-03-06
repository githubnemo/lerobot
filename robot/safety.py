"""
Torque-aware safety layer for SO-101.

Sits between any controller (RL agent, compliance spring, teleop) and the
motors.  Two independent defenses, both stateless per step:

  1. PROACTIVE — clamp the max position delta per step so the servo never
     receives a command that would require a large instantaneous torque.

  2. REACTIVE — read Present_Current from the *previous* step; if it is
     already high (gravity, collision, friction), scale down the commanded
     delta further so the servo doesn't pile on more torque.

Usage:
    safety = SafetyLayer(bus, motor_names)
    ...
    q_safe, info = safety(q_requested)   # call every control step
    bus.sync_write("Goal_Position", q_safe.tolist())
"""

import numpy as np


class SafetyLayer:
    """Stateless per-step torque limiter."""

    def __init__(
        self,
        bus,
        motor_names: list[str],
        *,
        max_delta_deg: float = 5.0,
        current_threshold_mA: float = 150.0,
        current_limit_mA: float = 400.0,
    ):
        """
        Args:
            bus:  robot.bus (has sync_read / sync_write / motors)
            motor_names:  ordered list of motor name strings
            max_delta_deg:  largest single-step position change allowed (degrees)
            current_threshold_mA:  below this current, full authority
            current_limit_mA:  at/above this current, zero authority (hold position)
        """
        self.bus = bus
        self.motor_names = motor_names
        self.motor_ids = [bus.motors[n].id for n in motor_names]
        self.max_delta = max_delta_deg
        self.threshold = current_threshold_mA
        self.limit = current_limit_mA

    def read_state(self):
        """Read actual positions and currents from hardware."""
        pos_dict = self.bus.sync_read("Present_Position")
        cur_dict = self.bus.sync_read("Present_Current")
        q_actual = np.array(
            [pos_dict.get(n, pos_dict.get(mid, 0))
             for n, mid in zip(self.motor_names, self.motor_ids)],
            dtype=np.float32,
        )
        currents = np.array(
            [cur_dict.get(n, cur_dict.get(mid, 0))
             for n, mid in zip(self.motor_names, self.motor_ids)],
            dtype=np.float32,
        )
        return q_actual, currents

    def __call__(self, q_requested: np.ndarray, q_actual: np.ndarray = None,
                 currents: np.ndarray = None, q_home: np.ndarray = None):
        """
        Filter a requested target position into a safe one.

        Args:
            q_requested:  desired target (from RL agent, spring, etc.)
            q_actual:     current joint positions (read if None)
            currents:     current readings in mA (read if None)
            q_home:       home/reference position. If provided, restoring
                          movements (toward home) bypass current attenuation
                          so gravity-loaded joints can return home.

        Returns:
            q_safe:   attenuated target position (ndarray)
            info:     dict with diagnostics
        """
        if q_actual is None or currents is None:
            q_actual_read, currents_read = self.read_state()
            if q_actual is None:
                q_actual = q_actual_read
            if currents is None:
                currents = currents_read

        # 1. PROACTIVE: clamp step size
        delta = q_requested - q_actual
        delta_clamped = np.clip(delta, -self.max_delta, self.max_delta)

        # 2. REACTIVE: attenuate by current magnitude (per-joint)
        abs_I = np.abs(currents)
        scale = np.clip(
            1.0 - (abs_I - self.threshold) / (self.limit - self.threshold),
            0.0, 1.0,
        )

        # 3. GRAVITY EXEMPTION: if q_home is given, don't attenuate
        #    movements that are heading toward home.  The max_delta clamp
        #    alone limits torque for restoring movements.
        if q_home is not None:
            toward_home = q_home - q_actual
            restoring = (np.sign(delta_clamped) == np.sign(toward_home)) | (np.abs(delta_clamped) < 0.1)
            scale[restoring] = 1.0

        q_safe = q_actual + scale * delta_clamped

        info = {
            "q_actual": q_actual,
            "currents": currents,
            "abs_currents": abs_I,
            "scale": scale,
            "delta_raw": q_requested - q_actual,
            "delta_clamped": delta_clamped,
            "delta_applied": scale * delta_clamped,
            "attenuation": 1.0 - scale,  # 0 = full authority, 1 = fully blocked
        }
        return q_safe, info
