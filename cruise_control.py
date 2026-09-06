"""
cruise_control.py
------------------
Adaptive Cruise Control (ACC): maintains either the driver's set speed,
or a safe following gap behind a detected lead vehicle -- whichever is
more restrictive -- using a simple PID controller on the speed error.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ACCResult:
    target_speed_mps: float
    commanded_accel_mps2: float
    mode: str  # "cruise" or "following"


class PID:
    def __init__(self, kp: float, ki: float, kd: float, output_limits=(-3.0, 2.0)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.output_limits = output_limits

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        lo, hi = self.output_limits
        return max(lo, min(hi, output))


class AdaptiveCruiseControl:
    def __init__(
        self,
        set_speed_mps: float = 25.0,     # ~90 km/h driver-set speed
        time_gap_s: float = 1.8,          # desired following time-gap
        min_gap_m: float = 5.0,
        kp: float = 0.6,
        ki: float = 0.05,
        kd: float = 0.15,
        dt: float = 1 / 30,
    ):
        self.set_speed_mps = set_speed_mps
        self.time_gap_s = time_gap_s
        self.min_gap_m = min_gap_m
        self.dt = dt
        self.pid = PID(kp, ki, kd)
        self.ego_speed_mps = set_speed_mps  # simulated ego speed state

    def update(self, lead_distance_m: Optional[float], lead_closing_speed_mps: Optional[float]) -> ACCResult:
        mode = "cruise"
        target_speed = self.set_speed_mps

        if lead_distance_m is not None and lead_distance_m != float("inf"):
            # lead vehicle speed relative to ground = ego_speed - closing_speed
            lead_speed = self.ego_speed_mps - (lead_closing_speed_mps or 0.0)
            desired_gap = max(self.min_gap_m, self.time_gap_s * self.ego_speed_mps)
            if lead_distance_m < desired_gap * 1.5:
                # Blend: slow to match lead speed, further reduced if gap is tight
                gap_error = lead_distance_m - desired_gap
                speed_adjust = max(-self.set_speed_mps, min(0.0, gap_error * 0.5))
                target_speed = max(0.0, min(self.set_speed_mps, lead_speed + speed_adjust))
                mode = "following"

        error = target_speed - self.ego_speed_mps
        accel = self.pid.step(error, self.dt)
        self.ego_speed_mps = max(0.0, self.ego_speed_mps + accel * self.dt)

        return ACCResult(target_speed_mps=target_speed, commanded_accel_mps2=accel, mode=mode)
