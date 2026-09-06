"""
collision_warning.py
---------------------
Forward Collision Warning (FCW) using Time-To-Collision (TTC).

TTC = distance / closing_speed

closing_speed is estimated by differentiating the tracked lead vehicle's
distance across frames. A warning is raised when TTC drops below a
configurable threshold (industry systems commonly warn around 2.0-2.7s
and brake-assist around 1.0-1.5s).
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque


@dataclass
class FCWResult:
    distance_m: Optional[float]
    closing_speed_mps: Optional[float]
    ttc_s: Optional[float]
    warning_level: str  # "none", "caution", "warning", "critical"


class CollisionWarningSystem:
    def __init__(
        self,
        dt: float = 1 / 30,           # seconds per frame
        caution_ttc: float = 3.0,
        warning_ttc: float = 2.0,
        critical_ttc: float = 1.0,
        smoothing_window: int = 5,
    ):
        self.dt = dt
        self.caution_ttc = caution_ttc
        self.warning_ttc = warning_ttc
        self.critical_ttc = critical_ttc
        self._history = deque(maxlen=smoothing_window)

    def update(self, distance_m: Optional[float]) -> FCWResult:
        if distance_m is None or distance_m == float("inf"):
            self._history.clear()
            return FCWResult(None, None, None, "none")

        self._history.append(distance_m)

        closing_speed = None
        ttc = None
        level = "none"

        if len(self._history) >= 2:
            # negative slope (distance shrinking) => positive closing speed
            closing_speed = (self._history[0] - self._history[-1]) / (self.dt * (len(self._history) - 1))
            if closing_speed > 0.05:  # only compute TTC when actually closing in
                ttc = distance_m / closing_speed
                if ttc <= self.critical_ttc:
                    level = "critical"
                elif ttc <= self.warning_ttc:
                    level = "warning"
                elif ttc <= self.caution_ttc:
                    level = "caution"

        return FCWResult(
            distance_m=distance_m,
            closing_speed_mps=closing_speed,
            ttc_s=ttc,
            warning_level=level,
        )
