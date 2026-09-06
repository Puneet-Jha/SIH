"""
lane_detection.py
------------------
Classical computer-vision lane detector.

Pipeline: grayscale -> blur -> Canny edges -> region-of-interest mask ->
probabilistic Hough transform -> average/extrapolate into a single
left-lane and right-lane line -> compute lane center & vehicle offset.

Works on real dashcam footage as well as the synthetic demo video, since
it makes no assumptions beyond "lanes are roughly straight, high-contrast
lines in the lower half of the frame".
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LaneResult:
    left_line: Optional[Tuple[int, int, int, int]]   # (x1, y1, x2, y2)
    right_line: Optional[Tuple[int, int, int, int]]
    lane_center_px: Optional[float]      # x-pixel of lane center at bottom of frame
    vehicle_offset_px: Optional[float]   # +right / -left of lane center
    vehicle_offset_m: Optional[float]    # approx offset in meters
    departure_warning: bool


class LaneDetector:
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_threshold: int = 20,
        min_line_len: int = 20,
        max_line_gap: int = 300,
        lane_px_to_m: float = 3.7 / 500,  # ~US lane width (3.7m) over ~500px lane in demo
        departure_threshold_m: float = 0.6,
        smoothing: float = 0.8,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.lane_px_to_m = lane_px_to_m
        self.departure_threshold_m = departure_threshold_m
        self.smoothing = smoothing

        # exponential smoothing state (slope, intercept) for left/right lanes
        self._left_avg = None
        self._right_avg = None

    def _region_of_interest(self, edges: np.ndarray) -> np.ndarray:
        h, w = edges.shape
        mask = np.zeros_like(edges)
        # trapezoid covering the road area (bottom half, narrowing toward horizon)
        polygon = np.array([[
            (int(0.05 * w), h),
            (int(0.45 * w), int(0.55 * h)),
            (int(0.55 * w), int(0.55 * h)),
            (int(0.95 * w), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _classify_and_average(self, lines, w, h):
        """Split raw Hough segments into left/right by slope sign, fit one
        representative line for each side."""
        left_pts, right_pts = [], []
        if lines is None:
            return None, None

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.3:  # ignore near-horizontal noise
                continue
            if slope < 0:
                left_pts.append((x1, y1))
                left_pts.append((x2, y2))
            else:
                right_pts.append((x1, y1))
                right_pts.append((x2, y2))

        def fit(points):
            if len(points) < 2:
                return None
            xs = np.array([p[0] for p in points])
            ys = np.array([p[1] for p in points])
            # fit x = m*y + b  (parameterize by y since lane lines are ~vertical)
            m, b = np.polyfit(ys, xs, 1)
            y_bottom, y_top = h, int(0.6 * h)
            x_bottom, x_top = int(m * y_bottom + b), int(m * y_top + b)
            return (x_bottom, y_bottom, x_top, y_top), (m, b)

        left = fit(left_pts)
        right = fit(right_pts)
        return left, right

    def _smooth(self, prev, current, alpha):
        if current is None:
            return prev
        if prev is None:
            return current
        (m0, b0) = prev
        (m1, b1) = current
        return (alpha * m0 + (1 - alpha) * m1, alpha * b0 + (1 - alpha) * b1)

    def process(self, frame: np.ndarray) -> LaneResult:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        roi = self._region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi,
            rho=2,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

        left, right = self._classify_and_average(lines, w, h)
        left_line, left_params = left if left else (None, None)
        right_line, right_params = right if right else (None, None)

        self._left_avg = self._smooth(self._left_avg, left_params, self.smoothing)
        self._right_avg = self._smooth(self._right_avg, right_params, self.smoothing)

        y_bottom = h
        left_x = right_x = None
        if self._left_avg is not None:
            m, b = self._left_avg
            left_x = m * y_bottom + b
            left_line = (int(left_x), y_bottom, int(m * int(0.6 * h) + b), int(0.6 * h))
        if self._right_avg is not None:
            m, b = self._right_avg
            right_x = m * y_bottom + b
            right_line = (int(right_x), y_bottom, int(m * int(0.6 * h) + b), int(0.6 * h))

        lane_center_px = None
        vehicle_offset_px = None
        vehicle_offset_m = None
        departure = False

        if left_x is not None and right_x is not None:
            lane_center_px = (left_x + right_x) / 2.0
            frame_center_px = w / 2.0
            vehicle_offset_px = frame_center_px - lane_center_px
            vehicle_offset_m = vehicle_offset_px * self.lane_px_to_m
            departure = abs(vehicle_offset_m) > self.departure_threshold_m

        return LaneResult(
            left_line=left_line,
            right_line=right_line,
            lane_center_px=lane_center_px,
            vehicle_offset_px=vehicle_offset_px,
            vehicle_offset_m=vehicle_offset_m,
            departure_warning=departure,
        )

    @staticmethod
    def draw(frame: np.ndarray, result: LaneResult) -> np.ndarray:
        overlay = frame.copy()
        if result.left_line:
            cv2.line(overlay, result.left_line[:2], result.left_line[2:], (0, 255, 0), 6)
        if result.right_line:
            cv2.line(overlay, result.right_line[:2], result.right_line[2:], (0, 255, 0), 6)
        if result.left_line and result.right_line:
            pts = np.array([
                result.left_line[:2], result.left_line[2:],
                result.right_line[2:], result.right_line[:2],
            ])
            cv2.fillPoly(overlay, [pts], (0, 200, 0))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

        if result.departure_warning:
            cv2.putText(frame, "LANE DEPARTURE WARNING", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        return frame
