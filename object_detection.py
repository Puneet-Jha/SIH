"""
object_detection.py
--------------------
Lightweight vehicle/obstacle detector using background subtraction (MOG2)
plus contour filtering. This needs no downloaded model weights, which
makes it fully self-contained and offline-friendly.

For production use you would typically swap this out for a trained
detector (e.g. YOLOv8 via `ultralytics`, or an ONNX/TensorRT model run
through cv2.dnn) -- see the `detect()` docstring for the drop-in
interface such a replacement would need to satisfy.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: int
    center: Tuple[int, int]
    est_distance_m: float


class VehicleDetector:
    def __init__(
        self,
        min_area: int = 500,
        known_width_m: float = 1.8,   # approx width of a car
        focal_length_px: float = 700.0,  # camera intrinsic (calibrate for real cameras)
        history: int = 200,
        var_threshold: float = 40,
    ):
        self.min_area = min_area
        self.known_width_m = known_width_m
        self.focal_length_px = focal_length_px
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True
        )

    def _estimate_distance(self, pixel_width: int) -> float:
        """Simple pinhole-camera distance estimate: distance = (real_width * focal_length) / pixel_width."""
        if pixel_width <= 0:
            return float("inf")
        return (self.known_width_m * self.focal_length_px) / pixel_width

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Return a list of Detection objects for this frame.

        A drop-in replacement (e.g. a YOLO wrapper) just needs to return
        the same List[Detection] shape so the rest of the pipeline
        (collision_warning.py, adas_pipeline.py) keeps working unchanged.
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # clean up shadows/noise
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            center = (x + w // 2, y + h // 2)
            distance = self._estimate_distance(w)
            detections.append(Detection(bbox=(x, y, w, h), area=int(area),
                                         center=center, est_distance_m=distance))
        return detections

    @staticmethod
    def draw(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x, y, w, h = det.bbox
            color = (255, 200, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{det.est_distance_m:.1f} m"
            cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return frame
