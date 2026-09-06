"""
adas_pipeline.py
-----------------
Orchestrates all ADAS subsystems on a video source (file or camera index)
and produces an annotated output video plus a per-frame event log.

Usage:
    python3 adas_pipeline.py --input road.mp4 --output annotated.mp4
    python3 adas_pipeline.py --input 0   # webcam
"""

import argparse
import time
import cv2

from lane_detection import LaneDetector
from object_detection import VehicleDetector
from collision_warning import CollisionWarningSystem
from cruise_control import AdaptiveCruiseControl


class ADASSystem:
    def __init__(self, fps: float = 30.0, set_speed_kph: float = 90.0):
        dt = 1.0 / fps
        self.lane_detector = LaneDetector()
        self.vehicle_detector = VehicleDetector()
        self.fcw = CollisionWarningSystem(dt=dt)
        self.acc = AdaptiveCruiseControl(set_speed_mps=set_speed_kph / 3.6, dt=dt)
        self.frame_idx = 0
        self.log = []

    def _closest_detection(self, detections):
        if not detections:
            return None
        return min(detections, key=lambda d: d.est_distance_m)

    def process_frame(self, frame):
        h, w = frame.shape[:2]

        lane_result = self.lane_detector.process(frame)
        detections = self.vehicle_detector.detect(frame)
        lead = self._closest_detection(detections)
        lead_distance = lead.est_distance_m if lead else None

        fcw_result = self.fcw.update(lead_distance)
        acc_result = self.acc.update(lead_distance, fcw_result.closing_speed_mps)

        frame = LaneDetector.draw(frame, lane_result)
        frame = VehicleDetector.draw(frame, detections)
        frame = self._draw_hud(frame, lane_result, fcw_result, acc_result)

        self.log.append({
            "frame": self.frame_idx,
            "lane_offset_m": lane_result.vehicle_offset_m,
            "lane_departure": lane_result.departure_warning,
            "lead_distance_m": fcw_result.distance_m,
            "ttc_s": fcw_result.ttc_s,
            "fcw_level": fcw_result.warning_level,
            "acc_mode": acc_result.mode,
            "acc_target_speed_kph": acc_result.target_speed_mps * 3.6,
        })
        self.frame_idx += 1
        return frame

    def _draw_hud(self, frame, lane_result, fcw_result, acc_result):
        h, w = frame.shape[:2]
        y = h - 20
        speed_txt = f"ACC: {acc_result.mode.upper()}  target {acc_result.target_speed_mps*3.6:5.1f} km/h"
        cv2.putText(frame, speed_txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if fcw_result.distance_m is not None:
            fcw_txt = f"Lead: {fcw_result.distance_m:4.1f} m"
            if fcw_result.ttc_s is not None:
                fcw_txt += f"  TTC: {fcw_result.ttc_s:4.1f} s"
            color = {"none": (255, 255, 255), "caution": (0, 255, 255),
                     "warning": (0, 140, 255), "critical": (0, 0, 255)}[fcw_result.warning_level]
            cv2.putText(frame, fcw_txt, (20, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if fcw_result.warning_level in ("warning", "critical"):
                label = "COLLISION WARNING" if fcw_result.warning_level == "warning" else "BRAKE NOW"
                cv2.putText(frame, label, (w // 2 - 160, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, color, 3)
        return frame

    def run(self, input_source, output_path=None, show_progress=True):
        cap = cv2.VideoCapture(int(input_source) if str(input_source).isdigit() else input_source)
        if not cap.isOpened():
            raise IOError(f"Could not open video source: {input_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.process_frame(frame)
            if writer:
                writer.write(annotated)
            if show_progress and total_frames > 0 and self.frame_idx % 30 == 0:
                pct = 100.0 * self.frame_idx / total_frames
                print(f"  processed {self.frame_idx}/{total_frames} frames ({pct:.0f}%)")

        cap.release()
        if writer:
            writer.release()
        elapsed = time.time() - start
        print(f"Done: {self.frame_idx} frames in {elapsed:.1f}s "
              f"({self.frame_idx / max(elapsed, 1e-6):.1f} fps)")
        return self.log


def main():
    parser = argparse.ArgumentParser(description="Run the ADAS pipeline on a video file or webcam.")
    parser.add_argument("--input", required=True, help="Path to video file, or webcam index (e.g. 0)")
    parser.add_argument("--output", default="adas_output.mp4", help="Path to save annotated video")
    parser.add_argument("--set-speed-kph", type=float, default=90.0, help="ACC driver-set speed (km/h)")
    args = parser.parse_args()

    system = ADASSystem(set_speed_kph=args.set_speed_kph)
    system.run(args.input, args.output)


if __name__ == "__main__":
    main()
