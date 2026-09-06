"""
synthetic_road.py
------------------
Generates a synthetic "dashcam" road video: a straight/gently-curving
two-lane road with lane markings, a lead vehicle that varies its distance
(including a hard-braking event to trigger FCW), and mild camera-shake
noise. This makes the whole ADAS pipeline runnable and demonstrable
without needing a real dashcam video file.
"""

import cv2
import numpy as np


def generate_synthetic_video(
    path: str,
    n_frames: int = 300,
    width: int = 960,
    height: int = 540,
    fps: int = 30,
):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    horizon_y = int(height * 0.55)
    lane_half_width_bottom = 260
    lane_half_width_top = 40
    center_x = width // 2

    # lead vehicle: distance in meters, oscillates then hard-brakes (gets close fast)
    for i in range(n_frames):
        frame = np.full((height, width, 3), (60, 60, 60), dtype=np.uint8)  # road/asphalt
        # sky
        cv2.rectangle(frame, (0, 0), (width, horizon_y), (180, 140, 100), -1)

        # slight horizontal wobble to simulate camera/road curvature
        wobble = int(15 * np.sin(i / 40.0))

        # lane lines (left, center-dashed, right)
        left_bottom = (center_x - lane_half_width_bottom + wobble, height)
        left_top = (center_x - lane_half_width_top + wobble, horizon_y)
        right_bottom = (center_x + lane_half_width_bottom + wobble, height)
        right_top = (center_x + lane_half_width_top + wobble, horizon_y)

        cv2.line(frame, left_bottom, left_top, (255, 255, 255), 8)
        cv2.line(frame, right_bottom, right_top, (255, 255, 255), 8)

        # dashed center line
        for t in np.linspace(0, 1, 12):
            y = int(height - t * (height - horizon_y))
            x = int(center_x + wobble - lane_half_width_bottom * 0 + (lane_half_width_top - lane_half_width_bottom) * t)
            if int(t * 12) % 2 == 0:
                cv2.line(frame, (x, y), (x, y + 10), (255, 255, 255), 4)

        # --- lead vehicle distance profile (meters) ---
        if i < 100:
            distance_m = 40 - i * 0.15          # slowly closing
        elif i < 160:
            distance_m = 25 - (i - 100) * 0.35  # hard braking event -> rapid close-in (triggers FCW)
        elif i < 220:
            distance_m = max(4.0, 4 + (i - 160) * 0.05)  # recovers / pulls away slowly
        else:
            distance_m = 7 + (i - 220) * 0.25

        distance_m = max(3.0, distance_m)

        # project a simple vehicle box based on distance (closer = bigger, lower in frame)
        focal_length_px = 700.0
        car_width_m = 1.8
        pixel_width = int((car_width_m * focal_length_px) / distance_m)
        pixel_width = min(pixel_width, width - 40)
        pixel_height = int(pixel_width * 0.7)
        cx = center_x + wobble
        cy = horizon_y + int((height - horizon_y) * (1 - min(distance_m / 45.0, 1.0)))
        x0, y0 = cx - pixel_width // 2, cy - pixel_height
        x1, y1 = cx + pixel_width // 2, cy
        y1 = min(y1, height - 5)
        y0 = max(y0, horizon_y + 5)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 200), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (10, 10, 120), 3)
        # taillights
        cv2.circle(frame, (x0 + 8, y1 - 8), 5, (0, 0, 255), -1)
        cv2.circle(frame, (x1 - 8, y1 - 8), 5, (0, 0, 255), -1)

        # mild sensor noise
        noise = np.random.randint(0, 12, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)

        writer.write(frame)

    writer.release()
    return path


if __name__ == "__main__":
    out = generate_synthetic_video("synthetic_road.mp4")
    print(f"Wrote {out}")
