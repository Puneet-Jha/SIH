# Python ADAS (Advanced Driver Assistance System)

A modular, self-contained ADAS demo built on OpenCV + NumPy. It implements
four classic ADAS features using classical computer vision and control
theory (no external model downloads required):

| Module | File | What it does |
|---|---|---|
| Lane Detection & Lane Departure Warning | `lane_detection.py` | Canny edges + Hough transform, averaged/smoothed into left & right lane lines; warns when the vehicle drifts from lane center. |
| Vehicle/Obstacle Detection | `object_detection.py` | Background subtraction (MOG2) + contour filtering; estimates distance via a pinhole-camera model. |
| Forward Collision Warning (FCW) | `collision_warning.py` | Computes Time-To-Collision (TTC) from the closing speed of the nearest detected vehicle; raises caution/warning/critical alerts. |
| Adaptive Cruise Control (ACC) | `cruise_control.py` | PID controller that holds a driver-set speed or falls back to a safe time-gap behind a lead vehicle. |
| Orchestrator | `adas_pipeline.py` | Runs all subsystems per-frame over a video file/webcam, draws the HUD, writes an annotated output video + event log. |
| Synthetic road simulator | `synthetic_road.py` | Generates a demo road video (lanes + a lead car with a scripted hard-braking event) so the whole system runs without needing a real dashcam clip. |
| Demo | `demo.py` | Runs everything end-to-end and prints a summary. |

## Quick start

```bash
pip install opencv-python numpy

# Run the full self-contained demo (generates synthetic video + processes it)
python3 demo.py

# Or run the pipeline on your own video / webcam
python3 adas_pipeline.py --input your_dashcam.mp4 --output annotated.mp4
python3 adas_pipeline.py --input 0    # webcam
```

Output: `adas_output.mp4` (annotated video) plus per-frame metrics
(`lane_offset_m`, `ttc_s`, `fcw_level`, `acc_mode`, etc.) returned from
`ADASSystem.run()`.

## How it fits together

```
video frame
   │
   ├─► LaneDetector.process()      → lane lines, offset, departure flag
   ├─► VehicleDetector.detect()    → bounding boxes + estimated distances
   │        │
   │        └─► nearest vehicle distance
   │                 │
   │                 ├─► CollisionWarningSystem.update()  → TTC, FCW level
   │                 └─► AdaptiveCruiseControl.update()   → target speed, accel
   │
   └─► draw overlays (lane, boxes, HUD) → annotated frame
```

## Design notes / limitations (read before using on real footage)

- **Distance estimation** uses a simple pinhole-camera formula
  (`distance = known_width * focal_length / pixel_width`). For real
  cameras you must calibrate `focal_length_px` in `object_detection.py`
  against your actual camera, or replace it with stereo/LiDAR/radar
  fusion for production-grade accuracy.
- **Vehicle detection** uses background subtraction, which works well
  for a moving-background/relatively-static-camera synthetic or
  slow-traffic scene, but is not a substitute for a trained detector.
  To upgrade: swap `VehicleDetector.detect()` for a YOLOv8/ONNX model —
  it just needs to return the same `List[Detection]` shape so
  `collision_warning.py` and `adas_pipeline.py` keep working unchanged.
- **Lane detection** assumes roughly straight lane markings in the
  lower half of the frame; sharp curves, heavy occlusion, or faded
  markings will degrade it. A production system would use a learned
  lane model (e.g. a segmentation network) instead of Canny/Hough.
- This is a **demonstration/educational system**, not a certified
  safety system. Real ADAS requires sensor fusion (camera + radar/LiDAR),
  redundancy, extensive validation, and compliance with automotive
  functional-safety standards (ISO 26262) before any real-world use.

## Tuning key parameters

- `CollisionWarningSystem(caution_ttc, warning_ttc, critical_ttc)` — TTC
  thresholds (seconds) for each alert level.
- `AdaptiveCruiseControl(set_speed_mps, time_gap_s, kp, ki, kd)` — cruise
  speed, desired following time-gap, and PID gains.
- `LaneDetector(departure_threshold_m)` — how far off-center (meters)
  before a lane-departure warning fires.
