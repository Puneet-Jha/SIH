"""
demo.py
--------
End-to-end demo: generates a synthetic road video, runs the full ADAS
pipeline (lane detection, vehicle detection, forward collision warning,
adaptive cruise control) over it, saves an annotated output video, and
prints a summary of key safety events.
"""

from synthetic_road import generate_synthetic_video
from adas_pipeline import ADASSystem


def main():
    print("Generating synthetic road video...")
    src = generate_synthetic_video("synthetic_road.mp4", n_frames=300)

    print("Running ADAS pipeline...")
    system = ADASSystem(fps=30.0, set_speed_kph=90.0)
    log = system.run(src, "adas_output.mp4")

    print("\n--- Summary ---")
    fcw_events = [e for e in log if e["fcw_level"] in ("warning", "critical")]
    departures = [e for e in log if e["lane_departure"]]
    print(f"Total frames processed: {len(log)}")
    print(f"Lane departure warnings: {len(departures)} frames")
    print(f"Forward collision warnings (warning/critical): {len(fcw_events)} frames")

    if fcw_events:
        first = fcw_events[0]
        print(f"  First FCW event at frame {first['frame']}: "
              f"distance={first['lead_distance_m']:.1f}m, TTC={first['ttc_s']:.2f}s, "
              f"level={first['fcw_level']}")

    following_frames = [e for e in log if e["acc_mode"] == "following"]
    print(f"ACC in 'following' mode for {len(following_frames)} frames "
          f"({100*len(following_frames)/len(log):.0f}%)")


if __name__ == "__main__":
    main()
