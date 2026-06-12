#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import pyrealsense2 as rs

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one RealSense frame and annotate ArUco corner camera coordinates.")
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "debug_images"), help="Directory for annotated images.")
    parser.add_argument("--marker-length", type=float, default=0.05, help="ArUco marker side length in meters.")
    parser.add_argument("--depth-radius", type=int, default=15, help="Median depth window radius around marker center.")
    parser.add_argument("--show-image", action="store_true", help="Show the annotated image in an OpenCV window.")
    return parser.parse_args()


def start_realsense():
    profiles = [
        (640, 480, 640, 480, 30),
        (640, 480, 1280, 720, 30),
        (640, 480, 1920, 1080, 30),
        (848, 480, 848, 480, 30),
    ]
    last_error = None
    for depth_w, depth_h, color_w, color_h, fps in profiles:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        try:
            profile = pipeline.start(config)
        except RuntimeError as exc:
            last_error = exc
            continue

        align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        print(f"[INFO] RealSense started: depth={depth_w}x{depth_h}@{fps}, color={color_w}x{color_h}@{fps}")
        return pipeline, align, intrinsics

    raise RuntimeError(f"No supported RealSense profile found. Last error: {last_error}")


def median_depth(depth_frame, x: int, y: int, radius: int):
    depths = []
    for yy in range(y - radius, y + radius + 1):
        for xx in range(x - radius, x + radius + 1):
            try:
                depth = float(depth_frame.get_distance(xx, yy))
            except RuntimeError:
                continue
            if 0.05 < depth < 5.0:
                depths.append(depth)
    if not depths:
        return None
    return float(np.median(depths))


def get_aligned_frame(pipeline, align):
    for _ in range(10):
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if color_frame and depth_frame:
            return np.asanyarray(color_frame.get_data()), depth_frame
    raise RuntimeError("Could not capture aligned RealSense color/depth frames.")


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, None


def detect_markers(image):
    dictionary, detector = make_aruco_detector()
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    return corners, ids, rejected


def draw_label(image, lines, origin, color):
    x, y = origin
    for idx, line in enumerate(lines):
        yy = y + idx * 18
        cv2.putText(image, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    pipeline, align, intrinsics = start_realsense()
    try:
        color_image, depth_frame = get_aligned_frame(pipeline, align)
    finally:
        pipeline.stop()

    corners, ids, _ = detect_markers(color_image)
    annotated = color_image.copy()

    if ids is None:
        print("[ARUCO] no markers detected")
    else:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        for corner, marker_id_arr in zip(corners, ids):
            marker_id = int(marker_id_arr[0])
            pts = corner.reshape(-1, 2)
            cx = int(round(float(pts[:, 0].mean())))
            cy = int(round(float(pts[:, 1].mean())))
            depth = median_depth(depth_frame, cx, cy, args.depth_radius)

            camera_xyz = None
            if depth is not None:
                camera_xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                camera_xyz = tuple(float(v) for v in camera_xyz)

            print(
                f"[ARUCO] id={marker_id} center=({cx},{cy}) "
                f"depth={None if depth is None else round(depth, 4)} camera_xyz={camera_xyz}"
            )

            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
            if camera_xyz is None:
                lines = [f"id={marker_id}", f"px=({cx},{cy})", "depth=None"]
            else:
                lines = [
                    f"id={marker_id}",
                    f"px=({cx},{cy}) d={depth:.3f}m",
                    f"cam=({camera_xyz[0]:.3f},{camera_xyz[1]:.3f},{camera_xyz[2]:.3f})",
                ]
            draw_label(annotated, lines, (cx + 8, max(18, cy - 30)), (0, 255, 255))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"aruco_corner_debug_{timestamp}.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"[ARUCO] debug_image={output_path}")

    if args.show_image:
        cv2.imshow("ArUco corner camera coordinates", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("ArUco corner camera coordinates")


if __name__ == "__main__":
    main()
