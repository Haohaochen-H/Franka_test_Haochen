#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.yolo_module import YoloObjectDetector
from single_test import default_weights_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one RealSense frame, run YOLO, and print object pose/yaw debug data.")
    parser.add_argument("--weights", default=default_weights_path(), help="Ultralytics YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--class-name", default="", help="Optional class/object_id filter for printed selected object.")
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "debug_images"), help="Directory for annotated output images.")
    parser.add_argument("--show-image", action="store_true", help="Show annotated image in an OpenCV window.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def draw_detection(image, detection, selected: bool = False) -> None:
    x1, y1, x2, y2 = detection.bbox_xyxy
    cx, cy = detection.center_pixel or (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))
    color = (0, 255, 0) if selected else (255, 180, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

    label = f"{detection.object_id} {detection.confidence:.2f}"
    depth = "None" if detection.center_depth_m is None else f"{detection.center_depth_m:.3f}m"
    yaw = "None" if detection.yaw_rad is None else f"{np.degrees(detection.yaw_rad):.1f}deg"
    lines = [label, f"px=({cx},{cy}) d={depth}", f"yaw={yaw}"]

    if detection.yaw_rad is not None:
        axis_len = max(25, int(0.4 * max(x2 - x1, y2 - y1)))
        dx = int(axis_len * np.cos(detection.yaw_rad))
        dy = int(axis_len * np.sin(detection.yaw_rad))
        cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 0), 3)

    text_x = x1
    text_y = max(18, y1 - 48)
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (text_x, text_y + index * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    camera_operations = CameraOperations()
    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)

    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detections = detector.detect(color_image=color_image, depth_frame=depth_frame, intrinsics=intrinsics)
    if args.class_name:
        requested = normalize_name(args.class_name)
        selected_ids = {
            det.object_id
            for det in detections
            if normalize_name(det.object_id) == requested or normalize_name(det.class_name) == requested
        }
    else:
        selected_ids = set()

    annotated = color_image.copy()
    if not detections:
        print("[YOLO_POSE_DEBUG] no detections")
    for det in detections:
        base_xy = pixel_to_base_xy(det.center_pixel) if det.center_pixel is not None else None
        yaw_deg = None if det.yaw_rad is None else float(np.degrees(det.yaw_rad))
        print(
            "[YOLO_POSE_DEBUG] object_id={object_id} class={class_name} conf={confidence:.3f} "
            "bbox={bbox} center={center} depth={depth} camera_xyz={camera_xyz} "
            "base_xy_from_pixel={base_xy} yaw_rad={yaw_rad} yaw_deg={yaw_deg}".format(
                object_id=det.object_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox_xyxy,
                center=det.center_pixel,
                depth=det.center_depth_m,
                camera_xyz=det.position_camera,
                base_xy=base_xy,
                yaw_rad=det.yaw_rad,
                yaw_deg=yaw_deg,
            )
        )
        draw_detection(annotated, det, selected=det.object_id in selected_ids)

    output_path = output_dir / f"yolo_pose_debug_{timestamp}.png"
    cv2.imwrite(str(output_path), annotated)
    print(f"[YOLO_POSE_DEBUG] debug_image={output_path}")

    if args.show_image:
        cv2.imshow("YOLO pose debug", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("YOLO pose debug")


if __name__ == "__main__":
    main()
