#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.vlm_yolo.yolo_module import YoloObjectDetector, print_detections
from single_test import choose_detection, default_weights_path, detection_to_base_point, write_debug_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one frame and log YOLO/object coordinate debug data.")
    parser.add_argument("--weights", default=default_weights_path(), help="Ultralytics YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--class-name", default="", help="Optional object class/object_id to select.")
    parser.add_argument(
        "--calibration",
        default="fixed",
        choices=["fixed", "aruco", "aruco_legacy"],
        help="Camera-to-base calibration source.",
    )
    parser.add_argument("--output-dir", default=str(PACKAGE_ROOT / "debug_images"), help="Directory for image/report output.")
    parser.add_argument("--show-image", action="store_true", help="Show annotated YOLO image in an OpenCV window.")
    return parser.parse_args()


def configure_calibration(transformer: PandaTransformations, calibration: str) -> None:
    if calibration == "fixed":
        transformer.use_fixed_camera_calibration()
    elif calibration == "aruco":
        transformer.calibrate_camera_from_aruco_3d()
    else:
        transformer.calibrate_camera()


def point_to_dict(point):
    return {
        "x": point.x,
        "y": point.y,
        "z": point.z,
        "roll": point.roll,
        "pitch": point.pitch,
        "yaw": point.yaw,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    camera_operations = CameraOperations()
    transformer = PandaTransformations(cam_operations=camera_operations)
    configure_calibration(transformer, args.calibration)

    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)

    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detections = detector.detect(color_image=color_image, depth_frame=depth_frame, intrinsics=intrinsics)
    print_detections(detections)

    selected = choose_detection(detections, args.class_name)
    image_path = output_dir / f"object_perception_debug_{timestamp}.png"
    write_debug_image(
        color_image=color_image,
        depth_frame=depth_frame,
        detections=detections,
        selected=selected,
        output_path=str(image_path),
        show_image=args.show_image,
    )

    object_base = detection_to_base_point(selected, transformer)
    transform_matrix = np.asarray(transformer.T_base_to_camera, dtype=float)
    report = {
        "timestamp": timestamp,
        "calibration": args.calibration,
        "transform_camera_to_base": transform_matrix.tolist(),
        "selected_object": {
            "object_id": selected.object_id,
            "class_name": selected.class_name,
            "confidence": selected.confidence,
            "bbox_xyxy": list(selected.bbox_xyxy),
            "center_pixel": list(selected.center_pixel) if selected.center_pixel else None,
            "center_depth_m": selected.center_depth_m,
            "position_camera": list(selected.position_camera) if selected.position_camera else None,
            "position_base": point_to_dict(object_base),
            "yaw_rad": selected.yaw_rad,
        },
        "debug_image": str(image_path),
    }
    report_path = output_dir / f"object_perception_debug_{timestamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[PERCEPTION_DEBUG] transform_camera_to_base=\n{np.round(transform_matrix, 4)}")
    print(f"[PERCEPTION_DEBUG] selected={selected.object_id} class={selected.class_name} conf={selected.confidence:.3f}")
    print(f"[PERCEPTION_DEBUG] center_pixel={selected.center_pixel} center_depth_m={selected.center_depth_m}")
    print(f"[PERCEPTION_DEBUG] position_camera={selected.position_camera}")
    print(f"[PERCEPTION_DEBUG] position_base={object_base}")
    print(f"[PERCEPTION_DEBUG] debug_image={image_path}")
    print(f"[PERCEPTION_DEBUG] report={report_path}")


if __name__ == "__main__":
    main()
