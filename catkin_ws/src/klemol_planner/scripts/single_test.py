#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import List

import cv2
import numpy as np
import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.yolo_module import YoloDetection, YoloObjectDetector, print_detections
from vlm_yolo_dynamic_demo import RRTGroundedExecutor


def default_weights_path() -> str:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[5]
    candidates = [
        package_root / "models" / "best.pt",
        package_root / "models" / "yolov8n.pt",
        Path("/home/haochenhe/YOLO_test/runs/detect/three_objects/weights/best.pt"),
        repo_root / "YOLO_test" / "runs" / "detect" / "runs" / "detect" / "three_objects" / "weights" / "best.pt",
        repo_root / "external" / "YOLO_test" / "runs" / "detect" / "three_objects" / "weights" / "best.pt",
        Path("/home/haochenhe/YOLO_test/yolov8n.pt"),
        repo_root / "external" / "YOLO_test" / "yolov8n.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-object YOLO + RRT pick test for the ROS1 Panda stack."
    )
    parser.add_argument("--weights", default=default_weights_path(), help="Ultralytics YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--class-name",
        default="",
        help="Optional object class/object_id to pick. If omitted, picks the highest-confidence detection.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually move the robot and close the gripper.")
    parser.add_argument(
        "--hover-only",
        action="store_true",
        help="With --execute, only move to a point above the detected object; do not pick or place.",
    )
    parser.add_argument("--hover-height", type=float, default=0.10, help="Height above the detected object for --hover-only.")
    parser.add_argument("--skip-place", action="store_true", help="Only pick and lift; do not place the object back.")
    parser.add_argument("--planner", default="rrt_with_connecting", choices=["rrt_with_connecting"])
    parser.add_argument("--post-processing", default="quintic_polynomial", choices=["quintic_polynomial"])
    parser.add_argument(
        "--calibration",
        default="fixed",
        choices=["fixed", "aruco"],
        help="Use fixed camera-to-base calibration by default, or recompute it from ArUco markers.",
    )
    parser.add_argument("--approach-height", type=float, default=0.12, help="Vertical approach offset in meters.")
    parser.add_argument("--grasp-height-offset", type=float, default=0.02, help="Offset above detected object for grasp.")
    parser.add_argument(
        "--debug-image",
        default="auto",
        help="Path for the annotated YOLO debug image. Use 'auto' for a timestamped debug_images file or an empty string to disable saving.",
    )
    parser.add_argument("--show-image", action="store_true", help="Show the annotated YOLO image in an OpenCV window.")
    return parser.parse_args()


def choose_detection(detections: List[YoloDetection], class_name: str = "") -> YoloDetection:
    if not detections:
        raise RuntimeError("No YOLO detections found.")

    if class_name:
        requested = normalize_name(class_name)
        matches = [
            det
            for det in detections
            if normalize_name(det.object_id) == requested or normalize_name(det.class_name) == requested
        ]
        if not matches:
            available = ", ".join(det.object_id for det in detections)
            raise RuntimeError(f"Requested object '{class_name}' was not detected. Available: {available}")
        return max(matches, key=lambda det: det.confidence)

    return max(detections, key=lambda det: det.confidence)


def detection_to_base_point(detection: YoloDetection, panda_transformations: PandaTransformations) -> PointWithOrientation:
    if detection.position_camera is None:
        raise RuntimeError(
            f"Detection '{detection.object_id}' has no 3D position. "
            "Check RealSense depth alignment and camera intrinsics."
        )

    x, y, z = detection.position_camera
    point_camera = PointWithOrientation(
        x=x,
        y=y,
        z=z,
        roll=0.0,
        pitch=0.0,
        yaw=detection.yaw_rad or 0.0,
    )
    return panda_transformations.transform_point(point_camera, "camera", "base")


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def depth_at(depth_frame, x: int, y: int):
    if depth_frame is None:
        return None
    if hasattr(depth_frame, "get_distance"):
        return float(depth_frame.get_distance(x, y))

    depth_array = np.asarray(depth_frame)
    height, width = depth_array.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        return None
    depth = float(depth_array[y, x])
    if depth > 20.0:
        depth *= 0.001
    return depth


def write_debug_image(
    color_image,
    depth_frame,
    detections: List[YoloDetection],
    selected: YoloDetection,
    output_path: str,
    show_image: bool,
) -> None:
    if not output_path and not show_image:
        return

    image = color_image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round((y1 + y2) * 0.5))
        depth = depth_at(depth_frame, cx, cy)
        is_selected = detection.object_id == selected.object_id
        color = (0, 255, 0) if is_selected else (255, 180, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
        label = f"{detection.object_id} {detection.confidence:.2f}"
        depth_label = "depth=None" if depth is None else f"depth={depth:.3f}m"
        cv2.putText(image, label, (x1, max(20, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(image, depth_label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(image, f"center=({cx},{cy})", (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    if output_path:
        if output_path == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output = PACKAGE_ROOT / "debug_images" / f"single_test_yolo_debug_{timestamp}.png"
        else:
            output = Path(output_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), image)
        print(f"[SINGLE_TEST] debug_image={output}")

    if show_image:
        cv2.imshow("single_test YOLO debug", image)
        cv2.waitKey(0)
        cv2.destroyWindow("single_test YOLO debug")


def main() -> None:
    args = parse_args()
    rospy.init_node("single_test", anonymous=True)

    camera_operations = CameraOperations()
    panda_transformations = PandaTransformations(cam_operations=camera_operations)
    if args.calibration == "fixed":
        panda_transformations.use_fixed_camera_calibration()
    else:
        panda_transformations.calibrate_camera()

    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)

    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detections = detector.detect(color_image=color_image, depth_frame=depth_frame, intrinsics=intrinsics)
    print_detections(detections)

    selected = choose_detection(detections, args.class_name)
    write_debug_image(
        color_image=color_image,
        depth_frame=depth_frame,
        detections=detections,
        selected=selected,
        output_path=args.debug_image,
        show_image=args.show_image,
    )
    object_point_base = detection_to_base_point(selected, panda_transformations)
    print(f"[SINGLE_TEST] selected={selected.object_id} class={selected.class_name} conf={selected.confidence:.3f}")
    print(f"[SINGLE_TEST] base_point={object_point_base}")

    if not args.execute:
        print("[DRY-RUN] Detection and base-frame grounding succeeded. Re-run with --execute to pick.")
        return

    executor = RRTGroundedExecutor(args.planner, args.post_processing)
    if args.hover_only:
        executor.execute_hover(
            object_id=selected.object_id,
            object_point=object_point_base,
            hover_height=args.hover_height,
        )
        print(f"[SINGLE_TEST] hover-only finished at {args.hover_height:.3f} m above {selected.object_id}")
        return

    executor.execute_pick(
        object_id=selected.object_id,
        object_point=object_point_base,
        approach_height=args.approach_height,
        grasp_height_offset=args.grasp_height_offset,
    )
    if args.skip_place:
        print("[SINGLE_TEST] pick sequence finished; skipping place")
        return

    executor.execute_place(
        object_id=selected.object_id,
        target_id=f"{selected.object_id}_original_position",
        target_point=object_point_base,
        approach_height=args.approach_height,
        place_height_offset=args.grasp_height_offset,
    )
    print("[SINGLE_TEST] pick-and-place-back sequence finished")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
