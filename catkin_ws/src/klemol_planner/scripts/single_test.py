#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

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
    parser.add_argument("--skip-place", action="store_true", help="Only pick and lift; do not place the object back.")
    parser.add_argument("--planner", default="rrt_with_connecting", choices=["rrt_with_connecting"])
    parser.add_argument("--post-processing", default="quintic_polynomial", choices=["quintic_polynomial"])
    parser.add_argument("--approach-height", type=float, default=0.12, help="Vertical approach offset in meters.")
    parser.add_argument("--grasp-height-offset", type=float, default=0.02, help="Offset above detected object for grasp.")
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


def main() -> None:
    args = parse_args()
    rospy.init_node("single_test", anonymous=True)

    camera_operations = CameraOperations()
    panda_transformations = PandaTransformations(cam_operations=camera_operations)
    panda_transformations.calibrate_camera()

    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)

    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detections = detector.detect(color_image=color_image, depth_frame=depth_frame, intrinsics=intrinsics)
    print_detections(detections)

    selected = choose_detection(detections, args.class_name)
    object_point_base = detection_to_base_point(selected, panda_transformations)
    print(f"[SINGLE_TEST] selected={selected.object_id} class={selected.class_name} conf={selected.confidence:.3f}")
    print(f"[SINGLE_TEST] base_point={object_point_base}")

    if not args.execute:
        print("[DRY-RUN] Detection and base-frame grounding succeeded. Re-run with --execute to pick.")
        return

    executor = RRTGroundedExecutor(args.planner, args.post_processing)
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
