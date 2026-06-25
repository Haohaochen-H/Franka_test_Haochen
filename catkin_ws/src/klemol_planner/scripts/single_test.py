#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import List, Optional

import cv2
import numpy as np
import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.box_corner import (
    DEFAULT_BOX_BASE_X,
    DEFAULT_BOX_BASE_Y,
    DEFAULT_BOX_TABLE_Z,
    DEFAULT_TOP_DOWN_PITCH,
    DEFAULT_TOP_DOWN_ROLL,
    DEFAULT_TOP_DOWN_YAW,
    box_pose_from_image,
    parse_id_list,
)
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.table_height import (
    TABLE_Z_BASE_OFFSET,
    place_z_for_object,
    table_z_for_object,
    target_z_from_table_height,
)
from klemol_planner.vlm_yolo.yaw_policy import DEFAULT_GRIPPER_YAW_OFFSET_DEG, target_yaw_from_detection
from klemol_planner.vlm_yolo.yolo_module import YoloDetection, YoloObjectDetector, print_detections
from vlm_yolo_dynamic_demo import RRTGroundedExecutor


def default_weights_path() -> str:
    package_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[5]
    candidates = [
        package_root / "models" / "best.pt",
        package_root / "models" / "yolov8n.pt",
        repo_root / "YOLO_test" / "runs" / "detect" / "runs" / "detect" / "five_objects" / "weights" / "best.pt",
        repo_root / "external" / "YOLO_test" / "runs" / "detect" / "five_objects" / "weights" / "best.pt",
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
        choices=["fixed", "aruco", "aruco_legacy"],
        help="Use fixed calibration, 3D ArUco point alignment, or the legacy ArUco calibration.",
    )
    parser.add_argument("--approach-height", type=float, default=0.12, help="Vertical approach offset in meters.")
    parser.add_argument("--grasp-height-offset", type=float, default=0.02, help="Offset above detected object for grasp.")
    parser.add_argument(
        "--gripper-yaw-offset-deg",
        type=float,
        default=DEFAULT_GRIPPER_YAW_OFFSET_DEG,
        help="Fixed yaw correction for the gripper in degrees. Use 0 to disable.",
    )
    parser.add_argument(
        "--xy-source",
        default="pixel",
        choices=["transform", "pixel"],
        help="Use camera->base transform or fixed pixel->base XY homography for object XY.",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="Override object grasp/place z with this height above the table plane. Internally adds 0.17 m.",
    )
    parser.add_argument(
        "--debug-image",
        default="auto",
        help="Path for the annotated YOLO debug image. Use 'auto' for a timestamped debug_images file or an empty string to disable saving.",
    )
    parser.add_argument("--show-image", action="store_true", help="Show the annotated YOLO image in an OpenCV window.")
    parser.add_argument("--box-hover", action="store_true", help="Move above the fixed box coordinate instead of running YOLO pick.")
    parser.add_argument("--box-detect", action="store_true", help="Detect box ArUco marker instead of using the fixed box coordinate.")
    parser.add_argument("--box-x", type=float, default=DEFAULT_BOX_BASE_X, help="Fixed box base-frame x coordinate.")
    parser.add_argument("--box-y", type=float, default=DEFAULT_BOX_BASE_Y, help="Fixed box base-frame y coordinate.")
    parser.add_argument("--box-id", type=int, default=None, help="Optional ArUco ID for the box marker when --box-detect is used.")
    parser.add_argument("--box-table-ids", default="0,1,2,3", help="Comma-separated ArUco IDs for table corner markers when --box-detect is used.")
    parser.add_argument("--box-marker-point", default="center", choices=["center", "tl", "tr", "br", "bl"], help="Box marker point to convert to base XY when --box-detect is used.")
    parser.add_argument("--box-table-z", type=float, default=DEFAULT_BOX_TABLE_Z, help="Box target Z above table plane; base z adds TABLE_Z_BASE_OFFSET.")
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


def detection_to_base_point(
    detection: YoloDetection,
    panda_transformations: PandaTransformations,
    gripper_yaw_offset: float,
) -> PointWithOrientation:
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
        yaw=0.0,
    )
    point_base = panda_transformations.transform_point(point_camera, "camera", "base")
    point_base.yaw = target_yaw_from_detection(detection, gripper_yaw_offset=gripper_yaw_offset)
    return point_base


def apply_planar_overrides(
    point_base: PointWithOrientation,
    detection: YoloDetection,
    xy_source: str,
    table_z: Optional[float],
) -> PointWithOrientation:
    if xy_source == "pixel":
        if detection.center_pixel is None:
            raise RuntimeError("Selected object has no center pixel for pixel XY transform.")
        point_base.x, point_base.y = pixel_to_base_xy(detection.center_pixel)

    if table_z is not None:
        point_base.z = target_z_from_table_height(table_z)

    return point_base


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
    log_prefix: str = "[SINGLE_TEST]",
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
        print(f"{log_prefix} debug_image={output}")

    if show_image:
        cv2.imshow("single_test YOLO debug", image)
        cv2.waitKey(0)
        cv2.destroyWindow("single_test YOLO debug")


def main() -> None:
    args = parse_args()
    rospy.init_node("single_test", anonymous=True)

    if args.box_hover and not args.box_detect:
        box_pose = PointWithOrientation(
            x=args.box_x,
            y=args.box_y,
            z=target_z_from_table_height(args.box_table_z),
            roll=DEFAULT_TOP_DOWN_ROLL,
            pitch=DEFAULT_TOP_DOWN_PITCH,
            yaw=DEFAULT_TOP_DOWN_YAW,
        )
        print(f"[SINGLE_TEST][BOX] coordinate_source=fixed base_pose={box_pose}")
        print(
            f"[SINGLE_TEST][BOX] table_offset={TABLE_Z_BASE_OFFSET:.3f} "
            f"box_table_z={args.box_table_z:.3f} hover_height={args.hover_height:.3f}"
        )
        if not args.execute:
            print("[DRY-RUN] Fixed box coordinate ready. Re-run with --execute --box-hover to move above it.")
            return
        executor = RRTGroundedExecutor(args.planner, args.post_processing)
        executor.execute_hover(
            object_id="box_fixed",
            object_point=box_pose,
            hover_height=args.hover_height,
        )
        print("[SINGLE_TEST][BOX] hover finished above fixed box coordinate")
        return

    camera_operations = CameraOperations()
    panda_transformations = PandaTransformations(cam_operations=camera_operations)
    if args.calibration == "fixed":
        panda_transformations.use_fixed_camera_calibration()
    elif args.calibration == "aruco":
        panda_transformations.calibrate_camera_from_aruco_3d()
    else:
        panda_transformations.calibrate_camera()

    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)

    if args.box_hover and args.box_detect:
        box_id, box_pixel, box_pose, marker_by_id = box_pose_from_image(
            image=color_image,
            table_ids=parse_id_list(args.box_table_ids),
            box_id=args.box_id,
            marker_point=args.box_marker_point,
            table_z=args.box_table_z,
        )
        print(f"[SINGLE_TEST][BOX] detected_marker_ids={sorted(marker_by_id.keys())}")
        print(f"[SINGLE_TEST][BOX] selected_id={box_id} marker_point={args.box_marker_point} pixel={box_pixel}")
        print(f"[SINGLE_TEST][BOX] base_pose={box_pose}")
        print(
            f"[SINGLE_TEST][BOX] table_offset={TABLE_Z_BASE_OFFSET:.3f} "
            f"box_table_z={args.box_table_z:.3f} hover_height={args.hover_height:.3f}"
        )
        if not args.execute:
            print("[DRY-RUN] Box coordinate grounding succeeded. Re-run with --execute --box-hover to move above it.")
            return
        executor = RRTGroundedExecutor(args.planner, args.post_processing)
        executor.execute_hover(
            object_id=f"box_marker_{box_id}",
            object_point=box_pose,
            hover_height=args.hover_height,
        )
        print(f"[SINGLE_TEST][BOX] hover finished above box marker {box_id}")
        return

    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detector_depth_frame = None if args.xy_source == "pixel" else depth_frame
    detector_intrinsics = None if args.xy_source == "pixel" else intrinsics
    detections = detector.detect(color_image=color_image, depth_frame=detector_depth_frame, intrinsics=detector_intrinsics)
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
    gripper_yaw_offset = float(np.deg2rad(args.gripper_yaw_offset_deg))
    pick_table_z = table_z_for_object(selected.class_name, selected.object_id, override_table_z=args.table_z)
    place_table_z = args.table_z if args.table_z is not None else place_z_for_object(selected.class_name, selected.object_id)
    if args.xy_source == "pixel" and pick_table_z is not None:
        if selected.center_pixel is None:
            raise RuntimeError("Selected object has no center pixel for pixel XY transform.")
        object_x, object_y = pixel_to_base_xy(selected.center_pixel)
        object_point_base = PointWithOrientation(
            x=object_x,
            y=object_y,
            z=target_z_from_table_height(pick_table_z),
            roll=DEFAULT_TOP_DOWN_ROLL,
            pitch=DEFAULT_TOP_DOWN_PITCH,
            yaw=target_yaw_from_detection(selected, gripper_yaw_offset=gripper_yaw_offset),
        )
    else:
        object_point_base = detection_to_base_point(selected, panda_transformations, gripper_yaw_offset)
    object_point_base = apply_planar_overrides(
        object_point_base,
        detection=selected,
        xy_source=args.xy_source,
        table_z=pick_table_z,
    )
    place_point_base = PointWithOrientation(
        object_point_base.x,
        object_point_base.y,
        object_point_base.z,
        object_point_base.roll,
        object_point_base.pitch,
        object_point_base.yaw,
    )
    if place_table_z is not None:
        place_point_base.z = target_z_from_table_height(place_table_z)
    grasp_height_offset = 0.0 if pick_table_z is not None else args.grasp_height_offset
    place_height_offset = 0.0 if place_table_z is not None else args.grasp_height_offset
    print(f"[SINGLE_TEST] selected={selected.object_id} class={selected.class_name} conf={selected.confidence:.3f}")
    print(f"[SINGLE_TEST] base_point={object_point_base}")
    print(
        f"[SINGLE_TEST] xy_source={args.xy_source} "
        f"table_z={args.table_z} "
        f"resolved_pick_table_z={pick_table_z} "
        f"resolved_place_table_z={place_table_z} "
        f"table_offset={TABLE_Z_BASE_OFFSET:.3f} "
        f"gripper_yaw_offset_deg={args.gripper_yaw_offset_deg:.1f}"
    )

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
        grasp_height_offset=grasp_height_offset,
    )
    if args.skip_place:
        print("[SINGLE_TEST] pick sequence finished; skipping place")
        return

    executor.execute_place(
        object_id=selected.object_id,
        target_id=f"{selected.object_id}_original_position",
        target_point=place_point_base,
        approach_height=args.approach_height,
        place_height_offset=place_height_offset,
    )
    print("[SINGLE_TEST] pick-and-place-back sequence finished")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

