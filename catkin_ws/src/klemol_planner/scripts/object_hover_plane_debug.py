#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.yolo_module import YoloObjectDetector, print_detections
from single_test import choose_detection, default_weights_path, detection_to_base_point, write_debug_image
from vlm_yolo_dynamic_demo import RRTGroundedExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the robot above the detected object's XY at a fixed height above the table plane."
    )
    parser.add_argument("--weights", default=default_weights_path(), help="Ultralytics YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--class-name", default="", help="Optional object class/object_id to select.")
    parser.add_argument(
        "--calibration",
        default="fixed",
        choices=["fixed", "aruco", "aruco_legacy"],
        help="Camera-to-base calibration source.",
    )
    parser.add_argument("--planner", default="rrt_with_connecting", choices=["rrt_with_connecting"])
    parser.add_argument("--post-processing", default="quintic_polynomial", choices=["quintic_polynomial"])
    parser.add_argument(
        "--xy-source",
        default="pixel",
        choices=["pixel", "transform"],
        help="Use fixed pixel->base XY homography or camera->base transform for target XY.",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=None,
        help="Direct target z in base frame. Defaults to old table height plus 0.50 m.",
    )
    parser.add_argument("--debug-image", default="auto", help="Annotated YOLO image path, 'auto', or empty string.")
    parser.add_argument("--show-image", action="store_true", help="Show annotated YOLO image in an OpenCV window.")
    return parser.parse_args()


def configure_calibration(transformer: PandaTransformations, calibration: str) -> None:
    if calibration == "fixed":
        transformer.use_fixed_camera_calibration()
    elif calibration == "aruco":
        transformer.calibrate_camera_from_aruco_3d()
    else:
        transformer.calibrate_camera()


def main() -> None:
    args = parse_args()
    rospy.init_node("object_hover_plane_debug", anonymous=True)

    camera_operations = CameraOperations()
    transformer = PandaTransformations(cam_operations=camera_operations)
    configure_calibration(transformer, args.calibration)

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

    object_base = detection_to_base_point(selected, transformer)
    if args.xy_source == "pixel":
        if selected.center_pixel is None:
            raise RuntimeError("Selected object has no center pixel for pixel XY transform.")
        target_x, target_y = pixel_to_base_xy(selected.center_pixel)
    else:
        target_x, target_y = object_base.x, object_base.y

    if args.table_z is None:
        table_height = float(sum(point[2] for point in transformer.table_corners_translations.values()) / 4.0)
        target_z = table_height + 0.50
    else:
        table_height = None
        target_z = args.table_z

    target_pose = PointWithOrientation(
        x=target_x,
        y=target_y,
        z=target_z,
        roll=object_base.roll,
        pitch=object_base.pitch,
        yaw=object_base.yaw,
    )

    print(f"[HOVER_PLANE_DEBUG] selected={selected.object_id} class={selected.class_name} conf={selected.confidence:.3f}")
    print(f"[HOVER_PLANE_DEBUG] center_pixel={selected.center_pixel} center_depth_m={selected.center_depth_m}")
    print(f"[HOVER_PLANE_DEBUG] object_base={object_base}")
    print(f"[HOVER_PLANE_DEBUG] xy_source={args.xy_source} target_xy=({target_x:.4f}, {target_y:.4f})")
    if table_height is None:
        print(f"[HOVER_PLANE_DEBUG] target_z={target_z:.4f} from --table-z")
    else:
        print(f"[HOVER_PLANE_DEBUG] target_z={target_z:.4f} from default table_height={table_height:.4f} + 0.5000")
    print(f"[HOVER_PLANE_DEBUG] target_pose={target_pose}")

    executor = RRTGroundedExecutor(args.planner, args.post_processing)
    executor.execute_move_to_pose(label=f"{selected.object_id}_xy_table_hover", target_pose=target_pose)
    print("[HOVER_PLANE_DEBUG] finished")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
