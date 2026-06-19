from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.table_height import TABLE_Z_BASE_OFFSET, target_z_from_table_height
from klemol_planner.vlm_yolo.yaw_policy import target_yaw_from_detection, yaw_policy_label
from klemol_planner.vlm_yolo.yolo_module import YoloDetection


def build_scene_objects(
    detections: list[YoloDetection],
    panda_transformations,
    xy_source: str = "pixel",
    table_z: Optional[float] = None,
    approach_height: float = 0.20,
    grasp_height_offset: float = 0.0,
) -> list[dict[str, Any]]:
    scene_objects = []
    for detection in detections:
        base_pose = detection_to_base_pose(detection, panda_transformations)
        if xy_source == "pixel":
            if detection.center_pixel is None:
                raise ValueError(f"Detection {detection.object_id} has no center pixel.")
            base_pose.x, base_pose.y = pixel_to_base_xy(detection.center_pixel)
        if table_z is not None:
            base_pose.z = target_z_from_table_height(table_z)

        grasp_pose = copy_pose(base_pose)
        grasp_pose.z += grasp_height_offset

        pre_grasp_pose = copy_pose(grasp_pose)
        pre_grasp_pose.z += approach_height

        lift_pose = copy_pose(grasp_pose)
        lift_pose.z += approach_height

        scene_objects.append(
            {
                "object_id": detection.object_id,
                "class_name": detection.class_name,
                "confidence": round_float(detection.confidence),
                "bbox_xyxy": list(detection.bbox_xyxy),
                "center_pixel": list(detection.center_pixel) if detection.center_pixel else None,
                "center_depth_m": round_float(detection.center_depth_m),
                "position_camera": round_sequence(detection.position_camera),
                "raw_yaw_rad": round_float(detection.yaw_rad),
                "yaw_policy": yaw_policy_label(detection),
                "base_pose": point_to_dict(base_pose),
                "pre_grasp_pose": point_to_dict(pre_grasp_pose),
                "grasp_pose": point_to_dict(grasp_pose),
                "lift_pose": point_to_dict(lift_pose),
                "table_z": round_float(table_z),
                "table_offset": round_float(TABLE_Z_BASE_OFFSET),
            }
        )
    return scene_objects


def detection_to_base_pose(detection: YoloDetection, panda_transformations) -> PointWithOrientation:
    if detection.position_camera is None:
        raise ValueError(f"Detection '{detection.object_id}' has no 3D camera position.")
    x, y, z = detection.position_camera
    point_camera = PointWithOrientation(x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=0.0)
    point_base = panda_transformations.transform_point(point_camera, "camera", "base")
    point_base.yaw = target_yaw_from_detection(detection)
    return point_base


def copy_pose(point: PointWithOrientation) -> PointWithOrientation:
    return PointWithOrientation(point.x, point.y, point.z, point.roll, point.pitch, point.yaw)


def point_to_dict(point: PointWithOrientation) -> dict[str, float]:
    return {
        "x": round_float(point.x),
        "y": round_float(point.y),
        "z": round_float(point.z),
        "roll": round_float(point.roll),
        "pitch": round_float(point.pitch),
        "yaw": round_float(point.yaw),
    }


def round_float(value: Optional[float], digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def round_sequence(values: Optional[tuple[float, ...]], digits: int = 4):
    if values is None:
        return None
    return [round(float(value), digits) for value in values]


def detections_to_jsonable(detections: list[YoloDetection]) -> list[dict[str, Any]]:
    output = []
    for detection in detections:
        data = asdict(detection)
        if detection.center_pixel is not None:
            data["center_pixel"] = list(detection.center_pixel)
        if detection.position_camera is not None:
            data["position_camera"] = list(detection.position_camera)
        return_item = dict(data)
        output.append(return_item)
    return output
