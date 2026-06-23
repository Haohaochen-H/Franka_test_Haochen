from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Optional

from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.table_height import (
    TABLE_Z_BASE_OFFSET,
    object_height_for_object,
    table_z_for_object,
    target_z_from_table_height,
)
from klemol_planner.vlm_yolo.yaw_policy import DEFAULT_GRIPPER_YAW_OFFSET_DEG, target_yaw_from_detection, yaw_policy_label
from klemol_planner.vlm_yolo.yolo_module import YoloDetection


TABLE_CENTER_BASE_X = 0.3900
TABLE_CENTER_BASE_Y = -0.0048


def build_scene_objects(
    detections: list[YoloDetection],
    panda_transformations,
    xy_source: str = "pixel",
    table_z: Optional[float] = None,
    approach_height: float = 0.20,
    grasp_height_offset: float = 0.0,
    gripper_yaw_offset: float = math.radians(DEFAULT_GRIPPER_YAW_OFFSET_DEG),
) -> list[dict[str, Any]]:
    scene_objects = []
    for detection in detections:
        base_pose = detection_to_base_pose(detection, panda_transformations, gripper_yaw_offset)
        object_table_z = table_z_for_object(
            class_name=detection.class_name,
            object_id=detection.object_id,
            override_table_z=table_z,
        )
        object_height = object_height_for_object(
            class_name=detection.class_name,
            object_id=detection.object_id,
        )
        if xy_source == "pixel":
            if detection.center_pixel is None:
                raise ValueError(f"Detection {detection.object_id} has no center pixel.")
            base_pose.x, base_pose.y = pixel_to_base_xy(detection.center_pixel)
        if object_table_z is not None:
            base_pose.z = target_z_from_table_height(object_table_z)

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
                "gripper_yaw_offset_deg": round_float(math.degrees(gripper_yaw_offset)),
                "base_pose": point_to_dict(base_pose),
                "pre_grasp_pose": point_to_dict(pre_grasp_pose),
                "grasp_pose": point_to_dict(grasp_pose),
                "lift_pose": point_to_dict(lift_pose),
                "table_z": round_float(object_table_z),
                "object_height": round_float(object_height),
                "table_offset": round_float(TABLE_Z_BASE_OFFSET),
            }
        )
    scene_objects.append(build_table_center_scene_object())
    return scene_objects


def build_table_center_scene_object() -> dict[str, Any]:
    base_pose = {
        "x": round_float(TABLE_CENTER_BASE_X),
        "y": round_float(TABLE_CENTER_BASE_Y),
        "z": round_float(TABLE_Z_BASE_OFFSET),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    return {
        "object_id": "table_center",
        "class_name": "table_center",
        "scene_role": "place_location",
        "place_mode": "on_table",
        "base_pose": base_pose,
        "pre_grasp_pose": base_pose,
        "grasp_pose": base_pose,
        "lift_pose": base_pose,
        "table_z": 0.0,
        "object_height": 0.0,
        "table_offset": round_float(TABLE_Z_BASE_OFFSET),
    }


def detection_to_base_pose(
    detection: YoloDetection,
    panda_transformations,
    gripper_yaw_offset: float = math.radians(DEFAULT_GRIPPER_YAW_OFFSET_DEG),
) -> PointWithOrientation:
    if detection.position_camera is None:
        raise ValueError(f"Detection '{detection.object_id}' has no 3D camera position.")
    x, y, z = detection.position_camera
    point_camera = PointWithOrientation(x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=0.0)
    point_base = panda_transformations.transform_point(point_camera, "camera", "base")
    point_base.yaw = target_yaw_from_detection(detection, gripper_yaw_offset=gripper_yaw_offset)
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


def build_executor_steps(plan: list[dict[str, Any]], scene_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scene_by_name = {}
    for scene_object in scene_objects:
        scene_by_name[normalize_name(scene_object["object_id"])] = scene_object
        scene_by_name[normalize_name(scene_object["class_name"])] = scene_object

    held_object_id = ""
    held_object_point = None
    held_object_height = 0.0
    steps = []
    for step in plan:
        action = str(step.get("action", "")).strip().lower()
        if action == "pick":
            source = scene_by_name[normalize_name(step["target"])]
            held_object_id = source["object_id"]
            held_object_point = source["base_pose"]
            held_object_height = float(source.get("object_height") or 0.0)
            steps.append(
                {
                    "skill": "pick",
                    "object_id": held_object_id,
                    "target_id": "",
                    "object_point_base": held_object_point,
                    "target_point_base": None,
                }
            )
        elif action == "place":
            target = scene_by_name[normalize_name(step["target_object"])]
            target_point = dict(target["base_pose"])
            stacking_height_offset = 0.0 if target.get("place_mode") in {"inside", "on_table", "location"} else held_object_height
            target_point["z"] = round_float(float(target_point["z"]) + stacking_height_offset)
            steps.append(
                {
                    "skill": "place",
                    "object_id": held_object_id,
                    "target_id": target["object_id"],
                    "object_point_base": held_object_point,
                    "target_point_base": target_point,
                    "stacking_height_offset": round_float(stacking_height_offset),
                    "place_mode": target.get("place_mode", "on_top"),
                }
            )
            held_object_id = ""
            held_object_point = None
            held_object_height = 0.0
        else:
            raise ValueError(f"Unsupported executor action: {step}")
    return steps


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")
