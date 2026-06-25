from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klemol_planner.vlm_yolo.yolo_module import YoloDetection


ROUND_OBJECTS = {"tomato_soup_can"}
LONG_OBJECTS = {"salt_box", "cleaner_bottle"}
SALT_BOX_OBJECTS = {"salt_box"}
CUBE_OBJECTS = {"orange_cube", "yellow_cube"}
DEFAULT_GRIPPER_YAW_OFFSET_DEG = -45.0


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def target_yaw_from_detection(
    detection: "YoloDetection",
    default_yaw: float = 0.0,
    gripper_yaw_offset: float = math.radians(DEFAULT_GRIPPER_YAW_OFFSET_DEG),
) -> float:
    name = normalize_name(detection.class_name or detection.object_id)
    raw_yaw = normalize_angle(detection.yaw_rad if detection.yaw_rad is not None else default_yaw)

    if name in ROUND_OBJECTS:
        return normalize_angle(gripper_yaw_offset)

    perpendicular_yaw = normalize_angle(raw_yaw - math.pi * 0.5)
    if name in CUBE_OBJECTS:
        candidates = [normalize_angle(raw_yaw + k * math.pi * 0.5 + gripper_yaw_offset) for k in range(-2, 3)]
        return min(candidates, key=lambda yaw: abs(normalize_angle(yaw)))

    if name in SALT_BOX_OBJECTS:
        return normalize_angle(perpendicular_yaw + gripper_yaw_offset)

    if name in LONG_OBJECTS:
        return normalize_angle(perpendicular_yaw + gripper_yaw_offset)

    return normalize_angle(perpendicular_yaw + gripper_yaw_offset)


def requires_yolo_yaw(detection: "YoloDetection") -> bool:
    name = normalize_name(detection.class_name or detection.object_id)
    return name not in ROUND_OBJECTS


def yaw_policy_label(detection: "YoloDetection") -> str:
    name = normalize_name(detection.class_name or detection.object_id)
    if name in ROUND_OBJECTS:
        return "round_fixed_0_plus_offset"
    if name in CUBE_OBJECTS:
        return "cube_90deg_equivalent_nearest_zero_after_offset"
    if name in SALT_BOX_OBJECTS:
        return "salt_box_raw_minus_90deg_plus_offset"
    if name in LONG_OBJECTS:
        return "long_raw_minus_90deg_plus_offset"
    return "default_raw_minus_90deg_plus_offset"
