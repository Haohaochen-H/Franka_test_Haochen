from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from klemol_planner.vlm_yolo.yolo_module import YoloDetection


ROUND_OBJECTS = {"tomato_soup_can", "tomato_can_small"}
LONG_OBJECTS = {"salt_box", "cleaner_bottle"}
CUBE_OBJECTS = {"orange_cube", "yellow_cube"}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def target_yaw_from_detection(detection: "YoloDetection", default_yaw: float = 0.0) -> float:
    name = normalize_name(detection.class_name or detection.object_id)
    raw_yaw = normalize_angle(detection.yaw_rad if detection.yaw_rad is not None else default_yaw)

    if name in ROUND_OBJECTS:
        return 0.0

    perpendicular_yaw = normalize_angle(raw_yaw - math.pi * 0.5)
    if name in CUBE_OBJECTS:
        return raw_yaw if abs(raw_yaw) <= abs(perpendicular_yaw) else perpendicular_yaw

    if name in LONG_OBJECTS:
        return perpendicular_yaw

    return perpendicular_yaw


def requires_yolo_yaw(detection: "YoloDetection") -> bool:
    name = normalize_name(detection.class_name or detection.object_id)
    return name not in ROUND_OBJECTS


def yaw_policy_label(detection: "YoloDetection") -> str:
    name = normalize_name(detection.class_name or detection.object_id)
    if name in ROUND_OBJECTS:
        return "round_fixed_0"
    if name in CUBE_OBJECTS:
        return "cube_min_abs(raw,raw_minus_90deg)"
    if name in LONG_OBJECTS:
        return "long_raw_minus_90deg"
    return "default_raw_minus_90deg"
