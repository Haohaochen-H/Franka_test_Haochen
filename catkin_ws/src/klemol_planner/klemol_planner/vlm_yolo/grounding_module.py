from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.yolo_module import YoloDetection


@dataclass(frozen=True)
class GroundedStep:
    skill: str
    object_id: str
    target_id: str
    object_point_base: Optional[PointWithOrientation]
    target_point_base: Optional[PointWithOrientation]


class PlanGrounder:
    def __init__(self, panda_transformations) -> None:
        self.panda_transformations = panda_transformations

    def ground(self, plan: list[dict[str, str]], detections: list[YoloDetection]) -> list[GroundedStep]:
        held_object_id = ""
        held_object_point = None
        grounded: list[GroundedStep] = []

        for step in plan:
            action = step["action"].lower()
            if action == "pick":
                det = self._find_detection(step["target"], detections)
                object_point = self._detection_to_base_point(det)
                grounded.append(
                    GroundedStep(
                        skill="pick",
                        object_id=det.object_id,
                        target_id="",
                        object_point_base=object_point,
                        target_point_base=None,
                    )
                )
                held_object_id = det.object_id
                held_object_point = object_point
            elif action == "place":
                det = self._find_detection(step["target_object"], detections)
                target_point = self._detection_to_base_point(det)
                grounded.append(
                    GroundedStep(
                        skill="place",
                        object_id=held_object_id,
                        target_id=det.object_id,
                        object_point_base=held_object_point,
                        target_point_base=target_point,
                    )
                )
                held_object_id = ""
                held_object_point = None
            else:
                raise ValueError(f"Unsupported action: {step}")
        return grounded

    def _find_detection(self, name: str, detections: list[YoloDetection]) -> YoloDetection:
        normalized = normalize_name(name)
        for det in detections:
            if normalize_name(det.object_id) == normalized:
                return det
        for det in detections:
            if normalize_name(det.class_name) == normalized:
                return det
        available = ", ".join(det.object_id for det in detections) or "none"
        raise ValueError(f"Object '{name}' not detected. Available: {available}")

    def _detection_to_base_point(self, detection: YoloDetection) -> PointWithOrientation:
        if detection.position_camera is None:
            raise ValueError(f"Detection '{detection.object_id}' has no 3D camera position.")
        x, y, z = detection.position_camera
        yaw = detection.yaw_rad or 0.0
        point_camera = PointWithOrientation(x=x, y=y, z=z, roll=0.0, pitch=0.0, yaw=yaw)
        return self.panda_transformations.transform_point(point_camera, "camera", "base")


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")
