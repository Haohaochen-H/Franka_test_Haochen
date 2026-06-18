#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict
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
from klemol_planner.srv import GenerateVlmPlan, GenerateVlmPlanResponse
from klemol_planner.vlm_yolo.grounding_module import PlanGrounder
from klemol_planner.vlm_yolo.vlm_module import VlmPlanner
from klemol_planner.vlm_yolo.yolo_module import YoloObjectDetector
from single_test import default_weights_path


class Ros1VlmPlannerNode:
    def __init__(self) -> None:
        self.model_name = rospy.get_param("~model_name", "gemma3:4b")
        self.critic_model_name = rospy.get_param("~critic_model_name", self.model_name)
        self.ollama_host = rospy.get_param("~ollama_host", "http://localhost:11434")
        self.timeout = int(rospy.get_param("~timeout", 120))
        self.max_rounds = int(rospy.get_param("~max_rounds", 3))
        self.weights = rospy.get_param("~weights", default_weights_path())
        self.confidence = float(rospy.get_param("~confidence", 0.25))
        self.calibration = rospy.get_param("~calibration", "fixed")
        self.demo_mode = bool(rospy.get_param("~demo_mode", False))

        rospy.loginfo("[VLM_NODE] loading camera, YOLO, and VLM clients")
        self.camera_operations = CameraOperations()
        self.transformer = PandaTransformations(cam_operations=self.camera_operations)
        self._configure_calibration()

        self.detector = YoloObjectDetector(
            weights_path=self.weights,
            confidence_threshold=self.confidence,
        )
        self.vlm = VlmPlanner(
            model_name=self.model_name,
            critic_model_name=self.critic_model_name,
            host=self.ollama_host,
            timeout=self.timeout,
            max_rounds=self.max_rounds,
        )
        self.grounder = PlanGrounder(self.transformer)

        self.service = rospy.Service("/generate_vlm_plan", GenerateVlmPlan, self.handle_generate_plan)
        rospy.loginfo("[VLM_NODE] ready: /generate_vlm_plan")

    def _configure_calibration(self) -> None:
        if self.calibration == "fixed":
            self.transformer.use_fixed_camera_calibration()
        elif self.calibration == "aruco":
            self.transformer.calibrate_camera_from_aruco_3d()
        elif self.calibration == "aruco_legacy":
            self.transformer.calibrate_camera()
        else:
            raise ValueError(f"Unsupported calibration mode: {self.calibration}")

    def handle_generate_plan(self, request):
        instruction = request.instruction.strip()
        if not instruction:
            return GenerateVlmPlanResponse(
                success=False,
                message="instruction is empty",
                plan_json="[]",
                detections_json="[]",
                grounded_json="[]",
            )

        try:
            color_image, depth_frame = self.camera_operations.get_image()
            intrinsics = getattr(self.camera_operations, "color_intrinsics", None)
            detections = self.detector.detect(
                color_image=color_image,
                depth_frame=depth_frame,
                intrinsics=intrinsics,
            )
            rospy.loginfo("[VLM_NODE] detections: %s", ", ".join(det.object_id for det in detections) or "none")

            if self.demo_mode:
                plan = self._demo_plan(detections)
                message = "demo plan generated"
            else:
                result = self.vlm.generate_plan_result(
                    instruction=instruction,
                    detections=detections,
                    color_image=color_image,
                )
                plan = result.plan
                if not plan:
                    raise RuntimeError(f"VLM planner produced no valid plan. Last feedback: {result.feedback}")
                for record in result.history:
                    rospy.loginfo(
                        "[VLM_NODE] critic round %d pass=%s feedback=%s",
                        record.iteration,
                        record.critic_pass,
                        record.critic_feedback,
                    )
                message = "VLM planner-critic plan generated" if result.success else (
                    "VLM planner-critic reached max rounds; using last valid plan. "
                    f"Last feedback: {result.feedback}"
                )

            grounded = self.grounder.ground(plan, detections)
            return GenerateVlmPlanResponse(
                success=True,
                message=message,
                plan_json=json.dumps(plan, ensure_ascii=False),
                detections_json=json.dumps([self._detection_to_dict(det) for det in detections], ensure_ascii=False),
                grounded_json=json.dumps([self._grounded_to_dict(step) for step in grounded], ensure_ascii=False),
            )
        except Exception as exc:
            rospy.logerr("[VLM_NODE] failed: %s", exc)
            return GenerateVlmPlanResponse(
                success=False,
                message=str(exc),
                plan_json="[]",
                detections_json="[]",
                grounded_json="[]",
            )

    def _demo_plan(self, detections):
        if len(detections) < 2:
            raise RuntimeError("demo_mode needs at least two detected objects")
        return [
            {"order": "01", "action": "Pick", "target": detections[0].object_id},
            {"order": "02", "action": "Place", "target_object": detections[1].object_id},
        ]

    def _detection_to_dict(self, detection):
        data = asdict(detection)
        if detection.center_pixel is not None:
            data["center_pixel"] = list(detection.center_pixel)
        if detection.position_camera is not None:
            data["position_camera"] = list(detection.position_camera)
        return data

    def _grounded_to_dict(self, step):
        return {
            "skill": step.skill,
            "object_id": step.object_id,
            "target_id": step.target_id,
            "object_point_base": self._point_to_dict(step.object_point_base),
            "target_point_base": self._point_to_dict(step.target_point_base),
        }

    def _point_to_dict(self, point):
        if point is None:
            return None
        return {
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "roll": point.roll,
            "pitch": point.pitch,
            "yaw": point.yaw,
        }


def main() -> None:
    rospy.init_node("vlm_planner_node")
    Ros1VlmPlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
