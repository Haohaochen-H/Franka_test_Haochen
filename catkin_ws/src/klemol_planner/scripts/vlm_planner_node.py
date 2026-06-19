#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ros_python_path import enable_generated_ros_modules

enable_generated_ros_modules(PACKAGE_ROOT)

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.srv import GenerateVlmPlan, GenerateVlmPlanResponse
from klemol_planner.vlm_yolo.scene_context import build_scene_objects, detections_to_jsonable
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
        self.xy_source = rospy.get_param("~xy_source", "pixel")
        table_z_param = rospy.get_param("~table_z", None)
        self.table_z = None if table_z_param in ("", None) else float(table_z_param)
        self.approach_height = float(rospy.get_param("~approach_height", 0.20))
        self.grasp_height_offset = float(rospy.get_param("~grasp_height_offset", 0.0))
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
            scene_objects = build_scene_objects(
                detections=detections,
                panda_transformations=self.transformer,
                xy_source=self.xy_source,
                table_z=self.table_z,
                approach_height=self.approach_height,
                grasp_height_offset=self.grasp_height_offset,
            )

            if self.demo_mode:
                plan = self._demo_plan(scene_objects)
                message = "demo low-level control plan generated"
            else:
                result = self.vlm.generate_plan_result(
                    instruction=instruction,
                    detections=detections,
                    color_image=color_image,
                    scene_objects=scene_objects,
                )
                plan = result.plan
                if not plan:
                    return GenerateVlmPlanResponse(
                        success=False,
                        message=f"VLM planner stopped: {result.feedback}",
                        plan_json="[]",
                        detections_json=json.dumps(detections_to_jsonable(detections), ensure_ascii=False),
                        grounded_json=json.dumps(
                            {
                                "scene_objects": scene_objects,
                                "control_plan": [],
                                "error": result.feedback,
                            },
                            ensure_ascii=False,
                        ),
                    )
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

            return GenerateVlmPlanResponse(
                success=True,
                message=message,
                plan_json=json.dumps(plan, ensure_ascii=False),
                detections_json=json.dumps(detections_to_jsonable(detections), ensure_ascii=False),
                grounded_json=json.dumps(
                    {
                        "scene_objects": scene_objects,
                        "control_plan": plan,
                    },
                    ensure_ascii=False,
                ),
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

    def _demo_plan(self, scene_objects):
        if len(scene_objects) < 2:
            raise RuntimeError("demo_mode needs at least two detected objects")
        source = scene_objects[0]
        target = scene_objects[1]
        return [
            {
                "order": "01",
                "action": "Move",
                "object_id": source["object_id"],
                "start": "current_robot_pose",
                "end": source["pre_grasp_pose"],
                "gripper": "open",
            },
            {
                "order": "02",
                "action": "Move",
                "object_id": source["object_id"],
                "start": source["pre_grasp_pose"],
                "end": source["grasp_pose"],
                "gripper": "open",
            },
            {
                "order": "03",
                "action": "Gripper",
                "object_id": source["object_id"],
                "position": source["grasp_pose"],
                "command": "close",
            },
            {
                "order": "04",
                "action": "Move",
                "object_id": source["object_id"],
                "start": source["grasp_pose"],
                "end": source["lift_pose"],
                "gripper": "closed",
            },
            {
                "order": "05",
                "action": "Move",
                "object_id": target["object_id"],
                "start": source["lift_pose"],
                "end": target["pre_grasp_pose"],
                "gripper": "closed",
            },
            {
                "order": "06",
                "action": "Move",
                "object_id": target["object_id"],
                "start": target["pre_grasp_pose"],
                "end": target["grasp_pose"],
                "gripper": "closed",
            },
            {
                "order": "07",
                "action": "Gripper",
                "object_id": target["object_id"],
                "position": target["grasp_pose"],
                "command": "open",
            },
            {
                "order": "08",
                "action": "Move",
                "object_id": target["object_id"],
                "start": target["grasp_pose"],
                "end": target["lift_pose"],
                "gripper": "open",
            },
        ]


def main() -> None:
    rospy.init_node("vlm_planner_node")
    Ros1VlmPlannerNode()
    rospy.spin()


if __name__ == "__main__":
    main()
