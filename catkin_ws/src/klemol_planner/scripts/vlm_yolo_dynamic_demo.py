#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from klemol_planner.camera_utils.camera_operations import CameraOperations
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.environment.environment_transformations import PandaTransformations
from klemol_planner.environment.robot_model import Robot
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.planners.rrt_with_connecting import RRTWithConnectingPlanner
from klemol_planner.post_processing.path_post_processing import PathPostProcessing
from klemol_planner.utils.config_loader import load_planner_params
from klemol_planner.vlm_yolo.grounding_module import GroundedStep, PlanGrounder
from klemol_planner.vlm_yolo.vlm_module import VlmPlanner
from klemol_planner.vlm_yolo.yolo_module import YoloObjectDetector, print_detections


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
    parser = argparse.ArgumentParser(description="Run ROS1 Panda RRT with YOLO + VLM grounding.")
    parser.add_argument("--instruction", default="", help="Natural-language task. Prompts interactively if omitted.")
    parser.add_argument("--weights", default=default_weights_path(), help="Ultralytics YOLO weights path.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama host URL.")
    parser.add_argument("--model-name", default="gemma3:4b", help="Ollama model name.")
    parser.add_argument("--planner", default="rrt_with_connecting", choices=["rrt_with_connecting"])
    parser.add_argument("--post-processing", default="quintic_polynomial", choices=["quintic_polynomial"])
    parser.add_argument(
        "--calibration",
        default="fixed",
        choices=["fixed", "aruco"],
        help="Use fixed camera-to-base calibration by default, or recompute it from ArUco markers.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually execute grounded pick/place steps.")
    parser.add_argument("--approach-height", type=float, default=0.12, help="Vertical approach offset in meters.")
    parser.add_argument("--grasp-height-offset", type=float, default=0.02, help="Offset above detected object for grasp.")
    parser.add_argument("--place-height-offset", type=float, default=0.04, help="Offset above target for place.")
    return parser.parse_args()


class RRTGroundedExecutor:
    def __init__(self, planner_name: str, post_processing_name: str) -> None:
        self.start_joint_config = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        self.robot_model = Robot()
        self.collision_checker = CollisionChecker(group_name="panda_arm")
        planner_params = load_planner_params(planner_name)
        self.planner = RRTWithConnectingPlanner(self.robot_model, self.collision_checker, planner_params)
        self.post_processing = PathPostProcessing(collision_checker=self.collision_checker)
        self.post_processing_method = self.post_processing.generate_quintic_polynomial_trajectory
        self.robot_model.move_to_joint_config(self.start_joint_config)

    def execute_steps(
        self,
        steps: list[GroundedStep],
        approach_height: float,
        grasp_height_offset: float,
        place_height_offset: float,
    ) -> None:
        for step in steps:
            if step.skill == "pick":
                if step.object_point_base is None:
                    raise ValueError(f"Pick step for {step.object_id} has no object point.")
                self.execute_pick(step.object_id, step.object_point_base, approach_height, grasp_height_offset)
            elif step.skill == "place":
                if step.target_point_base is None:
                    raise ValueError(f"Place step for {step.target_id} has no target point.")
                self.execute_place(step.object_id, step.target_id, step.target_point_base, approach_height, place_height_offset)
            else:
                raise ValueError(f"Unsupported grounded step: {step}")

    def execute_pick(
        self,
        object_id: str,
        object_point: PointWithOrientation,
        approach_height: float,
        grasp_height_offset: float,
    ) -> None:
        rospy.loginfo(f"[VLM-YOLO] picking {object_id} at {object_point}")
        self.robot_model.open_gripper()
        grasp = copy.deepcopy(object_point)
        grasp.z += grasp_height_offset
        approach_2 = copy.deepcopy(grasp)
        approach_2.z += approach_height * 0.5
        approach_1 = copy.deepcopy(grasp)
        approach_1.z += approach_height
        self._move_to_pose_sequence(goal=approach_1, post_goal_path=[approach_2, grasp])
        self.robot_model.close_gripper()
        lift = copy.deepcopy(grasp)
        lift.z += approach_height
        self._move_to_pose_sequence(goal=lift)

    def execute_hover(
        self,
        object_id: str,
        object_point: PointWithOrientation,
        hover_height: float,
    ) -> None:
        rospy.loginfo(f"[VLM-YOLO] hovering {hover_height:.3f} m above {object_id} at {object_point}")
        hover = copy.deepcopy(object_point)
        hover.z += hover_height
        self._move_to_pose_sequence(goal=hover)

    def execute_place(
        self,
        object_id: str,
        target_id: str,
        target_point: PointWithOrientation,
        approach_height: float,
        place_height_offset: float,
    ) -> None:
        rospy.loginfo(f"[VLM-YOLO] placing {object_id} at {target_id}: {target_point}")
        place = copy.deepcopy(target_point)
        place.z += place_height_offset
        approach_2 = copy.deepcopy(place)
        approach_2.z += approach_height * 0.5
        approach_1 = copy.deepcopy(place)
        approach_1.z += approach_height
        self._move_to_pose_sequence(goal=approach_1, post_goal_path=[approach_2, place])
        self.robot_model.open_gripper()
        retreat = copy.deepcopy(place)
        retreat.z += approach_height
        self._move_to_pose_sequence(goal=retreat)

    def _move_to_pose_sequence(
        self,
        goal: PointWithOrientation,
        post_goal_path: Optional[list[PointWithOrientation]] = None,
    ) -> None:
        current_config = np.array(self.robot_model.get_current_joint_values())
        current_pose = self.robot_model.fk(config=current_config)
        pre_start_path = []
        if current_pose.z < 0.10:
            wp = copy.deepcopy(current_pose)
            wp.z += 0.08
            pre_start_path.append(wp)

        self.robot_model.move_with_trajectory_planner(
            planner=self.planner,
            post_processing=self.post_processing,
            goal=goal,
            pre_start_path=pre_start_path or None,
            post_goal_path=post_goal_path,
            post_processing_method=self.post_processing_method,
        )


def main() -> None:
    args = parse_args()
    rospy.init_node("vlm_yolo_dynamic_demo", anonymous=True)

    instruction = args.instruction or input("Instruction: ").strip()
    if not instruction:
        raise ValueError("Instruction cannot be empty.")

    camera_operations = CameraOperations()
    panda_transformations = PandaTransformations(cam_operations=camera_operations)
    if args.calibration == "fixed":
        panda_transformations.use_fixed_camera_calibration()
    else:
        panda_transformations.calibrate_camera()

    color_image, depth_frame = camera_operations.get_image()
    intrinsics = getattr(camera_operations, "color_intrinsics", None)
    detector = YoloObjectDetector(weights_path=args.weights, confidence_threshold=args.conf)
    detections = detector.detect(color_image=color_image, depth_frame=depth_frame, intrinsics=intrinsics)
    print_detections(detections)
    if not detections:
        raise RuntimeError("No YOLO detections available for VLM planning.")

    vlm = VlmPlanner(model_name=args.model_name, host=args.ollama_host)
    plan = vlm.generate_plan(instruction=instruction, detections=detections)
    print(f"[VLM] plan: {plan}")

    grounder = PlanGrounder(panda_transformations)
    grounded_steps = grounder.ground(plan, detections)
    for step in grounded_steps:
        print(f"[GROUNDING] {step}")

    if not args.execute:
        print("[DRY-RUN] Grounded plan is ready. Re-run with --execute to move the robot.")
        return

    executor = RRTGroundedExecutor(args.planner, args.post_processing)
    executor.execute_steps(
        grounded_steps,
        approach_height=args.approach_height,
        grasp_height_offset=args.grasp_height_offset,
        place_height_offset=args.place_height_offset,
    )


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
