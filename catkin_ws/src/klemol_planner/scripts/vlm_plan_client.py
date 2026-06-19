#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import rospy

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ros_python_path import enable_generated_ros_modules

enable_generated_ros_modules(PACKAGE_ROOT)

from klemol_planner.srv import GenerateVlmPlan
from vlm_yolo_dynamic_demo import RRTGroundedExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the ROS1 VLM planner service.")
    parser.add_argument("--instruction", required=True, help="Natural-language task instruction.")
    parser.add_argument("--service", default="/generate_vlm_plan", help="Service name.")
    parser.add_argument("--execute", action="store_true", help="Execute returned executor_steps with RRT.")
    parser.add_argument("--planner", default="rrt_with_connecting", choices=["rrt_with_connecting"])
    parser.add_argument("--post-processing", default="quintic_polynomial", choices=["quintic_polynomial"])
    parser.add_argument("--approach-height", type=float, default=0.20, help="Vertical approach/lift offset in meters.")
    parser.add_argument("--grasp-height-offset", type=float, default=0.0, help="Offset above object pose for grasp.")
    parser.add_argument("--place-height-offset", type=float, default=0.0, help="Offset above target pose for place.")
    return parser.parse_args()


def pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except Exception:
        return raw


def extract_executor_steps(grounded_json: str) -> list[dict]:
    data = json.loads(grounded_json)
    steps = data.get("executor_steps", [])
    if not isinstance(steps, list):
        raise ValueError("grounded_json.executor_steps must be a list.")
    return steps


def main() -> None:
    args = parse_args()
    rospy.init_node("vlm_plan_client", anonymous=True)
    rospy.wait_for_service(args.service)
    proxy = rospy.ServiceProxy(args.service, GenerateVlmPlan)
    response = proxy(args.instruction)

    print(f"success: {response.success}")
    print(f"message: {response.message}")
    print("plan_json:")
    print(pretty_json(response.plan_json))
    print("detections_json:")
    print(pretty_json(response.detections_json))
    print("grounded_json:")
    print(pretty_json(response.grounded_json))

    if not args.execute:
        return
    if not response.success:
        raise RuntimeError(f"Cannot execute failed VLM plan: {response.message}")

    executor_steps = extract_executor_steps(response.grounded_json)
    if not executor_steps:
        raise RuntimeError("VLM plan succeeded but returned no executor_steps.")

    print("[VLM_EXECUTE] executing executor_steps with RRT...")
    executor = RRTGroundedExecutor(args.planner, args.post_processing)
    executor.execute_executor_steps(
        executor_steps,
        approach_height=args.approach_height,
        grasp_height_offset=args.grasp_height_offset,
        place_height_offset=args.place_height_offset,
    )
    print("[VLM_EXECUTE] finished")


if __name__ == "__main__":
    main()
