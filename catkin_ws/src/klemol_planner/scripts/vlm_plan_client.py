#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import rospy

from klemol_planner.srv import GenerateVlmPlan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call the ROS1 VLM planner service.")
    parser.add_argument("--instruction", required=True, help="Natural-language task instruction.")
    parser.add_argument("--service", default="/generate_vlm_plan", help="Service name.")
    return parser.parse_args()


def pretty_json(raw: str) -> str:
    try:
        return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
    except Exception:
        return raw


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


if __name__ == "__main__":
    main()
