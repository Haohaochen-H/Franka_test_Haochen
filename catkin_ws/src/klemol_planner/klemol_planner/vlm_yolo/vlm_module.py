from __future__ import annotations

import json
import re
import base64
from dataclasses import asdict, dataclass, field
from typing import Any, Optional
from urllib import request

import cv2
import numpy as np

from klemol_planner.vlm_yolo.yolo_module import YoloDetection


TASK_OBJECTS = [
    "Cleaner_bottle",
    "Salt_box",
    "tomato_soup_can",
    "Orange_cube",
    "Yellow_cube",
]


PLANNER_SYSTEM_PROMPT = """You are the planning VLM for a Franka Panda robot in a tabletop pick-and-place task.
Observe the current camera image, YOLO detections, and base-frame object coordinates, then convert the human instruction into an executor-level robot plan.

Executor action space:
- You may only use "Pick" and "Place".
- A Pick step means: move to the object's executor pose, grasp it, and lift it. The executor will generate approach/grasp/lift waypoints.
- A Place step means: move the currently held object to the target object's executor pose, release it, and retreat. The executor will generate approach/place/retreat waypoints.
- Do not output intermediate Move/Gripper waypoints.
- Do not output raw coordinates in the plan. Coordinates are provided in scene_objects and will be attached by the executor after your plan.

Available task object classes:
Cleaner_bottle, Salt_box, tomato_soup_can, Orange_cube, Yellow_cube.

Object naming:
- Use only object_id values from the scene objects list.
- If the instruction uses a class name, map it to the matching visible object_id.
- Do not invent objects, locations, bins, boxes, or unprovided coordinates.
- If the instruction asks for an object that is not in the scene objects list, do not produce a plan.
- If the instruction is ambiguous, impossible, unsafe, or any required object lacks coordinates, do not produce a plan.

Executor input convention:
- Pick target uses scene_objects[target].base_pose as object_point_base.
- Place target uses scene_objects[target_object].base_pose as target_point_base.
- The executor will expand each Pick/Place into all required intermediate waypoints.

Planner-critic mechanism:
Another VLM will act as a critic. If critic feedback is provided, revise the plan according to that feedback.

Output format:
If the task is feasible, return JSON only, with exactly this shape:
{
  "plan": [
    {"order": "01", "action": "Pick", "target": "Orange_cube"},
    {"order": "02", "action": "Place", "target_object": "Yellow_cube"}
  ]
}

If the task cannot be completed, return JSON only, with exactly this shape:
{
  "error": "Missing required object: object_name"
}
"""


CRITIC_SYSTEM_PROMPT = """You are the critic VLM for a Franka Panda robot.
Observe the current camera image, YOLO detections, base-frame object coordinates, the human instruction, and the planner executor-level action sequence.
Evaluate whether the sequence is physically feasible, safe, and logically correct for this tabletop task.

Evaluation standards:
1. Action order: every Pick must be followed by a Place before another Pick.
2. Holding state: the robot cannot Pick while already holding an object and cannot Place before Pick.
3. Object validity: every Pick target and Place target_object must be from the scene objects list.
4. Task completion: the final sequence must satisfy the human instruction.
5. Space constraints: if the target object is visibly occupied or blocked by another task object, the blocking object should be moved first.

Available task object classes:
Cleaner_bottle, Salt_box, tomato_soup_can, Orange_cube, Yellow_cube.

Response format:
Return JSON only:
{
  "pass": true,
  "feedback": "The plan is feasible."
}
or
{
  "pass": false,
  "feedback": "Concrete correction for the planner."
}

If the planner returned an error because required objects are missing or the task is impossible, pass the error through:
{
  "pass": false,
  "feedback": "Planner correctly stopped: Missing required object: object_name"
}
"""


@dataclass
class CriticRecord:
    iteration: int
    planner_raw: str
    plan: Optional[list[dict[str, Any]]]
    critic_raw: Optional[str]
    critic_pass: bool
    critic_feedback: str


@dataclass
class VlmPlannerResult:
    plan: list[dict[str, Any]]
    success: bool
    feedback: str = ""
    history: list[CriticRecord] = field(default_factory=list)


class VlmPlanner:
    def __init__(
        self,
        model_name: str = "gemma3:4b",
        critic_model_name: Optional[str] = None,
        host: str = "http://localhost:11434",
        timeout: int = 120,
        max_rounds: int = 3,
    ) -> None:
        self.model_name = model_name
        self.critic_model_name = critic_model_name or model_name
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_rounds = max_rounds

    def generate_plan(
        self,
        instruction: str,
        detections: list[YoloDetection],
        color_image: Optional[np.ndarray] = None,
        scene_objects: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        return self.generate_plan_result(instruction, detections, color_image, scene_objects).plan

    def generate_plan_result(
        self,
        instruction: str,
        detections: list[YoloDetection],
        color_image: Optional[np.ndarray] = None,
        scene_objects: Optional[list[dict[str, Any]]] = None,
    ) -> VlmPlannerResult:
        images = [_encode_image_base64(color_image)] if color_image is not None else None
        feedback = ""
        previous_plan_json = ""
        last_valid_plan: list[dict[str, Any]] = []
        history: list[CriticRecord] = []
        scene_objects = scene_objects or []

        for iteration in range(1, self.max_rounds + 1):
            planner_prompt = self._build_planner_prompt(
                instruction,
                detections,
                scene_objects,
                feedback,
                previous_plan_json,
            )
            planner_raw = self._ollama_generate(planner_prompt, model_name=self.model_name, images=images)
            try:
                planner_data = extract_json(planner_raw)
                planner_error = extract_planner_error(planner_data)
                if planner_error:
                    history.append(CriticRecord(iteration, planner_raw, None, None, False, planner_error))
                    return VlmPlannerResult(plan=[], success=False, feedback=planner_error, history=history)
                plan = normalize_plan(planner_data, detections, scene_objects)
                plan_json = json.dumps(plan, ensure_ascii=False)
                previous_plan_json = plan_json
                last_valid_plan = plan
            except Exception as exc:
                feedback = f"The planner output is invalid or unsafe: {exc}"
                history.append(CriticRecord(iteration, planner_raw, None, None, False, feedback))
                continue

            critic_prompt = self._build_critic_prompt(instruction, detections, scene_objects, plan_json)
            critic_raw = self._ollama_generate(critic_prompt, model_name=self.critic_model_name, images=images)
            critic = parse_critic_response(critic_raw)
            feedback = critic["feedback"]
            history.append(CriticRecord(iteration, planner_raw, plan, critic_raw, critic["pass"], feedback))
            if critic["pass"]:
                return VlmPlannerResult(plan=plan, success=True, feedback=feedback, history=history)

        return VlmPlannerResult(plan=last_valid_plan, success=False, feedback=feedback, history=history)

    def _visible_objects_json(self, detections: list[YoloDetection]) -> str:
        visible = []
        for det in detections:
            data = asdict(det)
            data["position_camera"] = det.position_camera
            visible.append(data)
        return json.dumps(visible, ensure_ascii=False)

    def _scene_objects_json(self, scene_objects: list[dict[str, Any]]) -> str:
        return json.dumps(scene_objects, ensure_ascii=False)

    def _build_planner_prompt(
        self,
        instruction: str,
        detections: list[YoloDetection],
        scene_objects: list[dict[str, Any]],
        critic_feedback: str = "",
        previous_plan_json: str = "",
    ) -> str:
        parts = [
            PLANNER_SYSTEM_PROMPT,
            f"YOLO detections: {self._visible_objects_json(detections)}",
            f"Scene objects with base-frame coordinates: {self._scene_objects_json(scene_objects)}",
            f"Human instruction: {instruction}",
        ]
        if previous_plan_json:
            parts.append(f"Previous invalid or incomplete plan: {previous_plan_json}")
        if critic_feedback:
            parts.append(f"Critic feedback to fix: {critic_feedback}")
        parts.append('Output only JSON with key "plan".')
        return "\n\n".join(parts)

    def _build_critic_prompt(
        self,
        instruction: str,
        detections: list[YoloDetection],
        scene_objects: list[dict[str, Any]],
        plan_json: str,
    ) -> str:
        return "\n\n".join(
            [
                CRITIC_SYSTEM_PROMPT,
                f"YOLO detections: {self._visible_objects_json(detections)}",
                f"Scene objects with base-frame coordinates: {self._scene_objects_json(scene_objects)}",
                f"Human instruction: {instruction}",
                f"Planner action sequence: {plan_json}",
                "Critic VLM:",
            ]
        )

    def _ollama_generate(
        self,
        prompt: str,
        model_name: str,
        images: Optional[list[str]] = None,
    ) -> str:
        try:
            return self._ollama_chat(prompt, model_name=model_name, images=images)
        except ImportError:
            return self._ollama_generate_http(prompt, model_name=model_name, images=images)

    def _ollama_chat(
        self,
        prompt: str,
        model_name: str,
        images: Optional[list[str]] = None,
    ) -> str:
        from ollama import Client

        message: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            message["images"] = images
        client = Client(host=self.host)
        try:
            response = client.chat(
                model=model_name,
                messages=[message],
                format="json",
                options={"temperature": 0.0},
            )
        except TypeError:
            response = client.chat(
                model=model_name,
                messages=[message],
                options={"temperature": 0.0},
            )
        message_obj = response.get("message") if isinstance(response, dict) else getattr(response, "message", None)
        if isinstance(message_obj, dict):
            return str(message_obj.get("content", ""))
        return str(getattr(message_obj, "content", ""))

    def _ollama_generate_http(
        self,
        prompt: str,
        model_name: str,
        images: Optional[list[str]] = None,
    ) -> str:
        payload_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        if images:
            payload_data["images"] = images
        payload = json.dumps(payload_data).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))


def _encode_image_base64(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Failed to encode camera image for VLM.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def extract_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text.strip(), flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in VLM output: {text}")
    return json.loads(match.group(1))


def normalize_plan(
    data: Any,
    detections: Optional[list[YoloDetection]] = None,
    scene_objects: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        data = data.get("plan", data)
    if not isinstance(data, list):
        raise ValueError("VLM plan must be a list or an object with key 'plan'.")

    valid_names = None
    if detections is not None:
        valid_names = {det.object_id for det in detections}
        valid_names.update(det.class_name for det in detections)

    normalized: list[dict[str, Any]] = []
    holding = False
    for index, step in enumerate(data, start=1):
        action = str(step.get("action", "")).strip()
        if action not in {"Pick", "Place"}:
            raise ValueError(f"Unsupported action at step {index}: {action!r}")

        out: dict[str, Any] = {"order": f"{index:02d}", "action": action}
        if action == "Pick":
            if holding:
                raise ValueError("Plan tries to Pick while already holding an object.")
            target = str(step.get("target") or step.get("object_id") or "").strip()
            if not target:
                raise ValueError(f"Pick step {index} is missing target.")
            if valid_names is not None and target not in valid_names:
                raise ValueError(f"Pick target {target!r} is not visible.")
            out["target"] = target
            holding = True
        else:
            if not holding:
                raise ValueError("Plan tries to Place before Pick.")
            target = str(step.get("target_object") or step.get("target") or "").strip()
            if not target:
                raise ValueError(f"Place step {index} is missing target_object.")
            if valid_names is not None and target not in valid_names:
                raise ValueError(f"Place target_object {target!r} is not visible.")
            out["target_object"] = target
            holding = False
        normalized.append(out)

    if holding:
        raise ValueError("Final Pick has no matching Place.")
    return normalized


def extract_planner_error(data: Any) -> str:
    if isinstance(data, dict):
        error = str(data.get("error") or "").strip()
        if error:
            return error
        status = str(data.get("status") or "").strip().lower()
        if status in {"error", "failed", "failure"}:
            return str(data.get("message") or data.get("feedback") or "Planner reported failure.").strip()
    return ""


def _normalize_pose(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a pose object.")
    pose = {}
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        if key not in value:
            raise ValueError(f"{label} is missing {key}.")
        try:
            pose[key] = float(value[key])
        except Exception as exc:
            raise ValueError(f"{label}.{key} must be numeric.") from exc
    return pose


def _poses_close(a: dict[str, float], b: dict[str, float], tolerance: float = 1e-3) -> bool:
    return all(abs(float(a[key]) - float(b[key])) <= tolerance for key in ("x", "y", "z", "roll", "pitch", "yaw"))


def _allowed_scene_points(scene_objects: list[dict[str, Any]]) -> list[dict[str, float]]:
    points = []
    for obj in scene_objects:
        for key in ("pre_grasp_pose", "grasp_pose", "lift_pose", "base_pose"):
            pose = obj.get(key)
            if isinstance(pose, dict):
                try:
                    points.append(_normalize_pose(pose, f"{obj.get('object_id', 'object')}.{key}"))
                except ValueError:
                    pass
    return points


def _matches_any_allowed_point(pose: dict[str, float], allowed_points: list[dict[str, float]]) -> bool:
    return any(_poses_close(pose, allowed, tolerance=2e-3) for allowed in allowed_points)


def parse_critic_response(text: str) -> dict[str, Any]:
    try:
        data = extract_json(text)
    except Exception as exc:
        return {"pass": False, "feedback": f"Critic output was invalid JSON: {exc}"}
    if not isinstance(data, dict):
        return {"pass": False, "feedback": "Critic did not return a JSON object."}
    feedback = str(data.get("feedback", "")).strip()
    if not feedback:
        feedback = "The plan is feasible." if bool(data.get("pass", False)) else "No correction was provided."
    return {"pass": bool(data.get("pass", False)), "feedback": feedback}
