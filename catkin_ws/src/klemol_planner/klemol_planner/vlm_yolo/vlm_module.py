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
Observe the current camera image and the YOLO detections, then convert the human instruction into a short low-level action sequence.

Action space:
- You may only use "Pick" and "Place".
- A Pick step must be {"order": "01", "action": "Pick", "target": "object_id"}.
- A Place step must be {"order": "02", "action": "Place", "target_object": "object_id"}.
- Every Pick must be followed by one Place before another Pick.

Available task object classes:
Cleaner_bottle, Salt_box, tomato_soup_can, Orange_cube, Yellow_cube.

Object naming:
- Use only object_id values from the visible objects list.
- If the instruction uses a class name, map it to the matching visible object_id.
- Do not invent objects, locations, bins, boxes, or coordinates.

Planner-critic mechanism:
Another VLM will act as a critic. If critic feedback is provided, revise the plan according to that feedback.

Output format:
Return JSON only, with exactly this shape:
{
  "plan": [
    {"order": "01", "action": "Pick", "target": "Cleaner_bottle"},
    {"order": "02", "action": "Place", "target_object": "Salt_box"}
  ]
}

Examples:
Human: Put the Cleaner_bottle on the Salt_box.
Robot:
{
  "plan": [
    {"order": "01", "action": "Pick", "target": "Cleaner_bottle"},
    {"order": "02", "action": "Place", "target_object": "Salt_box"}
  ]
}

Human: Move the Orange_cube to the Yellow_cube, then place the tomato_soup_can on the Salt_box.
Robot:
{
  "plan": [
    {"order": "01", "action": "Pick", "target": "Orange_cube"},
    {"order": "02", "action": "Place", "target_object": "Yellow_cube"},
    {"order": "03", "action": "Pick", "target": "tomato_soup_can"},
    {"order": "04", "action": "Place", "target_object": "Salt_box"}
  ]
}

Human: Clear the Salt_box for the Cleaner_bottle.
Critic VLM: The image shows Orange_cube already occupying the Salt_box. Move Orange_cube to Yellow_cube first, then place Cleaner_bottle on Salt_box.
Robot:
{
  "plan": [
    {"order": "01", "action": "Pick", "target": "Orange_cube"},
    {"order": "02", "action": "Place", "target_object": "Yellow_cube"},
    {"order": "03", "action": "Pick", "target": "Cleaner_bottle"},
    {"order": "04", "action": "Place", "target_object": "Salt_box"}
  ]
}
"""


CRITIC_SYSTEM_PROMPT = """You are the critic VLM for a Franka Panda robot.
Observe the current camera image, the YOLO detections, the human instruction, and the planner action sequence.
Evaluate whether the sequence is physically feasible, safe, and logically correct for this tabletop task.

Evaluation standards:
1. Robot arm state: the robot cannot Pick another object while already holding one.
2. Action order: check for reversed, missing, duplicated, or illogical actions.
3. Object validity: every Pick target and Place target_object must be a visible object_id.
4. Task completion: the final sequence must satisfy the human instruction.
5. Space constraints: if the target object is visibly occupied or blocked by another task object, the blocking object should be moved first.

Available task object classes:
Cleaner_bottle, Salt_box, tomato_soup_can, Orange_cube, Yellow_cube.

Example:
Task: Put the Cleaner_bottle on the Salt_box.
Robot:
[
  {"order": "01", "action": "Pick", "target": "Cleaner_bottle"},
  {"order": "02", "action": "Place", "target_object": "Salt_box"}
]
Critic VLM:
{
  "pass": false,
  "feedback": "The image shows Orange_cube on the Salt_box. Move Orange_cube to Yellow_cube first, then place Cleaner_bottle on Salt_box."
}

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
"""


@dataclass
class CriticRecord:
    iteration: int
    planner_raw: str
    plan: Optional[list[dict[str, str]]]
    critic_raw: Optional[str]
    critic_pass: bool
    critic_feedback: str


@dataclass
class VlmPlannerResult:
    plan: list[dict[str, str]]
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
    ) -> list[dict[str, str]]:
        return self.generate_plan_result(instruction, detections, color_image).plan

    def generate_plan_result(
        self,
        instruction: str,
        detections: list[YoloDetection],
        color_image: Optional[np.ndarray] = None,
    ) -> VlmPlannerResult:
        images = [_encode_image_base64(color_image)] if color_image is not None else None
        feedback = ""
        previous_plan_json = ""
        last_valid_plan: list[dict[str, str]] = []
        history: list[CriticRecord] = []

        for iteration in range(1, self.max_rounds + 1):
            planner_prompt = self._build_planner_prompt(instruction, detections, feedback, previous_plan_json)
            planner_raw = self._ollama_generate(planner_prompt, model_name=self.model_name, images=images)
            try:
                plan = normalize_plan(extract_json(planner_raw), detections)
                plan_json = json.dumps(plan, ensure_ascii=False)
                previous_plan_json = plan_json
                last_valid_plan = plan
            except Exception as exc:
                feedback = f"The planner output is invalid or unsafe: {exc}"
                history.append(CriticRecord(iteration, planner_raw, None, None, False, feedback))
                continue

            critic_prompt = self._build_critic_prompt(instruction, detections, plan_json)
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

    def _build_planner_prompt(
        self,
        instruction: str,
        detections: list[YoloDetection],
        critic_feedback: str = "",
        previous_plan_json: str = "",
    ) -> str:
        parts = [
            PLANNER_SYSTEM_PROMPT,
            f"Visible objects: {self._visible_objects_json(detections)}",
            f"Human instruction: {instruction}",
        ]
        if previous_plan_json:
            parts.append(f"Previous invalid or incomplete plan: {previous_plan_json}")
        if critic_feedback:
            parts.append(f"Critic feedback to fix: {critic_feedback}")
        parts.append('Output only JSON with key "plan".')
        return "\n\n".join(parts)

    def _build_critic_prompt(self, instruction: str, detections: list[YoloDetection], plan_json: str) -> str:
        return "\n\n".join(
            [
                CRITIC_SYSTEM_PROMPT,
                f"Visible objects: {self._visible_objects_json(detections)}",
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
        payload_data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
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


def normalize_plan(data: Any, detections: Optional[list[YoloDetection]] = None) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("plan", data)
    if not isinstance(data, list):
        raise ValueError("VLM plan must be a list or an object with key 'plan'.")

    valid_names = None
    if detections is not None:
        valid_names = {det.object_id for det in detections}
        valid_names.update(det.class_name for det in detections)

    normalized: list[dict[str, str]] = []
    holding = False
    for index, step in enumerate(data, start=1):
        action = str(step.get("action", "")).strip()
        if action not in {"Pick", "Place"}:
            raise ValueError(f"Unsupported action at step {index}: {action!r}")
        out = {"order": f"{index:02d}", "action": action}
        if action == "Pick":
            if holding:
                raise ValueError("Plan tries to Pick while already holding an object.")
            target = str(step.get("target", "")).strip()
            if not target:
                raise ValueError(f"Pick step {index} is missing target.")
            if valid_names is not None and target not in valid_names:
                raise ValueError(f"Pick target {target!r} is not a visible object_id.")
            out["target"] = target
            holding = True
        else:
            if not holding:
                raise ValueError("Plan tries to Place before Pick.")
            target = str(step.get("target_object") or step.get("target") or "").strip()
            if not target:
                raise ValueError(f"Place step {index} is missing target_object.")
            if valid_names is not None and target not in valid_names:
                raise ValueError(f"Place target_object {target!r} is not a visible object_id.")
            out["target_object"] = target
            holding = False
        normalized.append(out)

    if holding:
        raise ValueError("Final Pick has no matching Place.")
    return normalized


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
