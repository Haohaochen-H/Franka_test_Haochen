from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any
from urllib import request

from klemol_planner.vlm_yolo.yolo_module import YoloDetection


class VlmPlanner:
    def __init__(
        self,
        model_name: str = "gemma3:4b",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate_plan(self, instruction: str, detections: list[YoloDetection]) -> list[dict[str, str]]:
        prompt = self._build_prompt(instruction, detections)
        raw = self._ollama_generate(prompt)
        return normalize_plan(extract_json(raw))

    def _build_prompt(self, instruction: str, detections: list[YoloDetection]) -> str:
        visible = []
        for det in detections:
            data = asdict(det)
            data["position_camera"] = det.position_camera
            visible.append(data)

        return "\n\n".join(
            [
                "You control a Franka Panda robot. Create a short pick-and-place plan.",
                "Use only these actions: Pick and Place.",
                "Use only object_id values from the visible objects list.",
                "A Pick step is {\"order\":\"01\",\"action\":\"Pick\",\"target\":\"object_id\"}.",
                "A Place step is {\"order\":\"02\",\"action\":\"Place\",\"target_object\":\"object_id\"}.",
                "Return JSON only with key plan.",
                f"Visible objects: {json.dumps(visible, ensure_ascii=False)}",
                f"Human instruction: {instruction}",
            ]
        )

    def _ollama_generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            }
        ).encode("utf-8")
        req = request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))


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


def normalize_plan(data: Any) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("plan", data)
    if not isinstance(data, list):
        raise ValueError("VLM plan must be a list or an object with key 'plan'.")

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
            out["target"] = target
            holding = True
        else:
            if not holding:
                raise ValueError("Plan tries to Place before Pick.")
            target = str(step.get("target_object") or step.get("target") or "").strip()
            if not target:
                raise ValueError(f"Place step {index} is missing target_object.")
            out["target_object"] = target
            holding = False
        normalized.append(out)

    if holding:
        raise ValueError("Final Pick has no matching Place.")
    return normalized

