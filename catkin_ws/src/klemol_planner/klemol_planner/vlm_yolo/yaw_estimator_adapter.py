from __future__ import annotations

import math
from pathlib import Path
import sys


def _add_external_yolo_path() -> None:
    current = Path(__file__).resolve()
    repo_root = current.parents[5]
    external = repo_root / "external" / "YOLO_test"
    if external.exists() and str(external) not in sys.path:
        sys.path.insert(0, str(external))


def estimate_yaw_rad(image, bbox_xyxy: tuple[int, int, int, int], padding: int = 8) -> float:
    _add_external_yolo_path()
    from yaw_estimator import estimate_yaw_from_bbox

    result = estimate_yaw_from_bbox(image=image, bbox_xyxy=bbox_xyxy, padding=padding)
    return math.radians(float(result.yaw_deg))

