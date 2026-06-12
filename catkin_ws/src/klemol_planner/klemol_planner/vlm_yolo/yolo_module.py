from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class YoloDetection:
    object_id: str
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    center_pixel: Optional[tuple[int, int]] = None
    center_depth_m: Optional[float] = None
    position_camera: Optional[tuple[float, float, float]] = None
    yaw_rad: Optional[float] = None


class YoloObjectDetector:
    def __init__(
        self,
        weights_path: Union[str, Path],
        confidence_threshold: float = 0.25,
        yaw_padding: int = 8,
    ) -> None:
        self.weights_path = Path(weights_path).expanduser()
        if not self.weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install the copied requirements in external/YOLO_test/requirements.txt"
            ) from exc

        self.model = YOLO(str(self.weights_path))
        self.confidence_threshold = confidence_threshold
        self.yaw_padding = yaw_padding

    def detect(
        self,
        color_image: np.ndarray,
        depth_frame: Optional[Any] = None,
        intrinsics: Optional[Any] = None,
    ) -> list[YoloDetection]:
        results = self.model.predict(color_image, conf=self.confidence_threshold, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []

        result = results[0]
        names = result.names
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        detections: list[YoloDetection] = []
        seen_by_class: dict[str, int] = {}
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            bbox = tuple(int(round(value)) for value in box)
            class_name = str(names.get(int(class_id), int(class_id)))
            seen_by_class[class_name] = seen_by_class.get(class_name, 0) + 1
            object_id = self._make_object_id(class_name, seen_by_class[class_name])

            center_pixel = self._center_pixel(bbox)
            center_depth_m = self._depth_at(depth_frame, *center_pixel) if depth_frame is not None else None
            position_camera = self._estimate_position_camera(
                bbox,
                depth_frame,
                intrinsics,
                center_pixel=center_pixel,
                center_depth_m=center_depth_m,
            )
            yaw_rad = self._estimate_yaw(color_image, bbox)

            detections.append(
                YoloDetection(
                    object_id=object_id,
                    class_name=class_name,
                    confidence=float(confidence),
                    bbox_xyxy=bbox,
                    center_pixel=center_pixel,
                    center_depth_m=center_depth_m,
                    position_camera=position_camera,
                    yaw_rad=yaw_rad,
                )
            )
        return detections

    def _center_pixel(self, bbox_xyxy: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox_xyxy
        return int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5))

    def _estimate_position_camera(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        depth_frame: Optional[Any],
        intrinsics: Optional[Any],
        center_pixel: Optional[tuple[int, int]] = None,
        center_depth_m: Optional[float] = None,
    ) -> Optional[tuple[float, float, float]]:
        if depth_frame is None or intrinsics is None:
            return None

        cx, cy = center_pixel or self._center_pixel(bbox_xyxy)
        depth = center_depth_m if center_depth_m is not None else self._depth_at(depth_frame, cx, cy)
        if depth is None or not (0.05 < depth < 5.0):
            return None

        try:
            import pyrealsense2 as rs

            x, y, z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
            return float(x), float(y), float(z)
        except Exception:
            fx = float(getattr(intrinsics, "fx"))
            fy = float(getattr(intrinsics, "fy"))
            ppx = float(getattr(intrinsics, "ppx"))
            ppy = float(getattr(intrinsics, "ppy"))
            x = (cx - ppx) * depth / fx
            y = (cy - ppy) * depth / fy
            return float(x), float(y), float(depth)

    def _depth_at(self, depth_frame: Any, x: int, y: int) -> Optional[float]:
        if hasattr(depth_frame, "get_distance"):
            return float(depth_frame.get_distance(x, y))

        depth_array = np.asarray(depth_frame)
        height, width = depth_array.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return None
        depth = float(depth_array[y, x])
        if depth > 20.0:
            depth *= 0.001
        return depth

    def _estimate_yaw(self, image: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> Optional[float]:
        try:
            from klemol_planner.vlm_yolo.yaw_estimator_adapter import estimate_yaw_rad

            return estimate_yaw_rad(image, bbox_xyxy, padding=self.yaw_padding)
        except Exception:
            return None

    def _make_object_id(self, class_name: str, instance_index: int) -> str:
        normalized = class_name.strip().replace(" ", "_")
        return normalized if instance_index == 1 else f"{normalized}_{instance_index}"


def print_detections(detections: list[YoloDetection]) -> None:
    if not detections:
        print("[YOLO] no detections")
        return
    for det in detections:
        print(
            "[YOLO] {object_id} class={class_name} conf={confidence:.3f} bbox={bbox} "
            "center={center} center_depth={center_depth} position_camera={position} yaw={yaw}".format(
                object_id=det.object_id,
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox_xyxy,
                center=det.center_pixel,
                center_depth=None if det.center_depth_m is None else round(det.center_depth_m, 3),
                position=det.position_camera,
                yaw=det.yaw_rad,
            )
        )
