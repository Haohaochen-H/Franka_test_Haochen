from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.table_height import TABLE_Z_BASE_OFFSET, target_z_from_table_height



DEFAULT_BOX_BASE_X = 0.28536411992536775
DEFAULT_BOX_BASE_Y = 0.33404660024272576
DEFAULT_BOX_TABLE_Z = 0.0
MARKER_POINT_INDEX = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}


def parse_id_list(value) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.split(",")
    else:
        items = value
    return {int(item) for item in items if str(item).strip()}


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        parameters = cv2.aruco.DetectorParameters()
        return dictionary, cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, None


def detect_markers(image) -> dict[int, np.ndarray]:
    dictionary, detector = make_aruco_detector()
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(image)
    else:
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    if ids is None:
        return {}
    return {int(marker_id[0]): corner.reshape(-1, 2) for marker_id, corner in zip(ids, corners)}


def marker_center(pts: np.ndarray) -> tuple[int, int]:
    cx = int(round(float(pts[:, 0].mean())))
    cy = int(round(float(pts[:, 1].mean())))
    return cx, cy


def marker_point_pixel(pts: np.ndarray, marker_point: str) -> tuple[int, int]:
    if marker_point == "center":
        return marker_center(pts)
    if marker_point not in MARKER_POINT_INDEX:
        raise ValueError(f"Unsupported marker point: {marker_point}")
    point = pts[MARKER_POINT_INDEX[marker_point]]
    return int(round(float(point[0]))), int(round(float(point[1])))


def choose_box_marker(marker_ids: Iterable[int], table_ids: set[int], box_id: Optional[int] = None) -> int:
    detected_ids = set(marker_ids)
    if box_id is not None:
        if box_id not in detected_ids:
            raise RuntimeError(f"Requested box marker id={box_id} was not detected. Detected IDs: {sorted(detected_ids)}")
        return box_id

    candidates = sorted(detected_ids - table_ids)
    if not candidates:
        raise RuntimeError(
            f"No non-table ArUco marker found. Detected IDs: {sorted(detected_ids)}; table IDs: {sorted(table_ids)}"
        )
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple non-table markers found: {candidates}. Re-run with a specific box id.")
    return candidates[0]


def box_pose_from_image(
    image,
    table_ids: set[int] | None = None,
    box_id: Optional[int] = None,
    marker_point: str = "center",
    table_z: float = 0.0,
) -> tuple[int, tuple[int, int], PointWithOrientation, dict[int, np.ndarray]]:
    table_ids = table_ids if table_ids is not None else {0, 1, 2, 3}
    marker_by_id = detect_markers(image)
    selected_id = choose_box_marker(marker_by_id.keys(), table_ids, box_id)
    pixel = marker_point_pixel(marker_by_id[selected_id], marker_point)
    base_x, base_y = pixel_to_base_xy(pixel)
    pose = PointWithOrientation(
        x=base_x,
        y=base_y,
        z=target_z_from_table_height(table_z),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    return selected_id, pixel, pose, marker_by_id


def build_box_scene_object(
    image,
    table_ids: set[int] | None = None,
    box_id: Optional[int] = None,
    marker_point: str = "center",
    table_z: float = 0.0,
) -> dict:
    selected_id, pixel, pose, marker_by_id = box_pose_from_image(
        image=image,
        table_ids=table_ids,
        box_id=box_id,
        marker_point=marker_point,
        table_z=table_z,
    )
    base_pose = point_to_dict(pose)
    return {
        "object_id": "box",
        "class_name": "box",
        "scene_role": "container",
        "place_mode": "inside",
        "aruco_id": selected_id,
        "marker_point": marker_point,
        "center_pixel": [int(pixel[0]), int(pixel[1])],
        "detected_marker_ids": sorted(marker_by_id.keys()),
        "base_pose": base_pose,
        "pre_grasp_pose": base_pose,
        "grasp_pose": base_pose,
        "lift_pose": base_pose,
        "table_z": float(table_z),
        "object_height": 0.0,
        "table_offset": TABLE_Z_BASE_OFFSET,
    }



def build_fixed_box_scene_object(
    x: float = DEFAULT_BOX_BASE_X,
    y: float = DEFAULT_BOX_BASE_Y,
    table_z: float = DEFAULT_BOX_TABLE_Z,
) -> dict:
    pose = PointWithOrientation(
        x=float(x),
        y=float(y),
        z=target_z_from_table_height(float(table_z)),
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
    )
    base_pose = point_to_dict(pose)
    return {
        "object_id": "box",
        "class_name": "box",
        "scene_role": "container",
        "place_mode": "inside",
        "coordinate_source": "fixed",
        "base_pose": base_pose,
        "pre_grasp_pose": base_pose,
        "grasp_pose": base_pose,
        "lift_pose": base_pose,
        "table_z": float(table_z),
        "object_height": 0.0,
        "table_offset": TABLE_Z_BASE_OFFSET,
    }

def point_to_dict(point: PointWithOrientation) -> dict[str, float]:
    return {
        "x": float(point.x),
        "y": float(point.y),
        "z": float(point.z),
        "roll": float(point.roll),
        "pitch": float(point.pitch),
        "yaw": float(point.yaw),
    }