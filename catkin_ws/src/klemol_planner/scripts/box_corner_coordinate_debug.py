#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Iterable

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aruco_corner_debug import detect_markers, get_aligned_frame, start_realsense
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.vlm_yolo.pixel_xy_transform import pixel_to_base_xy
from klemol_planner.vlm_yolo.table_height import TABLE_Z_BASE_OFFSET, target_z_from_table_height


MARKER_POINT_INDEX = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}


def parse_id_list(value: str) -> set[int]:
    ids = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            ids.add(int(item))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect an extra ArUco corner marker on a box, convert its image XY to robot base XY, "
            "and use a fixed table-plane Z."
        )
    )
    parser.add_argument(
        "--table-ids",
        default="0,1,2,3",
        help="Comma-separated ArUco IDs used by the four table-corner markers.",
    )
    parser.add_argument(
        "--box-id",
        type=int,
        default=None,
        help="Optional ArUco ID for the box corner. If omitted, the only non-table marker is used.",
    )
    parser.add_argument(
        "--marker-point",
        default="center",
        choices=["center", "tl", "tr", "br", "bl"],
        help="Which point of the box marker to convert. Use center for marker center, or a marker corner.",
    )
    parser.add_argument(
        "--table-z",
        type=float,
        default=0.0,
        help="Z height above the table plane in meters. Base Z is TABLE_Z_BASE_OFFSET + this value.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGE_ROOT / "debug_images"),
        help="Directory for annotated debug images.",
    )
    parser.add_argument("--show-image", action="store_true", help="Show the annotated image in an OpenCV window.")
    return parser.parse_args()


def marker_center(pts: np.ndarray) -> tuple[int, int]:
    cx = int(round(float(pts[:, 0].mean())))
    cy = int(round(float(pts[:, 1].mean())))
    return cx, cy


def marker_point_pixel(pts: np.ndarray, marker_point: str) -> tuple[int, int]:
    if marker_point == "center":
        return marker_center(pts)
    point = pts[MARKER_POINT_INDEX[marker_point]]
    return int(round(float(point[0]))), int(round(float(point[1])))


def choose_box_marker(marker_ids: Iterable[int], table_ids: set[int], box_id: int | None) -> int:
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
        raise RuntimeError(f"Multiple non-table markers found: {candidates}. Re-run with --box-id <id>.")
    return candidates[0]


def draw_text(image, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    for idx, line in enumerate(lines):
        cv2.putText(image, line, (x, y + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    table_ids = parse_id_list(args.table_ids)
    target_z = target_z_from_table_height(args.table_z)

    pipeline, align, _ = start_realsense()
    try:
        color_image, _ = get_aligned_frame(pipeline, align)
    finally:
        pipeline.stop()

    corners, ids, _ = detect_markers(color_image)
    annotated = color_image.copy()

    if ids is None:
        raise RuntimeError("No ArUco markers detected.")

    marker_by_id = {}
    for corner, marker_id_arr in zip(corners, ids):
        marker_id = int(marker_id_arr[0])
        marker_by_id[marker_id] = corner.reshape(-1, 2)

    box_marker_id = choose_box_marker(marker_by_id.keys(), table_ids, args.box_id)
    box_pts = marker_by_id[box_marker_id]
    pixel = marker_point_pixel(box_pts, args.marker_point)
    base_x, base_y = pixel_to_base_xy(pixel)
    base_pose = PointWithOrientation(x=base_x, y=base_y, z=target_z, roll=0.0, pitch=0.0, yaw=0.0)

    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    for marker_id, pts in marker_by_id.items():
        center = marker_center(pts)
        color = (0, 255, 255) if marker_id == box_marker_id else (180, 180, 180)
        label = "BOX" if marker_id == box_marker_id else "TABLE" if marker_id in table_ids else "OTHER"
        cv2.circle(annotated, center, 4, color, -1)
        draw_text(annotated, [f"{label} id={marker_id}", f"px={center}"], (center[0] + 8, max(18, center[1] - 24)), color)

    cv2.drawMarker(annotated, pixel, (0, 0, 255), cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)
    draw_text(
        annotated,
        [
            f"BOX id={box_marker_id} point={args.marker_point}",
            f"px=({pixel[0]},{pixel[1]})",
            f"base=({base_x:.4f},{base_y:.4f},{target_z:.4f})",
        ],
        (pixel[0] + 10, max(18, pixel[1] + 18)),
        (0, 0, 255),
    )

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"box_corner_coordinate_debug_{timestamp}.png"
    cv2.imwrite(str(output_path), annotated)

    print(f"[BOX_CORNER] table_ids={sorted(table_ids)} detected_ids={sorted(marker_by_id)}")
    print(f"[BOX_CORNER] selected_id={box_marker_id} marker_point={args.marker_point} pixel=({pixel[0]},{pixel[1]})")
    print(
        f"[BOX_CORNER] base_xy=({base_x:.6f}, {base_y:.6f}) "
        f"base_z={target_z:.6f} table_offset={TABLE_Z_BASE_OFFSET:.6f} table_z={args.table_z:.6f}"
    )
    print(f"[BOX_CORNER] base_pose={base_pose}")
    print(f"[BOX_CORNER] debug_image={output_path}")

    if args.show_image:
        cv2.imshow("box corner coordinate debug", annotated)
        cv2.waitKey(0)
        cv2.destroyWindow("box corner coordinate debug")


if __name__ == "__main__":
    main()