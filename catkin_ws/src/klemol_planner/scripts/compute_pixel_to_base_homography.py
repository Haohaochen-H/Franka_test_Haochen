#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aruco_corner_debug import detect_markers, get_aligned_frame, start_realsense


TABLE_MARKER_IDS = [0, 1, 2, 3]
BASE_POINTS_BY_ID = {
    0: (0.687, -0.385),
    1: (0.674, 0.366),
    2: (0.100, 0.412),
    3: (0.099, -0.412),
}


def marker_center(pts: np.ndarray) -> tuple[float, float]:
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def compute_homography(pixel_points: np.ndarray, base_points: np.ndarray) -> np.ndarray:
    homography, _ = cv2.findHomography(pixel_points, base_points, method=0)
    if homography is None:
        raise RuntimeError("cv2.findHomography failed. Check marker order and corner detection.")
    return (homography / homography[2, 2]).astype(float)


def transform_pixel(homography: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float]:
    mapped = homography @ np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=float)
    if abs(mapped[2]) < 1e-12:
        raise RuntimeError(f"Invalid homography denominator for pixel {pixel}: {mapped}")
    xy = mapped[:2] / mapped[2]
    return float(xy[0]), float(xy[1])


def format_matrix_for_python(homography: np.ndarray) -> str:
    rows = []
    for row in homography:
        rows.append("    [" + ", ".join(f"{value:.8e}" for value in row) + "],")
    return "PIXEL_TO_BASE_XY_H = np.array([\n" + "\n".join(rows) + "\n], dtype=float)"


def draw_text(image, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = origin
    for index, line in enumerate(lines):
        cv2.putText(image, line, (x, y + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def main() -> None:
    pipeline, align, _ = start_realsense()
    try:
        color_image, _ = get_aligned_frame(pipeline, align)
    finally:
        pipeline.stop()

    corners, ids, _ = detect_markers(color_image)
    if ids is None:
        raise RuntimeError("No ArUco markers detected. Make sure table corner IDs 0,1,2,3 are visible.")

    marker_by_id = {int(marker_id[0]): corner.reshape(-1, 2) for corner, marker_id in zip(corners, ids)}
    missing = [marker_id for marker_id in TABLE_MARKER_IDS if marker_id not in marker_by_id]
    if missing:
        raise RuntimeError(f"Missing table corner marker IDs: {missing}. Detected IDs: {sorted(marker_by_id)}")

    pixel_points = np.array([marker_center(marker_by_id[marker_id]) for marker_id in TABLE_MARKER_IDS], dtype=np.float64)
    base_points = np.array([BASE_POINTS_BY_ID[marker_id] for marker_id in TABLE_MARKER_IDS], dtype=np.float64)
    homography = compute_homography(pixel_points, base_points)

    annotated = color_image.copy()
    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
    for marker_id in TABLE_MARKER_IDS:
        center = marker_center(marker_by_id[marker_id])
        cx, cy = int(round(center[0])), int(round(center[1]))
        base_xy = BASE_POINTS_BY_ID[marker_id]
        cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        draw_text(
            annotated,
            [f"id={marker_id}", f"px=({cx},{cy})", f"base=({base_xy[0]:.3f},{base_xy[1]:.3f})"],
            (cx + 8, max(18, cy - 28)),
            (0, 255, 255),
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = PACKAGE_ROOT / "debug_images" / f"pixel_to_base_homography_debug_{timestamp}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    print("[HOMOGRAPHY] detected_table_marker_ids=", TABLE_MARKER_IDS)
    print("[HOMOGRAPHY] pixel_points_by_id:")
    for marker_id, pixel, base_xy in zip(TABLE_MARKER_IDS, pixel_points, base_points):
        print(
            f"  id={marker_id}: pixel=({pixel[0]:.3f}, {pixel[1]:.3f}) "
            f"-> base=({base_xy[0]:.6f}, {base_xy[1]:.6f})"
        )
    print("[HOMOGRAPHY] homography pixel -> base XY:")
    print(homography)
    print()
    print(format_matrix_for_python(homography))

    print("\n[HOMOGRAPHY] reprojection_check:")
    for marker_id, pixel, expected in zip(TABLE_MARKER_IDS, pixel_points, base_points):
        actual = transform_pixel(homography, tuple(pixel))
        error = np.array(actual) - expected
        print(
            f"  id={marker_id}: actual=({actual[0]:.6f}, {actual[1]:.6f}) "
            f"error=({error[0]:+.6e}, {error[1]:+.6e})"
        )
    print(f"[HOMOGRAPHY] debug_image={output}")


if __name__ == "__main__":
    main()