import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class YawResult:
    yaw_deg: float
    area: float
    bbox_xyxy: tuple[int, int, int, int]
    contour: np.ndarray
    mask: np.ndarray
    roi: np.ndarray


def normalize_angle_180(angle: float) -> float:
    """Normalize angle to [-90, 90)."""
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return angle


def clip_bbox(
    bbox_xyxy: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    padding: int = 8,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(width - 1, x1 - padding))
    y1 = max(0, min(height - 1, y1 - padding))
    x2 = max(0, min(width, x2 + padding))
    y2 = max(0, min(height, y2 + padding))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid bbox after clipping: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def build_binary_mask(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_normal = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, mask_inverse = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Prefer the threshold direction that does not select most of the crop.
    normal_area = cv2.countNonZero(mask_normal)
    inverse_area = cv2.countNonZero(mask_inverse)
    roi_area = roi.shape[0] * roi.shape[1]
    mask = mask_normal
    if abs(inverse_area - roi_area * 0.35) < abs(normal_area - roi_area * 0.35):
        mask = mask_inverse

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def contour_yaw_deg(contour: np.ndarray) -> float:
    rect = cv2.minAreaRect(contour)
    (_, _), (width, height), angle = rect

    # minAreaRect reports the rectangle edge angle. Use the long side as the
    # object axis so tall and wide crops share the same yaw convention.
    if width < height:
        angle += 90.0
    return normalize_angle_180(angle)


def estimate_yaw_from_bbox(
    image: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    padding: int = 8,
    min_area_ratio: float = 0.02,
) -> YawResult:
    x1, y1, x2, y2 = clip_bbox(bbox_xyxy, image.shape, padding=padding)
    roi = image[y1:y2, x1:x2].copy()
    mask = build_binary_mask(roi)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contour found in the object crop.")

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    min_area = roi.shape[0] * roi.shape[1] * min_area_ratio
    if area < min_area:
        raise RuntimeError(
            f"Largest contour is too small: area={area:.1f}, min_area={min_area:.1f}"
        )

    yaw_deg = contour_yaw_deg(contour)
    return YawResult(
        yaw_deg=yaw_deg,
        area=area,
        bbox_xyxy=(x1, y1, x2, y2),
        contour=contour,
        mask=mask,
        roi=roi,
    )


def save_debug_images(result: YawResult, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    margin = 90
    contour_view = cv2.copyMakeBorder(
        result.roi,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=(30, 30, 30),
    )
    rect = cv2.minAreaRect(result.contour)
    shifted_contour = result.contour + np.array([[[margin, margin]]], dtype=np.int32)
    shifted_rect = (
        (rect[0][0] + margin, rect[0][1] + margin),
        rect[1],
        rect[2],
    )
    box = cv2.boxPoints(shifted_rect).astype(int)
    cv2.drawContours(contour_view, [shifted_contour], -1, (0, 255, 0), 2)
    cv2.drawContours(contour_view, [box], -1, (0, 0, 255), 2)

    center = tuple(map(int, shifted_rect[0]))
    axis_len = max(35, int(max(rect[1]) * 0.48))
    yaw_rad = np.deg2rad(result.yaw_deg)
    axis_start = (
        int(center[0] - axis_len * np.cos(yaw_rad)),
        int(center[1] - axis_len * np.sin(yaw_rad)),
    )
    axis_end = (
        int(center[0] + axis_len * np.cos(yaw_rad)),
        int(center[1] + axis_len * np.sin(yaw_rad)),
    )
    x_axis_end = (min(contour_view.shape[1] - 1, center[0] + axis_len), center[1])
    cv2.line(contour_view, center, x_axis_end, (255, 255, 255), 2)
    cv2.line(contour_view, axis_start, axis_end, (255, 0, 0), 3)

    arc_radius = max(18, int(axis_len * 0.42))
    start_angle = 0
    end_angle = int(result.yaw_deg)
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    cv2.ellipse(
        contour_view,
        center,
        (arc_radius, arc_radius),
        0,
        start_angle,
        end_angle,
        (0, 255, 255),
        2,
    )
    cv2.circle(contour_view, center, 4, (0, 255, 255), -1)

    text = f"yaw={result.yaw_deg:.1f} deg"
    text_origin = (16, 34)
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    )
    cv2.rectangle(
        contour_view,
        (text_origin[0] - 6, text_origin[1] - text_height - 8),
        (text_origin[0] + text_width + 6, text_origin[1] + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        contour_view,
        text,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(debug_dir / "roi.jpg"), result.roi)
    cv2.imwrite(str(debug_dir / "mask.jpg"), result.mask)
    cv2.imwrite(str(debug_dir / "contour.jpg"), contour_view)


def parse_bbox(raw_bbox: list[int]) -> tuple[int, int, int, int]:
    if len(raw_bbox) != 4:
        raise ValueError("--bbox needs exactly four numbers: x1 y1 x2 y2")
    x1, y1, x2, y2 = raw_bbox
    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate tabletop object yaw from an image crop using contours."
    )
    parser.add_argument("--image", required=True, help="Path to the source image.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        required=True,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Detection bbox in pixel xyxy format.",
    )
    parser.add_argument("--padding", type=int, default=8, help="Crop padding in pixels.")
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory for roi/mask/contour debug images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    result = estimate_yaw_from_bbox(
        image=image,
        bbox_xyxy=parse_bbox(args.bbox),
        padding=args.padding,
    )
    print(f"yaw_deg: {result.yaw_deg:.2f}")
    print(f"contour_area: {result.area:.1f}")
    print(f"crop_bbox_xyxy: {result.bbox_xyxy}")

    if args.debug_dir:
        save_debug_images(result, Path(args.debug_dir))
        print(f"debug_dir: {args.debug_dir}")


if __name__ == "__main__":
    main()
