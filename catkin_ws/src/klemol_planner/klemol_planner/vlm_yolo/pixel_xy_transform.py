from __future__ import annotations

from typing import Sequence

import numpy as np


# Homography from color-image pixel coordinates [u, v, 1] to robot base XY [x, y].
# Fitted from the four table ArUco corner correspondences:
# corner_0/id=0: pixel (123.00, 409.75) -> base (0.687, -0.385)
# corner_1/id=1: pixel (484.25, 389.00) -> base (0.674,  0.366)
# corner_2/id=2: pixel (495.50, 112.75) -> base (0.100,  0.412)
# corner_3/id=3: pixel ( 97.50, 123.00) -> base (0.099, -0.412)
PIXEL_TO_BASE_XY_H = np.array([
    [5.00850041e-05, 2.02520256e-03, -1.55542296e-01],
    [2.03771044e-03, -8.23533669e-05, -5.98220571e-01],
    [-4.59442508e-05, -9.49467895e-06, 1.00000000e+00],
], dtype=float)


def pixel_to_base_xy(pixel: Sequence[float]) -> tuple[float, float]:
    u, v = float(pixel[0]), float(pixel[1])
    homogeneous = PIXEL_TO_BASE_XY_H @ np.array([u, v, 1.0], dtype=float)
    if abs(homogeneous[2]) < 1e-9:
        raise ValueError(f"Invalid homography denominator for pixel {pixel}: {homogeneous}")
    x, y = homogeneous[:2] / homogeneous[2]
    return float(x), float(y)
