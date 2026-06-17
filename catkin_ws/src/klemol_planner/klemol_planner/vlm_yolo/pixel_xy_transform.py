from __future__ import annotations

from typing import Sequence

import numpy as np


# Homography from color-image pixel coordinates [u, v, 1] to robot base XY [x, y].
# Fitted from the four table ArUco corner correspondences:
# corner_0/id=0: pixel (108, 400) -> base (0.687, -0.385)
# corner_1/id=1: pixel (469, 392) -> base (0.674,  0.366)
# corner_2/id=2: pixel (491, 117) -> base (0.100,  0.412)
# corner_3/id=3: pixel ( 92, 112) -> base (0.099, -0.412)
PIXEL_TO_BASE_XY_H = np.array([
    [-2.82884029e-05,  2.00639907e-03, -1.23916948e-01],
    [ 2.02531711e-03, -8.16084924e-06, -5.94074276e-01],
    [-5.59839916e-05, -2.64144047e-05,  1.00000000e+00],
], dtype=float)


def pixel_to_base_xy(pixel: Sequence[float]) -> tuple[float, float]:
    u, v = float(pixel[0]), float(pixel[1])
    homogeneous = PIXEL_TO_BASE_XY_H @ np.array([u, v, 1.0], dtype=float)
    if abs(homogeneous[2]) < 1e-9:
        raise ValueError(f"Invalid homography denominator for pixel {pixel}: {homogeneous}")
    x, y = homogeneous[:2] / homogeneous[2]
    return float(x), float(y)
