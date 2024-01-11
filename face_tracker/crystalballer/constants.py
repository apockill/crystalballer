import math

import open3d as o3d

# All distances are in meters
GAKKEN_CIRCUMFERENCE = 0.7874  # Meters
GAKKEN_RADIUS = GAKKEN_CIRCUMFERENCE / (2 * math.pi)
GAKKEN_DIAMETER = GAKKEN_RADIUS * 2

GAKKEN_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(
    width=800, height=600, cx=400, cy=300, fx=500, fy=500
)
