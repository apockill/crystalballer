import cv2
import numpy as np
import open3d as o3d

from crystalballer import constants
from crystalballer.constants import GAKKEN_RADIUS
from crystalballer.o3d_utils import VirtualCameraRenderer, Visualizer
from crystalballer.resources import GLOBE

PROJECTOR_DISTANCE = GAKKEN_RADIUS * 25.5
"""Distance from the projector to the center of the globe in meters, with the default
open3d camera intrinsics
"""


def main() -> None:
    projector_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=800, height=600, fx=800, fy=800, cx=400, cy=300
    )

    renderer = VirtualCameraRenderer(intrinsics=projector_intrinsics)

    visualizer = Visualizer()
    visualizer.set_camera(
        eye=(0, 0, 0),
        center=(0, 0, -constants.PROJECTOR_DISTANCE),
        up=(0, -1, 0),
    )

    geometries = [
        GLOBE.create_mesh(),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01),
    ]
    while True:
        camera_position = np.eye(4)
        camera_position[:3, 3] = [0, 0, constants.PROJECTOR_DISTANCE]

        color, depth = renderer.render(camera_position, geometries)
        cv2.imshow("Color", color)
        cv2.waitKey(1)

        visualizer.draw(geometries)
