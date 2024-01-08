import cv2
import numpy as np
import open3d as o3d

from crystalballer import constants
from crystalballer.o3d_utils import VirtualCameraRenderer, Visualizer
from crystalballer.resources import GLOBE

PROJECTOR_DISTANCE = constants.GAKKEN_RADIUS * 1.666
"""Distance from the projector to the center of the globe in meters, with the default
open3d camera intrinsics
"""

def main() -> None:

    renderer = VirtualCameraRenderer(intrinsics=constants.GAKKEN_INTRINSICS)

    visualizer = Visualizer()
    visualizer.set_camera(
        eye=(0, 0, 0),
        center=(0, 0, -PROJECTOR_DISTANCE),
        up=(0, -1, 0),
    )

    eyeball = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    eyeball.translate(
        [
            constants.GAKKEN_RADIUS / 2,
            constants.GAKKEN_RADIUS / 2,
            constants.GAKKEN_RADIUS + 0.01,
        ]
    )
    eyeball.paint_uniform_color([1, 0, 0])

    geometries = [
        GLOBE.create_mesh(),
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01),
        eyeball,
    ]

    WINDOW_NAME = "Projector"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        camera_position = np.eye(4)
        camera_position[:3, 3] = [0, 0, PROJECTOR_DISTANCE]

        color_bgr, depth = renderer.render(camera_position, geometries)
        cv2.imshow(WINDOW_NAME, color_bgr[..., ::-1])

        # Compensate for the titlebar by moving the window up
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        cv2.setWindowProperty(
            WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.waitKey(1)

        visualizer.draw(geometries)
