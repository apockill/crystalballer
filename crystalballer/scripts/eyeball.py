from copy import copy

import numpy as np
import open3d as o3d

from crystalballer import constants
from crystalballer.depthai_pipelines import FacePositionPipeline
from crystalballer.o3d_utils import VirtualCameraRenderer, Visualizer
from crystalballer.o3d_utils.fullscreen_visualizer import FullScreenVisualizer
from crystalballer.resources import GLOBE

PROJECTOR_DISTANCE = constants.GAKKEN_RADIUS * 1.666
"""Distance from the projector to the center of the globe in meters, with the default
open3d camera intrinsics
"""


def main() -> None:

    # Create a camera, a fullscreen visualizer for the Gakken, and a 3d visualizer
    # for easy debugging
    cv2_visualizer = FullScreenVisualizer("Gakken View")
    renderer = VirtualCameraRenderer(intrinsics=constants.GAKKEN_INTRINSICS)
    visualizer = Visualizer()
    visualizer.set_camera(
        eye=(0, 0, 0),
        center=(0, 0, -PROJECTOR_DISTANCE),
        up=(0, -1, 0),
    )

    # Create geometries for the scene
    eyeball = GLOBE.create_mesh()
    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01),
    ]


    while True:
        camera_position = np.eye(4)
        camera_position[:3, 3] = [0, 0, PROJECTOR_DISTANCE]

        color_rgb, depth = renderer.render(camera_position, geometries)
        cv2_visualizer.show(color_rgb)

        visualizer.draw(geometries)
