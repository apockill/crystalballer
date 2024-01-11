from copy import copy

import numpy as np
import open3d as o3d

from crystalballer import constants, linalg
from crystalballer.depthai_pipelines import FacePositionPipeline
from crystalballer.o3d_utils import VirtualCameraRenderer
from crystalballer.o3d_utils.fullscreen_visualizer import FullScreenVisualizer
from crystalballer.resources import GLOBE

PROJECTOR_DISTANCE = constants.GAKKEN_RADIUS * 1.666
"""Distance from the projector to the center of the globe in meters, with the default
open3d camera intrinsics
"""


def main() -> None:
    # Create a camera, a fullscreen visualizer for the Gakken, and a 3d visualizer
    # for easy debugging
    face_pipeline = FacePositionPipeline()
    cv2_visualizer = FullScreenVisualizer("Gakken View")
    renderer = VirtualCameraRenderer(intrinsics=constants.GAKKEN_INTRINSICS)

    try:
        run_rendering_loop(
            renderer=renderer,
            cv2_visualizer=cv2_visualizer,
            face_pipeline=face_pipeline,
        )
    except Exception as e:
        raise e
    finally:
        del renderer
        face_pipeline.close()
        cv2_visualizer.close()


def run_rendering_loop(
    renderer: VirtualCameraRenderer,
    cv2_visualizer: FullScreenVisualizer,
    face_pipeline: FacePositionPipeline,
) -> None:
    # Create geometries for the scene
    eyeball = GLOBE.create_mesh()
    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01),
    ]

    with face_pipeline:
        while True:
            face = face_pipeline.get_latest_face()

            # Only update the Gakken when face detections are found
            if face is None:
                continue

            # Take the vector from origin to face.centroid and get a rotation matrix
            oriented_eyeball = copy(eyeball)
            rotation_from_z = linalg.rotation_between_vectors(
                np.array([0, 0, 1]), face.centroid
            )
            oriented_eyeball.rotate(R=rotation_from_z, center=(0, 0, 0))

            camera_position = np.eye(4)
            camera_position[:3, 3] = [0, 0, PROJECTOR_DISTANCE]

            render_geometries = [*geometries, oriented_eyeball]
            color_rgb, depth = renderer.render(camera_position, render_geometries)
            cv2_visualizer.show(color_rgb)
