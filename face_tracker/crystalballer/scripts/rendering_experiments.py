import cv2
import numpy as np
import open3d as o3d

from face_tracker.crystalballer.depthai_pipelines import FacePositionPipeline
from face_tracker.crystalballer.o3d_utils import VirtualCameraRenderer


def main() -> None:
    face_pipeline = FacePositionPipeline()

    geometries = [
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2),
    ]

    headtracking_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=1920,
        height=1080,
        fx=1920,
        fy=1920,
        cx=1920 / 2,
        cy=1080 / 2,
    )

    renderer = VirtualCameraRenderer(intrinsics=headtracking_intrinsics)

    with face_pipeline:
        while True:
            print("Loop")
            face = face_pipeline.get_latest_face()

            if not face:
                continue

            camera_position = np.eye(4)
            camera_position[:3, 3] = face.centroid
            color, depth = renderer.render(camera_position, geometries)

            cv2.imshow("Color", color)
            cv2.waitKey(1)
