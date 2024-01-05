from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import open3d as o3d
from open3d.visualization.rendering import OffscreenRenderer


# TODO: Try without the executor
class VirtualCameraRenderer:
    """
    A class that can render point clouds from a virtual camera.

    This is useful for creating Mock camera drivers that can be used for testing without
    the robot.
    """

    def __init__(
        self,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        truncate_depth: float = 99999.0,
    ):
        """
        :param intrinsics: The intrinsics of the virtual camera
        :param truncate_depth: The maximum depth value to render. Anything beyond this
            value will be set to NaN in the depth readings.
        """
        self.truncate_depth = truncate_depth

        self._width = intrinsics.width
        self._height = intrinsics.height
        self._o3d_intrinsics = intrinsics
        # For some reason, the open3d renderer MUST be used in the same thread it was
        # instantiated on. For that purpose, a thread pool with 1 worker is used.
        self._render_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MockDriver Render Thread"
        )
        self._o3d_renderer = self._render_executor.submit(
            self._create_renderer
        ).result()

    def render(
        self,
        transform: npt.NDArray[np.float64],
        geometries: list[o3d.geometry.Geometry],
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float64]]:
        """Render the scene from the perspective of the given transform

        :param transform: A (4, 4) transformation matrix
        :param geometries: A list of open3d geometries to render
        :return: A tuple of (color, depth) images where color is (height, width, 3)
            and depth is a (height, width)
        """
        self._o3d_renderer.scene.clear_geometry()
        for geometry in geometries:
            self._render_executor.submit(
                self._o3d_renderer.scene.add_geometry,
                name="Plank-Cloud",
                geometry=geometry,
                material=o3d.visualization.rendering.MaterialRecord(),
                add_downsampled_copy_for_fast_rendering=False,
            ).result()

        # For some reason the following two functions are okay to run in the main thread
        self._o3d_renderer.setup_camera(self._o3d_intrinsics, transform)
        self._o3d_renderer.scene.camera.set_projection(
            self._o3d_intrinsics.intrinsic_matrix,
            0.01,
            10000.0,
            self._width,
            self._height,
        )

        color = self._render_executor.submit(
            self._o3d_renderer.render_to_image
        ).result()
        depth = self._render_executor.submit(
            self._o3d_renderer.render_to_depth_image, z_in_view_space=True
        ).result()

        # Open3d gives distances as negative values, so we flip those values here
        depth = np.asarray(depth)

        # Truncate depth
        depth[depth > self.truncate_depth] = np.nan

        return np.asarray(color), depth

    def close(self) -> None:
        """Close the renderer thread"""
        self._render_executor.shutdown(wait=True)

    def _create_renderer(self) -> OffscreenRenderer:
        """Create an offscreen renderer"""
        renderer = OffscreenRenderer(self._width, self._height)

        # Set the background to black so that black "holes" in apriltags show up black
        renderer.scene.set_background(np.array([0, 0, 0, 1]))

        return renderer
