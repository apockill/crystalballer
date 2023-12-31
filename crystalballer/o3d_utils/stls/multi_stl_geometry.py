from pathlib import Path

import numpy as np
import open3d as o3d


class MultiSTLGeometry:
    def __init__(
        self,
        stl_paths: list[Path],
        offset: tuple[float, float, float] = (0, 0, 0),
        largest_side_length: float = 1,
    ):
        self.stl_paths = stl_paths
        self._largest_side_length = largest_side_length

        # Keep track of an internal 'transform' to apply to all meshes
        self._default_transform = np.eye(4)
        self._default_transform[:3, 3] = offset

    def create_mesh(self) -> o3d.geometry.TriangleMesh:
        meshes = [
            o3d.io.read_triangle_mesh(str(stl_path)) for stl_path in self.stl_paths
        ]

        # Apply the initial default transform to the meshes
        final_mesh: o3d.geometry.TriangleMesh = sum(meshes, o3d.geometry.TriangleMesh())

        # Set the center of the mesh to origin
        final_mesh.translate(-final_mesh.get_center())

        # Apply the transform
        final_mesh.transform(self._default_transform)

        # Normalize the mesh so the largest axis-aligned side is _largest_side_length
        bounding_box: o3d.geometry.AxisAlignedBoundingBox = (
            final_mesh.get_axis_aligned_bounding_box()
        )
        largest_side = bounding_box.get_max_extent()
        final_mesh.scale(
            self._largest_side_length / largest_side, final_mesh.get_center()
        )

        return final_mesh
