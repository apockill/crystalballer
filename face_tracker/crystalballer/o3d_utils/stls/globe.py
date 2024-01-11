from pathlib import Path

import open3d as o3d

from face_tracker.crystalballer.constants import GAKKEN_DIAMETER

from .multi_stl_geometry import MultiSTLGeometry


class GlobeGeometry(MultiSTLGeometry):
    def __init__(self, stl_paths: list[Path]):
        super().__init__(
            stl_paths=stl_paths,
            largest_side_length=GAKKEN_DIAMETER,
        )

    def create_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = super().create_mesh()

        # Align the globe so the lowest point is on the Z ground plane
        mesh.translate([0, 0, -mesh.get_min_bound()[2]])
        return mesh
