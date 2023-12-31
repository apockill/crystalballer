import faulthandler
from copy import deepcopy
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt
import open3d as o3d
from open3d.visualization import ViewControl

ARRAYABLE_POINT = Union[tuple[float, float, float], npt.NDArray[np.float64]]


class NonblockingVisualizer:
    def __init__(self, name: str = "Window") -> None:
        faulthandler.enable(all_threads=True)

        self.vis: o3d.visualization.Visualizer = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=name)

        self.view: ViewControl = self.vis.get_view_control()

    def draw(self, geometries: Optional[List[o3d.geometry.Geometry]] = None) -> None:
        if geometries is not None:
            self.vis.clear_geometries()

            for i, geometry in enumerate(geometries):
                self.vis.add_geometry(geometry=geometry, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()
