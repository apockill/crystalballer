from __future__ import annotations

import numpy as np
import numpy.typing as npt
import open3d as o3d
from open3d.visualization import gui, rendering

ARRAYABLE_POINT = tuple[float, float, float] | npt.NDArray[np.float64]


class Visualizer:
    def __init__(self, name: str = "Window", width: int = 1920, height: int = 1080):
        # We need to initialize the application, which finds the necessary shaders
        # for rendering and prepares the cross-platform window abstraction.
        self.app: gui.Application = gui.Application.instance
        self.app.initialize()

        self.window: gui.PyWindow = self.app.create_window(name, width, height)
        self.widget_3d = gui.SceneWidget()
        self.scene: rendering.Open3DScene = rendering.Open3DScene(self.window.renderer)

        # Link things together
        self.widget_3d.scene = self.scene
        self.window.add_child(self.widget_3d)

        self.default_material = rendering.MaterialRecord()

    def draw(self, geometries: list[o3d.geometry.Geometry] | None = None) -> None:
        if geometries is not None:
            self.scene.clear_geometry()

            for i, geometry in enumerate(geometries):
                self.scene.add_geometry(
                    name=str(i),
                    geometry=geometry,
                    material=self.default_material,
                    add_downsampled_copy_for_fast_rendering=False,
                )

        # Without this, `run_one_tick` will halt until mouse movement or keyboard press
        self.window.post_redraw()
        self.app.run_one_tick()

    def set_camera(
        self, center: ARRAYABLE_POINT, eye: ARRAYABLE_POINT, up: ARRAYABLE_POINT
    ) -> None:
        self.widget_3d.look_at(
            np.array(center).reshape(3, 1).astype(np.float32),
            np.array(eye).reshape(3, 1).astype(np.float32),
            np.array(up).reshape(3, 1).astype(np.float32),
        )

    def close(self) -> None:
        self.app.quit()
