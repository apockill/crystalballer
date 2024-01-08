import cv2
import numpy as np
import numpy.typing as npt


class FullScreenVisualizer:
    """A handler for OpenCV windows that comes preconfigured for fullscreen"""

    def __init__(self, name: str, x: int = 0, y: int = 0):
        self.window_name = name

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def show(self, image_rgb: npt.NDArray[np.uint8]) -> None:
        """Show an RGB image in the window"""
        cv2.imshow(self.window_name, image_rgb[..., ::-1])
        cv2.moveWindow(self.window_name, 0, 0)
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        cv2.waitKey(1)
