from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

POINT = tuple[float, float, float]


@dataclass
class FaceDetection:
    """A face position in meters, in the cameras coordinate frame"""

    centroid: npt.NDArray[np.float64]
    """(3,) The centoid of the face in meters, camera frame"""

    landmarks: npt.NDArray[np.float64]
    """(5, 3) The landmarks found on the face in meters, camera frame"""

    left_frame: npt.NDArray[np.uint8]
    """(h, w) The left camera image"""

    right_frame: npt.NDArray[np.uint8]
    """(h, w) The right camera image"""

    @property
    def x(self) -> float:
        return cast(float, self.centroid[0])

    @property
    def y(self) -> float:
        return cast(float, self.centroid[1])

    @property
    def z(self) -> float:
        return cast(float, self.centroid[2])
