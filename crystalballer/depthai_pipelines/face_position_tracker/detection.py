from dataclasses import dataclass

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
