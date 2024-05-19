from typing import cast

import numpy as np
import numpy.typing as npt
from depthai import ImageManipConfig

from .detection import FaceDetection
from .stereo import StereoInference


def calculate_face_detection_from_landmarks(
    left_landmarks: npt.NDArray[np.float32],
    right_landmarks: npt.NDArray[np.float32],
    left_manip_config: ImageManipConfig,
    right_manip_config: ImageManipConfig,
    stereo: StereoInference,
    left_frame: npt.NDArray[np.uint8],
    right_frame: npt.NDArray[np.uint8],
) -> FaceDetection:
    """

    :param left_landmarks: A (5, 2) array of the left camera's facial landmarks, in
        0-1 range.
    :param right_landmarks: A (5, 2) array of the right camera's facial landmarks, in
        0-1 range.
    :param left_manip_config: The image manipulation (crop, resize) config to the nn
    :param right_manip_config: The image manipulation (crop, resize) config to the nn
    :param stereo: A helper for extracting depth from the disparity
    :param left_frame: The left image, to pack with the FaceDetection
    :param right_frame: The right image, to pack with the FaceDetection
    :return: The average depth of the centroid of the face
    """
    assert left_landmarks.shape == (5, 2)
    assert right_landmarks.shape == (5, 2)
    assert left_frame.shape == right_frame.shape
    crop_size = (left_frame.shape[0], left_frame.shape[1])

    # TODO: Calculate spatials of the average of the face
    spatials: list[tuple[float, float, float]] = []
    left_landmarks_pix = []
    right_landmarks_pix = []
    for left_landmark, right_landmark in zip(left_landmarks, right_landmarks):
        assert left_landmark[0] >= 0.0 and left_landmark[0] <= 1.0
        assert right_landmark[1] >= 0.0 and right_landmark[1] <= 1.0

        left_landmark_pix = landmark_to_pixels(
            left_landmark, left_manip_config, crop_size
        )
        right_landmark_pix = landmark_to_pixels(
            right_landmark, right_manip_config, crop_size
        )
        for pix in (left_landmark_pix, right_landmark_pix):
            assert pix[0] >= 0 and pix[0] <= crop_size[0]
            assert pix[1] >= 0 and pix[1] <= crop_size[1]

        xyz = calculate_landmark_depth(
            landmark_cam_right=right_landmark_pix,
            landmark_cam_left=left_landmark_pix,
            stereo=stereo,
        )
        left_landmarks_pix.append(left_landmark_pix)
        right_landmarks_pix.append(right_landmark_pix)
        spatials.append(xyz)

    # Next, normalize to meter units and calculate the average of all the landmarks
    spatials_np = np.array(spatials)
    spatials_np = spatials_np / 1000.0
    return FaceDetection(
        centroid=np.average(spatials_np, axis=0),
        landmarks=spatials_np,
        left_landmarks_pix=np.array(left_landmarks_pix),
        right_landmarks_pix=np.array(right_landmarks_pix),
        left_frame=left_frame,
        right_frame=right_frame,
        stereo=stereo,
    )


def landmark_to_pixels(
    landmark: tuple[float, float],
    manip_config: ImageManipConfig,
    crop_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert a landmark from 0-1 range to pixel coordinates"""
    width = manip_config.getCropXMax() - manip_config.getCropXMin()
    height = manip_config.getCropYMax() - manip_config.getCropYMin()

    x = int(round((landmark[0] * width + manip_config.getCropXMin()) * crop_size[0]))
    y = int(round((landmark[1] * height + manip_config.getCropYMin()) * crop_size[1]))
    return (x, y)


def calculate_landmark_depth(
    landmark_cam_right: tuple[int, int],
    landmark_cam_left: tuple[int, int],
    stereo: StereoInference,
) -> tuple[float, float, float]:
    disparity = stereo.calculate_distance(landmark_cam_right, landmark_cam_left)
    depth = stereo.calculate_depth(disparity)

    # Averaging the two landmarks ensures the depth is "centered" around the center of
    # the camera, and not one of the cameras.
    landmark_cam_avg = tuple(
        int(round((landmark_cam_right[i] + landmark_cam_left[i]) / 2)) for i in range(2)
    )
    return stereo.calc_spatials(cast(tuple[int, int], landmark_cam_avg), depth)
