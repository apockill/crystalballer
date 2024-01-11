import math
from typing import cast

import depthai as dai
import numpy as np


class StereoInference:
    def __init__(
        self,
        device: dai.Device,
        resolution: tuple[float, float],
        width: int,
        height: int,
    ) -> None:
        calib_data: dai.CalibrationHandler = device.readCalibration()
        baseline = calib_data.getBaselineDistance(useSpecTranslation=False) * 10  # mm

        # Original mono frames shape
        # assert resolution == (640, 400)  # This was what the original used
        self.original_width, self.original_height = resolution

        self.hfov = calib_data.getFov(dai.CameraBoardSocket.RIGHT, useSpec=False)

        focal_length = self.get_focal_length_pixels(self.original_width, self.hfov)
        self.disp_scale_factor = baseline * focal_length

        # Cropped frame shape
        self.mono_width = width
        self.mono_height = height
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 to get the
        # correct disparity.
        # TODO: This should be self.original_height / self.mono_height, but for some
        #       reason tweaking this to 400 makes the distance pretty bang on accurate
        #       Must fix!
        self.resize_factor = 400 / self.mono_height  # This is way more accurate

    def get_focal_length_pixels(self, pixel_width: float, hfov: float) -> float:
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi / 180)

    def calculate_depth(self, disparity_pixels: float) -> float:
        try:
            return cast(float, self.disp_scale_factor / disparity_pixels)
        except ZeroDivisionError:
            print("Warning: Zero division error!")
            return 0.0  # Or inf?

    def calculate_distance(self, c1: tuple[int, int], c2: tuple[int, int]) -> float:
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 (if using 720P
        # resolution) to get the correct disparity.
        c1_arr = np.array(c1) * self.resize_factor
        c2_arr = np.array(c2) * self.resize_factor

        x_delta = c1_arr[0] - c2_arr[0]
        y_delta = c1_arr[1] - c2_arr[1]
        return math.sqrt(x_delta**2 + y_delta**2)

    def calc_angle(self, offset: float) -> float:
        return math.atan(
            math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0)
        )

    def calc_spatials(
        self, coords: tuple[int, int], depth: float
    ) -> tuple[float, float, float]:
        x, y = coords
        bb_x_pos = x - self.mono_width / 2
        bb_y_pos = y - self.mono_height / 2

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z_world = depth
        x_world = z_world * math.tan(angle_x)
        y_world = -z_world * math.tan(angle_y)
        return (x_world, y_world, z_world)
