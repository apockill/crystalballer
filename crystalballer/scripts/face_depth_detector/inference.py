import math

import depthai as dai
import numpy as np


class StereoInference:
    def __init__(self, device: dai.Device, resolution_num, width, height) -> None:
        calibData = device.readCalibration()
        baseline = calibData.getBaselineDistance(useSpecTranslation=True) * 10  # mm

        # Original mono frames shape
        self.original_height = resolution_num
        self.original_width = 640 if resolution_num == 400 else 1280
        assert self.original_width == 640
        assert self.original_height == 400
        self.original_height = 300
        self.original_width = 300
        self.hfov = calibData.getFov(dai.CameraBoardSocket.RIGHT)

        focalLength = self.get_focal_length_pixels(self.original_width, self.hfov)
        self.dispScaleFactor = baseline * focalLength

        # Cropped frame shape
        self.mono_width = width
        self.mono_height = height
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from
        # 720x720 (by ImageManip), so we need to multiple coords by 2.4 to get the correct disparity.
        self.resize_factor = self.original_height / self.mono_height

    def get_focal_length_pixels(self, pixel_width, hfov):
        return pixel_width * 0.5 / math.tan(hfov * 0.5 * math.pi / 180)

    def calculate_depth(self, disparity_pixels: float):
        try:
            return self.dispScaleFactor / disparity_pixels
        except ZeroDivisionError:
            return 0  # Or inf?

    def calculate_distance(self, c1, c2):
        # Our coords are normalized for 300x300 image. 300x300 was downscaled from 720x720 (by ImageManip),
        # so we need to multiple coords by 2.4 (if using 720P resolution) to get the correct disparity.
        c1 = np.array(c1) * self.resize_factor
        c2 = np.array(c2) * self.resize_factor

        x_delta = c1[0] - c2[0]
        y_delta = c1[1] - c2[1]
        return math.sqrt(x_delta**2 + y_delta**2)

    def calc_angle(self, offset):
        return math.atan(
            math.tan(self.hfov / 2.0) * offset / (self.original_width / 2.0)
        )

    def calc_spatials(self, coords, depth):
        x, y = coords
        bb_x_pos = x - self.mono_width / 2
        bb_y_pos = y - self.mono_height / 2

        angle_x = self.calc_angle(bb_x_pos)
        angle_y = self.calc_angle(bb_y_pos)

        z = depth
        x = z * math.tan(angle_x)
        y = -z * math.tan(angle_y)
        # print(f"x {x}, y {y}, z {z}")
        return [x, y, z]
