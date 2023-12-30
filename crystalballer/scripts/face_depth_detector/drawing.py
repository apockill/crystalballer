from typing import Union

import cv2
import numpy as np
import numpy.typing as npt


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def draw_text(
        self,
        frame: npt.NDArray[np.uint8],
        text: Union[str, list[str]],
        coords: tuple[int, int],
    ):
        """Draw multiline strings on a frame, on a specific top-left coordinate"""
        if isinstance(text, str):
            text = text.split("\n")

        x, y = coords
        y_delta = 18
        for i, line in enumerate(text):
            cv2.putText(
                frame,
                line,
                (x, y),
                self.text_type,
                0.5,
                self.bg_color,
                4,
                self.line_type,
            )
            cv2.putText(
                frame, line, (x, y), self.text_type, 0.5, self.color, 1, self.line_type
            )
            y += y_delta
