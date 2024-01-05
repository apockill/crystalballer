from typing import Union

import cv2
import numpy as np
import numpy.typing as npt


def draw_text(
    frame: npt.NDArray[np.uint8],
    text: Union[str, list[str]],
    coords: tuple[int, int],
    bg_color: tuple[int, int, int] = (0, 0, 0),
    color: tuple[int, int, int] = (255, 255, 255),
    text_type: int = cv2.FONT_HERSHEY_SIMPLEX,
    line_type: int = cv2.LINE_AA,
) -> None:
    """Draw multiline strings on a frame, on a specific top-left coordinate"""
    if isinstance(text, str):
        text = text.split("\n")

    x, y = coords
    y_delta = 18
    for line in text:
        cv2.putText(frame, line, (x, y), text_type, 0.5, bg_color, 4, line_type)
        cv2.putText(frame, line, (x, y), text_type, 0.5, color, 1, line_type)
        y += y_delta
