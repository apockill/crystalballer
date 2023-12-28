import cv2
import numpy as np


def draw_landmarks(
    img: np.ndarray, bboxes: np.ndarray, landmarks: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """
    This function draws bounding boxes and landmarks on the image and return the result.

    :param img: Image to draw on.
    :param scores: Scores of shape [n, 1]. 'n' for number of bboxes.
    :param bboxes: bboxes of shape [n, 5]. 'n' for number of bboxes, '5'
        for coordinate and confidence (x1, y1, x2, y2, c).
    :param landmarks: Landmarks of shape [n, 5, 2]. 'n' for number of bboxes, '5' for 5
        landmarks (two for eyes center, one for nose tip, two for mouth corners),
        '2' for coordinate on the image.
    :return: Image with bounding boxes and landmarks drawn
    """

    # draw bounding boxes
    if bboxes is not None:
        color = (0, 255, 0)
        thickness = 2
        for idx in range(bboxes.shape[0]):
            bbox = bboxes[idx].astype(np.int16)
            cv2.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                color,
                thickness,
            )
            cv2.putText(
                img,
                f"{scores[idx]:.4f}",
                (bbox[0], bbox[1] + 12),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
            )

    # draw landmarks
    if landmarks is not None:
        radius = 2
        thickness = 2
        colors = [
            (255, 0, 0),  # right eye
            (0, 0, 255),  # left eye
            (0, 255, 0),  # nose tip
            (255, 0, 255),  # mouth right
            (0, 255, 255),  # mouth left
        ]
        for idx in range(landmarks.shape[0]):
            face_landmarks = landmarks[idx].astype(np.int16)
            for idx, landmark in enumerate(face_landmarks):
                cv2.circle(
                    img,
                    (int(landmark[0]), int(landmark[1])),
                    radius,
                    colors[idx],
                    thickness,
                )
    return img
