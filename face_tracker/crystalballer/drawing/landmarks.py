import cv2
import numpy as np
import numpy.typing as npt

from crystalballer.depthai_pipelines import FaceDetection

from .text import draw_text


def draw_face_detection(face: FaceDetection) -> npt.NDArray[np.uint8]:
    left_color = (255, 0, 0)
    right_color = (0, 255, 0)

    combined = cv2.addWeighted(face.left_frame, 0.5, face.right_frame, 0.5, 0)
    left_frame = face.left_frame.copy()
    right_frame = face.right_frame.copy()

    # Draw the face distance on the combined image
    y = 0
    y_delta = 18
    strings = [
        f"X: {face.centroid[0]:.2f} m",
        f"Y: {face.centroid[1]:.2f} m",
        f"Z: {face.centroid[2]:.2f} m",
    ]
    for s in strings:
        y += y_delta
        draw_text(combined, s, (10, y))

    spatials = []
    for left_landmark, right_landmark in zip(
        face.left_landmarks_pix, face.right_landmarks_pix, strict=False
    ):
        cv2.circle(left_frame, left_landmark, 3, left_color)
        cv2.circle(right_frame, right_landmark, 3, right_color)
        cv2.circle(combined, left_landmark, 3, left_color)
        cv2.circle(combined, right_landmark, 3, right_color)

        # Visualize disparity line frame
        cv2.line(combined, right_landmark, left_landmark, (0, 0, 255), 1)

        disparity = face.stereo.calculate_distance(right_landmark, left_landmark)
        depth = face.stereo.calculate_depth(disparity)
        # print(f"Disp {disparity}, depth {depth}")
        spatial = face.stereo.calc_spatials(right_landmark, depth)
        spatials.append(spatial)

    # Combine the frames and return them
    return np.concatenate((left_frame, combined, right_frame), axis=1)
