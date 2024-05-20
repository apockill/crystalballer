from __future__ import annotations

import time
from collections import deque

import numpy as np

from .detection import FaceDetection
from .pipeline import FacePositionPipeline


class SingleFacePositionSmoother:
    """This class chooses a single face out of multiple faces and smooths its position
    over time.
    """

    def __init__(
        self,
        pipeline: FacePositionPipeline,
        distance_thresh: float = 0.3,
        time_thresh: float = 0.5,
        smooth_samples: int = 5,
    ):
        self.pipeline: FacePositionPipeline = pipeline
        self.targeted_trail: deque[FaceDetection] | None = None
        self.last_seen_time: float | None = None

        self.distance_thresh: float = distance_thresh
        self.time_thresh: float = time_thresh
        self.smooth_samples: int = smooth_samples

    def get_smoothed_face(self) -> FaceDetection | None:
        face: FaceDetection | None = self.pipeline.get_latest_face()

        if face is not None:
            if self.targeted_trail is None or self._is_time_too_long():
                # If we got a new face AND it's time to reset the trail
                self.targeted_trail = deque([face], maxlen=self.smooth_samples)
                self.last_seen_time = time.time()
                return face
            elif not self._is_face_too_far(face):
                # If the face is close enough to the current target
                self.last_seen_time = time.time()
                self.targeted_trail.append(face)

        if self.targeted_trail is None:
            return None

        else:
            return self._create_smoothed_position(list(self.targeted_trail))

    def _is_face_too_far(self, face: FaceDetection) -> bool:
        if self.targeted_trail is None:
            return False

        latest_face = self.targeted_trail[-1]
        distance: float = latest_face.distance(face)
        return distance > self.distance_thresh

    def _is_time_too_long(self) -> bool:
        assert self.last_seen_time is not None
        current_time: float = time.time()
        return current_time - self.last_seen_time > self.time_thresh

    def _create_smoothed_position(self, faces: list[FaceDetection]) -> FaceDetection:
        # Calculate the average position of all faces
        avg_centroid = np.mean(np.stack([face.centroid for face in faces]), axis=0)

        # Create a new FaceDetection object with the averaged position
        # and other attributes from the first face in the list
        first_face = faces[0]
        smoothed_face = FaceDetection(
            centroid=avg_centroid,
            landmarks=first_face.landmarks,
            left_landmarks_pix=first_face.left_landmarks_pix,
            right_landmarks_pix=first_face.right_landmarks_pix,
            left_frame=first_face.left_frame,
            right_frame=first_face.right_frame,
            stereo=first_face.stereo,
        )

        return smoothed_face
