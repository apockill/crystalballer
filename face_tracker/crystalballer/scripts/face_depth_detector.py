import cv2

from crystalballer import drawing
from crystalballer.depthai_pipelines import FacePositionPipeline


def main() -> None:
    face_pipeline = FacePositionPipeline()

    with face_pipeline:
        while True:
            face_detection = face_pipeline.get_latest_face()

            if not face_detection:
                continue

            render = drawing.draw_face_detection(face_detection)
            cv2.imshow("Combined stereo", render)
            if cv2.waitKey(1) == ord("q"):
                break
