import cv2

from crystalballer.depthai_pipelines import FacePositionPipeline
from crystalballer.depthai_pipelines.drawing import TextHelper


def main() -> None:
    face_pipeline = FacePositionPipeline()
    text_helper = TextHelper()

    with face_pipeline:
        while True:
            # 300x300 Mono image frames
            left_frame = face_pipeline.left_frame_queue.get().getCvFrame()  # type: ignore
            right_frame = face_pipeline.right_frame_queue.get().getCvFrame()  # type: ignore

            # Combine the two mono frames
            combined = cv2.addWeighted(left_frame, 0.5, right_frame, 0.5, 0)

            # Draw face information
            face_detection = face_pipeline.get_latest_face(drain_frame_queues=False)
            if face_detection is not None:
                strings = [
                    f"X: {face_detection.centroid[0]:.2f} m",
                    f"Y: {face_detection.centroid[1]:.2f} m",
                    f"Z: {face_detection.centroid[2]:.2f} m",
                ]
                text_helper.draw_text(combined, strings, (10, 10))

            cv2.imshow("Combined stereo", combined)
            if cv2.waitKey(1) == ord("q"):
                break
