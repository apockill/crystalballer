import cv2

from crystalballer.depthai_pipelines import FacePositionPipeline
from crystalballer.depthai_pipelines.drawing import TextHelper


def main() -> None:
    face_pipeline = FacePositionPipeline()

    # Pipeline is defined, now we can connect to the device
    with face_pipeline:  # tODO: Used to check openvino version
        # Set device log level - to see logs from the Script node
        text_helper = TextHelper()

        while True:
            # 300x300 Mono image frames
            left_frame = face_pipeline.left_frame_queue.get().getCvFrame()  # type: ignore
            right_frame = face_pipeline.right_frame_queue.get().getCvFrame()  # type: ignore

            # Combine the two mono frames
            combined = cv2.addWeighted(left_frame, 0.5, right_frame, 0.5, 0)

            # 3D visualization
            face_detection = face_pipeline.latest_face_detection

            if face_detection is not None:
                strings = [
                    "X: {:.2f} m".format(face_detection.centroid[0]),
                    "Y: {:.2f} m".format(face_detection.centroid[1]),
                    "Z: {:.2f} m".format(face_detection.centroid[2]),
                ]
                text_helper.draw_text(combined, strings, (10, 10))

            cv2.imshow("Combined stereo", combined)
            if cv2.waitKey(1) == ord("q"):
                break
