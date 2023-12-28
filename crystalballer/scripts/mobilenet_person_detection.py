"""A helper script for connecting to the Oak-D and visualizing output"""


import blobconverter
import cv2
import depthai
import numpy as np
import numpy.typing as npt

BBOX = tuple[float, float, float, float]


def normalize_bbox(frame: npt.NDArray[np.uint8], bbox: BBOX) -> npt.NDArray[np.int64]:
    """Normalize the bounding box coordinates from 0..1 to the frame size in pixels"""
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


def main() -> None:
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)

    # Link the camera to the neural network, and create the neural network
    detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
    detection_nn.setBlobPath(blobconverter.from_zoo(name="mobilenet-ssd", shaves=6))
    detection_nn.setConfidenceThreshold(0.5)
    cam_rgb.preview.link(detection_nn.input)

    # Link output for the camera images
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Link output for the neural network detections
    xout_nn = pipeline.create(depthai.node.XLinkOut)
    xout_nn.setStreamName("nn")
    detection_nn.out.link(xout_nn.input)

    with depthai.Device(pipeline) as device:
        # TODO: These can be blocking, if so desired
        q_rgb = device.getOutputQueue(name="rgb")
        q_nn = device.getOutputQueue(name="nn")

        frame = None
        detections = []

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None:
                detections = in_nn.detections

            if frame is not None:
                for detection in detections:
                    bbox = normalize_bbox(
                        frame,
                        (
                            detection.xmin,
                            detection.ymin,
                            detection.xmax,
                            detection.ymax,
                        ),
                    )
                    cv2.rectangle(
                        frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2
                    )
                cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord("q"):
                break


if __name__ == "__main__":
    main()
