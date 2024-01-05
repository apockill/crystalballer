from functools import cached_property
from typing import Any, Optional

import blobconverter
import depthai as dai
import numpy as np
from depthai import Device, Pipeline

from .calculations import calculate_face_detection_from_landmarks
from .detection import FaceDetection
from .stereo import StereoInference

# Create a script to pipe into a Script node
# TODO: Try resizing to 400x400 in pipeline, to see if it's more accurate
IMAGE_MANIPULATION_SCRIPT = """
import time
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999

while True:
    face_dets = node.io['nn_in'].get().detections
    # node.warn(f"Faces detected: {len(face_dets)}")
    for det in face_dets:
        limit_roi(det)
        # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
        cfg = ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        cfg.setResize(48, 48)
        cfg.setKeepAspectRatio(False)
        node.io['to_manip'].send(cfg)
        # node.warn(f"1 from nn_in: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
"""

FACE_DETECTION_MODEL = "face-detection-retail-0004"
LANDMARKS_MODEL = "landmarks-regression-retail-0009"
OPENVINO_VERSION = "2021.4"
CAMERA_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_480_P


class FacePositionPipeline:
    RESOLUTION = (640, 480)
    """This is a constant used throughout the pipeline"""

    MONO_CROP_SIZE = (300, 300)
    """The size of the crop of the full mono image before image manip"""

    # TODO: One possible optimization is to stop creating unecessary streams that go to the host
    def __init__(self) -> None:
        self.pipeline = self._create_pipeline()
        self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        self.device = Device(self.pipeline.getOpenVINOVersion())

    def __enter__(self) -> "FacePositionPipeline":
        """Open up the device and start outputting results to the queues"""
        self.device.__enter__()
        self.device.startPipeline(self.pipeline)

        self.device.setLogLevel(dai.LogLevel.INFO)
        self.device.setLogOutputLevel(dai.LogLevel.INFO)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close the device"""
        self.device.__exit__(exc_type, exc_val, exc_tb)

    def get_latest_face(self) -> Optional[FaceDetection]:
        """The latest face detections from the NN

        :return: The latest face detection, or None if no face was detected
        """

        # For some reason, if we don't drain the queues then the NNs will freeze. So
        # regardless of whether we want to pack frames into the FaceDetection, we need
        # to drain the queues.
        # These should be 300x300 mono images
        left_image = self.left_frame_queue.get().getCvFrame()  # type: ignore
        right_image = self.right_frame_queue.get().getCvFrame()  # type: ignore

        assert left_image.shape == (*self.MONO_CROP_SIZE, 3), f"{left_image.shape=}"
        assert right_image.shape == (*self.MONO_CROP_SIZE, 3), f"{right_image.shape=}"

        left_config = self.left_config_queue.tryGet()
        if left_config is not None:
            left_landmarks_nn_layer = (
                self.left_landmarks_queue.get().getFirstLayerFp16()  # type: ignore
            )

        right_config = self.right_config_queue.tryGet()
        if right_config is not None:
            right_landmarks_nn_layer = (
                self.right_landmarks_queue.get().getFirstLayerFp16()  # type: ignore
            )

        # TODO: This is a very hacky way of draining all the queues
        self.left_config_queue.tryGetAll()
        self.right_config_queue.tryGetAll()
        self.left_landmarks_queue.tryGetAll()
        self.right_landmarks_queue.tryGetAll()

        if left_config is None or right_config is None:
            return None

        return calculate_face_detection_from_landmarks(
            left_landmarks=np.array(left_landmarks_nn_layer).reshape(5, 2),
            right_landmarks=np.array(right_landmarks_nn_layer).reshape(5, 2),
            left_manip_config=left_config,  # type: ignore
            right_manip_config=right_config,  # type: ignore
            stereo=self.stereo_inference,
            left_frame=left_image,
            right_frame=right_image,
        )

    @cached_property
    def stereo_inference(self) -> StereoInference:
        return StereoInference(
            self.device,
            resolution=self.RESOLUTION,
            width=self.MONO_CROP_SIZE[0],
            height=self.MONO_CROP_SIZE[1],
        )

    @cached_property
    def left_frame_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("mono_left", maxSize=4, blocking=False)

    @cached_property
    def right_frame_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("mono_right", maxSize=4, blocking=False)

    @cached_property
    def left_landmarks_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("landmarks_left", maxSize=4, blocking=False)

    @cached_property
    def right_landmarks_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("landmarks_right", maxSize=4, blocking=False)

    @cached_property
    def left_crop_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("crop_left", maxSize=4, blocking=False)

    @cached_property
    def right_crop_queue(self) -> dai.DataOutputQueue:
        """The queue that will contain the frames from the left camera"""
        return self.device.getOutputQueue("crop_right", maxSize=4, blocking=False)

    @cached_property
    def left_config_queue(self) -> dai.DataOutputQueue:
        return self.device.getOutputQueue("config_left", maxSize=4, blocking=False)

    @cached_property
    def right_config_queue(self) -> dai.DataOutputQueue:
        return self.device.getOutputQueue("config_right", maxSize=4, blocking=False)

    def _create_pipeline(self) -> Pipeline:
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        # Set resolution of mono cameras
        self._create_mono_pipeline(pipeline, "right", dai.CameraBoardSocket.RIGHT)
        self._create_mono_pipeline(pipeline, "left", dai.CameraBoardSocket.LEFT)
        return pipeline

    def _create_mono_pipeline(
        self, pipeline: Pipeline, name: str, camera_socket: dai.CameraBoardSocket
    ) -> None:
        """Create a face detection + landmark recognition pipeline for a single mono"""

        # Pretty sure the Oak-D lite only supports up to 480p, check the logs
        resolution = dai.MonoCameraProperties.SensorResolution.THE_480_P

        cam = pipeline.create(dai.node.MonoCamera)
        cam.setBoardSocket(camera_socket)
        cam.setResolution(resolution)
        actual_resolution = cam.getResolutionSize()
        assert actual_resolution == self.RESOLUTION, f"{actual_resolution=}"

        # ImageManip for cropping (face detection NN requires input image of
        # MONO_CROP_SIZE) and to change frame type
        face_manip = pipeline.create(dai.node.ImageManip)
        face_manip.initialConfig.setResize(*self.MONO_CROP_SIZE)
        # The NN model expects BGR input. By default ImageManip output type would be
        # same as input (gray in this case)
        face_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
        cam.out.link(face_manip.inputImage)

        # NN that detects faces in the image
        face_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        face_nn.setConfidenceThreshold(0.2)
        face_nn.setBlobPath(
            blobconverter.from_zoo(
                FACE_DETECTION_MODEL, shaves=6, version=OPENVINO_VERSION
            )
        )
        face_manip.out.link(face_nn.input)

        # Send mono frames to the host via XLink
        cam_xout = pipeline.create(dai.node.XLinkOut)
        cam_xout.setStreamName("mono_" + name)
        face_nn.passthrough.link(cam_xout.input)

        # Script node will take the output from the NN as an input, get the first
        # bounding box
        # and send ImageManipConfig to the manip_crop
        image_manip_script = pipeline.create(dai.node.Script)
        image_manip_script.inputs["nn_in"].setBlocking(False)
        image_manip_script.inputs["nn_in"].setQueueSize(1)
        face_nn.out.link(image_manip_script.inputs["nn_in"])
        image_manip_script.setScript(IMAGE_MANIPULATION_SCRIPT)

        # This ImageManip will crop the mono frame based on the NN detections. Resulting
        # image will be the cropped face that was detected by the face-detection NN.
        manip_crop = pipeline.create(dai.node.ImageManip)
        face_nn.passthrough.link(manip_crop.inputImage)
        image_manip_script.outputs["to_manip"].link(manip_crop.inputConfig)
        manip_crop.initialConfig.setResize(48, 48)
        manip_crop.inputConfig.setWaitForMessage(False)

        # Send ImageManipConfig to host so it can visualize the landmarks
        config_xout = pipeline.create(dai.node.XLinkOut)
        config_xout.setStreamName("config_" + name)
        image_manip_script.outputs["to_manip"].link(config_xout.input)

        # crop_xout = pipeline.create(dai.node.XLinkOut)
        # crop_xout.setStreamName("crop_" + name)
        # manip_crop.out.link(crop_xout.input)

        # Second NN that detcts landmarks from the cropped 48x48 face
        landmarks_nn = pipeline.create(dai.node.NeuralNetwork)
        landmarks_nn.setBlobPath(
            blobconverter.from_zoo(LANDMARKS_MODEL, shaves=6, version=OPENVINO_VERSION)
        )
        manip_crop.out.link(landmarks_nn.input)

        landmarks_nn_xout = pipeline.create(dai.node.XLinkOut)
        landmarks_nn_xout.setStreamName("landmarks_" + name)
        landmarks_nn.out.link(landmarks_nn_xout.input)
