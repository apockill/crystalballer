import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

from crystalballer.depthai_pipelines import FacePositionPipeline
from crystalballer.depthai_pipelines.face_position_tracker import (
    SingleFacePositionSmoother,
)


class FaceLocationPacket(BaseModel):
    location: tuple[float, float, float]


class FaceUpdatePacket(BaseModel):
    face_locations: list[FaceLocationPacket]


def create_api(
    face_pipeline: FacePositionPipeline, face_smoother: SingleFacePositionSmoother
) -> FastAPI:
    app = FastAPI()

    @app.websocket("/faces")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()

        while True:
            # Get the latest face and package it into a packet
            face = face_pipeline.get_latest_face()
            faces = (
                []
                if face is None
                else [FaceLocationPacket(location=face.centroid.tolist())]
            )
            packet = FaceUpdatePacket(face_locations=faces)

            await websocket.send_text(packet.model_dump_json())
            while not face_pipeline.new_results_ready:
                await asyncio.sleep(0.01)

    return app


def main() -> None:
    """This is the main entrypoint for the face socket server

    It will start the face pipeline and then start the server on port 6942. It exposes
    a websocket endpoint at /faces that will send face updates to any connected clients.

    A quick way to test if it's working is to run the following in a terminal:
        websocat ws://0.0.0.:6942/faces
    """
    face_pipeline = FacePositionPipeline()
    face_smoother = SingleFacePositionSmoother(face_pipeline)
    app = create_api(face_pipeline, face_smoother)

    # Start the face pipeline and server the app
    with face_pipeline:
        uvicorn.run(app, host="0.0.0.0", port=6942)  # noqa: S104
