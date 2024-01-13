from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
import asyncio
import uvicorn

class DetectedFace(BaseModel):
    location: tuple[float, float, float]

class FaceUpdatePacket(BaseModel):
    face_locations: list[DetectedFace]

app = FastAPI()

def get_faces():
    # Dummy data for simulation
    return [DetectedFace(location=(0.5, 0.5, 0.5))]

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        faces = get_faces()
        packet = FaceUpdatePacket(face_locations=faces)
        await websocket.send_text(packet.json())
        await asyncio.sleep(1)  # Send data every second

def main():
    uvicorn.run(app, host="0.0.0.0", port=6942)