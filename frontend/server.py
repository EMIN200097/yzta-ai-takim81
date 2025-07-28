from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from typing import List

import cv2
import numpy as np

app = FastAPI()

# ilk posecorelu anlamda backend denemesi.

def bytes_to_cv2_image(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cv2_image_to_bytes(img: np.ndarray, format: str = ".jpg") -> bytes:
    success, encoded_img = cv2.imencode(format, img)
    if not success:
        raise ValueError("Encoding failed")
    return encoded_img.tobytes()


@app.get("/")
async def get():
    with open("index.html") as f:
        return HTMLResponse(f.read())

@app.websocket("/backend/upload")
async def upload_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            img = bytes_to_cv2_image(data)

            # Edit the image (draw a red circle)
            cv2.circle(img, center=(100, 100), radius=50, color=(0, 0, 255), thickness=3)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Encode back to bytes
            data = cv2_image_to_bytes(img, ".jpg")

            await websocket.send_bytes(data)

    except WebSocketDisconnect:
        print("websocket quit on life fr")

