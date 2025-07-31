from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from typing import List

import json

from app.img_utils import bytes_to_cv2_image, cv2_image_to_bytes, yuv420p_bytes_to_bgr
from app.ai_engine import AIEngine

import cv2
import numpy as np

app = FastAPI()

engine = AIEngine()

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
        print("generic websocket quit on life fr")

@app.websocket("/backend/sitting/upload")
async def upload_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            img = bytes_to_cv2_image(data)

            # Edit the image (draw a red circle)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img, status = engine.process_sitting(img)

            # Encode back to bytes

            await websocket.send_bytes(cv2_image_to_bytes(img))
            # print("shouldve sent bytes")
            await websocket.send_text(json.dumps({"status": status}))
            # print("shouldve sent status", status)

    except WebSocketDisconnect:
        print("sitting websocket quit fr")

@app.websocket("/backend/squatting/upload")
async def upload_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()

            img = yuv420p_bytes_to_bgr(data)

            # Edit the image (draw a red circle)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img, score = engine.process_squatting(img)

            # Encode back to bytes

            await websocket.send_bytes(cv2_image_to_bytes(img))
            print("shouldve sent bytes")
            await websocket.send_text(json.dumps({"score": score}))
            print("shouldve sent score", score)

    except WebSocketDisconnect:
        print("squat websocket quit fr")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8765)
