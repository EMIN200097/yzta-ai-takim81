from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
import json
import base64
import asyncio
import os
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import logging
from pydantic import BaseModel
import io
from PIL import Image

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import geo_engine
except ImportError:
    logger.error("geo_engine.py modülü bulunamadı!")

app = FastAPI(
    title="AI Pose Detection API",
    description="Real-time pose detection ve analiz sistemi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class PoseData(BaseModel):
    timestamp: datetime
    exercise_type: str
    angles: List[float]
    is_correct: bool
    confidence: float


class AnalysisResult(BaseModel):
    pose_detected: bool
    exercise_type: str
    angles: Dict[str, float]
    feedback: str
    score: float
    is_correct: bool


class PoseAnalyzer:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.script_dir, 'models', 'pose_landmarker.task')
        self.pose_landmarker = None
        self.start_time = time.time()
        self.setup_model()

        # ML modelleri
        self.sitting_model = None
        self.squat_model = None
        self.load_ml_models()

    def setup_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model dosyası bulunamadı: {self.model_path}")

            base_options = python.BaseOptions(model_asset_path=self.model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1,
                running_mode=RunningMode.VIDEO
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            logger.info(" Pose Landmarker modeli yüklendi!")

        except Exception as e:
            logger.error(f" Model yükleme hatası: {e}")
            raise

    def load_ml_models(self):

        try:
            # Sitting model
            sitting_csv = os.path.join(self.script_dir, 'output', 'sitting_records.csv')
            if os.path.exists(sitting_csv):
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.model_selection import train_test_split

                df = pd.read_csv(sitting_csv)
                if len(df) > 5:
                    X = df.iloc[:, 1:]
                    y = df.iloc[:, 0]

                    self.sitting_model = KNeighborsClassifier(n_neighbors=3)
                    self.sitting_model.fit(X, y)
                    logger.info(" Sitting ML modeli yüklendi!")



        except Exception as e:
            logger.error(f"ML model yükleme hatası: {e}")

    def process_frame(self, image: np.ndarray) -> Optional[List]:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((time.time() - self.start_time) * 1000)
            result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                return result.pose_landmarks[0]
            return None

        except Exception as e:
            logger.error(f"Frame işleme hatası: {e}")
            return None

    def analyze_pose(self, landmarks, exercise_type: str = "sitting") -> AnalysisResult:
        try:
            if not landmarks:
                return AnalysisResult(
                    pose_detected=False,
                    exercise_type=exercise_type,
                    angles={},
                    feedback="Pose tespit edilemedi",
                    score=0.0,
                    is_correct=False
                )


            angles = self.calculate_angles(landmarks, exercise_type)

            is_correct, confidence = self.predict_pose(angles, exercise_type)

            feedback = self.generate_feedback(angles, is_correct, exercise_type)

            score = confidence * 100 if is_correct else (1 - confidence) * 100

            return AnalysisResult(
                pose_detected=True,
                exercise_type=exercise_type,
                angles=angles,
                feedback=feedback,
                score=score,
                is_correct=is_correct
            )

        except Exception as e:
            logger.error(f"Pose analiz hatası: {e}")
            return AnalysisResult(
                pose_detected=False,
                exercise_type=exercise_type,
                angles={},
                feedback=f"Analiz hatası: {str(e)}",
                score=0.0,
                is_correct=False
            )

    def calculate_angles(self, landmarks, exercise_type: str) -> Dict[str, float]:
        angles = {}

        try:
            if exercise_type == "sitting":

                desired_indexes = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
                left_indexes = [7, 11, 23, 25, 27]
                right_indexes = [8, 12, 24, 26, 28]

                left_score = sum(landmarks[i].visibility for i in left_indexes)
                right_score = sum(landmarks[i].visibility for i in right_indexes)

                selected_indexes = left_indexes if left_score > right_score else right_indexes
                selected_landmarks = [landmarks[i] for i in selected_indexes]

                for j in range(len(selected_landmarks) - 2):
                    angle = geo_engine.angle_between_points_cv2(
                        (selected_landmarks[j].x, selected_landmarks[j].y),
                        (selected_landmarks[j + 1].x, selected_landmarks[j + 1].y),
                        (selected_landmarks[j + 2].x, selected_landmarks[j + 2].y)
                    )
                    angle_names = ['shoulder', 'hip', 'ankle']
                    if j < len(angle_names):
                        angles[angle_names[j]] = round(angle, 2)

                angles['selected_side'] = 'left' if left_score > right_score else 'right'

        except Exception as e:
            logger.error(f"Açı hesaplama hatası: {e}")

        return angles

    def predict_pose(self, angles: Dict[str, float], exercise_type: str) -> tuple[bool, float]:
        """ML modeli ile pose prediction yap"""
        try:
            if exercise_type == "sitting" and self.sitting_model:
                # Angles'ı model için hazırla
                angle_values = []
                for key in ['shoulder', 'hip', 'ankle']:
                    angle_values.append(angles.get(key, 0.0))

                if len(angle_values) == 3:
                    prediction = self.sitting_model.predict([angle_values])[0]
                    probabilities = self.sitting_model.predict_proba([angle_values])[0]
                    confidence = max(probabilities)

                    return bool(prediction), float(confidence)

            # Varsayılan basit kural tabanlı analiz
            return self.rule_based_analysis(angles, exercise_type)

        except Exception as e:
            logger.error(f"Prediction hatası: {e}")
            return False, 0.5

    def rule_based_analysis(self, angles: Dict[str, float], exercise_type: str) -> tuple[bool, float]:
        if exercise_type == "sitting":
            shoulder_angle = angles.get('shoulder', 0)
            hip_angle = angles.get('hip', 0)

            is_correct = (140 <= shoulder_angle <= 180) and (80 <= hip_angle <= 120)
            confidence = 0.8 if is_correct else 0.6

            return is_correct, confidence

        return False, 0.5

    def generate_feedback(self, angles: Dict[str, float], is_correct: bool, exercise_type: str) -> str:
        if not angles:
            return "Pose tespit edilemedi. Kameraya daha yakın olun."

        if exercise_type == "sitting":
            if is_correct:
                return " Mükemmel duruş"
            else:
                feedback_parts = []

                shoulder_angle = angles.get('shoulder', 0)
                if shoulder_angle < 140:
                    feedback_parts.append("🔺 Omuzlarınızı geri çekin")

                hip_angle = angles.get('hip', 0)
                if hip_angle < 80:
                    feedback_parts.append("🔺 Sırtınızı daha dik tutun")
                elif hip_angle > 120:
                    feedback_parts.append("🔺 Fazla öne eğilmeyin")

                return " | ".join(feedback_parts) if feedback_parts else "⚠️ Duruşunuzu düzeltin"

        return "Analiz tamamlandı"


# Global analyzer instance
analyzer = PoseAnalyzer()


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"🔌 Yeni WebSocket bağlantısı: {len(self.active_connections)} aktif")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"🔌 WebSocket bağlantısı kesildi: {len(self.active_connections)} aktif")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)


manager = ConnectionManager()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    """Ana sayfa"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Pose Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 50px; background: #f0f2f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #1a73e8; text-align: center; }
            .endpoint { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .method { color: #28a745; font-weight: bold; }
            .url { color: #6c757d; font-family: monospace; }
            a { color: #1a73e8; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1> AI Pose Detection </h1>
            <p>Real-time pose detection ve analiz </p>

            <h2>📡 API Endpoints:</h2>

            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/docs</span></div>
                <div>Interactive API documentation (Swagger UI)</div>
            </div>

            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/health</span></div>
                <div>API health check</div>
            </div>

            <div class="endpoint">
                <div><span class="method">POST</span> <span class="url">/analyze-image</span></div>
                <div>Tek görüntü analizi</div>
            </div>

            <div class="endpoint">
                <div><span class="method">WebSocket</span> <span class="url">/ws/pose-analysis</span></div>
                <div>Real-time pose analysis</div>
            </div>

            <div class="endpoint">
                <div><span class="method">GET</span> <span class="url">/models/status</span></div>
                <div>Model durumu</div>
            </div>

            <h2>🚀 Hızlı Başlangıç:</h2>
            <p>
                <a href="/docs" target="_blank">🔗 API Documentation</a> |  
                <a href="/static/demo.html" target="_blank">🎬 Live Demo</a> |
                <a href="/health" target="_blank">❤️ Health Check</a>
            </p>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """API sağlık kontrolü"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models_loaded": {
            "pose_landmarker": analyzer.pose_landmarker is not None,
            "sitting_model": analyzer.sitting_model is not None
        }
    }


@app.get("/models/status")
async def models_status():
    """Model durumları"""
    return {
        "pose_landmarker": {
            "loaded": analyzer.pose_landmarker is not None,
            "model_path": analyzer.model_path,
            "exists": os.path.exists(analyzer.model_path)
        },
        "ml_models": {
            "sitting_model": analyzer.sitting_model is not None,
            "squat_model": analyzer.squat_model is not None
        }
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...), exercise_type: str = "sitting"):
    """Tek görüntü analizi"""
    try:

        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Sadece görüntü dosyaları kabul edilir")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        landmarks = analyzer.process_frame(image_np)
        result = analyzer.analyze_pose(landmarks, exercise_type)

        return {
            "filename": file.filename,
            "analysis": result.dict(),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Görüntü analiz hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {str(e)}")


@app.websocket("/ws/pose-analysis")
async def websocket_pose_analysis(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            # Kullanıcıdan veri al (base64 encoded image)
            data = await websocket.receive_text()

            try:
                # JSON parse et
                message = json.loads(data)

                if message.get("type") == "frame":

                    image_data = base64.b64decode(message["image"].split(",")[1])
                    image = Image.open(io.BytesIO(image_data))
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    landmarks = analyzer.process_frame(image_np)
                    exercise_type = message.get("exercise_type", "sitting")
                    result = analyzer.analyze_pose(landmarks, exercise_type)

                    response = {
                        "type": "analysis_result",
                        "data": result.dict(),
                        "timestamp": datetime.now().isoformat()
                    }

                    await manager.send_personal_message(json.dumps(response), websocket)

            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": "Invalid JSON format"}),
                    websocket
                )
            except Exception as e:
                await manager.send_personal_message(
                    json.dumps({"type": "error", "message": str(e)}),
                    websocket
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket hatası: {e}")
        manager.disconnect(websocket)


@app.post("/save-training-data")
async def save_training_data(data: PoseData):
    """Training verisi kaydet"""
    try:

        csv_path = os.path.join(analyzer.script_dir, 'output', f'{data.exercise_type}_records.csv')

        row_data = [data.is_correct] + data.angles
        df = pd.DataFrame([row_data])

        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:

            columns = ['correct'] + [f'angle_{i}' for i in range(len(data.angles))]
            df.columns = columns
            df.to_csv(csv_path, index=False)

        return {"status": "success", "message": "Veri kaydedildi"}

    except Exception as e:
        logger.error(f"Veri kaydetme hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("🚀 AI Pose Detection API başlatılıyor...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )