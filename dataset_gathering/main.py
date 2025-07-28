
#main.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
import numpy as np
import time
import csv
import os

import geo_engine


class PoseLandmarkerWrapper:

    def __init__(self, model_name='pose_landmarker.task', video_name=None, correct=True):
        print("🚀 PoseLandmarkerWrapper başlatılıyor...")

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_correct = correct

        base_options = python.BaseOptions(model_asset_path=self.script_dir + '/' + 'models/' + model_name)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=RunningMode.VIDEO
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("✅ Pose landmarker modeli yüklendi!")

        self.start_time = time.time()

        # Initialize video capture
        if video_name:
            video_path = self.script_dir + '/' + 'input' + '/' + video_name
            print(f" Video dosyası yükleniyor: {video_path}")
            self.input = cv2.VideoCapture(video_path)
        else:
            print(" Webcam açılıyor")
            self.input = cv2.VideoCapture(0)

        # Webcam kontrolü
        if not self.input.isOpened():
            print(" HATA: Webcam açılamadı!")
            return
        else:
            print(" Webcam açıldı")

        self.frame_shape = None

    def process_frame(self, image: cv2.typing.MatLike):
        if self.frame_shape is None:
            self.frame_shape = image.shape
            print(f"📐 Frame boyutu: {self.frame_shape}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        detection_result: PoseLandmarkerResult = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            return landmarks
        else:
            return None

    def display_all_nodes_loop(self):
        print(" Display all nodes loop başlatılıyor...")
        print(" Kontroller: 'q' tuşu ile çıkış")

        frame_count = 0

        while self.input.isOpened():
            success, image = self.input.read()
            if not success:
                print(" Frame okunamadı!")
                break

            frame_count += 1
            if frame_count % 30 == 0:  # Her 30 frame'de bir bilgi ver
                print(f" Frame #{frame_count} işleniyor...")

            landmarks = self.process_frame(image=image)

            if landmarks:
                for landmark in landmarks:
                    cx = int(landmark.x * self.frame_shape[1])
                    cy = int(landmark.y * self.frame_shape[0])
                    cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            else:
                cv2.putText(image, "Pose bulunamadi", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Debug: All Landmarks', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Kullanıcı çıkış yaptı")
                break

        self.input.release()
        cv2.destroyAllWindows()
        print("Program sonlandı!")

    def process_sitting(self):
        print(" Sitting process başlatılıyor...")
        print(" Kontroller:")
        print("  'l' tuşu: Logging aç/kapat")
        print("  'q' tuşu: Çıkış")

        log = False
        desired_indexes = [7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
        left_indexes = [7, 11, 23, 25, 27]
        right_indexes = [8, 12, 24, 26, 28]
        frame_count = 0

        while self.input.isOpened():
            success, image = self.input.read()
            if not success:
                print("Frame okunamadı!")
                break

            frame_count += 1
            landmarks = self.process_frame(image=image)

            log_text = "Logging: ON" if log else "Logging: OFF"
            log_color = (0, 255, 0) if log else (0, 0, 255)
            cv2.putText(image, log_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, log_color, 2)

            if landmarks == None:
                cv2.putText(image, "Pose bulunamadi", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Pose Landmarker', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            left_direction_score = 0
            left_direction_landmarks = list()
            right_direction_score = 0
            right_direction_landmarks = list()
            landmarks_of_interest = list()

            for i, landmark in enumerate(landmarks):
                if i not in desired_indexes:
                    continue
                if i in left_indexes:
                    left_direction_score += landmark.visibility
                    left_direction_landmarks.append(landmark)
                elif i in right_indexes:
                    right_direction_score += landmark.visibility
                    right_direction_landmarks.append(landmark)

            if left_direction_score > right_direction_score:
                landmarks_of_interest = left_direction_landmarks
                side_text = "Taraf: SOL"
            else:
                landmarks_of_interest = right_direction_landmarks
                side_text = "Taraf: SAG"

            # Taraf bilgisini göster
            cv2.putText(image, side_text, (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            for landmark in landmarks_of_interest:
                cx = int(landmark.x * self.frame_shape[1])
                cy = int(landmark.y * self.frame_shape[0])
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # Calculate angles between the landmarks
            new_row = list()
            new_row.append(self.input_correct)
            try:
                for j in range(len(landmarks_of_interest) - 2):
                    angle = geo_engine.angle_between_points_cv2(
                        (landmarks_of_interest[j].x, landmarks_of_interest[j].y),
                        (landmarks_of_interest[j + 1].x, landmarks_of_interest[j + 1].y),
                        (landmarks_of_interest[j + 2].x, landmarks_of_interest[j + 2].y)
                    )
                    new_row.append(angle)

                # Açıları ekranda göster
                for i, angle in enumerate(new_row[1:]):  # İlk eleman correct flag'i
                    cv2.putText(image, f"Aci {i + 1}: {angle:.1f}", (10, 150 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            except Exception as e:
                print(f" Açı hesaplama hatası: {e}")

            # If logging is enabled, append to the CSV file
            if log and len(new_row) > 1:
                csv_path = self.script_dir + '/' + 'output/sitting_records.csv'
                try:
                    with open(csv_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)

                        # Write header if file is empty
                        if os.path.getsize(csv_path) == 0:
                            writer.writerow(['correct', 'shoulder', 'hip', 'ankle'])

                        writer.writerow(new_row)
                        print(f"📊 Kayıt: {new_row}")
                except Exception as e:
                    print(f"❌ CSV yazma hatası: {e}")

            cv2.imshow('Pose Landmarker', image)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('l'):
                log = not log
                status = "AÇILDI" if log else "KAPANDI"
                print(f" Logging {status}")

            elif key == ord('q'):
                print("🚪 Kullanıcı çıkış yaptı")
                break

        self.input.release()
        cv2.destroyAllWindows()
        print("✅ Program sonlandı!")


if __name__ == "__main__":
    print("=" * 50)
    print("🚀 AI POSE DETECTION BAŞLATIILIYOR")
    print("=" * 50)

    try:
        pose_landmarker = PoseLandmarkerWrapper(video_name=None, correct=True)  # None = webcam

        # İstediğiniz modu seçin:
        # pose_landmarker.display_all_nodes_loop()  # Tüm noktaları göster
        pose_landmarker.process_sitting()  # Oturma analizi (varsayılan)

    except Exception as e:
        print(f"💥 Program hatası: {e}")
        input("Çıkmak için Enter'a basın...")