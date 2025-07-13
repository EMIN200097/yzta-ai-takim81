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

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import geo_engine

class MLPlayground:

    def __init__(self, model_name='pose_landmarker.task', video_name=None):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        base_options = python.BaseOptions(model_asset_path=self.script_dir + '/' + 'models/' + model_name)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
            running_mode=RunningMode.VIDEO
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

        self.start_time = time.time()

        # Initialize video capture
        # Use 0 for webcam, or replace with video file path
        if video_name:
            self.input = cv2.VideoCapture(self.script_dir + '/' + 'input' + '/' + video_name)
        else:
            # Default to webcam
            self.input = cv2.VideoCapture(0)
        
        self.frame_shape = None  # Initialize frame shape to None


        ## ACTUAL ML STUFF
        # Step 1: Load CSV
        knn_df = pd.read_csv(self.script_dir + '/output/' + 'sitting_records.csv')  # Replace with your actual CSV path

        # Step 2: Separate features and labels
        knn_y = knn_df.iloc[:, 0]    # Label is the first column
        knn_X = knn_df.iloc[:, 1:]   # Features are the rest

        # Step 3: Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(knn_X, knn_y, test_size=0.2, random_state=42)

        # Step 4: Train KNN
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(X_train, y_train)

        # Step 5: Predict and evaluate
        y_pred = self.knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("KNN Accuracy (training data):", accuracy)

    def process_frame(self, image: cv2.typing.MatLike):

        if self.frame_shape is None:
            # Beware that this for some goddamn reason returns height first, not width
            # Keep the weird opposite way thingy in order to help LLMs advise you
            self.frame_shape = image.shape

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Use timestamp in millis
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        # Process the image with the pose landmarker
        detection_result: PoseLandmarkerResult = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        # Only return the best detection
        if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            return landmarks
        else:
            return None

    def display_all_nodes_loop(self):
        
        while self.input.isOpened():

            success, image = self.input.read()
            if not success:
                assert False, "Failed to read frame from input"

            landmarks = self.process_frame(image=image)
            
            for landmark in landmarks:
                cx = int(landmark.x * self.frame_shape[1])
                cy = int(landmark.y * self.frame_shape[0])
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
            
            cv2.imshow('Debug: All Landmarks', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def process_sitting(self):

        # Define the indexes of the landmarks we are interested in
        desired_indexes = [7,8,11,12,23,24,25,26,27,28]
        left_indexes = [7, 11, 23, 25, 27]
        right_indexes = [8, 12, 24, 26, 28]

        while self.input.isOpened():

            success, image = self.input.read()
            if not success:
                assert False, "Failed to read frame from input"

            landmarks = self.process_frame(image=image)

            # Pick the best side based on the visibility of the landmarks
            left_direction_score = 0
            left_direction_landmarks = list()
            right_direction_score = 0
            right_direction_landmarks = list()

            landmarks_of_interest = list()

            if landmarks == None:
                continue

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
            else:
                landmarks_of_interest = right_direction_landmarks

            for landmark in landmarks_of_interest:
                cx = int(landmark.x * self.frame_shape[1])
                cy = int(landmark.y * self.frame_shape[0])
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

            # Calculate angles between the landmarks
            new_row = list()
            for j in range(len(landmarks_of_interest)-2):
                new_row.append(
                    geo_engine.angle_between_points_cv2(
                        (landmarks_of_interest[j].x, landmarks_of_interest[j].y),
                        (landmarks_of_interest[j + 1].x, landmarks_of_interest[j + 1].y),
                        (landmarks_of_interest[j + 2].x, landmarks_of_interest[j + 2].y)
                    )
                )
            column_names = ['shoulder', 'hip', 'ankle']
            this_frame_df = pd.DataFrame([new_row], columns=column_names)
            print(self.knn.predict(this_frame_df))

            cv2.imshow('Pose Landmarker', image)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break


if __name__ == "__main__":
    ml_playground = MLPlayground()

    ml_playground.process_sitting()

