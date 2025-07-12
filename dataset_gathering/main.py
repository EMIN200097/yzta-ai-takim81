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

        self.start_time = time.time()

        # Initialize video capture
        # Use 0 for webcam, or replace with video file path
        if video_name:
            self.input = cv2.VideoCapture(self.script_dir + '/' + 'input' + '/' + video_name)
        else:
            # Default to webcam
            self.input = cv2.VideoCapture(0)
        
        self.frame_shape = None  # Initialize frame shape to None
    
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
        # This method calculates the sitting position based on the landmarks
        # 'log' is a boolean to control whether to log the results to the csv file to be used for training.
        log = False # TBC by the user in runtime

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
            new_row.append(self.input_correct)
            for j in range(len(landmarks_of_interest)-2):
                new_row.append(
                    geo_engine.angle_between_points_cv2(
                        (landmarks_of_interest[j].x, landmarks_of_interest[j].y),
                        (landmarks_of_interest[j + 1].x, landmarks_of_interest[j + 1].y),
                        (landmarks_of_interest[j + 2].x, landmarks_of_interest[j + 2].y)
                    )
                )
            
            # If logging is enabled, append to the CSV file
            # Since we have 5 landmarks we can account for 3 angles
            # This is a very inefficient way to do it, but it will do for local testing & gathering schenanigans
            if log:
                with open(self.script_dir + '/' + 'output/sitting_records.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
            
                    # Write header if file is empty
                    csvfile.seek(0, 2)
                    if csvfile.tell() == 0:
                        writer.writerow(['correct', 'shoulder', 'hip', 'ankle'])
                        
                    writer.writerow(new_row)
                    print(f"Logged angles: {new_row}")

            cv2.imshow('Pose Landmarker', image)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('l'):
                log = not log
                if log:
                    print("Logging enabled. Angles will be saved to output/sitting_records.csv")
                else:
                    print("Logging disabled. Angles will not be saved.")

            elif key == ord('q'):
                break

if __name__ == "__main__":
    pose_landmarker = PoseLandmarkerWrapper(video_name='1.mp4', correct=True)

    # Comment in and out the desired method to run since this is a demo
    # pose_landmarker.display_all_nodes_loop()
    pose_landmarker.process_sitting()