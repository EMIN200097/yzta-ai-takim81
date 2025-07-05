import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker
import numpy as np

def angle_between_points_cv2(A, B, C):
    # Convert points to np arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    # vectors
    BA = A - B
    BC = C - B

    # Calculate angle of vectors w.r.t x-axis in degrees
    angle1 = cv2.fastAtan2(BA[1], BA[0])
    angle2 = cv2.fastAtan2(BC[1], BC[0])

    # Calculate difference
    angle = angle2 - angle1

    # Normalize angle to [0, 360]
    if angle < 0:
        angle += 360

    # Smaller angle between vectors (<= 180)
    if angle > 180:
        angle = 360 - angle

    return angle