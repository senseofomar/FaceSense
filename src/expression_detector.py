import cv2
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector


# --------------------------------------------
# Eye Aspect Ratio (EAR)
# --------------------------------------------

def eye_aspect_ratio(eye_points):
    """
    Calculates eye aspect ratio from 6 landmark points (x,y).
    EAR drops when eyes close, increases when open wide.
    """
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


# --------------------------------------------
# Mouth Curvature (smile vs frown)
# --------------------------------------------

def mouth_curvature(left, right, top):
    """
    Positive = smile (Happy)
    Negative = frown (Sad)
    Near zero = Neutral
    """
    mid_y = (left[1] + right[1]) / 2
    return mid_y - top[1]


# --------------------------------------------
# Expression Classification
# --------------------------------------------

def classify_expression(ear, curve):
    """
    Expression rules using EAR + Mouth curvature.
    """
    if curve > 6:        # Clear smile
        return "Happy"
    if curve < -4:       # Clear frown
        return "Sad"
    if ear > 0.31:       # Eyes wide open
        return "Angry"
    return "Neutral"


# --------------------------------------------
# FACE SENSE DETECTOR (for webcam or images)
# --------------------------------------------

class FaceSense:
    def __init__(self):
        # Initialize the 468-landmark face mesh detector
        self.detector = FaceMeshDetector(maxFaces=1)

        # Landmark indices (FaceMesh reference)
        self.left_eye_ids = [33, 160, 158, 133, 153, 144]
        self.right_eye_ids = [362, 385, 387, 263, 373, 380]

        # Mouth points (simple version)
        self.mouth_left = 308
        self.mouth_right = 78
        self.mouth_top = 13

    def detect_expression(self, img):
        """
        Returns (expression, img_with_landmarks)
        """
        img, faces = self.detector.findFaceMesh(img, draw=False)

        if not faces:
            return "No Face", img

        face = faces[0]  # 468 points

        # Eye landmarks
        left_eye = np.array([face[i] for i in self.left_eye_ids])
        right_eye = np.array([face[i] for i in self.right_eye_ids])

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2

        # Mouth landmarks
        left = np.array(face[self.mouth_left])
        right = np.array(face[self.mouth_right])
        top = np.array(face[self.mouth_top])

        curve = mouth_curvature(left, right, top)

        # Final classification
        expression = classify_expression(ear, curve)

        return expression, img
