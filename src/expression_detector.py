import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FaceSense:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)

        # smoothing to prevent flickering
        self.history = deque(maxlen=5)

    def get_expression(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Convert landmarks to array
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])

        # Bounding box
        xs = points[:, 0]
        ys = points[:, 1]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # ---------- EXTRACT IMPORTANT LANDMARKS ----------
        left_mouth = points[61]
        right_mouth = points[291]
        upper_lip = points[13]
        lower_lip = points[14]

        left_brow = points[70]
        right_brow = points[300]

        # ---------- COMPUTE FEATURES ----------

        # Smile width (bigger = smile)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)

        # Lip openness (bigger = smile)
        lip_gap = lower_lip[1] - upper_lip[1]

        # Mouth curve: positive = smile, negative = sad
        curve = ((left_mouth[1] + right_mouth[1]) / 2) - upper_lip[1]

        # Eyebrow height (lower = angry)
        brow_drop = ((left_brow[1] + right_brow[1]) / 2) - upper_lip[1]

        # ---------- SCORE SYSTEM ----------
        happy_score = (mouth_width * 0.02) + (lip_gap * 0.04) + (curve * 0.3)
        sad_score = (-curve * 0.4)
        angry_score = (-brow_drop * 0.4)

        # Normalize scores to positive only
        scores = {
            "Happy": max(happy_score, 0),
            "Sad": max(sad_score, 0),
            "Angry": max(angry_score, 0),
            "Neutral": 0.3  # baseline
        }

        # Best expression
        expression = max(scores, key=scores.get)
        confidence = min(scores[expression] / 5, 1.0)
        confidence = round(confidence, 2)

        # Smooth prediction
        self.history.append(expression)
        smooth_expression = max(set(self.history), key=self.history.count)

        return smooth_expression, confidence, bbox


    # blink detection, drowsiness / eye-closure
    def eye_aspect_ratio(self, eye_points): #unused right now
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        return (v1 + v2) / (2.0 * h)

    def mouth_curvature(self, left, right, top):
        mid_y = (left[1] + right[1]) / 2
        return mid_y - top[1]

    def classify_expression(self, ear, curve):
        if curve > 4:
            return "Happy"
        if curve < -3:
            return "Sad"
        if ear > 0.30:
            return "Angry"
        return "Neutral"
