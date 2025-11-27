import cv2
import mediapipe as mp
import numpy as np

class FaceSense:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)

    def get_expression(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Convert to 2D points
        points = []
        for lm in landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))
        points = np.array(points)

        # Eye EAR
        left_eye = points[[33, 160, 158, 133, 153, 144]]
        right_eye = points[[362, 385, 387, 263, 373, 380]]

        left_ear = self.eye_aspect_ratio(left_eye)
        right_ear = self.eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2

        # Mouth curvature
        left = points[61]   # left lip corner
        right = points[291] # right lip corner
        top = points[0]     # upper lip
        curve = self.mouth_curvature(left, right, top)

        return self.classify_expression(ear, curve)

    def eye_aspect_ratio(self, eye_points):
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
