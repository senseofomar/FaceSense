import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class FaceSense:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)
        self.debug_counter = 0
        # smoothing to prevent flickering
        self.history = deque(maxlen=5)

    def get_expression(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # self.face_mesh.process(rgb) → runs Mediapipe FaceMesh model
        # results.multi_face_landmarks → list of all detected faces
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Convert landmarks to array
        points = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark])

        # debugging test 1 - checking the landmark points
        for i, (x, y) in enumerate(points):
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
            if i in (61, 291, 13, 14, 70, 300):
                cv2.putText(frame, str(i), (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        # Bounding box, fix - normalized the face size
        xs = points[:, 0]
        ys = points[:, 1]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        bbox = (x1, y1, x2, y2)

        face_w = max(1, x2 - x1)  # avoid division by zero
        face_h = max(1, y2 - y1)

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
        curve = self.mouth_curvature(left_mouth, right_mouth, upper_lip)

        # Eyebrow height (lower = angry)
        brow = ((left_brow[1] + right_brow[1]) / 2) - upper_lip[1]

        # normalize by face size
        mouth_width_n = mouth_width / face_w
        lip_gap_n = lip_gap / face_h
        curve_n = curve / face_h
        brow_n = brow / face_h

        mouth_width_n = max(0.0, mouth_width_n)
        lip_gap_n = max(0.0, lip_gap_n)

        # Separate positive / negative curvature
        curve_up = max(0.0, curve_n)  # smile
        curve_down = max(0.0, -curve_n)  # frown

        # ---------- SCORE SYSTEM ----------
        happy_score = 3.0 * mouth_width_n + 2.5 * lip_gap_n + 5.0 * curve_up
        sad_score = 4.0 * curve_down  # mouth pulled down
        angry_score = 1.5 * curve_down  # also lower mouth / tension, but weaker

        # Normalize scores to positive only
        scores = {
            "Happy": max(happy_score, 0),
            "Sad": max(sad_score, 0),
            "Angry": max(angry_score, 0),
            "Neutral": 0.3  # baseline
        }

        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        total = sum(scores.values()) + 1e-6
        confidence = best_score / total  # like softmax-ish normalisation
        confidence = float(round(min(max(confidence, 0.0), 1.0), 2))

        # Best expression
        expression = max(scores, key=scores.get)
        confidence = min(scores[expression] / 5, 1.0)
        confidence = round(confidence, 2)

        # Smooth prediction
        self.history.append(expression)
        smooth_expression = max(set(self.history), key=self.history.count)

        # Reducing the prints on every call from 30 fps to 3 fps
        self.debug_counter += 1
        if self.debug_counter % 10 ==0:
        #debugging test 2 - raw feature values
            print(f"mouth_width={mouth_width:.1f}, lip_gap={lip_gap:.1f}, curve={curve:.2f}, brow={brow:.1f}")

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


