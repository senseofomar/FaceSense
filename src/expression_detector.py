import cv2
import mediapipe as mp
import numpy as np
from collections import deque

from utils.draw_debug_features import draw_debug_features
from utils.draw_feature_bars import draw_feature_bars


class FaceSense:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(refine_landmarks=True)

        # smoothing to prevent flickering
        self.history = deque(maxlen=5)

        # debug
        self.debug = False
        self.debug_counter = 0

        # expose last raw features (for CSV logging, dashboard, etc.)
        self.last_mouth_width = 0.0
        self.last_lip_gap = 0.0
        self.last_curve = 0.0
        self.last_brow = 0.0

    def set_debug(self, mode: bool):
        self.debug = mode


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
        # for i, (x, y) in enumerate(points):
        #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        #     if i in (61, 291, 13, 14, 70, 300):
        #         cv2.putText(frame, str(i), (x + 3, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

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

        # raw feature overlay on live frames
        draw_debug_features(frame, mouth_width, lip_gap, curve, brow)

        # ---------- SCORE SYSTEM ----------
        happy_score = 2.0 * mouth_width_n + 3.0 * lip_gap_n + 2.0 * curve_up
        sad_score = 8.0 * curve_down  # mouth pulled down
        angry_score = 7.0 * curve_down  # also lower mouth / tension, but weaker

        # Hard guard: if mouth isn't really "smiley", kill happy_score
        if lip_gap_n < 0.01 and curve_up < 0.01:
            happy_score = 0.0

        # Normalize scores to positive only
        scores = {
            "Happy": max(happy_score, 0),
            "Sad": max(sad_score, 0),
            "Angry": max(angry_score, 0),
            "Neutral": 0.3  # baseline
        }

        # computing confidence
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        total = sum(scores.values()) + 1e-6

        if best_score < 0.07:
            expression = "Neutral"
        else :
            expression = best_label

        confidence = best_score / total  # like softmax-ish normalisation
        confidence = float(round(min(max(confidence, 0.0), 1.0), 2))
        # expose for external logging
        self.last_mouth_width = mouth_width
        self.last_lip_gap = lip_gap
        self.last_curve = curve
        self.last_brow = brow

        # If debug mode enabled, draw landmarks (if you had them) & debug overlays
        if self.debug:
            try:
                # draw numeric lines
                draw_debug_features(frame, mouth_width, lip_gap, curve, brow)
                # compute normalized values for bars using face size to be in 0..1 range
                mouth_width_n = mouth_width / max(1.0, face_w)
                lip_gap_n = lip_gap / max(1.0, face_h)
                curve_n = curve / max(1.0, face_h)
                brow_n = (brow / max(1.0, face_h)) * -1.0  # if brow negative for "down", invert for visualization
                draw_feature_bars(frame, mouth_width_n, lip_gap_n, curve_n, brow_n)
            except Exception as e:
                # debug drawing must not break detection
                print("Debug draw error:", e)

        # Smooth prediction (existing code)
        self.history.append(best_label)
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


