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

        # state memory
        self.current_expression = "Neutral"

        self.curve_history = deque(maxlen=15)
        self.lip_gap_history = deque(maxlen=15)
        self.brow_history = deque(maxlen=15)

        self.angry_counter = 0
        self.sad_counter = 0

        self.brow_baseline = None
        self.baseline_frames = 0

        self.curve_baseline = None
        self.curve_baseline_frames = 0

        # debug
        self.debug = False
        self.debug_counter = 0

        # expose last raw features (for CSV logging, dashboard, etc.)
        self.last_mouth_width = 0.0
        self.last_lip_gap = 0.0
        self.last_curve = 0.0
        self.last_brow = 0.0

    def set_debug(self, mode: bool):
        self.debug = bool(mode)


    def get_expression(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # self.face_mesh.process(rgb) → runs Mediapipe FaceMesh model
        # results.multi_face_landmarks → list of all detected faces
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape


        # Convert landmarks to pixel coordinates
        points = np.array([
            (int(lm.x * w), int(lm.y * h))
            for lm in landmarks.landmark
        ])

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

        # ---------- COMPUTE RAW FEATURES ----------

        # Smile width (bigger = smile)
        mouth_width = np.linalg.norm(left_mouth - right_mouth)

        # Lip openness (bigger = smile)
        lip_gap = lower_lip[1] - upper_lip[1]

        # Mouth curve: positive = smile, negative = sad
        curve = self.mouth_curvature(left_mouth, right_mouth, upper_lip)

        # Eyebrow height (lower = angry)
        brow = ((left_brow[1] + right_brow[1]) / 2) - upper_lip[1]

        # expose raw values for external logging
        self.last_mouth_width = mouth_width
        self.last_lip_gap = lip_gap
        self.last_curve = curve
        self.last_brow = brow

        # ---------- NORMALIZATION (distance-invariant) ----------
        mouth_width_n = max(0.0, mouth_width / face_w)
        lip_gap_n = max(0.0, lip_gap / face_h)

        curve_n = curve / face_h

        # --- Curve baseline calibration ---
        if self.curve_baseline is None:
            self.curve_baseline_frames += 1
            if self.curve_baseline_frames <= 30:
                if self.curve_baseline is None:
                    self.curve_baseline = curve_n
                else:
                    self.curve_baseline = (
                            0.9 * self.curve_baseline + 0.1 * curve_n
                    )

        # Brow drop: positive = eyebrows lowered (anger)
        brow_drop_n = max(0.0, -brow / face_h)

        # --- Baseline calibration (first ~30 frames) ---
        if self.brow_baseline is None:
            self.baseline_frames += 1
            if self.baseline_frames <= 30:
                if self.brow_baseline is None:
                    self.brow_baseline = brow_drop_n
                else:
                    self.brow_baseline = (
                            0.9 * self.brow_baseline + 0.1 * brow_drop_n
                    )
            else:
                # baseline locked
                pass

        self.brow_history.append(brow_drop_n)
        avg_brow_drop = np.mean(self.brow_history)

        # Separate positive / negative curvature
        curve_up = max(0.0, curve_n) # smile
        curve_down = max(0.0, -curve_n)  # frown

        curve_delta = curve_n - self.curve_baseline

        brow_delta = brow_drop_n - self.brow_baseline
        brow_delta = max(0.0, brow_delta)

        self.curve_history.append(curve_n)
        self.lip_gap_history.append(lip_gap_n)


        # ---------- DEBUG (PRINT NORMALIZED VALUES) ----------
        self.debug_counter += 1
        if self.debug and self.debug_counter % 15 == 0:
            print(
                f"[NORM] mw={mouth_width_n:.3f}, "
                f"lg={lip_gap_n:.3f}, "
                f"curve={curve_n:.3f}, "
                f"brow={brow_drop_n:.3f}"
            )

        # ---------- SCORE SYSTEM ----------
        happy_score = 4.0 * mouth_width_n + 3.0 * lip_gap_n + 2.0 * curve_up
        sad_score = 8.0 * curve_down  # mouth pulled down
        angry_score = 7.0 * curve_down  # also lower mouth / tension, but weaker

        # Normalize scores to positive only
        scores = {
            "Happy": max(happy_score, 0),
            "Sad": max(sad_score, 0),
            "Angry": max(angry_score, 0),
            "Neutral": 0.3  # baseline
        }

        # ---------- TEMPORAL SIGNALS ----------
        avg_curve = np.mean(self.curve_history)
        avg_lip_gap = np.mean(self.lip_gap_history)

        # --- Emotion signals ---
        angry_signal = (
                brow_delta > 0.025 and
                lip_gap_n < 0.01 and
                mouth_width_n < 0.36
        )

        if angry_signal:
            self.angry_counter += 1
        else:
            self.angry_counter = max(0, self.angry_counter - 1)

        is_angry = self.angry_counter >= 6

        sad_signal = (
                curve_delta < -0.025 and
                lip_gap_n < 0.01 and
                brow_delta < 0.01
        )

        if sad_signal:
            self.sad_counter += 1
        else:
            self.sad_counter = max(0, self.sad_counter - 1)

        is_sad = self.sad_counter >= 8

        # ---------- TEMPORAL HYSTERESIS + EMOTION LOGIC ----------

        if self.current_expression == "Happy":
            if mouth_width_n < 0.35:
                self.current_expression = "Neutral"

        elif self.current_expression == "Angry":
            if self.angry_counter < 3:
                self.current_expression = "Neutral"

        elif self.current_expression == "Sad":
            if self.sad_counter < 3:
                self.current_expression = "Neutral"


        else:
            # Neutral state → look for strong signals
            if mouth_width_n > 0.42 and lip_gap_n > 0.015:
                self.current_expression = "Happy"

            elif is_angry:
                self.current_expression = "Angry"

            elif is_sad:
                self.current_expression = "Sad"

            else:
                self.current_expression = "Neutral"


        expression = self.current_expression

        # Computing confidence
        if expression == "Happy":
            confidence = (mouth_width_n - 0.35) / 0.10


        elif expression == "Angry":
            confidence = min((avg_brow_drop - 0.08) / 0.08, 1.0)


        elif expression == "Sad":
            confidence = min((0.36 - mouth_width_n) / 0.08, 1.0)

        else:
            confidence = 0.6

        confidence = round(max(0.0, min(confidence, 1.0)), 2)

        print("brow_delta:", round(brow_delta, 3))

        # If debug mode enabled, draw landmarks (if you had them) & debug overlays
        if self.debug:
            try:
                # draw numeric lines / raw values distance dependent
                draw_debug_features(frame,
                                    mouth_width,
                                    lip_gap,
                                    curve,
                                    brow
                                )
                # clamp for visualization
                mw_v = min(max(mouth_width_n, 0.0), 1.0)
                lg_v = min(max(lip_gap_n, 0.0), 1.0)
                cv_v = min(max(curve_n, -1.0), 1.0)
                br_v = min(max(-brow_drop_n, 0.0), 1.0)

                # distance independent normalized values
                draw_feature_bars(
                    frame,
                    mw_v,
                    lg_v,
                    cv_v,
                    br_v
                )
            except Exception as e:
                # debug drawing must not break detection
                print("Debug draw error:", e)

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


