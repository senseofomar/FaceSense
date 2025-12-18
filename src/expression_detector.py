import cv2
import mediapipe as mp
import numpy as np


class FaceSense:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def get_face(self, frame):
        """
        Returns:
        - face_crop (BGR) or None
        - bbox (x1, y1, x2, y2) or None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None, None

        h, w, _ = frame.shape
        det = results.detections[0]
        box = det.location_data.relative_bounding_box

        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        x2 = x1 + bw
        y2 = y1 + bh

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Tighten crop slightly (important for FER models)
        pad = int(0.1 * (x2 - x1))
        x1 = max(0, x1 + pad)
        y1 = max(0, y1 + pad)
        x2 = min(w, x2 - pad)
        y2 = min(h, y2 - pad)


        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None

        return face_crop, (x1, y1, x2, y2)
