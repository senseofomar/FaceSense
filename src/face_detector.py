import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=1,  # Change to 1 for full-range (better for different distances)
            min_detection_confidence=0.5
        )

    def detect(self, image_bgr, padding_pct=0.15):
        """
        Detects face and returns a padded crop.
        padding_pct: Percentage of padding to add (0.15 = 15%)
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        if not results.detections:
            return None, None

        # Get the largest face if multiple are present
        det = max(results.detections, key=lambda d: d.location_data.relative_bounding_box.width)

        h, w, _ = image_bgr.shape
        box = det.location_data.relative_bounding_box

        # 1. Calculate base coordinates
        x1 = int(box.xmin * w)
        y1 = int(box.ymin * h)
        bw = int(box.width * w)
        bh = int(box.height * h)

        # 2. Add Padding (Context is crucial for emotions)
        dx = int(bw * padding_pct)
        dy = int(bh * padding_pct)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x1 + bw + (2 * dx))
        y2 = min(h, y1 + bh + (2 * dy))

        # 3. Crop
        face_crop = image_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None, None

        return face_crop, (x1, y1, x2, y2)