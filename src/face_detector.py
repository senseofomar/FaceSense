import cv2
import numpy as np


class FaceDetector:
    def __init__(self):
        # Using OpenCV's built-in Haar Cascade detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, image_bgr, padding_pct=0.15):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None, None

        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])

        # Add Padding
        dx = int(w * padding_pct)
        dy = int(h * padding_pct)

        # Calculate coordinates with padding and boundary checks
        img_h, img_w = image_bgr.shape[:2]
        y1 = max(0, y - dy)
        y2 = min(img_h, y + h + dy)
        x1 = max(0, x - dx)
        x2 = min(img_w, x + w + dx)

        face_crop = image_bgr[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2 - x1, y2 - y1)