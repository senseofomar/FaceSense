import cv2
import numpy as np

EMOTIONS = [
    "Neutral",
    "Happy",
    "Surprise",
    "Sad",
    "Angry",
    "Disgust",
    "Fear"
]


class EmotionModel:
    def __init__(self, model_path: str):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, face_bgr):
        # Grayscale
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        # Resize to FER+ input
        gray = cv2.resize(gray, (64, 64))

        # Normalize
        gray = gray.astype(np.float32) / 255.0

        # Shape: (1, 1, 64, 64)
        blob = gray.reshape(1, 1, 64, 64)

        self.net.setInput(blob)
        preds = self.net.forward()

        probs = preds.flatten()
        probs = probs / probs.sum()  # safety normalize

        idx = int(np.argmax(probs))
        return EMOTIONS[idx], float(probs[idx]), probs
