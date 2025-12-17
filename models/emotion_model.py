import cv2
import numpy as np
from tensorflow.keras.models import load_model

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, face_img):
        """
        face_img: BGR face crop
        returns: (label, confidence, probs)
        """

        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to FER2013 input size
        gray = cv2.resize(gray, (48, 48))

        # Normalize
        gray = gray / 255.0

        # Shape: (1, 48, 48, 1)
        gray = gray.reshape(1, 48, 48, 1)

        preds = self.model.predict(gray, verbose=0)[0]

        idx = np.argmax(preds)
        label = EMOTIONS[idx]
        confidence = float(preds[idx])

        return label, confidence, preds
