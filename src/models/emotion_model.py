from deepface import DeepFace
import cv2
import numpy as np


class EmotionModel:
    def __init__(self, model_name="sfer"):
        # "race" and "gender" are disabled to speed it up
        # "sfer" (Simplified FER) is fast and accurate, or use "emotion" (default)
        self.model_name = "DeepFace"

    def predict(self, face_bgr):
        # DeepFace expects path or numpy array
        try:
            # enforce_detection=False because we already cropped the face
            objs = DeepFace.analyze(
                img_path=face_bgr,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
        except ValueError:
            return "Unknown", 0.0, {}

        result = objs[0]
        dominant_emotion = result['dominant_emotion']
        confidence = result['emotion'][dominant_emotion]

        # Normalize probs to 0-1 range if they aren't
        probs = result['emotion']

        return dominant_emotion, confidence, probs