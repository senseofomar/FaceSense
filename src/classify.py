# src/classify.py
from __future__ import annotations
import os
import numpy as np

EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Surprise"]

class EmotionClassifier:
    """
    Minimal pluggable classifier.
    - For now: a simple heuristic based on brightness/contrast.
    - Later: load a real CNN (Keras/PyTorch) and replace `predict`.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self.backend = "dummy"

        # Example of how you’d load a real model (commented out for now):
        # if model_path and os.path.exists(model_path):
        #     from tensorflow.keras.models import load_model
        #     self.model = load_model(model_path)
        #     self.backend = "keras"

    def predict(self, face_batch: np.ndarray) -> list[str]:
        """
        face_batch: (N, H, W) grayscale normalized to [0, 1]
        Returns list of string labels.
        """
        if face_batch.size == 0:
            return []

        if self.backend == "dummy":
            labels = []
            for face in face_batch:
                mean = float(face.mean())
                std  = float(face.std())
                # Silly but useful to wire the pipeline:
                if mean > 0.55 and std > 0.18:
                    labels.append("Happy")
                elif mean < 0.30 and std < 0.12:
                    labels.append("Sad")
                elif std > 0.30:
                    labels.append("Surprise")
                else:
                    labels.append("Neutral")
            return labels

        # Example if you switch to Keras:
        # preds = self.model.predict(face_batch[..., None])  # add channel dim
        # idxs = preds.argmax(axis=1)
        # return [EMOTIONS[i] for i in idxs]

        return ["Neutral"] * face_batch.shape[0]
