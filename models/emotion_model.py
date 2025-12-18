import cv2
import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class EmotionModel:
    def __init__(self, model_path: str):
        """
        model_path: path to FER2013 ONNX model
        """
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def pre_process(self, face_img):
        """
        face_img: BGR face crop
        returns: blob ready for DNN
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        gray = gray.astype(np.float32) / 255.0

        # Shape: (1, 1, 48, 48)
        blob = np.expand_dims(gray, axis=(0, 1))
        return blob

    def predict(self, face_img):
        """
        returns: (label, confidence, probs)
        """
        blob = self.pre_process(face_img)

        self.net.setInput(blob)
        preds = self.net.forward()[0]

        idx = int(np.argmax(preds))
        label = EMOTIONS[idx]
        confidence = float(preds[idx])

        return label, confidence, preds
