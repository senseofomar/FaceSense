import cv2
import numpy as np

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


class EmotionModel:
    def __init__(self, model_path):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, face_img):
        """
        face_img: BGR face crop
        returns: (label, confidence)
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        gray = gray.astype("float32") / 255.0

        blob = cv2.dnn.blobFromImage(
            gray,
            scalefactor=1.0,
            size=(48, 48),
            mean=(0,),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        preds = self.net.forward()[0]

        idx = int(np.argmax(preds))
        label = EMOTIONS[idx]
        confidence = float(preds[idx])

        return label, confidence
