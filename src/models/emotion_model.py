import cv2
import numpy as np

EMOTIONS = [
    "Neutral",   # 0
    "Happy",     # 1
    "Surprise",  # 2
    "Sad",       # 3
    "Angry",     # 4
    "Disgust",   # 5
    "Fear"       # 6
]

def softmax(x):
    x = x - np.max(x)  # numerical stability
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class EmotionModel:
    def __init__(self, model_path: str):
        # Load ONNX model
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def predict(self, face_bgr):
        """
        face_bgr: BGR face crop from OpenCV
        returns: (label, confidence)
        """

        # 1. Convert to grayscale
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 2. Resize to FER+ expected size
        gray = cv2.resize(gray, (64, 64))

        # 3. Normalize to [0,1]
        gray = gray.astype(np.float32) / 255.0

        # 4. Create DNN blob (NCHW)
        blob = gray.reshape(1, 1, 64, 64)

        # 5. Feed network
        self.net.setInput(blob)
        raw = self.net.forward()

        # 6. FIX: flatten output safely
        logits = raw.reshape(-1)

        probs = softmax(logits)
        # Ignore Neutral when competing
        neutral_idx = EMOTIONS.index("Neutral")
        probs_wo_neutral = probs.copy()
        probs_wo_neutral[neutral_idx] = 0.0

        best_idx = int(np.argmax(probs_wo_neutral))
        best_conf = float(probs_wo_neutral[best_idx])

        # Threshold gating
        if best_conf < 0.35:
            return "Neutral", float(probs[neutral_idx])

        return EMOTIONS[best_idx], best_conf


