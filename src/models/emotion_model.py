import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

# FER2013 class order (VERY IMPORTANT)
EMOTIONS = [
    "Angry",     # 0
    "Disgust",   # 1
    "Fear",      # 2
    "Happy",     # 3
    "Sad",       # 4
    "Surprise",  # 5
    "Neutral"    # 6
]


# ----------------------------
# VGG-style FER2013 model
# ----------------------------
class VGG_FER(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ----------------------------
# EmotionModel wrapper
# ----------------------------
class EmotionModel:
    def __init__(self, model_path: str):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.device = torch.device("cpu")

        self.model = VGG_FER().to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device),
            strict=False
        )

        self.model.eval()

    def preprocess(self, face_bgr):
        """
        FER2013 preprocessing
        """
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))

        gray = gray.astype(np.float32) / 255.0
        gray = torch.tensor(gray).unsqueeze(0).unsqueeze(0)

        return gray.to(self.device)

    @torch.no_grad()
    def predict(self, face_bgr):
        """
        Returns:
        - label (str)
        - confidence (float)
        - probs (dict)
        """

        if face_bgr is None or face_bgr.size == 0:
            return None, 0.0, None

        x = self.preprocess(face_bgr)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0]

        probs_np = probs.cpu().numpy()

        idx = int(np.argmax(probs_np))
        label = EMOTIONS[idx]
        confidence = float(probs_np[idx])

        prob_map = {
            EMOTIONS[i]: float(probs_np[i])
            for i in range(len(EMOTIONS))
        }

        return label, confidence, prob_map
