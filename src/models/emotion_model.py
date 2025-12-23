# src/models/emotion_model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import os

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class EmotionModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FER model not found at:\n{model_path}")

        # Load checkpoint
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "net" in state:
            state = state["net"]

        # Build EXACT matching model
        base = models.vgg19_bn(weights=None)

        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, 7)

        self.load_state_dict(state, strict=True)
        self.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def predict(self, face_bgr):
        face_rgb = face_bgr[:, :, ::-1]
        img = Image.fromarray(face_rgb)

        x = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1).squeeze()

        idx = probs.argmax().item()
        return EMOTIONS[idx], float(probs[idx]), probs.tolist()
