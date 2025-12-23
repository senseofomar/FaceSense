import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# FER2013 label order (this matters!)
EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

class EmotionModel:
    def __init__(self, model_path: str):
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FER model not found at:\n{model_path}")

        # --- Build VGG19 ---
        self.model = models.vgg19(weights=None)

        # Change input conv: 3 → 1 channel
        self.model.features[0] = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Change classifier head → 7 emotions
        self.model.classifier[6] = nn.Linear(4096, 7)

        # Load trained weights
        state = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(state, strict=True)

        self.model.eval()

        # Preprocessing MUST match training
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, image_path: str):
        """
        image_path: path to image file
        returns: (label, confidence, probs_dict)
        """
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        idx = int(torch.argmax(probs))
        confidence = float(probs[idx])

        probs_dict = {
            EMOTIONS[i]: float(probs[i])
            for i in range(len(EMOTIONS))
        }

        return EMOTIONS[idx], confidence, probs_dict
