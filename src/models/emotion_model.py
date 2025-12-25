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
        # ... (Keep your VGG loading code exactly as is) ...
        # (Assuming your VGG loading logic matches your specific .pth file)

        # --- FIX: New Transform ---
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            # REMOVED ImageNet Normalization. 
            # Most FER models just want 0-1 tensors (which ToTensor does).
            # If your model specifically expects Grayscale, you might need T.Grayscale(num_output_channels=3)
        ])

        # If the model continues to fail, try uncommenting this normalization:
        # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def predict(self, face_bgr):
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb)

        # Transform
        x = self.transform(img).unsqueeze(0)

        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1).squeeze()

        idx = probs.argmax().item()
        return EMOTIONS[idx], float(probs[idx]), probs.tolist()