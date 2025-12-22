import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

# Label order for FER2013
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

class FERModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # Replace with the exact architecture that matches the .pth
        self.model = ... # define VGG19 or CNN
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, img):
        # img is PIL Image
        transform = T.Compose([
            T.Grayscale(),
            T.Resize((48,48)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
        x = transform(img).unsqueeze(0)  # batch
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1).squeeze()
            idx = torch.argmax(probs).item()
        return EMOTIONS[idx], probs[idx].item()
