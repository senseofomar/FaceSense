import torch
import torch.nn as nn
from torchvision import models


class FaceSenseVGG19(nn.Module):
    def __init__(self, num_classes=7):
        super(FaceSenseVGG19, self).__init__()
        # 1. Load the standard VGG19 features (convolutional layers)
        vgg19_base = models.vgg19(weights=None)
        self.features = vgg19_base.features

        # 2. Add Global Average Pooling to ensure output is 512
        # This is likely how your 512-input classifier was achieved
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 3. Define the Classifier (Matches your shape [7, 512])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- Loading the weights ---
model = FaceSenseVGG19(num_classes=7)
checkpoint = torch.load('fer2013_vgg19.pth', map_location=torch.device('cpu'))

# Since your file uses the 'net' key, extract it:
if 'net' in checkpoint:
    model.load_state_dict(checkpoint['net'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✅ Model loaded successfully!")