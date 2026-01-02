import torch
import torch.nn as nn
from torchvision import models

class FaceSenseVGG19(nn.Module):
    def __init__(self, num_classes=7):
        super(FaceSenseVGG19, self).__init__()
        # 1. Use Pre-trained weights (Transfer Learning) - HUGE Accuracy Boost
        # Instead of weights=None, we use ImageNet weights as a starting point.
        weights = models.VGG19_BN_Weights.DEFAULT
        vgg19_base = models.vgg19_bn(weights=weights)

        self.features = vgg19_base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Add Dropout to the classifier
        self.classifier = nn.Sequential(
            nn.Identity(),  # classifier.0
            nn.Linear(512, num_classes)  # classifier.1 (This matches your error!)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

