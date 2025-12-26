import torch
import torch.nn as nn
from torchvision import models


class FaceSenseVGG19(nn.Module):
    def __init__(self, num_classes=7):
        super(FaceSenseVGG19, self).__init__()
        # Use vgg19_bn (Batch Normalization) instead of standard vgg19
        # This matches the "running_mean" and "running_var" keys in your file
        vgg19_base = models.vgg19_bn(weights=None)

        self.features = vgg19_base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # This matches the [7, 512] shape we confirmed earlier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x