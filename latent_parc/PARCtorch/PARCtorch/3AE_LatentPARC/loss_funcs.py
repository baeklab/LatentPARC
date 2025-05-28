import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
import torch.nn.functional as F


class LpLoss(torch.nn.Module):
    def __init__(self, p=10):
        super(LpLoss, self).__init__()
        self.p = p

    def forward(self, input, target):
        # Compute element-wise absolute difference
        diff = torch.abs(input - target)
        # Raise the differences to the power of p, sum them, and raise to the power of 1/p
        return (torch.sum(diff ** self.p) ** (1 / self.p))

    # PERCEPTUAL LOSS FUNCTIONS
# Load Pretrained Feature Extractor (VGG19)
class FeatureExtractor(nn.Module):
    def __init__(self, layers=[2, 7, 12]):  # Example layers: relu1_2, relu2_2, relu3_2
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features  # Updated weights usage
        self.selected_layers = layers
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:max(layers) + 1])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

# Perceptual Loss Function
class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, loss_fn=nn.L1Loss()):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn  # Can use L1 or MSE loss

    def forward(self, input, target):
        with torch.no_grad():  # Save memory by disabling gradient computation
            input_features = self.feature_extractor(input)
            target_features = self.feature_extractor(target)
        
        loss = sum(self.loss_fn(inp, tgt) for inp, tgt in zip(input_features, target_features))
        return loss