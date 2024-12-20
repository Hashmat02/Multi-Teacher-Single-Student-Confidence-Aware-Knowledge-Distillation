import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torchvision.models import vgg11
from torchvision import models


class VGG11(nn.Module):
    def __init__(self, num_classes=16, pretrained=True):
        super(VGG11, self).__init__()
        self.vgg11 = models.vgg11(pretrained=pretrained)

        for param in self.vgg11.features.parameters():
            param.requires_grad = False
        self.vgg11.classifier[6] = nn.Linear(self.vgg11.classifier[6].in_features, num_classes)
    
    def forward(self, x):
        return self.vgg11(x)

    def get_model(self):
        return self.vgg11

