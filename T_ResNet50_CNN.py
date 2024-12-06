# resnet50_model.py

import torch
from torchvision import models

def load_resnet50(pretrained=True):
    model = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
    model.eval() 
    return model