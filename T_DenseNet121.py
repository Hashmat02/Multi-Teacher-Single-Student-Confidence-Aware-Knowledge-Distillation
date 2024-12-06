# densenet_model.py

import torch
from torchvision import models

def load_densenet121(pretrained=True):
    model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
    model.eval() 
    return model