# vit_model.py

import torch
from torchvision import models

def load_vit(pretrained=True):
    model = models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None)
    model.eval()  
    return model