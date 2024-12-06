# vgg16_model.py

import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def load_vgg16(pretrained=True):
    model = models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
    model.eval() 
    return model