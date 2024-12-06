# resnet50_model.py

import torch
from torchvision import models

def load_resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained) 
    model.eval() 
    return model