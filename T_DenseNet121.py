# densenet_model.py

import torch
from torchvision import models

def load_densenet121(pretrained=True):
    model = models.densenet121(pretrained=pretrained) 
    model.eval() 
    return model