# vit_model.py

import torch
from torchvision import models

def load_vit(pretrained=True):
    model = models.vit_b_16(pretrained=pretrained) 
    model.eval()  
    return model