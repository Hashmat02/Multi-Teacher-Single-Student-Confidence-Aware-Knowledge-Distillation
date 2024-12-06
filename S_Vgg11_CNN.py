import torch
from torchvision import models

def load_vgg11(pretrained=True):
    model = models.vgg11(pretrained=pretrained)
    model.eval() 
    return model