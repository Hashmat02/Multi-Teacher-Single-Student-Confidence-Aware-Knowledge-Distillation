# vit_model.py

import torch
from torchvision import models

class ViTModel:
    def __init__(self, num_classes, pretrained=True):
        self.model = models.vit_b_16(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad = False
            
        in_features = self.model.heads.head.in_features 
        self.model.heads.head = torch.nn.Linear(in_features, num_classes)

        for param in self.model.heads.head.parameters():
            param.requires_grad = True

    def get_model(self):
        return self.model