import torch
from torchvision import models
import torch.nn as nn

class ResNet50Model:
    def __init__(self, num_classes, pretrained=True):
        self.model = models.resnet50(pretrained=pretrained)
        
        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.fc.in_features  
        self.model.fc = nn.Linear(in_features, num_classes)

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def get_model(self):
        return self.model

# num_classes = 10 
# resnet_model = ResNet50Model(num_classes=num_classes, pretrained=True)
# model_instance = resnet_model.get_model()
# print(model_instance)
