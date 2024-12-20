import torch
from torchvision import models
import torch.nn as nn

class DenseNet121Model:
    def __init__(self, num_classes, pretrained=True):
        self.model = models.densenet121(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad = False
            
        in_features = self.model.classifier.in_features 
        self.model.classifier = nn.Linear(in_features, num_classes)

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def get_model(self):
        return self.model

# num_classes = 10  
# densenet_model = DenseNet121Model(num_classes=num_classes, pretrained=True)
# model_instance = densenet_model.get_model()

# print(model_instance)
