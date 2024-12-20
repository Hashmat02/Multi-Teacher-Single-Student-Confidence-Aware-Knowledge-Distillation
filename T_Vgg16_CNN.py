import torch
import torchvision.models as models
import torch.nn as nn

class VGG16Model:
    def __init__(self, num_classes, pretrained=True):
        self.model = models.vgg16(pretrained=pretrained)

        for param in self.model.parameters():
            param.requires_grad = False

        in_features = self.model.classifier[6].in_features 
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

        for param in self.model.classifier[6].parameters():
            param.requires_grad = True

    def get_model(self):
        return self.model

# num_classes = 10  
# vgg_model = VGG16Model(num_classes=num_classes, pretrained=True)
# model_instance = vgg_model.get_model()

# print(model_instance)
