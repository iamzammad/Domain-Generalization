import timm
import torch
import torch.nn as nn


class EfficientNetB3Model(nn.Module):
    def __init__(self):
        super(EfficientNetB3Model, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=True)

    def forward(self, x):
        x = self.model(x)
        return x

def load_efficientnetb3_model(num_classes, device):
    model = ViTClassifier(num_classes)
    model = model.to(device)
    return model
