import timm
import torch
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ViTClassifier, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)

def load_vit_model(num_classes, device):
    model = ViTClassifier(num_classes)
    model = model.to(device)
    return model